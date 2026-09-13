"""Single-consumer durable spool worker. Run with python -m src.worker."""

import argparse
import hashlib
import logging
import time
from datetime import UTC, datetime
from threading import Event

from pydantic import ValidationError
from pymongo.errors import AutoReconnect, NetworkTimeout, PyMongoError

from src.config import Settings
from src.db import close_mongo_client, get_calls_collection
from src.repositories import AuditRepository, EventConflict, MongoAuditRepository
from src.schemas import CallAuditRecord
from src.storage import exclusive_lock, write_json

logger = logging.getLogger("altur.worker")


def process_record(raw_line: str, repository: AuditRepository) -> bool:
    """Validate and persist; failures propagate so callers cannot acknowledge them."""
    return repository.save(CallAuditRecord.model_validate_json(raw_line))


def process_pending_logs(
    repository: AuditRepository | None = None,
    settings: Settings | None = None,
    *,
    sleep=time.sleep,
) -> int:
    """Process complete spool JSON files. Return acknowledged events, excluding failures."""
    settings = settings or Settings.from_env()
    with exclusive_lock(settings.processed_dir / "worker.lock"):
        if repository is None:
            repository = MongoAuditRepository(get_calls_collection())
            repository.ensure_indexes()
        completed = 0
        attempted = 0
        for path in sorted(settings.spool_dir.glob("*.json")):
            if path.is_symlink():
                continue
            receipt = settings.processed_dir / "receipts" / path.name
            rejected = settings.processed_dir / "rejected" / path.name
            if receipt.exists() or rejected.exists():
                continue
            if attempted >= settings.batch_size:
                break
            attempted += 1
            started = time.monotonic()
            try:
                with path.open("rb") as handle:
                    raw = handle.read(settings.max_json_bytes + 1)
                if len(raw) > settings.max_json_bytes:
                    raise ValueError("JSON exceeds configured limit")
                record = CallAuditRecord.model_validate_json(raw)
            except (ValidationError, ValueError) as exc:
                write_json(rejected, {"status": "invalid", "error_type": type(exc).__name__})
                logger.warning("event=validation_failed file=%s", path.name)
                continue
            for attempt in range(settings.retry_attempts):
                try:
                    inserted = repository.save(record)
                    write_json(
                        receipt,
                        {
                            "status": "stored",
                            "processed_at": datetime.now(UTC).isoformat(),
                            "event_id": record.event_id,
                            "raw_sha256": hashlib.sha256(raw).hexdigest(),
                            "record": record.model_dump(mode="json"),
                        },
                    )
                    completed += 1
                    logger.info(
                        "event=stored document=%s inserted=%s elapsed_ms=%.1f",
                        record.event_id,
                        inserted,
                        (time.monotonic() - started) * 1000,
                    )
                    break
                except EventConflict:
                    write_json(rejected, {"status": "conflict", "event_id": record.event_id})
                    logger.error("event=id_conflict document=%s", record.event_id)
                    break
                except (AutoReconnect, NetworkTimeout) as exc:
                    logger.warning(
                        "event=retry document=%s attempt=%d error_type=%s",
                        record.event_id,
                        attempt + 1,
                        type(exc).__name__,
                    )
                    if attempt + 1 == settings.retry_attempts:
                        raise
                    sleep(settings.retry_seconds * (2**attempt))
                except PyMongoError as exc:
                    # Authentication, validation or write-concern errors require inspection.
                    # Keep the source pending, including ambiguous acknowledgements.
                    logger.error("event=database_failed error_type=%s", type(exc).__name__)
                    raise
        return completed


def start_worker(*, once: bool = False) -> None:
    import signal

    from dotenv import load_dotenv

    load_dotenv()
    settings = Settings.from_env()
    logging.basicConfig(
        level=settings.log_level, format="%(asctime)s %(levelname)s %(name)s %(message)s"
    )
    stop = Event()
    for name in (signal.SIGINT, signal.SIGTERM):
        signal.signal(name, lambda *_: stop.set())
    try:
        while not stop.is_set():
            try:
                count = process_pending_logs(settings=settings, sleep=stop.wait)
                logger.info("event=batch_completed stored=%d", count)
            except (PyMongoError, OSError, ValueError) as exc:
                logger.error("event=batch_failed error_type=%s", type(exc).__name__)
                if once:
                    raise SystemExit(1) from None
            if once:
                break
            stop.wait(settings.poll_seconds)
    finally:
        close_mongo_client()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--once", action="store_true")
    start_worker(once=parser.parse_args().once)
