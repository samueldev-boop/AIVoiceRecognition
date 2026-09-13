"""Explicit JSONL migration. Originals stay untouched and legacy labels are not trusted."""

import argparse
import hashlib
import json
import logging
from pathlib import Path

from src.config import Settings
from src.schemas import CallAuditRecord
from src.storage import atomic_write, exclusive_lock


def normalize_legacy_record(raw: dict) -> CallAuditRecord:
    if "schema_version" in raw:
        return CallAuditRecord.model_validate(raw)
    digest = hashlib.sha256(json.dumps(raw, sort_keys=True, allow_nan=False).encode()).hexdigest()
    analysis = raw.get("analysis") or {}
    # Earlier versions fabricated turns/acoustics/ASR; do not certify those as measurements.
    return CallAuditRecord(
        event_id=digest,
        call_id=hashlib.sha256(str(raw.get("call_id", digest)).encode()).hexdigest(),
        timestamp=raw["timestamp"],
        source="legacy-jsonl",
        decision=raw.get("decision") or raw.get("prediction"),
        analysis={
            k: analysis[k] for k in ("stage", "latency_s", "audio_duration_s") if k in analysis
        },
        metadata={"legacy_payload_sha256": digest},
    )


def migrate(path: Path, settings: Settings) -> tuple[int, int]:
    published = rejected = 0
    with exclusive_lock(settings.processed_dir / "migration.lock"):
        with path.open("rb") as source:
            line_number = 0
            while line := source.readline(settings.max_json_bytes + 1):
                line_number += 1
                if len(line) > settings.max_json_bytes:
                    # Stop at oversized lines: do not interpret the remainder as another event.
                    raise ValueError("legacy JSONL line exceeds configured size")
                if not line.strip():
                    continue
                try:
                    record = normalize_legacy_record(json.loads(line.decode("utf-8-sig")))
                    destination = settings.spool_dir / f"{record.event_id}.json"
                    content = record.model_dump_json().encode()
                    if destination.exists() and destination.read_bytes() != content:
                        raise ValueError("migration event ID conflict")
                    if not destination.exists():
                        atomic_write(destination, content)
                    published += 1
                except (ValueError, KeyError, TypeError) as exc:
                    rejected += 1
                    logging.warning(
                        "event=migration_rejected line=%d error_type=%s",
                        line_number,
                        type(exc).__name__,
                    )
    return published, rejected


def main() -> None:
    from dotenv import load_dotenv

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "jsonl", type=Path, nargs="?", help="JSONL heredado; por defecto AUDIT_LOG_PATH"
    )
    args = parser.parse_args()
    load_dotenv()
    settings = Settings.from_env()
    logging.basicConfig(level=settings.log_level)
    published, rejected = migrate(args.jsonl or settings.legacy_audit_log, settings)
    logging.info("event=migration_completed published=%d rejected=%d", published, rejected)
    if rejected:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
