"""Export immutable dataset snapshots from MongoDB without changing source documents."""

import argparse
import json
import logging
from datetime import UTC, datetime

from src.config import Settings
from src.data_processing import prepare_dataset
from src.db import close_mongo_client, get_calls_collection
from src.repositories import MongoAuditRepository
from src.storage import exclusive_lock, write_json


def export_active_learning_batch(repository=None, settings=None, *, until=None):
    settings = settings or Settings.from_env()
    repository = repository or MongoAuditRepository(get_calls_collection())
    cutoff = until or datetime.now(UTC)
    with exclusive_lock(settings.training_dir / "export.lock"):
        registry_path = settings.training_dir / "split_registry.json"
        registry = json.loads(registry_path.read_text()) if registry_path.exists() else {}
        dataset = prepare_dataset(
            repository.training_records(until=cutoff), split_registry=registry
        )
        destination = settings.training_dir / dataset["dataset_id"]
        # Freeze identities before publishing a dataset that can be consumed by training.
        write_json(registry_path, dataset["split_registry"])
        if not (destination / "dataset.json").exists():
            write_json(
                destination / "manifest.json",
                {
                    "created_at": datetime.now(UTC).isoformat(),
                    "cutoff": cutoff.isoformat(),
                    "dataset_id": dataset["dataset_id"],
                    "source_events": dataset["source_events"],
                    "distribution": dataset["distribution"],
                    "excluded": dataset["excluded"],
                    "pipeline_version": settings.pipeline_version,
                },
            )
            write_json(destination / "dataset.json", dataset)
    logging.getLogger(__name__).info(
        "event=dataset_exported dataset=%s source_events=%d",
        dataset["dataset_id"],
        dataset["source_events"],
    )
    return destination / "dataset.json"


def main() -> None:
    from dotenv import load_dotenv

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--until", type=datetime.fromisoformat)
    args = parser.parse_args()
    if args.until is not None and args.until.tzinfo is None:
        parser.error("--until requires a timezone")
    load_dotenv()
    logging.basicConfig(level=Settings.from_env().log_level)
    try:
        export_active_learning_batch(until=args.until)
    finally:
        close_mongo_client()


if __name__ == "__main__":
    main()
