"""Verified-label selection. Predictions are never promoted to ground truth."""

from src.data_processing import prepare_dataset
from src.db import get_calls_collection
from src.repositories import MongoAuditRepository


def select_training_batch(max_per_class: int = 100, repository=None) -> list[dict]:
    """Compatibility helper; partitions must still be prepared before fitting."""
    if max_per_class < 1:
        raise ValueError("max_per_class must be positive")
    repository = repository or MongoAuditRepository(get_calls_collection())
    dataset = prepare_dataset(repository.training_records())
    selected = []
    counts = {0: 0, 1: 0}
    for row in dataset["splits"]["train"]:
        if counts[row["label"]] < max_per_class:
            selected.append(row)
            counts[row["label"]] += 1
    return selected
