"""MongoDB persistence; raw events are immutable and replay cannot reset labels."""

import hashlib
import json
from datetime import UTC, datetime
from typing import Protocol

from pymongo import ASCENDING, DESCENDING
from pymongo.errors import DuplicateKeyError

from src.schemas import CallAuditRecord


class EventConflict(ValueError):
    """The same event ID was reused with different content."""


class AuditRepository(Protocol):
    def save(self, record: CallAuditRecord) -> bool:
        """Return True for an insert, False for an identical replay."""
        ...


class MongoAuditRepository:
    def __init__(self, collection):
        self.collection = collection

    def ensure_indexes(self) -> None:
        self.collection.create_index([("event_id", ASCENDING)], unique=True)
        self.collection.create_index([("call_id", ASCENDING)])
        self.collection.create_index(
            [("status_for_training", ASCENDING), ("timestamp", DESCENDING)]
        )
        self.collection.create_index([("input_sha256", ASCENDING)])

    def save(self, record: CallAuditRecord) -> bool:
        record = CallAuditRecord.model_validate(record.model_dump())
        canonical = json.dumps(record.model_dump(mode="json"), sort_keys=True, allow_nan=False)
        digest = hashlib.sha256(canonical.encode()).hexdigest()
        payload = record.model_dump(mode="python")
        payload.update(
            _id=record.event_id,
            payload_sha256=digest,
            ingested_at=datetime.now(UTC),
            processing_status="stored",
        )
        try:
            result = self.collection.update_one(
                {"_id": record.event_id}, {"$setOnInsert": payload}, upsert=True
            )
        except DuplicateKeyError:
            result = None
        if result is not None and result.upserted_id is not None:
            return True
        existing = self.collection.find_one({"_id": record.event_id}, {"payload_sha256": 1})
        if not existing or existing.get("payload_sha256") != digest:
            raise EventConflict("event ID reused with different content")
        return False

    def training_records(self, since: datetime | None = None, until: datetime | None = None):
        # Include unlabelled records: they can connect identities across labelled samples.
        # Readiness filtering belongs to dataset preparation, after grouping.
        query = {}
        if since is not None or until is not None:
            query["timestamp"] = {}
            if since is not None:
                query["timestamp"]["$gte"] = since
            if until is not None:
                query["timestamp"]["$lt"] = until
        projection = {k: 0 for k in ("_id", "payload_sha256", "ingested_at", "processing_status")}
        yield from self.collection.find(query, projection).sort("event_id", ASCENDING)
