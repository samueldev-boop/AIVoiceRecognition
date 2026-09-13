"""Persistencia en MongoDB; los eventos son inmutables y repetirlos no los altera."""

import hashlib
import json
import logging
from datetime import UTC, datetime
from typing import Protocol

from pymongo import ASCENDING
from pymongo.errors import DuplicateKeyError, OperationFailure

from src.schemas import CallAuditRecord

logger = logging.getLogger("altur.repositories")

# IndexOptionsConflict e IndexKeySpecsConflict: el indice ya existe con otras opciones.
INDEX_CONFLICTS = {85, 86}


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
        # Parcial: la coleccion puede tener documentos del auditor anterior sin event_id, y
        # un indice unico total los trataria a todos como event_id null repetido.
        self._create_index(
            [("event_id", ASCENDING)],
            unique=True,
            partialFilterExpression={"event_id": {"$exists": True}},
        )
        self._create_index([("call_id", ASCENDING)])
        self._create_index([("timestamp", ASCENDING)])
        self._create_index([("input_sha256", ASCENDING)])

    def _create_index(self, keys, **options) -> None:
        """Respeta un indice que ya existe con otras opciones en vez de abortar el worker.

        La coleccion del auditor anterior tiene call_id_1 unico. Pedir el mismo indice sin
        unique falla con IndexOptionsConflict (85) o IndexKeySpecsConflict (86), y eso
        dejaba al worker sin guardar nada. La idempotencia no depende de estos indices:
        _id es event_id.
        """
        try:
            self.collection.create_index(keys, **options)
        except OperationFailure as exc:
            if exc.code not in INDEX_CONFLICTS:
                raise
            logger.warning("event=index_kept keys=%s code=%s", keys, exc.code)

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
