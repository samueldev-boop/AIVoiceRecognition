"""Pipeline settings; importing this module performs no I/O or database connection."""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel, Field, SecretStr


class Settings(BaseModel):
    mongodb_uri: SecretStr = SecretStr("")
    audit_id_key: SecretStr = SecretStr("")
    mongodb_database: str = "altur_defense"
    mongodb_collection: str = "calls_v1"
    mongodb_timeout_ms: int = Field(default=5000, ge=100, le=120000)
    spool_dir: Path = Path("data/raw")
    # JSONL del auditor anterior: solo lo lee la migracion explicita, la API ya no escribe ahi.
    legacy_audit_log: Path = Path("data/audit.jsonl")
    processed_dir: Path = Path("data/processed")
    training_dir: Path = Path("data/training")
    model_dir: Path = Path("model/candidates")
    poll_seconds: float = Field(default=2, gt=0, le=300)
    retry_attempts: int = Field(default=3, ge=1, le=10)
    retry_seconds: float = Field(default=1, ge=0, le=60)
    batch_size: int = Field(default=100, ge=1, le=10000)
    max_json_bytes: int = Field(default=1048576, ge=1024, le=8388608)
    environment: str = "development"
    log_level: str = "INFO"
    pipeline_version: str = "audit-v1"
    model_version: str = "unknown"

    @classmethod
    def from_env(cls) -> Settings:
        aliases = {
            # Los nombres MONGO_* y COLLECTION_CALLS son los del .env del equipo.
            "mongodb_uri": ("MONGODB_URI", "MONGO_URI"),
            "mongodb_database": ("MONGODB_DATABASE", "MONGO_DB", "DATABASE_NAME"),
            "mongodb_collection": ("MONGODB_COLLECTION", "COLLECTION_CALLS", "COLLECTION_NAME"),
            "spool_dir": ("AUDIT_SPOOL_DIR",),
            "legacy_audit_log": ("AUDIT_LOG_PATH",),
            "poll_seconds": ("WORKER_POLL_SECONDS",),
            "retry_attempts": ("WORKER_RETRY_ATTEMPTS",),
            "retry_seconds": ("WORKER_RETRY_SECONDS",),
            "batch_size": ("WORKER_BATCH_SIZE",),
        }
        values = {}
        for field in cls.model_fields:
            for name in aliases.get(field, (field.upper(),)):
                if name in os.environ:
                    values[field] = os.environ[name]
                    break
        return cls(**values)
