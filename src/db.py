"""Lazy, reusable MongoDB client. No network activity at module import."""

from threading import Lock

from pymongo import MongoClient

from src.config import Settings

_client: MongoClient | None = None
_lock = Lock()


def get_mongo_client() -> MongoClient:
    global _client
    with _lock:
        if _client is None:
            settings = Settings.from_env()
            uri = settings.mongodb_uri.get_secret_value()
            if not uri:
                raise ValueError("MONGODB_URI is required for database commands")
            candidate = MongoClient(
                uri,
                timeoutMS=settings.mongodb_timeout_ms,
                serverSelectionTimeoutMS=settings.mongodb_timeout_ms,
                connectTimeoutMS=settings.mongodb_timeout_ms,
                socketTimeoutMS=settings.mongodb_timeout_ms,
                retryWrites=True,
                w="majority",
                tz_aware=True,
                appname="altur-pipeline",
            )
            try:
                candidate.admin.command("ping")
            except Exception:
                candidate.close()
                raise
            _client = candidate
        return _client


def close_mongo_client() -> None:
    global _client
    with _lock:
        if _client is not None:
            _client.close()
            _client = None


def get_calls_collection():
    settings = Settings.from_env()
    return get_mongo_client()[settings.mongodb_database][settings.mongodb_collection]


def init_db_indexes() -> None:
    from src.repositories import MongoAuditRepository

    MongoAuditRepository(get_calls_collection()).ensure_indexes()
