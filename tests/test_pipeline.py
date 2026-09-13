"""Referencias a llamadas inciertas: contrato, worker y repositorio con ficheros y fakes."""

import copy
import hashlib
import importlib
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from pydantic import ValidationError
from pymongo.errors import AutoReconnect

from src.config import Settings
from src.repositories import EventConflict, MongoAuditRepository
from src.schemas import CallAuditRecord
from src.storage import exclusive_lock, write_json
from src.worker import process_pending_logs


def document(i=0):
    return {
        "event_id": f"event-{i}",
        "call_id": f"call-{i}",
        "source": "test",
        "timestamp": "2026-09-13T00:00:00Z",
        "input_sha256": hashlib.sha256(str(i).encode()).hexdigest(),
        "decision": {"is_synthetic": True, "confidence": 0.6, "probability_synthetic": 0.6},
        "analysis": {
            "stage": "full",
            "audio_duration_s": 10,
            "turns": [{"channel": 0, "start": 0, "end": 2}],
            "uncertainty_reasons": ["probabilidad_ambigua"],
        },
    }


@pytest.fixture
def settings(tmp_path):
    return Settings(
        spool_dir=tmp_path / "raw",
        processed_dir=tmp_path / "processed",
        retry_seconds=0,
    )


@pytest.mark.parametrize(
    "change",
    [
        {"event_id": "../escape"},
        {"schema_version": 2},
        {"timestamp": "2026-01-01"},
        {"decision": {"is_synthetic": True, "confidence": float("nan")}},
        {"analysis": {"turns": [{"channel": 0, "start": 2, "end": 1}]}},
        {"label": {"value": "human"}},
        {"status_for_training": "ready"},
        {"audio_uri": "s3://bucket/a?token=secret"},
        {"password": "must-not-be-accepted"},
    ],
)
def test_invalid_json_contract(change):
    with pytest.raises(ValidationError):
        CallAuditRecord.model_validate(document() | change)


def test_worker_keeps_raw_and_acknowledges_only_after_retry(settings):
    write_json(settings.spool_dir / "a.json", document())
    before = (settings.spool_dir / "a.json").read_bytes()
    repository = Mock()
    repository.save.side_effect = [AutoReconnect("secret must not be logged"), True]
    assert process_pending_logs(repository, settings, sleep=lambda _: None) == 1
    assert repository.save.call_count == 2
    assert (settings.spool_dir / "a.json").read_bytes() == before
    assert process_pending_logs(repository, settings) == 0


def test_worker_outage_survives_restart(settings, caplog):
    write_json(settings.spool_dir / "a.json", document())
    repository = Mock()
    repository.save.side_effect = AutoReconnect("private-uri")
    with pytest.raises(AutoReconnect):
        process_pending_logs(repository, settings, sleep=lambda _: None)
    assert not (settings.processed_dir / "receipts/a.json").exists()
    assert "private-uri" not in caplog.text
    repository.save.side_effect = None
    repository.save.return_value = False  # MongoDB may have committed before the timeout.
    assert process_pending_logs(repository, settings) == 1


def test_invalid_file_does_not_block_following_event(settings):
    settings.spool_dir.mkdir()
    (settings.spool_dir / "a.json").write_text("{broken")
    (settings.spool_dir / ".pending-unfinished").write_text("{unfinished")
    write_json(settings.spool_dir / "b.json", document())
    assert process_pending_logs(Mock(), settings) == 1
    assert (settings.processed_dir / "rejected/a.json").exists()
    assert (settings.spool_dir / "a.json").read_text() == "{broken"


def test_worker_conflicting_event_is_quarantined(settings):
    write_json(settings.spool_dir / "a.json", document())
    repository = Mock()
    repository.save.side_effect = EventConflict()
    assert process_pending_logs(repository, settings) == 0
    assert (settings.processed_dir / "rejected/a.json").exists()


def test_only_one_worker_can_process_spool(settings):
    with exclusive_lock(settings.processed_dir / "worker.lock"):
        with pytest.raises(OSError):
            process_pending_logs(Mock(), settings)


class FakeCollection:
    def __init__(self):
        self.docs = {}

    def update_one(self, query, update, upsert):
        key = query["_id"]
        exists = key in self.docs
        if not exists:
            self.docs[key] = copy.deepcopy(update["$setOnInsert"])
        return SimpleNamespace(upserted_id=None if exists else key)

    def find_one(self, query, projection):
        return self.docs.get(query["_id"])


def test_repository_replay_cannot_reset_curated_document():
    collection = FakeCollection()
    repository = MongoAuditRepository(collection)
    record = CallAuditRecord.model_validate(document())
    assert repository.save(record)
    collection.docs[record.event_id]["revisado"] = True
    assert not repository.save(record)
    assert collection.docs[record.event_id]["revisado"] is True
    changed = record.model_copy(update={"source": "different"})
    with pytest.raises(EventConflict):
        repository.save(changed)


def test_existing_index_with_other_options_does_not_stop_the_worker():
    from pymongo.errors import OperationFailure

    class LegacyIndexes:
        def __init__(self):
            self.created = []

        def create_index(self, keys, **options):
            if keys == [("call_id", 1)]:
                raise OperationFailure("An existing index has the same name", code=86)
            self.created.append(keys)

    collection = LegacyIndexes()
    MongoAuditRepository(collection).ensure_indexes()
    assert [("input_sha256", 1)] in collection.created, "los indices siguientes se crean"


def test_other_index_errors_still_fail():
    from pymongo.errors import OperationFailure

    class Unauthorized:
        def create_index(self, keys, **options):
            raise OperationFailure("not authorized", code=13)

    with pytest.raises(OperationFailure):
        MongoAuditRepository(Unauthorized()).ensure_indexes()


def test_failed_client_is_closed_and_next_call_retries(monkeypatch):
    import src.db as db

    monkeypatch.setenv("MONGODB_URI", "mongodb://localhost")
    monkeypatch.setattr(db, "_client", None)
    bad, good = Mock(), Mock()
    bad.admin.command.side_effect = AutoReconnect()
    factory = Mock(side_effect=[bad, good])
    monkeypatch.setattr(db, "MongoClient", factory)
    with pytest.raises(AutoReconnect):
        db.get_mongo_client()
    bad.close.assert_called_once()
    assert db.get_mongo_client() is good
    assert db.get_mongo_client() is good
    assert factory.call_count == 2
    db.close_mongo_client()
    good.close.assert_called_once()


def test_pipeline_imports_do_not_connect(monkeypatch):
    import src.db as db

    factory = Mock(side_effect=AssertionError("network on import"))
    monkeypatch.setattr(db, "get_calls_collection", factory)
    for name in ("src.worker", "src.migrate_audit"):
        importlib.reload(importlib.import_module(name))
    factory.assert_not_called()


def test_audit_publishes_true_probability_without_inventing_features(tmp_path, monkeypatch):
    from app.audit import log_detection_event

    monkeypatch.setenv("AUDIT_SPOOL_DIR", str(tmp_path))
    identifier = log_detection_event(confidence=0.95, probability_synthetic=0.05)
    record = CallAuditRecord.model_validate_json((tmp_path / f"{identifier}.json").read_bytes())
    assert record.decision.probability_synthetic == 0.05
    assert record.analysis.turns == []
    assert record.audio_uri is None, "solo se guarda una referencia, nunca el audio"


def test_migration_is_repeatable_and_does_not_trust_legacy_labels(settings, tmp_path):
    from src.migrate_audit import migrate

    source = tmp_path / "audit.jsonl"
    source.write_text(
        json.dumps(
            {
                "call_id": "opaque-legacy",
                "timestamp": "2026-09-13T00:00:00Z",
                "decision": {"is_synthetic": True, "confidence": 0.99},
                "analysis": {"audio_duration_s": 10, "acoustics": {"pitch_mean_hz": 165.4}},
                "status_for_training": "ready",
            }
        )
        + "\n"
    )
    original = source.read_bytes()
    assert migrate(source, settings) == (1, 0)
    assert migrate(source, settings) == (1, 0)
    assert source.read_bytes() == original
    files = list(settings.spool_dir.glob("*.json"))
    assert len(files) == 1
    record = CallAuditRecord.model_validate_json(files[0].read_bytes())
    assert "status_for_training" not in record.model_dump()
    assert record.analysis.turns == []


def test_receipt_write_failure_can_replay_acknowledged_insert(settings, monkeypatch):
    import src.worker as worker

    write_json(settings.spool_dir / "a.json", document())
    repository = MongoAuditRepository(FakeCollection())
    real_write = worker.write_json
    monkeypatch.setattr(worker, "write_json", Mock(side_effect=OSError("disk full")))
    with pytest.raises(OSError):
        worker.process_pending_logs(repository, settings)
    monkeypatch.setattr(worker, "write_json", real_write)
    assert worker.process_pending_logs(repository, settings) == 1
    assert len(repository.collection.docs) == 1


def test_size_limit_rejects_without_losing_source(settings):
    settings.spool_dir.mkdir()
    path = settings.spool_dir / "large.json"
    path.write_bytes(b" " * (settings.max_json_bytes + 1))
    repository = Mock()
    assert process_pending_logs(repository, settings) == 0
    repository.save.assert_not_called()
    assert path.exists()


def test_config_legacy_names_and_secret_repr(monkeypatch):
    monkeypatch.delenv("MONGODB_URI", raising=False)
    monkeypatch.setenv("MONGO_URI", "private-placeholder")
    settings = Settings.from_env()
    assert settings.mongodb_uri.get_secret_value() == "private-placeholder"
    assert "private-placeholder" not in repr(settings)
    monkeypatch.setenv("MONGODB_URI", "preferred-placeholder")
    assert Settings.from_env().mongodb_uri.get_secret_value() == "preferred-placeholder"


def test_config_accepts_team_env_names(monkeypatch):
    for name in ("MONGODB_DATABASE", "DATABASE_NAME", "MONGODB_COLLECTION", "COLLECTION_NAME"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("MONGO_DB", "altur_defense")
    monkeypatch.setenv("COLLECTION_CALLS", "calls")
    monkeypatch.setenv("AUDIT_LOG_PATH", "data/audit.jsonl")
    settings = Settings.from_env()
    assert (settings.mongodb_database, settings.mongodb_collection) == ("altur_defense", "calls")
    assert str(settings.legacy_audit_log) == "data/audit.jsonl"
    monkeypatch.setenv("MONGODB_COLLECTION", "calls_v1")
    assert Settings.from_env().mongodb_collection == "calls_v1"


def test_audit_external_ids_use_hmac_or_no_correlation(tmp_path, monkeypatch):
    from app.audit import log_detection_event

    monkeypatch.setenv("AUDIT_SPOOL_DIR", str(tmp_path))
    monkeypatch.setenv("AUDIT_ID_KEY", "")
    first = log_detection_event(call_id="private-input-id")
    record = json.loads((tmp_path / f"{first}.json").read_text())
    assert record["call_id"] == first
    monkeypatch.setenv("AUDIT_ID_KEY", "test-only-key")
    ids = [log_detection_event(call_id="private-input-id") for _ in range(2)]
    records = [json.loads((tmp_path / f"{i}.json").read_text()) for i in ids]
    assert records[0]["call_id"] == records[1]["call_id"]
    assert "private-input-id" not in json.dumps(records)
