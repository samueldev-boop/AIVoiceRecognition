"""Failure and leakage regression tests; all persistence uses isolated temp files/fakes."""

import copy
import hashlib
import importlib
import json
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from pydantic import ValidationError
from pymongo.errors import AutoReconnect

from src.config import Settings
from src.data_processing import extract_features, prepare_dataset, validate_training_splits
from src.repositories import EventConflict, MongoAuditRepository
from src.schemas import CallAuditRecord
from src.storage import exclusive_lock, write_json
from src.worker import process_pending_logs


def document(i=0, *, label="human", group=None):
    return {
        "event_id": f"event-{i}",
        "call_id": f"call-{i}",
        "source": "test",
        "timestamp": "2026-09-13T00:00:00Z",
        "input_sha256": hashlib.sha256(str(i).encode()).hexdigest(),
        "group_ids": [group or f"speaker-{i}"],
        "decision": {"is_synthetic": label == "synthetic", "confidence": 0.9},
        "analysis": {
            "audio_duration_s": 10,
            "turns": [{"channel": 0, "start": 0, "end": 2}],
            "acoustics": {
                "pitch_mean_hz": 150,
                "jitter": 0,
                "shimmer": 0.02,
                "spectral_centroid": 2000,
            },
            "conversation": {"caller_response_latency_s": 0.3, "interruption_count": 0},
            "asr": {"avg_logprob": -0.2},
        },
        "label": {"value": label, "provenance": "reviewer-opaque", "verified": True},
        "quality": "accepted",
        "status_for_training": "ready",
    }


@pytest.fixture
def settings(tmp_path):
    return Settings(
        spool_dir=tmp_path / "raw",
        processed_dir=tmp_path / "processed",
        training_dir=tmp_path / "training",
        model_dir=tmp_path / "models",
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
        {"label": None},
        {"audio_uri": "s3://bucket/a?token=secret"},
        {"password": "must-not-be-accepted"},
    ],
)
def test_invalid_json_contract(change):
    with pytest.raises(ValidationError):
        CallAuditRecord.model_validate(document() | change)


def test_prediction_is_not_label():
    raw = document() | {"label": None, "status_for_training": "unlabeled"}
    result = prepare_dataset([raw])
    assert not any(result["splits"].values())


def test_zero_measurements_are_preserved_and_missing_is_rejected():
    assert extract_features(document()["analysis"])[1] == 0
    with pytest.raises(ValueError, match="missing measured"):
        extract_features({})


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
    collection.docs[record.event_id]["status_for_training"] = "excluded"
    assert not repository.save(record)
    assert collection.docs[record.event_id]["status_for_training"] == "excluded"
    changed = record.model_copy(update={"source": "different"})
    with pytest.raises(EventConflict):
        repository.save(changed)


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
    for name in ("src.worker", "src.curation", "src.export_training", "src.analytics"):
        importlib.reload(importlib.import_module(name))
    factory.assert_not_called()


def all_rows(dataset):
    return [r for rows in dataset["splits"].values() for r in rows]


def test_transitive_groups_are_joined_before_audio_deduplication():
    a, b, c = document(1, group="speaker-a"), document(2, group="speaker-b"), document(3)
    b["input_sha256"] = a["input_sha256"]
    c["group_ids"] = ["speaker-b"]
    dataset = prepare_dataset([a, b, c])
    assert len(all_rows(dataset)) == 2
    assert len({r["group"] for r in all_rows(dataset)}) == 1
    assert len([rows for rows in dataset["splits"].values() if rows]) == 1


def test_conflicting_ground_truth_fails_closed():
    a, b = document(1), document(2, label="synthetic")
    b["input_sha256"] = a["input_sha256"]
    with pytest.raises(ValueError, match="conflicting verified"):
        prepare_dataset([a, b])


def test_splits_reproducible_and_frozen_across_new_data():
    documents = [document(i, label="human" if i % 2 else "synthetic") for i in range(200)]
    dataset = prepare_dataset(documents)
    assert dataset == prepare_dataset(reversed(documents))
    validate_training_splits(dataset)
    updated = prepare_dataset(documents + [document(999)], split_registry=dataset["split_registry"])
    for name, rows in dataset["splits"].items():
        assert {r["event_id"] for r in rows} <= {r["event_id"] for r in updated["splits"][name]}


def test_new_link_across_frozen_partitions_is_rejected():
    docs = [document(i, label="human" if i % 2 else "synthetic") for i in range(200)]
    dataset = prepare_dataset(docs)
    train_id = dataset["splits"]["train"][0]["event_id"]
    test_id = dataset["splits"]["test"][0]["event_id"]
    selected = [d for d in docs if d["event_id"] in {train_id, test_id}]
    bridge = document(999)
    bridge["group_ids"] = [g for d in selected for g in d["group_ids"]]
    with pytest.raises(ValueError, match="frozen partitions"):
        prepare_dataset(docs + [bridge], split_registry=dataset["split_registry"])


def test_audit_publishes_true_probability_without_inventing_features(tmp_path, monkeypatch):
    from app.audit import log_detection_event

    monkeypatch.setenv("AUDIT_SPOOL_DIR", str(tmp_path))
    identifier = log_detection_event(confidence=0.95, probability_synthetic=0.05)
    record = CallAuditRecord.model_validate_json((tmp_path / f"{identifier}.json").read_bytes())
    assert record.decision.probability_synthetic == 0.05
    assert record.label is None
    assert record.analysis.acoustics is None
    assert record.analysis.turns == []


def test_training_creates_report_without_promoting(settings):
    from src.retrain_pipeline import execute_retraining

    docs = [document(i, label="human" if i % 2 else "synthetic") for i in range(200)]
    dataset = prepare_dataset(docs)
    path = settings.training_dir / "dataset.json"
    write_json(path, dataset)
    output = execute_retraining(path, settings=settings)
    report = json.loads((output / "report.json").read_text())
    assert not report["promotion"]["eligible"]
    assert not report["promotion"]["automatic_promotion"]
    assert report["dataset_id"] == dataset["dataset_id"]
    assert set(report["metrics"]) == {"validation", "test"}
    assert not (settings.model_dir / "classifier_latest.pkl").exists()


def test_export_uses_injected_repository_and_frozen_cutoff(settings):
    from src.export_training import export_active_learning_batch

    repository = Mock()
    repository.training_records.return_value = [document()]
    cutoff = datetime(2026, 9, 14, tzinfo=UTC)
    path = export_active_learning_batch(repository, settings, until=cutoff)
    repository.training_records.assert_called_once_with(until=cutoff)
    assert path.exists()
    assert (path.parent / "manifest.json").exists()
    assert (settings.training_dir / "split_registry.json").exists()


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
    assert record.status_for_training == "unlabeled"
    assert record.analysis.acoustics is None


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


@pytest.mark.parametrize("metric,value", [("auc", 0.8), ("fpr", 0.3), ("auc", float("nan"))])
def test_promotion_rejects_bad_or_invalid_metrics(metric, value):
    from src.retrain_pipeline import promotion_decision

    baseline = {"validation": {"auc": 0.90, "brier": 0.1, "fpr": 0.02}}
    candidate = {"validation": {"auc": 0.95, "brier": 0.05, "fpr": 0.01}}
    candidate["validation"][metric] = value
    assert not promotion_decision(candidate, baseline, compatible=True)["eligible"]


def test_promotion_requires_compatibility_and_comparable_baseline():
    from src.retrain_pipeline import promotion_decision

    baseline = {"validation": {"auc": 0.90, "brier": 0.1, "fpr": 0.02}}
    candidate = {"validation": {"auc": 0.95, "brier": 0.05, "fpr": 0.01}}
    assert not promotion_decision(candidate, None, compatible=True)["eligible"]
    assert not promotion_decision(candidate, baseline, compatible=False)["eligible"]
    result = promotion_decision(candidate, baseline, compatible=True)
    assert result["eligible"]
    assert not result["automatic_promotion"]


def test_dataset_checksum_detects_modified_rows():
    dataset = prepare_dataset(
        [document(i, label="human" if i % 2 else "synthetic") for i in range(200)]
    )
    dataset["splits"]["train"][0]["features"][0] += 1
    with pytest.raises(ValueError, match="checksum"):
        validate_training_splits(dataset)


def test_config_legacy_names_and_secret_repr(monkeypatch):
    monkeypatch.delenv("MONGODB_URI", raising=False)
    monkeypatch.setenv("MONGO_URI", "private-placeholder")
    settings = Settings.from_env()
    assert settings.mongodb_uri.get_secret_value() == "private-placeholder"
    assert "private-placeholder" not in repr(settings)
    monkeypatch.setenv("MONGODB_URI", "preferred-placeholder")
    assert Settings.from_env().mongodb_uri.get_secret_value() == "preferred-placeholder"


def test_agent_only_speech_is_excluded():
    raw = document()
    raw["analysis"]["turns"][0]["channel"] = 1
    assert not all_rows(prepare_dataset([raw]))


def test_unlabelled_record_still_connects_related_samples():
    a, b, bridge = document(1), document(2), document(3)
    bridge["group_ids"] = a["group_ids"] + b["group_ids"]
    bridge["label"] = None
    bridge["status_for_training"] = "unlabeled"
    result = prepare_dataset([a, b, bridge])
    assert len({r["group"] for r in all_rows(result)}) == 1


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
