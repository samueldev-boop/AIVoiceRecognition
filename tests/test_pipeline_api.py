"""API audit contract with a fake model and no database or ASR downloads."""

import json
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app import main
from app.schemas import DetectRequest
from tests.test_cascade import wav_base64


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setenv("AUDIT_SPOOL_DIR", str(tmp_path / "raw"))
    monkeypatch.setattr(main.model, "cargar", lambda: SimpleNamespace(version="test-model-v1"))
    monkeypatch.setattr(
        main.cascade,
        "decidir",
        lambda *args, **kwargs: {
            "is_synthetic": False,
            "confidence": 0.95,
            "probability_synthetic": 0.05,
            "stage": "first_turn",
            "layer_scores": {},
            "budget_scores": {},
            "ms": 1,
            "turns": [{"channel": 0, "start": 0, "end": 1}],
            "disagreement": False,
        },
    )
    with TestClient(main.app) as value:
        yield value


def test_api_publishes_measured_telemetry(client, tmp_path):
    response = client.post("/detect", json={"audio_base64": wav_base64()})
    assert response.status_code == 200
    files = list((tmp_path / "raw").glob("*.json"))
    assert len(files) == 1
    record = json.loads(files[0].read_text())
    assert record["decision"]["probability_synthetic"] == 0.05
    assert record["analysis"]["turns"][0]["end"] == 1
    assert record["model_version"] == "test-model-v1"
    assert record["input_sha256"]


def test_api_does_not_acknowledge_when_audit_storage_fails(client, monkeypatch):
    def fail(**kwargs):
        raise OSError("private filesystem details")

    monkeypatch.setattr(main, "log_detection_event", fail)
    response = client.post("/detect", json={"audio": wav_base64()})
    assert response.status_code == 503
    assert "private" not in response.text


def test_audio_aliases_cannot_disagree():
    with pytest.raises(ValueError, match="deben coincidir"):
        DetectRequest(audio="a", audio_base64="b")
