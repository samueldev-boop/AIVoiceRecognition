"""Referencias de llamadas inciertas desde la API, con un modelo de pega y sin base de datos."""

import json
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app import main
from app.schemas import DetectRequest
from tests.test_cascade import wav_base64


def resultado(p=0.45, *, stage="full", disagreement=False):
    return {
        "is_synthetic": p >= 0.5,
        "confidence": max(p, 1 - p),
        "probability_synthetic": p,
        "stage": stage,
        "layer_scores": {},
        "budget_scores": {},
        "ms": 1,
        "turns": [{"channel": 0, "start": 0, "end": 1}],
        "disagreement": disagreement,
    }


@pytest.fixture
def decision(monkeypatch):
    """Lo que devolvera la cascada de pega; por defecto, una llamada incierta."""
    actual = {"valor": resultado()}
    monkeypatch.setattr(main.cascade, "decidir", lambda *a, **kw: dict(actual["valor"]))
    return actual


@pytest.fixture
def client(monkeypatch, tmp_path, decision):
    monkeypatch.setenv("AUDIT_SPOOL_DIR", str(tmp_path / "raw"))
    monkeypatch.setattr(main.model, "cargar", lambda: SimpleNamespace(version="test-model-v1"))
    with TestClient(main.app) as value:
        yield value


def eventos(tmp_path):
    return [json.loads(f.read_text()) for f in (tmp_path / "raw").glob("*.json")]


def test_an_uncertain_call_leaves_a_reference(client, tmp_path):
    response = client.post("/detect", json={"audio_base64": wav_base64()})
    assert response.status_code == 200
    assert set(response.json()) == {"is_synthetic", "confidence"}
    [record] = eventos(tmp_path)
    assert record["decision"]["probability_synthetic"] == 0.45
    assert record["analysis"]["uncertainty_reasons"] == ["probabilidad_ambigua"]
    assert record["analysis"]["turns"][0]["end"] == 1
    assert record["model_version"] == "test-model-v1"
    assert record["input_sha256"]
    assert "audio" not in json.dumps(record).replace("audio_duration_s", "").replace(
        "audio_uri", ""
    ), "la referencia no lleva el audio"


@pytest.mark.parametrize("p", [0.02, 0.97])
def test_a_clear_call_is_not_stored(client, tmp_path, decision, p):
    decision["valor"] = resultado(p, stage="first_turn")
    assert client.post("/detect", json={"audio_base64": wav_base64()}).status_code == 200
    assert eventos(tmp_path) == []


@pytest.mark.parametrize(
    "valor,motivo",
    [
        (resultado(0.95, disagreement=True), "desacuerdo"),
        (resultado(0.5, stage="sin_habla"), "abstencion"),
        (resultado(0.03, stage="first_turn+limite"), "abstencion"),
    ],
)
def test_disagreement_and_abstentions_are_uncertain(client, tmp_path, decision, valor, motivo):
    decision["valor"] = valor
    client.post("/detect", json={"audio_base64": wav_base64()})
    [record] = eventos(tmp_path)
    assert motivo in record["analysis"]["uncertainty_reasons"]


def test_the_band_is_configurable(client, tmp_path, decision, monkeypatch):
    monkeypatch.setattr(main.config, "AUDIT_BANDA", (0.3, 0.7))
    decision["valor"] = resultado(0.2)
    client.post("/detect", json={"audio_base64": wav_base64()})
    assert eventos(tmp_path) == []


def _falla(**kwargs):
    raise OSError("private filesystem details")


def test_a_failed_audit_does_not_break_detection_by_default(client, monkeypatch):
    monkeypatch.setattr(main, "log_detection_event", _falla)
    response = client.post("/detect", json={"audio": wav_base64()})
    assert response.status_code == 200
    assert response.json()["is_synthetic"] is False


def test_api_does_not_acknowledge_when_audit_is_required_and_fails(client, monkeypatch):
    monkeypatch.setattr(main, "log_detection_event", _falla)
    monkeypatch.setattr(main.config, "AUDIT_REQUIRED", True)
    response = client.post("/detect", json={"audio": wav_base64()})
    assert response.status_code == 503
    assert "private" not in response.text


@pytest.mark.parametrize("call_id", ["call-001", 12345, "", "x" * 300, None])
def test_any_call_id_is_accepted(client, call_id):
    response = client.post("/detect", json={"audio_base64": wav_base64(), "call_id": call_id})
    assert response.status_code == 200, response.text


def test_audio_base64_is_the_primary_field():
    assert DetectRequest(audio_base64="a", audio="b").audio == "a"
