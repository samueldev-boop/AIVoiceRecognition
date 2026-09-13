"""Criterio de aceptacion de #2: el servicio arranca y /health responde 200."""

from fastapi.testclient import TestClient

from app.main import app


def test_health_responde_200():
    with TestClient(app) as c:
        r = c.get("/health")
        assert r.status_code == 200
        cuerpo = r.json()
        assert cuerpo["ok"] is True
        assert "version" in cuerpo


def test_detect_rechaza_base64_invalido():
    with TestClient(app) as c:
        r = c.post("/detect", json={"audio": "esto-no-es-base64!!"})
        assert r.status_code == 422


def test_detect_acepta_formato_del_juez_y_formato_anterior(monkeypatch):
    import base64
    import io
    from types import SimpleNamespace

    import numpy as np
    import soundfile as sf

    from app import cascade, model

    monkeypatch.setattr(model, "cargar", lambda: SimpleNamespace(version="test"))
    monkeypatch.setattr(
        cascade,
        "decidir",
        lambda *a, **kw: {
            "is_synthetic": False,
            "confidence": 0.87,
            "probability_synthetic": 0.13,
            "stage": "first_turn",
        },
    )
    x = np.zeros((8000 * 10, 2), dtype=np.int16)
    x[:, 0] = (1000 * np.sin(np.arange(len(x)) / 20)).astype(np.int16)
    buffer = io.BytesIO()
    sf.write(buffer, x, 8000, format="WAV", subtype="PCM_16")
    encoded = base64.b64encode(buffer.getvalue()).decode()
    with TestClient(app) as client:
        for field in ("audio_base64", "audio"):
            response = client.post(
                "/detect",
                json={
                    "call_id": "call_contract_test",
                    field: encoded,
                    "sample_rate": 8000,
                    "channels": 2,
                },
            )
            assert response.status_code == 200
            assert response.json()["is_synthetic"] is False
            assert response.json()["confidence"] == 0.87


def test_detect_sin_modelo_responde_503(tmp_path, monkeypatch):
    # Sin artefacto entrenado, /detect no adivina: informa que no hay modelo.
    import base64
    import io

    import numpy as np
    import soundfile as sf

    from app import config

    monkeypatch.setattr(config, "MODEL_PATH", str(tmp_path / "missing.joblib"))

    sr = 8000
    n = sr * 10
    x = np.zeros((n, 2), dtype=np.int16)
    x[:, 0] = (1000 * np.sin(np.arange(n) / 20.0)).astype(np.int16)
    x[:, 1] = (800 * np.sin(np.arange(n) / 15.0)).astype(np.int16)
    buf = io.BytesIO()
    sf.write(buf, x, sr, subtype="PCM_16", format="WAV")

    with TestClient(app) as c:
        r = c.post("/detect", json={"audio": base64.b64encode(buf.getvalue()).decode()})
        assert r.status_code == 503


def test_detect_rechaza_canales_identicos():
    import base64
    import io

    import numpy as np
    import soundfile as sf

    sr = 8000
    mono = (1000 * np.sin(np.arange(sr * 10) / 20.0)).astype(np.int16)
    x = np.stack([mono, mono], axis=1)
    buf = io.BytesIO()
    sf.write(buf, x, sr, subtype="PCM_16", format="WAV")

    with TestClient(app) as c:
        r = c.post("/detect", json={"audio": base64.b64encode(buf.getvalue()).decode()})
        assert r.status_code == 422
        assert "identicos" in r.json()["detail"]
