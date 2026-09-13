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
