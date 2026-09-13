"""Cascada de decision: salida temprana, fusion, desacuerdo y watchdog.

La mecanica se prueba con un modelo y un extractor de pega, que es determinista y rapido;
lo que se esta fijando es la logica de la cascada, no el modelo. Al final hay una prueba de
integracion con el artefacto real a traves del endpoint.
"""

import base64
import io

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from app import cascade, config

SR = config.SAMPLE_RATE
CAPAS = ("ganancia", "codec_bw", "silencio", "prosodia", "conducta", "razon_canal")


class ModeloDePega:
    """Devuelve las probabilidades que se le pasan, en el orden de los presupuestos."""

    def __init__(self, probabilidades, scores=0.9):
        self.probabilidades = dict(probabilidades)
        self.scores = scores
        self.llamadas = []

    def predecir(self, muestra):
        presupuesto = muestra["budget"]
        self.llamadas.append(presupuesto)
        p = self.probabilidades[presupuesto]
        return {
            "probability_synthetic": p,
            "layer_scores": dict.fromkeys(CAPAS, self.scores),
            "available_layers": list(CAPAS),
            "audio_used_s": 10.0,
            "stage": presupuesto,
        }


@pytest.fixture
def sin_audio(monkeypatch):
    """Evita el VAD y el extractor: aqui no se prueban ellos."""
    monkeypatch.setattr(cascade.vad, "turnos", lambda x, sr: [])
    monkeypatch.setattr(cascade, "extract_sample",
                        lambda x, sr, budget, **kw: {"budget": budget})
    return np.zeros((SR * 10, 2), dtype=np.int16)


def test_sale_en_la_primera_etapa_si_esta_fuera_de_la_banda(sin_audio):
    modelo = ModeloDePega({"first_turn": 0.97, "20s": 0.5, "full": 0.5})
    r = cascade.decidir(modelo, sin_audio, SR)
    assert modelo.llamadas == ["first_turn"], "no deberia mirar mas audio"
    assert r["stage"] == "first_turn"
    assert list(r["budget_scores"]) == ["first_turn"]
    assert r["is_synthetic"] is True


def test_sale_temprano_tambien_hacia_humano(sin_audio):
    modelo = ModeloDePega({"first_turn": 0.02, "20s": 0.5, "full": 0.5})
    r = cascade.decidir(modelo, sin_audio, SR)
    assert modelo.llamadas == ["first_turn"]
    assert r["is_synthetic"] is False


def test_escala_cuando_esta_en_la_banda_y_fusiona(sin_audio):
    modelo = ModeloDePega({"first_turn": 0.5, "20s": 0.6, "full": 0.7})
    r = cascade.decidir(modelo, sin_audio, SR)
    assert modelo.llamadas == ["first_turn", "20s", "full"]
    assert r["stage"] == "full"
    assert set(r["budget_scores"]) == {"first_turn", "20s", "full"}
    # media de logits de 0.5, 0.6 y 0.7
    assert r["probability_synthetic"] == pytest.approx(0.602906, abs=1e-4)


def test_el_desacuerdo_entre_presupuestos_baja_la_confianza(sin_audio):
    modelo = ModeloDePega({"first_turn": 0.55, "20s": 0.2, "full": 0.05})
    r = cascade.decidir(modelo, sin_audio, SR)
    assert r["disagreement"] is True
    assert r["confidence"] <= cascade.TOPE_DESACUERDO


def test_sin_desacuerdo_no_se_topa_la_confianza(sin_audio):
    modelo = ModeloDePega({"first_turn": 0.6, "20s": 0.7, "full": 0.8})
    r = cascade.decidir(modelo, sin_audio, SR)
    assert r["disagreement"] is False
    assert r["confidence"] > cascade.TOPE_DESACUERDO


def test_el_limite_de_tiempo_corta_entre_etapas(sin_audio):
    modelo = ModeloDePega({"first_turn": 0.5, "20s": 0.5, "full": 0.5})
    r = cascade.decidir(modelo, sin_audio, SR, limite_s=0.0)
    assert modelo.llamadas == ["first_turn"], "la primera etapa siempre se completa"
    assert "limite" in r["stage"]


def test_una_sola_capa_disponible_topa_la_confianza(sin_audio):
    modelo = ModeloDePega({"first_turn": 0.99, "20s": 0.5, "full": 0.5})

    def una_capa(muestra):
        salida = ModeloDePega.predecir(modelo, muestra)
        salida["available_layers"] = ["conducta"]
        return salida

    modelo.predecir = una_capa
    r = cascade.decidir(modelo, sin_audio, SR)
    assert r["confidence"] <= 0.9


def test_abstencion_no_acusa_y_lo_dice():
    r = cascade.abstencion(1234.0, "watchdog")
    assert r["is_synthetic"] is False, "el error caro es acusar a una persona"
    assert r["confidence"] == 0.5
    assert r["stage"] == "watchdog"
    assert r["ms"] == 1234.0


def wav_base64(segundos=12.0, con_habla=True):
    rng = np.random.default_rng(0)
    n = int(SR * segundos)
    x = np.zeros((n, 2), dtype=np.int16)
    if con_habla:
        x[SR:SR * 4, 1] = np.clip(rng.normal(0, 3000, SR * 3), -32768, 32767)
        x[SR * 5:SR * 8, 0] = np.clip(rng.normal(0, 3000, SR * 3), -32768, 32767)
        x[SR * 9:int(SR * 11), 0] = np.clip(rng.normal(0, 2000, int(SR * 2)), -32768, 32767)
    else:
        x[:, 1] = np.clip(rng.normal(0, 30, n), -32768, 32767)
        x[:, 0] = np.clip(rng.normal(0, 20, n), -32768, 32767)
    buf = io.BytesIO()
    sf.write(buf, x, SR, subtype="PCM_16", format="WAV")
    return base64.b64encode(buf.getvalue()).decode()


def test_endpoint_responde_con_el_artefacto_real():
    """Integracion: /detect devuelve 200 y el esquema completo, sin adivinar nada."""
    from app.main import app

    with TestClient(app) as c:
        if not c.get("/health").json()["model_loaded"]:
            pytest.skip("sin artefacto entrenado en model/model.joblib")
        r = c.post("/detect", json={"audio": wav_base64()})
        assert r.status_code == 200, r.text
        cuerpo = r.json()
        assert isinstance(cuerpo["is_synthetic"], bool)
        assert 0.0 <= cuerpo["confidence"] <= 1.0
        assert cuerpo["stage"]
        assert cuerpo["ms"] > 0
        assert cuerpo["ms"] < 1000 * config.DETECT_TIMEOUT_S
