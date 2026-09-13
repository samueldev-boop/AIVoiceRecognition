"""La transcripcion es un extra de la interfaz: va aparte de /detect y nunca altera su respuesta."""

import asyncio
import base64
import io
import re
from types import SimpleNamespace

import httpx
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from app import config, main, transcripcion
from app.schemas import DetectResponse
from tests.test_cascade import wav_base64

# Forma de la respuesta de ElevenLabs con use_multi_channel: un transcript por canal.
RESPUESTA = {
    "transcripts": [
        {
            "channel_index": 0,
            "words": [
                {"text": "Hola,", "start": 2.0, "end": 2.3, "type": "word"},
                {"text": " ", "start": 2.3, "end": 2.4, "type": "spacing"},
                {"text": "quiero", "start": 2.4, "end": 2.7, "type": "word"},
                {"text": "(tos)", "start": 2.8, "end": 3.0, "type": "audio_event"},
                # Mas de PAUSA_S callado sin que hable el otro canal: intervencion nueva.
                {"text": "saber.", "start": 4.5, "end": 4.9, "type": "word"},
            ],
        },
        {
            "channel_index": 1,
            "words": [
                {"text": "Buenas", "start": 0.1, "end": 0.4, "type": "word"},
                {"text": "tardes.", "start": 0.5, "end": 0.9, "type": "word"},
                {"text": "Claro.", "start": 6.0, "end": 6.4, "type": "word"},
            ],
        },
    ]
}


def test_segmentos_intercala_los_canales_y_corta_por_pausa():
    segmentos = transcripcion.segmentos(RESPUESTA)
    assert [(s["channel"], s["text"]) for s in segmentos] == [
        (1, "Buenas tardes."),
        (0, "Hola, quiero"),
        (0, "saber."),
        (1, "Claro."),
    ]
    assert (segmentos[0]["start"], segmentos[0]["end"]) == (0.1, 0.9)


def elevenlabs_simulado(monkeypatch, respuesta):
    """Sustituye la red: guarda cada peticion y contesta `respuesta` (o la lanza si es error)."""
    vistas = []

    def responder(peticion):
        peticion.read()
        vistas.append(peticion)
        if isinstance(respuesta, Exception):
            raise respuesta
        return respuesta

    cliente_real = httpx.AsyncClient
    monkeypatch.setattr(
        transcripcion.httpx,
        "AsyncClient",
        lambda **kw: cliente_real(transport=httpx.MockTransport(responder), **kw),
    )
    return vistas


def test_pide_a_elevenlabs_un_transcript_por_canal(monkeypatch):
    monkeypatch.setattr(config, "ELEVENLABS_API_KEY", "clave-de-prueba")
    vistas = elevenlabs_simulado(monkeypatch, httpx.Response(200, json=RESPUESTA))

    resultado = asyncio.run(transcripcion.transcribir(b"RIFF\x00\x00\x00\x00WAVE"))

    [peticion] = vistas
    assert str(peticion.url) == transcripcion.URL
    assert peticion.headers["xi-api-key"] == "clave-de-prueba"
    cuerpo = peticion.content.decode("latin-1")
    campos = {
        "model_id": config.ELEVENLABS_STT_MODEL,
        "use_multi_channel": "true",
        "language_code": "es",
    }
    for campo, valor in campos.items():
        assert f'name="{campo}"\r\n\r\n{valor}\r\n' in cuerpo, campo
    assert 'name="file"; filename="llamada.wav"' in cuerpo
    assert len(resultado["segments"]) == 4


@pytest.mark.parametrize(
    ("respuesta", "motivo"),
    [
        (
            httpx.Response(401, json={"detail": {"status": "invalid_api_key", "message": "no"}}),
            "401 (invalid_api_key)",
        ),
        (httpx.Response(500, text="caido"), "500"),
        (httpx.Response(200, text="no es json"), "ilegible"),
        (httpx.ConnectTimeout("lento"), "sin respuesta de ElevenLabs (ConnectTimeout)"),
    ],
)
def test_un_fallo_de_elevenlabs_se_informa_con_su_motivo(monkeypatch, respuesta, motivo):
    elevenlabs_simulado(monkeypatch, respuesta)
    with pytest.raises(transcripcion.TranscripcionFallida, match=re.escape(motivo)):
        asyncio.run(transcripcion.transcribir(b"RIFF\x00\x00\x00\x00WAVE"))


@pytest.fixture
def cliente(monkeypatch):
    monkeypatch.setattr(config, "ELEVENLABS_API_KEY", "clave-de-prueba")
    monkeypatch.setattr(main.model, "cargar", lambda: SimpleNamespace(version="test"))
    with TestClient(main.app) as c:
        yield c


def sin_red(monkeypatch):
    """Si algo intenta llamar a ElevenLabs, el test falla."""

    async def prohibido(wav):
        raise AssertionError("no deberia llamar a ElevenLabs")

    monkeypatch.setattr(transcripcion, "transcribir", prohibido)


def flac_base64():
    """La misma llamada en FLAC: /detect la acepta, pero no es un WAV."""
    x, sr = sf.read(io.BytesIO(base64.b64decode(wav_base64())), dtype="int16", always_2d=True)
    buf = io.BytesIO()
    sf.write(buf, x, sr, format="FLAC", subtype="PCM_16")
    return base64.b64encode(buf.getvalue()).decode()


def test_sin_clave_responde_503_sin_llamar_a_elevenlabs(cliente, monkeypatch):
    monkeypatch.setattr(config, "ELEVENLABS_API_KEY", "")
    sin_red(monkeypatch)
    assert cliente.post("/transcribe", json={"audio": wav_base64()}).status_code == 503


def test_solo_se_transcriben_wav_validos(cliente, monkeypatch):
    sin_red(monkeypatch)
    assert cliente.post("/transcribe", json={"audio": "esto-no-es-base64!!"}).status_code == 422
    r = cliente.post("/transcribe", json={"audio": flac_base64()})
    assert r.status_code == 422
    assert "WAV" in r.json()["detail"]


def test_una_llamada_demasiado_larga_no_se_manda(cliente, monkeypatch):
    sin_red(monkeypatch)
    monkeypatch.setattr(config, "TRANSCRIPCION_MAX_S", 10.0)
    r = cliente.post("/transcribe", json={"audio": wav_base64(segundos=12.0)})
    assert r.status_code == 413


def test_devuelve_las_intervenciones_del_wav_original(cliente, monkeypatch):
    enviado = {}

    async def simulado(wav):
        enviado["wav"] = wav
        return {"segments": transcripcion.segmentos(RESPUESTA)}

    monkeypatch.setattr(transcripcion, "transcribir", simulado)
    audio_b64 = wav_base64()
    r = cliente.post("/transcribe", json={"audio": audio_b64})

    assert r.status_code == 200, r.text
    segmentos = r.json()["segments"]
    assert [s["channel"] for s in segmentos] == [1, 0, 0, 1]
    assert set(segmentos[0]) == {"channel", "start", "end", "text"}
    # A ElevenLabs llega el WAV tal cual, sin remuestrear ni recodificar.
    assert enviado["wav"] == base64.b64decode(audio_b64)


def test_un_fallo_de_elevenlabs_responde_502(cliente, monkeypatch):
    async def falla(wav):
        raise transcripcion.TranscripcionFallida("ElevenLabs respondio 429 (quota_exceeded)")

    monkeypatch.setattr(transcripcion, "transcribir", falla)
    r = cliente.post("/transcribe", json={"audio": wav_base64()})
    assert r.status_code == 502
    assert "quota_exceeded" in r.json()["detail"]


def test_detect_no_transcribe_ni_cambia_su_respuesta(cliente, monkeypatch):
    sin_red(monkeypatch)
    monkeypatch.setattr(
        main.cascade,
        "decidir",
        lambda *a, **kw: {
            "is_synthetic": True,
            "confidence": 0.97,
            "probability_synthetic": 0.97,
            "stage": "first_turn",
        },
    )
    r = cliente.post("/detect", json={"audio_base64": wav_base64()})
    assert r.status_code == 200, r.text
    assert set(r.json()) == set(DetectResponse.model_fields)
