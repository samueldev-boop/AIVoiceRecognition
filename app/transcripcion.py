"""Transcripcion de la llamada con ElevenLabs Speech to Text, para la interfaz.

No participa en la decision: /detect no la usa y su respuesta no cambia. Cada canal se
transcribe por separado (ch0 = llamante, ch1 = agente) y las palabras se agrupan en
intervenciones ordenadas en el tiempo, que es lo que pinta la pagina.
"""

import httpx

from app import config

URL = "https://api.elevenlabs.io/v1/speech-to-text"

# Silencio a partir del cual un mismo canal abre una intervencion nueva.
PAUSA_S = 1.0


class TranscripcionFallida(RuntimeError):
    """ElevenLabs no devolvio una transcripcion utilizable."""


def es_wav(datos: bytes) -> bool:
    """Cabecera RIFF/WAVE: solo los WAV se transcriben."""
    return datos[:4] == b"RIFF" and datos[8:12] == b"WAVE"


async def transcribir(wav: bytes) -> dict:
    """WAV -> {"segments": [...]}. Lanza TranscripcionFallida si ElevenLabs no responde bien."""
    try:
        async with httpx.AsyncClient(timeout=config.ELEVENLABS_TIMEOUT_S) as cliente:
            r = await cliente.post(
                URL,
                headers={"xi-api-key": config.ELEVENLABS_API_KEY},
                data={
                    "model_id": config.ELEVENLABS_STT_MODEL,
                    "language_code": "es",
                    # Un transcript por canal: la llamada ya viene separada por hablante.
                    "use_multi_channel": "true",
                    "tag_audio_events": "false",
                },
                files={"file": ("llamada.wav", wav, "audio/wav")},
            )
    except httpx.HTTPError as e:
        raise TranscripcionFallida(f"sin respuesta de ElevenLabs ({type(e).__name__})") from e

    if r.status_code != 200:
        raise TranscripcionFallida(f"ElevenLabs respondio {r.status_code}{_motivo(r)}")
    try:
        return {"segments": segmentos(r.json())}
    except (ValueError, AttributeError, TypeError) as e:
        raise TranscripcionFallida("respuesta de ElevenLabs ilegible") from e


def _motivo(r: httpx.Response) -> str:
    """El codigo de error de ElevenLabs (invalid_api_key, quota_exceeded...), si lo trae."""
    try:
        return f" ({r.json()['detail']['status']})"
    except (ValueError, KeyError, TypeError):
        return ""


def segmentos(datos: dict) -> list[dict]:
    """Palabras de todos los canales en orden temporal, agrupadas en intervenciones.

    Se abre una nueva cuando cambia el canal o cuando el mismo canal calla mas de PAUSA_S.
    Con varios canales ElevenLabs devuelve un transcript por canal; con uno, el objeto plano.
    """
    palabras = []
    for i, transcript in enumerate(datos.get("transcripts") or [datos]):
        canal = transcript.get("channel_index")
        canal = i if canal is None else canal
        for w in transcript.get("words") or []:
            texto = (w.get("text") or "").strip()
            if w.get("type") == "word" and w.get("start") is not None and texto:
                palabras.append((w["start"], w.get("end") or w["start"], canal, texto))
    palabras.sort(key=lambda p: p[0])

    salida: list[dict] = []
    for inicio, fin, canal, texto in palabras:
        ultimo = salida[-1] if salida else None
        if ultimo and ultimo["channel"] == canal and inicio - ultimo["end"] <= PAUSA_S:
            ultimo["text"] += " " + texto
            ultimo["end"] = max(ultimo["end"], fin)
        else:
            salida.append({"channel": canal, "start": inicio, "end": fin, "text": texto})
    return salida
