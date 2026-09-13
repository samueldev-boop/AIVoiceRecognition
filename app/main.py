"""Servicio HTTP: POST /detect, POST /transcribe y el frontend estatico.

Un solo proceso sirve la API y la interfaz: un despliegue, sin CORS, sin build.
"""

import asyncio
import base64
import hashlib
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles

from app import __version__, audio, cascade, config, model, transcripcion, vad
from app.audit import log_detection_event, motivos_de_incertidumbre
from app.schemas import (
    DetectDiagnosticsResponse,
    DetectRequest,
    DetectResponse,
    HealthResponse,
    TranscribeRequest,
    TranscribeResponse,
)

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
log = logging.getLogger("detect")

ESTADO: dict = {"modelo": None}

# Clip minimo para calentar el VAD al arrancar: 1 s de silencio estereo.
_CLIP_DE_CALENTAMIENTO = np.zeros((config.SAMPLE_RATE, config.CHANNELS), dtype=np.int16)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Todo lo caro se carga una vez al arrancar, nunca por peticion.
    ESTADO["modelo"] = model.cargar()
    model_path = Path(config.MODEL_PATH)
    ESTADO["model_artifact_sha256"] = (
        hashlib.sha256(model_path.read_bytes()).hexdigest() if model_path.is_file() else None
    )
    if ESTADO["modelo"] is None:
        log.warning("sin modelo en %s: /detect respondera 503", config.MODEL_PATH)
    else:
        log.info("modelo cargado, version %s", ESTADO["modelo"].version)
    # El VAD se calienta al arrancar: la sesion de webrtcvad y el extractor no deben
    # cargarse dentro de la primera peticion.
    vad.turnos(_CLIP_DE_CALENTAMIENTO, config.SAMPLE_RATE)
    log.info("vad listo")
    yield


app = FastAPI(
    title="AIVoiceRecognition",
    version=__version__,
    summary="Detecta si quien llama es una persona o un agente autonomo",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    m = ESTADO["modelo"]
    return HealthResponse(
        ok=True,
        version=__version__,
        model_loaded=m is not None,
        model_version=m.version if m else None,
    )


@app.post("/detect", response_model=DetectResponse)
@app.post("/detect/details", response_model=DetectDiagnosticsResponse)
async def detect(req: DetectRequest) -> DetectDiagnosticsResponse:
    # Ambas rutas comparten inferencia y auditoria. El response_model de /detect
    # filtra la salida a los dos campos del jurado; /detect/details sirve la interfaz.
    t0 = time.perf_counter()

    # Etapa 0: validacion de entrada.
    try:
        x, sr = audio.decodificar(req.audio)
        duracion = audio.validar(x, sr)
    except audio.AudioInvalido as e:
        # Entrada que no cumple el formato: se rechaza con el motivo, no se adivina.
        raise HTTPException(status_code=422, detail=str(e)) from e

    m = ESTADO["modelo"]
    if m is None:
        raise HTTPException(
            status_code=503,
            detail=(
                f"sin modelo entrenado en {config.MODEL_PATH}. "
                "La cascada de decision se implementa en #6 sobre el modelo de #5."
            ),
        )

    # Etapas 1 a 3: la cascada corre en un hilo porque es trabajo de CPU, y el watchdog
    # acota el total. Un hilo no se puede matar, asi que la cascada tambien comprueba el
    # limite entre etapas; este wait_for es el techo duro.
    restante = max(1.0, config.DETECT_TIMEOUT_S - (time.perf_counter() - t0))
    try:
        resultado = await asyncio.wait_for(
            asyncio.to_thread(cascade.decidir, m, x, sr, limite_s=restante),
            timeout=restante,
        )
    except TimeoutError:
        log.warning("watchdog: %.1fs agotados, se responde con abstencion", restante)
        resultado = cascade.abstencion((time.perf_counter() - t0) * 1000, "watchdog")
    except ValueError as e:
        # Clip valido en formato pero sin intervenciones utilizables del llamante.
        log.warning("sin puntuacion posible: %s", e)
        resultado = cascade.abstencion((time.perf_counter() - t0) * 1000, "sin_habla")

    resultado["ms"] = (time.perf_counter() - t0) * 1000
    log.info(
        "detect clip=%.1fs stage=%s p=%.4f conf=%.3f %.0fms",
        duracion,
        resultado["stage"],
        resultado["probability_synthetic"],
        resultado["confidence"],
        resultado["ms"],
    )

    # Solo las llamadas inciertas dejan una referencia para el ciclo de reentrenamiento.
    motivos = motivos_de_incertidumbre(resultado)
    if not motivos:
        return DetectDiagnosticsResponse(**resultado)

    # La referencia se publica fuera del event loop antes de responder. Si falla, por
    # defecto se responde igual: la deteccion es el contrato y la auditoria es un extra.
    # Con AUDIT_REQUIRED=true se exige conservarla y, si no se puede, se responde 503.
    try:
        await asyncio.to_thread(
            log_detection_event,
            uncertainty_reasons=motivos,
            call_id=None if req.call_id is None else str(req.call_id),
            is_synthetic=resultado.get(
                "is_synthetic",
                bool(resultado.get("probability_synthetic", 0.0) >= 0.5),
            ),
            confidence=resultado.get("confidence"),
            probability_synthetic=resultado.get("probability_synthetic"),
            turns=resultado.get("turns"),
            disagreement=resultado.get("disagreement", False),
            layer_scores=resultado.get("layer_scores"),
            budget_scores=resultado.get("budget_scores"),
            input_sha256=hashlib.sha256(x.tobytes()).hexdigest(),
            model_version=m.version,
            model_artifact_sha256=ESTADO.get("model_artifact_sha256"),
            stage=resultado.get("stage", 1),
            latency_s=(time.perf_counter() - t0),
            audio_duration_s=duracion,
        )
    except Exception as exc:
        log.error("event=audit_failed error_type=%s", type(exc).__name__)
        if config.AUDIT_REQUIRED:
            raise HTTPException(
                status_code=503, detail="No se pudo conservar la auditoria"
            ) from None

    return DetectDiagnosticsResponse(**resultado)


@app.post("/transcribe", response_model=TranscribeResponse)
async def transcribe(req: TranscribeRequest) -> TranscribeResponse:
    """Transcripcion por canal para la interfaz. Va aparte: /detect no la usa ni la espera."""
    t0 = time.perf_counter()
    if not config.ELEVENLABS_API_KEY:
        raise HTTPException(status_code=503, detail="transcripcion no configurada")

    # Las mismas reglas de formato que /detect: lo que el detector rechaza no se transcribe.
    try:
        x, sr = await asyncio.to_thread(audio.decodificar, req.audio)
        duracion = audio.validar(x, sr)
    except audio.AudioInvalido as e:
        raise HTTPException(status_code=422, detail=str(e)) from e

    wav = base64.b64decode(req.audio)
    if not transcripcion.es_wav(wav):
        raise HTTPException(status_code=422, detail="solo se transcriben archivos WAV")
    if duracion > config.TRANSCRIPCION_MAX_S:
        raise HTTPException(
            status_code=413,
            detail=(
                f"llamada de {duracion:.0f} s: se transcriben hasta "
                f"{config.TRANSCRIPCION_MAX_S:.0f} s"
            ),
        )

    try:
        resultado = await transcripcion.transcribir(wav)
    except transcripcion.TranscripcionFallida as e:
        log.warning("transcribe fallo: %s", e)
        raise HTTPException(status_code=502, detail=str(e)) from e

    ms = (time.perf_counter() - t0) * 1000
    log.info("transcribe clip=%.1fs segmentos=%d %.0fms", duracion, len(resultado["segments"]), ms)
    return TranscribeResponse(**resultado, ms=ms)


class EstaticosSinCache(StaticFiles):
    """Sin Cache-Control, el navegador aplica cache heuristica y tras un despliegue puede
    seguir usando el app.js anterior. Con no-cache revalida siempre por ETag: si el archivo
    no cambio recibe un 304 sin cuerpo."""

    def file_response(self, *args, **kwargs):
        respuesta = super().file_response(*args, **kwargs)
        respuesta.headers["Cache-Control"] = "no-cache"
        return respuesta


# El frontend (#13) se sirve desde el mismo proceso. Se monta al final para que no
# capture las rutas de la API.
if os.path.isdir(config.STATIC_DIR):
    app.mount("/", EstaticosSinCache(directory=config.STATIC_DIR, html=True), name="static")
