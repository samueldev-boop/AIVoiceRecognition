"""Servicio HTTP: POST /detect y el frontend estatico.

Un solo proceso sirve la API y la interfaz: un despliegue, sin CORS, sin build.
"""

import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles

from app import __version__, audio, cascade, config, model, vad
from app.schemas import DetectRequest, DetectResponse, HealthResponse

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
async def detect(req: DetectRequest) -> DetectResponse:
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
    log.info("detect clip=%.1fs stage=%s p=%.4f conf=%.3f %.0fms",
             duracion, resultado["stage"], resultado["probability_synthetic"],
             resultado["confidence"], resultado["ms"])
    return DetectResponse(**resultado)


# El frontend (#13) se sirve desde el mismo proceso. Se monta al final para que no
# capture las rutas de la API.
if os.path.isdir(config.STATIC_DIR):
    app.mount("/", StaticFiles(directory=config.STATIC_DIR, html=True), name="static")
