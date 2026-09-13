"""Servicio HTTP: POST /detect y el frontend estatico.

Un solo proceso sirve la API y la interfaz: un despliegue, sin CORS, sin build.
"""

import logging
import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles

from app import __version__, audio, config, model
from app.schemas import DetectRequest, DetectResponse, HealthResponse

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
log = logging.getLogger("detect")

ESTADO: dict = {"modelo": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Todo lo caro se carga una vez al arrancar, nunca por peticion.
    ESTADO["modelo"] = model.cargar()
    if ESTADO["modelo"] is None:
        log.warning("sin modelo en %s: /detect respondera 503", config.MODEL_PATH)
    else:
        log.info("modelo cargado, version %s", ESTADO["modelo"].version)
    # Pendiente #6: precargar aqui el VAD y el modelo de ASR para que la primera
    # peticion no pague la carga.
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
def detect(req: DetectRequest) -> DetectResponse:
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

    # Pendiente #6: cascada por etapas con salida temprana y watchdog.
    raise HTTPException(
        status_code=503,
        detail=(
            f"clip valido ({duracion:.1f}s) pero la cascada no esta implementada todavia: "
            f"ver #6. ms={1000 * (time.perf_counter() - t0):.1f}"
        ),
    )


# El frontend (#13) se sirve desde el mismo proceso. Se monta al final para que no
# capture las rutas de la API.
if os.path.isdir(config.STATIC_DIR):
    app.mount("/", StaticFiles(directory=config.STATIC_DIR, html=True), name="static")
