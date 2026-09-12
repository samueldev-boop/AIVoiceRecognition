"""Configuracion por variables de entorno. Sin fichero de settings ni capas."""

import os

MODEL_PATH = os.getenv("MODEL_PATH", "model/model.joblib")

WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")
WHISPER_COMPUTE = os.getenv("WHISPER_COMPUTE", "int8")
CPU_THREADS = int(os.getenv("CPU_THREADS", "4"))

# El reto limita la respuesta a 30 s. Cortamos antes para dejar margen.
DETECT_TIMEOUT_S = float(os.getenv("DETECT_TIMEOUT_S", "25"))

STATIC_DIR = os.getenv("STATIC_DIR", "static")

# Restricciones del formato de entrada, fijadas por el dataset.
SAMPLE_RATE = 8000
CHANNELS = 2
MIN_DURATION_S = 5.0
MAX_DURATION_S = 600.0
