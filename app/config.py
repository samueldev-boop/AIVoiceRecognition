"""Configuracion por variables de entorno. Sin fichero de settings ni capas."""

import os

MODEL_PATH = os.getenv("MODEL_PATH", "model/model.joblib")

WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")
WHISPER_COMPUTE = os.getenv("WHISPER_COMPUTE", "int8")
CPU_THREADS = int(os.getenv("CPU_THREADS", "4"))

# El reto limita la respuesta a 30 s. Cortamos antes para dejar margen.
DETECT_TIMEOUT_S = float(os.getenv("DETECT_TIMEOUT_S", "25"))

STATIC_DIR = os.getenv("STATIC_DIR", "static")

# Si es true, /detect responde 503 cuando no puede conservar el evento de auditoria. Por
# defecto no: un disco lleno o un volumen sin permisos no debe tumbar la deteccion.
AUDIT_REQUIRED = os.getenv("AUDIT_REQUIRED", "false").strip().lower() in ("1", "true", "yes")

# Banda de probabilidad en la que una llamada se considera incierta y deja su referencia.
# Por defecto coincide con la banda que hace escalar a la cascada (app/cascade.py).
AUDIT_BANDA = (
    float(os.getenv("AUDIT_BANDA_MIN", "0.10")),
    float(os.getenv("AUDIT_BANDA_MAX", "0.90")),
)

# Restricciones del formato de entrada, fijadas por el dataset.
SAMPLE_RATE = 8000
CHANNELS = 2
MIN_DURATION_S = 5.0
# El dataset no pasa de 273 s, pero en la evaluacion pueden llegar llamadas mas largas: se
# aceptan hasta una hora. El tamano del cuerpo lo acota Caddy (request_body en Caddyfile).
MAX_DURATION_S = float(os.getenv("MAX_DURATION_S", "3600"))

# Audio que analiza la cascada. La etapa full crece con la llamada (medido: ~0.55 s de
# computo por minuto; 16 s con 30 min y 34 s con una hora) y agotaria el watchdog. Mas alla
# de este tope se analiza solo el principio: el modelo nunca vio llamadas mas largas y, con
# la validacion agrupada, mas audio no mejora la generalizacion.
ANALISIS_MAX_S = float(os.getenv("ANALISIS_MAX_S", "300"))
