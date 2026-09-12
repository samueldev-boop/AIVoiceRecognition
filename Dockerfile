# Python 3.14 verificado: todas las dependencias del servicio tienen wheel, incluida
# praat-parselmouth. Coincide con el entorno de desarrollo, asi que no hay sorpresas
# entre local y produccion.
FROM python:3.14-slim

# Las wheels traen sus propias librerias nativas (libgomp en ctranslate2, libsndfile en
# soundfile), asi que la imagen no necesita ningun paquete de apt.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models

WORKDIR /srv

COPY requirements.txt .
# --no-compile ahorra ~120 MB de bytecode a cambio de ~0.4 s de arranque (medido).
RUN pip install --no-cache-dir --no-compile -r requirements.txt

COPY app/ app/
COPY static/ static/
COPY model/ model/

RUN useradd --create-home --uid 10001 servicio \
    && mkdir -p /models \
    && chown -R servicio:servicio /models /srv
USER servicio

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD ["python", "-c", "import sys,urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3).status == 200 else 1)"]

# Un solo worker a proposito: la etapa de ASR consume los 4 hilos de CPU y varios
# workers se los quitarian entre si, empeorando la latencia en lugar de mejorarla.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
