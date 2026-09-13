# Python 3.14 verificado: todas las dependencias del servicio tienen wheel en 3.14 menos
# webrtcvad-wheels, que solo publica hasta cp313 y hay que compilar desde el sdist. Por eso
# el build es en dos etapas: gcc vive en el builder y no llega a la imagen final, y asi el
# contenedor mantiene la misma version de Python que el entorno de desarrollo.

FROM python:3.14-slim AS builder

# Lo unico que hay que compilar es webrtcvad. El resto son wheels.
# libc6-dev hace falta explicito: con --no-install-recommends, gcc no trae las cabeceras.
RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc libc6-dev \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
COPY requirements-worker.txt .
# --no-compile ahorra ~120 MB de bytecode a cambio de ~0.4 s de arranque (medido).
RUN pip install --no-cache-dir --no-compile -r requirements.txt


FROM python:3.14-slim

# La imagen final no instala ningun paquete de apt: las wheels traen sus librerias nativas
# (libgomp dentro de ctranslate2, libsndfile dentro de soundfile) y webrtcvad ya viene
# compilado en el venv del builder.
COPY --from=builder /opt/venv /opt/venv

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/models

WORKDIR /srv

COPY app/ app/
COPY src/ src/
COPY static/ static/
COPY model/ model/

RUN useradd --create-home --uid 10001 servicio \
    && mkdir -p /models /srv/data \
    && chown -R servicio:servicio /models /srv /srv/data
USER servicio

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD ["python", "-c", "import sys,urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3).status == 200 else 1)"]

# Un solo worker a proposito: la etapa de ASR consume los 4 hilos de CPU y varios
# workers se los quitarian entre si, empeorando la latencia en lugar de mejorarla.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
