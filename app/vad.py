"""Actividad de voz por canal con webrtcvad, nativo a 8 kHz.

El endpoint recibe solo el WAV, nunca los turns/*.json del dataset, asi que las features
de conducta dependen de este modulo. Los turnos que devuelve tienen exactamente la misma
forma que los del dataset: {"channel", "start", "end"} en segundos.

Por que webrtcvad y no Silero (medido en scripts/vad_acuerdo.py, sobre las 353 llamadas):

    fronteras             IoU ch0  IoU ch1   CV-logloss   AUC val   FP
    turns/*.json (ref)      1.000    1.000       0.0944    0.9976    2
    webrtcvad agr=3         0.842    0.935       0.1169    1.0000    0
    Silero ONNX             0.618    0.880       0.1995    0.9833    4

La pista estaba en el dataset: los tiempos de turns/*.json caen al 100 % en una rejilla de
20 ms con duracion minima de 0.2 s. Asi trabaja webrtcvad (tramas de 10/20/30 ms), no
Silero (ventanas de 32 ms a 8 kHz). Reproducir el sesgo del generador vale mas que usar el
VAD con mejor reputacion.

Ademas webrtcvad no necesita onnxruntime ni un archivo de modelo, y es mas rapido.
"""

import numpy as np
import webrtcvad

from app import config, intervalos

# Trama de 20 ms: es la rejilla en la que caen todos los tiempos de turns/*.json.
TRAMA = 160  # muestras a 8 kHz
SALTO_S = TRAMA / config.SAMPLE_RATE  # 0.02 s

# Calibrado barriendo agresividad x silencio minimo x padding contra el rendimiento de las
# features de conducta, eligiendo por validacion cruzada repetida (10x5 sobre train) y no por
# el AUC de val, que con 71 llamadas no distingue 0.01 del ruido.
#
# La agresividad 3 gana en las dos cosas a la vez: es la que mas se parece a turns/*.json
# (IoU 0.84/0.94, frente a 0.71/0.76 con agresividad 1) y la que mejor rinde. Eso refuerza
# que el dataset se genero con este mismo VAD.
AGRESIVIDAD = 3         # 0 = permisivo, 3 = estricto
MIN_HABLA_S = 0.20      # igual que el minimo observado en turns/*.json
MIN_SILENCIO_S = 0.30   # silencio mas corto que esto no parte el turno
PADDING_S = 0.06        # margen a cada lado: medido, mejora
REJILLA_S = 0.02


def _decisiones(canal: np.ndarray, agresividad: int) -> np.ndarray:
    """Decision de habla por trama de 20 ms para un canal en int16."""
    vad = webrtcvad.Vad(agresividad)
    n = (len(canal) // TRAMA) * TRAMA
    if n == 0:
        return np.zeros(0, dtype=bool)
    crudo = np.ascontiguousarray(canal[:n], dtype=np.int16).tobytes()
    paso = TRAMA * 2  # 2 bytes por muestra
    return np.fromiter(
        (vad.is_speech(crudo[i:i + paso], config.SAMPLE_RATE) for i in range(0, len(crudo), paso)),
        dtype=bool,
        count=n // TRAMA,
    )


def _segmentar(
    decisiones: np.ndarray,
    duracion_s: float,
    *,
    min_habla_s: float = MIN_HABLA_S,
    min_silencio_s: float = MIN_SILENCIO_S,
    padding_s: float = PADDING_S,
) -> list[tuple[float, float]]:
    """Tramas de habla -> segmentos, uniendo silencios cortos y descartando los breves.

    Los parametros son argumentos para poder barrerlos (ver scripts/vad_acuerdo.py).
    """
    min_silencio = max(1, round(min_silencio_s / SALTO_S))
    tramos: list[tuple[int, int]] = []
    en_habla = False
    inicio = 0
    primer_silencio: int | None = None

    for i, hay_habla in enumerate(decisiones):
        if not en_habla:
            if hay_habla:
                en_habla, inicio, primer_silencio = True, i, None
        elif hay_habla:
            primer_silencio = None
        elif primer_silencio is None:
            primer_silencio = i
        elif i - primer_silencio + 1 >= min_silencio:
            tramos.append((inicio, primer_silencio))
            en_habla, primer_silencio = False, None
    if en_habla:
        tramos.append((inicio, len(decisiones)))

    fuera = []
    for a, b in tramos:
        ini = max(0.0, a * SALTO_S - padding_s)
        fin = min(duracion_s, b * SALTO_S + padding_s)
        if fin - ini >= min_habla_s:
            fuera.append((round(ini, 2), round(fin, 2)))

    # portion fusiona los solapes que pueda haber creado el padding.
    return intervalos.tramos(intervalos.union(fuera))


def turnos(x: np.ndarray, sr: int, *, agresividad: int = AGRESIVIDAD, **parametros) -> list[dict]:
    """Turnos de habla por canal, con la misma forma que turns/*.json.

    x: (n, canales) en int16. Devuelve [{"channel", "start", "end"}, ...] por inicio.
    """
    if sr != config.SAMPLE_RATE:
        raise ValueError(f"este VAD trabaja a {config.SAMPLE_RATE} Hz nativos, llego {sr}")
    if x.ndim != 2:
        raise ValueError(f"se esperaba (n, canales), llego {x.shape}")

    duracion_s = len(x) / sr
    salida = []
    for canal in range(x.shape[1]):
        decisiones = _decisiones(x[:, canal], agresividad)
        for inicio, fin in _segmentar(decisiones, duracion_s, **parametros):
            salida.append({"channel": canal, "start": inicio, "end": fin})
    salida.sort(key=lambda t: (t["start"], t["channel"]))
    return salida
