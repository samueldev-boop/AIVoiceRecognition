"""Deteccion de actividad de voz por canal.

Pendiente: #3. El endpoint recibe solo el WAV, nunca los turns/*.json, asi que las
features de conducta dependen de este modulo.

Medicion que condiciona la implementacion: los tiempos de turns/*.json caen al 100 %
en una rejilla de 20 ms con duracion minima de segmento de 0.2 s. Silero a 8 kHz usa
ventanas de 32 ms, asi que hay que ajustar parametros y medir el acuerdo (IoU) antes
de re-derivar las features de entrenamiento.
"""

import numpy as np


def turnos(x: np.ndarray, sr: int) -> list[dict]:
    """Devuelve [{"channel": 0|1, "start": float, "end": float}, ...] en segundos.

    Misma forma que turns/*.json para que las features sean intercambiables entre el
    dataset y produccion.
    """
    raise NotImplementedError("ver #3: Silero VAD por ONNX en modo 8 kHz nativo")
