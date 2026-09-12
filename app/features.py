"""Extractor unico de features, con presupuesto de audio.

Pendiente: #4. Un solo punto de entrada para que el entrenamiento y el endpoint usen
exactamente el mismo codigo, y para que la cascada de #6 pueda pedir el subconjunto
que le cabe en su etapa.
"""

from typing import Literal

import numpy as np

Presupuesto = Literal["first_turn", "20s", "full"]


def extraer(x: np.ndarray, sr: int, turnos: list[dict], presupuesto: Presupuesto) -> dict:
    """Vector de features del llamante para el presupuesto de audio indicado."""
    raise NotImplementedError("ver #4: unificar audio_probe y turns_probe aqui")
