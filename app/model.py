"""Carga del modelo entrenado y puntuacion por capas.

Pendiente: #5. Aqui solo esta la carga del artefacto, para que /health pueda informar
si hay modelo disponible y el endpoint responda de forma honesta cuando no lo hay.
"""

import os

import joblib

from app import config


class Modelo:
    """Envoltorio del artefacto entrenado."""

    def __init__(self, bundle: dict) -> None:
        self.bundle = bundle
        self.version: str = bundle.get("version", "desconocida")
        self.features: list[str] = bundle.get("features", [])

    def puntuar(self, features: dict) -> tuple[float, dict[str, float]]:
        """Devuelve (probabilidad de sintetico, score por capa)."""
        raise NotImplementedError("ver #5: fusion calibrada sobre scores de capa")


def cargar() -> Modelo | None:
    """Carga el modelo si el artefacto existe. Devuelve None si todavia no hay."""
    ruta = config.MODEL_PATH
    if not os.path.exists(ruta):
        return None
    return Modelo(joblib.load(ruta))
