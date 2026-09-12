"""Decodificacion y validacion del clip de entrada.

Etapa 0 de la cascada: si el clip no cumple el formato, se responde con confianza baja
en lugar de adivinar. Ver #6.
"""

import base64
import binascii
import io

import numpy as np
import soundfile as sf

from app import config


class AudioInvalido(ValueError):
    """El clip no cumple el formato esperado."""


def decodificar(audio_b64: str) -> tuple[np.ndarray, int]:
    """base64 -> (muestras int16 de forma (n, 2), sample rate).

    Devuelve el array tal cual viene: canal 0 = llamante, canal 1 = agente.
    """
    try:
        crudo = base64.b64decode(audio_b64, validate=True)
    except (binascii.Error, ValueError) as e:
        raise AudioInvalido(f"base64 invalido: {e}") from e

    if not crudo:
        raise AudioInvalido("cuerpo vacio")

    try:
        x, sr = sf.read(io.BytesIO(crudo), dtype="int16", always_2d=True)
    except Exception as e:
        raise AudioInvalido(f"no es un WAV legible: {e}") from e

    return x, sr


def leer_wav(ruta: str) -> tuple[np.ndarray, int]:
    """Lee un WAV del disco. Misma salida que decodificar(), para reusar validar()."""
    try:
        return sf.read(ruta, dtype="int16", always_2d=True)
    except Exception as e:
        raise AudioInvalido(f"no puedo leer {ruta}: {e}") from e


def validar(x: np.ndarray, sr: int) -> float:
    """Comprueba el formato y devuelve la duracion en segundos.

    Lanza AudioInvalido con el motivo concreto, que el endpoint traduce a una respuesta
    degradada en lugar de a un error opaco.
    """
    if sr != config.SAMPLE_RATE:
        raise AudioInvalido(f"sample rate {sr}, se esperaba {config.SAMPLE_RATE}")

    if x.ndim != 2 or x.shape[1] != config.CHANNELS:
        raise AudioInvalido(f"se esperaban {config.CHANNELS} canales, hay {x.shape[-1]}")

    duracion = len(x) / sr
    if not config.MIN_DURATION_S <= duracion <= config.MAX_DURATION_S:
        raise AudioInvalido(
            f"duracion {duracion:.1f}s fuera del rango "
            f"[{config.MIN_DURATION_S}, {config.MAX_DURATION_S}]"
        )

    # Un mono duplicado en los dos canales no es una llamada: no hay nada que comparar.
    if np.array_equal(x[:, 0], x[:, 1]):
        raise AudioInvalido("los dos canales son identicos")

    if not np.any(x[:, 0]):
        raise AudioInvalido("el canal del llamante esta en silencio absoluto")

    return duracion
