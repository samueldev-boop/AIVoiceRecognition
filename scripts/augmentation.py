"""Augmentacion telefonica de #11. Solo se importa durante entrenamiento."""

import hashlib
import random
import subprocess
from functools import lru_cache
from pathlib import Path

import numpy as np
import soundfile as sf
from audiomentations import BandPassFilter, ClippingDistortion, Compose, Gain
from scipy.signal import fftconvolve

from app.audio import remuestrear
from scripts.rawboost import ISD_additive_noise, LnL_convolutive_noise

CODECS = {
    "gsm": ("libgsm", "gsm", 8000),
    "amrnb": ("libopencore_amrnb", "amr", 8000),
    "g722": ("g722", "g722", 16000),
    "alaw": ("pcm_alaw", "alaw", 8000),
    "ulaw": ("pcm_mulaw", "mulaw", 8000),
}


def seed_for(call_id, purpose, epoch=0):
    return int(hashlib.sha256(f"11:{call_id}:{purpose}:{epoch}".encode()).hexdigest()[:8], 16)


def pcm(samples):
    return np.rint(np.clip(samples, -1, 32767 / 32768) * 32768).astype(np.int16)


def codec_roundtrip(samples, sr, codec):
    """Codifica y decodifica el llamante; G.722 usa soxr_hq en ambos sentidos."""
    encoder, container, rate = CODECS[codec]
    source = pcm(remuestrear(np.asarray(samples, dtype=np.float32), sr, rate))
    command = ["ffmpeg", "-v", "error", "-nostdin", "-threads", "1"]
    extra = ["-b:a", "12.2k"] if codec == "amrnb" else []
    encoded = subprocess.run(
        command
        + [
            "-f",
            "s16le",
            "-ar",
            str(rate),
            "-ac",
            "1",
            "-i",
            "pipe:0",
            "-c:a",
            encoder,
            *extra,
            "-f",
            container,
            "pipe:1",
        ],
        input=source.tobytes(),
        capture_output=True,
        check=True,
    ).stdout
    # Los formatos PCM crudos no guardan la frecuencia.
    input_rate = ["-ar", str(rate), "-ac", "1"] if codec in ("alaw", "ulaw") else []
    decoded = subprocess.run(
        command
        + [
            "-f",
            container,
            *input_rate,
            "-i",
            "pipe:0",
            "-f",
            "s16le",
            "-c:a",
            "pcm_s16le",
            "pipe:1",
        ],
        input=encoded,
        capture_output=True,
        check=True,
    ).stdout
    result = remuestrear(np.frombuffer(decoded, dtype="<i2").astype(np.float32) / 32768, rate, sr)
    return np.pad(result[: len(samples)], (0, max(0, len(samples) - len(result))))


def resource_files(root, kind, heldout=False):
    folder = "pointsource_noises" if kind == "noise" else "real_rirs_isotropic_noises"
    listing = Path(root) / "RIRS_NOISES" / folder / ("rir_list" if kind == "rir" else "noise_list")
    files = sorted(
        {Path(root) / line.split()[-1] for line in listing.read_text().splitlines() if line.strip()}
    )
    if len(files) < 8:
        raise ValueError(f"faltan recursos OpenSLR 28 en {root}: {kind}")
    # Un archivo de ruido/RIR nunca aparece en entrenamiento y estres a la vez.
    return [p for i, p in enumerate(files) if (i % 4 == 0) == heldout]


@lru_cache(maxsize=32)
def load_resource(path, sr):
    signal, rate = sf.read(path, dtype="float32", always_2d=True)
    return remuestrear(signal, rate, sr)


def room(samples, sr, resources, rng, heldout=False):
    noises = resource_files(resources, "noise", heldout)
    rirs = resource_files(resources, "rir", heldout)
    noise = load_resource(str(noises[int(rng.integers(len(noises)))]), sr)[:, 0]
    impulse = load_resource(str(rirs[int(rng.integers(len(rirs)))]), sr)
    impulse = impulse[:, int(rng.integers(impulse.shape[1]))]
    peak = int(np.argmax(np.abs(impulse)))
    impulse = impulse[peak : peak + sr]
    impulse = impulse / max(float(np.linalg.norm(impulse)), 1e-8)
    wet = fftconvolve(samples, impulse)[: len(samples)]
    wet *= np.linalg.norm(samples) / max(float(np.linalg.norm(wet)), 1e-8)
    offset = int(rng.integers(len(noise)))
    background = np.resize(np.roll(noise, offset), len(samples))
    snr = float(rng.uniform(10, 25))
    background = background * (
        np.linalg.norm(wet) / max(float(np.linalg.norm(background)), 1e-8) / 10 ** (snr / 20)
    )
    return (wet + background).astype(np.float32)


def rawboost(samples, sr):
    """Algoritmo 5 oficial: LnL + ISD; bandas limitadas a Nyquist de 8 kHz."""
    result = LnL_convolutive_noise(
        samples, 5, 5, 20, sr / 2 - 100, 100, 1000, 10, 100, 0, 0, 5, 20, sr
    )
    return ISD_additive_noise(result, 10, 2).astype(np.float32)


def augment(x, sr, label, resources, seed, *, codec=None, heldout=False, use_rawboost=False):
    # Las librerias usan random/np.random globales; cada trabajo corre en su proceso.
    random.seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    caller = x[:, 0].astype(np.float32) / 32768
    if use_rawboost:
        caller = rawboost(caller, sr)
    elif label == 0:
        codec = codec or tuple(CODECS)[int(rng.integers(len(CODECS)))]
        caller = codec_roundtrip(caller, sr, codec)
    else:
        caller = room(caller, sr, resources, rng, heldout)
    if not heldout and not use_rawboost:
        transform = Compose(
            [
                Gain(min_gain_db=-6, max_gain_db=6, p=1),
                ClippingDistortion(min_percentile_threshold=0, max_percentile_threshold=8, p=0.3),
                BandPassFilter(
                    min_center_freq=1500,
                    max_center_freq=2200,
                    min_bandwidth_fraction=1.4,
                    max_bandwidth_fraction=1.8,
                    p=0.3,
                ),
            ]
        )
        caller = transform(samples=caller, sample_rate=sr)
    result = x.copy()
    result[:, 0] = pcm(caller)
    return result
