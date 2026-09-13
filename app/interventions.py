"""Unidad de entrenamiento e inferencia: un turno VAD del llamante.

El contexto de silencio termina en el propio turno: nunca lee el turno siguiente.
Los ceros fisicos se conservan; una medicion no disponible se representa con NaN.
"""

from functools import lru_cache

import numpy as np

from app import config, features, vad

EXTRACTOR_VERSION = "interventions-1"
TURN_GROUPS = ("ganancia", "codec_bw", "silencio", "prosodia")
CALL_GROUPS = ("conducta", "razon_canal")
LAYER_NAMES = TURN_GROUPS + CALL_GROUPS
# Alias exactos por definicion del extractor; no se seleccionan mirando val.
ALIASES = ("f0_jitter_praat", "dur_usada_s")


@lru_cache(maxsize=1)
def feature_names() -> tuple[str, ...]:
    return tuple(k for k in features.claves("full") if k not in ALIASES)


def vector(values: dict) -> np.ndarray:
    return np.array([values.get(k, np.nan) for k in feature_names()], dtype=float)


def mark_unavailable(values: dict, x: np.ndarray, sr: int, turns: list[dict]) -> dict:
    """Distingue ausencia de evidencia de una observacion de valor cero."""
    result = dict(values)
    m0 = features._mascara(turns, 0, len(x), sr)
    m1 = features._mascara(turns, 1, len(x), sr)
    no_silence = int((~m0 & ~m1).sum()) < 256
    no_pitch = result.get("f0_mean", 0.0) == 0.0
    for k in result:
        if no_pitch and (
            k.startswith(("f0_", "ac_peak")) or k in ("jitter_local", "shimmer_local", "hnr_db")
        ):
            result[k] = np.nan
        if no_silence and (
            k.startswith(("sil_", "zero_frac_sil", "nuniq_sil"))
            or k in ("rms_noise_db", "noise_std_db", "snr_db", "noise_rms_cv")
        ):
            result[k] = np.nan
        if k.startswith("mod_") and m0.sum() < 256 + 128 * 128:
            result[k] = np.nan
        if k in ("ag_cen", "ag_rms_db") and m1.sum() <= sr:
            result[k] = np.nan
        if k in ("leak_rms_db", "leak_vs_noise", "xcorr") and (m1 & ~m0).sum() <= sr:
            result[k] = np.nan
    if result.get("n_lat", 0) == 0:
        for k in (
            "lat_mean",
            "lat_med",
            "lat_std",
            "lat_min",
            "lat_max",
            "lat_mad",
            "frac_lat_fast",
            "frac_lat_slow",
            "lat_cv",
            "lat_iqr",
        ):
            if k in result:
                result[k] = np.nan
    if result.get("n_lat", 0) < 6:
        for k in ("lat_slope", "lat_drift"):
            if k in result:
                result[k] = np.nan
    return result


def extract_turns(x: np.ndarray, sr: int, turns: list[dict], cache: dict | None = None):
    """Devuelve matriz de turnos y sus fronteras; cero filas si no hay llamante."""
    cache = {} if cache is None else cache
    rows, bounds = [], []
    previous_end = 0.0
    for t in sorted((t for t in turns if t["channel"] == 0), key=lambda t: t["start"]):
        start, end = float(t["start"]), float(t["end"])
        a = max(0, round(max(previous_end, start - 0.30) * sr))
        b = min(len(x), round(end * sr))
        previous_end = end
        if b <= a:
            continue
        key = (a, b, round(start * sr))
        if key not in cache:
            local = [
                {
                    "channel": q["channel"],
                    "start": max(q["start"] - a / sr, 0),
                    "end": min(q["end"] - a / sr, (b - a) / sr),
                }
                for q in turns
                if q["end"] > a / sr and q["start"] < b / sr
            ]
            # Solo el turno actual en ch0; el prefijo puede contener padding del anterior.
            local = [q for q in local if q["channel"] == 1] + [
                {"channel": 0, "start": max(start - a / sr, 0), "end": (b - a) / sr}
            ]
            clip = x[a:b]
            values = features._acustica(clip, sr, local)
            cache[key] = vector(mark_unavailable(values, clip, sr, local))
        rows.append(cache[key])
        bounds.append((start, min(end, len(x) / sr)))
    return np.array(rows, dtype=float).reshape(-1, len(feature_names())), bounds


def extract_sample(
    x: np.ndarray,
    sr: int,
    budget: str = "full",
    *,
    turns: list[dict] | None = None,
    cache: dict | None = None,
) -> dict:
    """Misma ruta para WAV de entrenamiento y scoring del artefacto.

    first_turn necesita el cierre VAD de la primera intervencion (lookahead de silencio
    del VAD). 20s ejecuta el VAD sobre el prefijo, sin conocer fronteras futuras.
    """
    if sr != config.SAMPLE_RATE or x.ndim != 2 or x.shape[1] != config.CHANNELS or not len(x):
        raise ValueError("se esperaba audio estereo no vacio a 8000 Hz")
    if budget not in features.Presupuesto:
        raise ValueError(f"presupuesto desconocido: {budget}")
    if budget == "20s":
        x = x[: 20 * sr]
        turns = vad.turnos(x, sr)  # incluye el turno truncado al llegar al limite
    elif turns is None:
        turns = vad.turnos(x, sr)
    x, turns, duration = features._recortar(x, sr, turns, budget)
    values = features.extraer(x, sr, turns, "full", faltantes=True)
    values = mark_unavailable(values, x, sr, turns)
    rows, bounds = extract_turns(x, sr, turns, cache)
    return {
        "call_features": vector(values),
        "turn_features": rows,
        "turn_bounds": bounds,
        "audio_used_s": duration,
        "budget": budget,
    }
