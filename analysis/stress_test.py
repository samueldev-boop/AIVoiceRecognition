"""Banco de estres #5: ataques deterministas, evaluados fuera de muestra en train.

Los ataques de audio vuelven a pasar por el VAD y extractor de produccion.
bot_ritmo es una ablacion de metadatos de conducta; NO simula audio retemporizado.
Ejecutar: python -m analysis.stress_test
"""

import numpy as np

SR = 8000


def mask_from_turns(turns, channel, n):
    mask = np.zeros(n, bool)
    for t in turns:
        if t["channel"] == channel:
            mask[max(0, round(t["start"] * SR)) : min(n, round(t["end"] * SR))] = True
    return mask


def lowpass(x, fc=3400):
    spectrum = np.fft.rfft(x.astype(float), axis=0)
    frequency = np.fft.rfftfreq(len(x), 1 / SR)
    gain = np.ones(len(frequency))
    gain[frequency > fc] = 0
    taper = (frequency > fc - 200) & (frequency <= fc)
    gain[taper] = np.cos((frequency[taper] - (fc - 200)) / 200 * np.pi / 2) ** 2
    return np.fft.irfft(spectrum * gain[:, None], n=len(x), axis=0)


def humano_limpio(x, turns):
    y = lowpass(x)
    speech = mask_from_turns(turns, 0, len(x))
    y[~speech, 0] = 0
    if speech.any():
        peak = np.max(np.abs(y[speech, 0]))
        if peak > 0:
            y[:, 0] *= 32000 / peak
    return np.clip(y, -32768, 32767).astype(np.int16)


def bot_evasivo(x, turns, *, seed=7):
    rng = np.random.default_rng(seed)
    y = x.astype(float).copy()
    noise = np.fft.rfft(rng.standard_normal(len(y)))
    frequency = np.fft.rfftfreq(len(y), 1 / SR)
    noise /= np.sqrt(np.maximum(frequency, 20))
    pink = np.fft.irfft(noise, n=len(y))
    pink *= (10 ** (-56 / 20) * 32768) / (pink.std() + 1e-9)
    y[:, 0] += pink
    for t in turns:
        if t["channel"] == 0:
            a, b = max(0, round(t["start"] * SR)), min(len(y), round(t["end"] * SR))
            y[a:b, 0] *= 10 ** (rng.uniform(-3, 3) / 20)
    return np.clip(y, -32768, 32767).astype(np.int16)


def bot_ritmo(turns, duration, *, seed=7):
    rng = np.random.default_rng(seed)
    caller = sorted((dict(t) for t in turns if t["channel"] == 0), key=lambda t: t["start"])
    agent = [dict(t) for t in turns if t["channel"] == 1]
    if not caller or not agent:
        return turns
    shift = caller[0]["start"] - (min(t["end"] for t in agent) + 0.6)
    for t in caller:
        delta = -shift + rng.uniform(-0.8, 0.8)
        length = max(0.15, t["end"] - t["start"])
        t["start"] = max(0, t["start"] + delta)
        t["end"] = min(duration, t["start"] + length)
    extra = []
    for t in agent:
        if t["end"] - t["start"] > 4 and rng.random() < 0.65:
            start = rng.uniform(t["start"] + 1, t["end"] - 0.6)
            extra.append(
                {"channel": 0, "start": start, "end": min(duration, start + rng.uniform(0.2, 0.45))}
            )
    return sorted(
        (t for t in caller + extra + agent if 0 <= t["start"] < t["end"] <= duration),
        key=lambda t: (t["start"], t["channel"]),
    )


if __name__ == "__main__":
    import sys
    from pathlib import Path

    if __package__ in (None, ""):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from scripts.train_issue5 import main

    main()
