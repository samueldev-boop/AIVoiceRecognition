"""Prueba de estres: cuanto de la precision en val es real y cuanto es atajo.

Entrena con el train LIMPIO y evalua el val ATACADO:

  humano_limpio : al humano se le quita lo que lo delata como humano
                  (pasa-bajos 3.4 kHz, normalizacion de pico, silencio a cero
                  digital) -> simula un humano con otro telefono/codec.
                  Lo que salga aqui son FALSOS POSITIVOS.

  bot_evasivo   : al bot se le añade lo que le falta (ruido de sala, jitter de
                  ganancia por turno) -> simula un atacante que sabe que
                  miramos el canal. Lo que salga aqui son FALSOS NEGATIVOS.

  bot_ritmo     : ataque sobre el TIEMPO, no el audio (entra antes, latencias
                  con jitter humano, backchannels cortos interrumpiendo).

Cada ataque es realista: nada aqui requiere mas que ffmpeg y 20 lineas de codigo
por parte del atacante.
"""

import csv
import json
import os
import random

import numpy as np
import soundfile as sf

import audio_probe as AP
import turns_probe as TP

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SR = 8000
random.seed(7)
rng = np.random.default_rng(7)


# ---------------------------------------------------------------- ataques audio
def lowpass(x, fc=3400):
    X = np.fft.rfft(x.astype(np.float64), axis=0)
    fr = np.fft.rfftfreq(len(x), 1 / SR)
    g = np.ones(len(fr))
    g[fr > fc] = 0.0
    taper = (fr > fc - 200) & (fr <= fc)
    g[taper] = np.cos((fr[taper] - (fc - 200)) / 200 * np.pi / 2) ** 2
    return np.fft.irfft(X * g[:, None], n=len(x), axis=0)


def humano_limpio(x, turns):
    """Le quita al humano sus pistas de canal: banda, nivel y ruido de fondo."""
    y = lowpass(x)
    sp = AP.mask_from_turns(turns, 0, len(x))
    # silencio a cero digital (como un noise gate agresivo de VoIP)
    y[~sp, 0] = 0.0
    # normalizacion de pico agresiva, como la salida de un TTS
    peak = np.abs(y[sp, 0]).max() + 1e-9
    y[:, 0] *= 32000 / peak
    return np.clip(y, -32768, 32767).astype(np.int16)


def bot_evasivo(x, turns):
    """Le añade al bot ruido de sala y variacion de nivel por turno."""
    y = x.astype(np.float64).copy()
    n = len(y)
    # ruido rosa a ~-56 dBFS en todo el canal del caller
    w = rng.standard_normal(n)
    W = np.fft.rfft(w)
    fr = np.fft.rfftfreq(n, 1 / SR)
    W /= np.sqrt(np.maximum(fr, 20))
    pink = np.fft.irfft(W, n=n)
    pink *= (10 ** (-56 / 20) * 32768) / (pink.std() + 1e-9)
    y[:, 0] += pink
    # jitter de ganancia por turno (+-3 dB), como un hablante que se mueve
    for t in turns:
        if t["channel"] != 0:
            continue
        a, b = int(t["start"] * SR), min(n, int(t["end"] * SR))
        if b > a:
            y[a:b, 0] *= 10 ** (rng.uniform(-3, 3) / 20)
    return np.clip(y, -32768, 32767).astype(np.int16)


# --------------------------------------------------------------- ataque ritmo
def bot_ritmo(turns, duration):
    """Reescribe los tiempos del caller para imitar conducta humana."""
    ch0 = [dict(t) for t in turns if t["channel"] == 0]
    ch1 = [dict(t) for t in turns if t["channel"] == 1]
    if not ch0 or not ch1:
        return turns
    # 1) entrar antes: pegar el primer turno al final del primer turno del agente
    first_ag_end = min(t["end"] for t in ch1)
    shift = ch0[0]["start"] - (first_ag_end + 0.6)
    for t in ch0:
        t["start"] = max(0.0, t["start"] - shift)
        t["end"] = max(t["start"] + 0.15, t["end"] - shift)
    # 2) jitter humano en las latencias
    for t in ch0:
        j = random.uniform(-0.8, 0.8)
        t["start"] = max(0.0, t["start"] + j)
        t["end"] = max(t["start"] + 0.15, t["end"] + j)
    # 3) backchannels cortos encima del agente (interrupciones)
    extra = []
    for t in ch1:
        if t["end"] - t["start"] > 4 and random.random() < 0.65:
            s = random.uniform(t["start"] + 1, t["end"] - 0.6)
            extra.append({"channel": 0, "start": round(s, 2), "end": round(s + random.uniform(0.2, 0.45), 2)})
    out = sorted(ch0 + extra + ch1, key=lambda t: t["start"])
    return [t for t in out if t["end"] <= duration]


# ------------------------------------------------------------------- modelo
def auc(score, y):
    o = np.argsort(score)
    r = np.empty(len(score), float)
    r[o] = np.arange(1, len(score) + 1)
    n1, n0 = y.sum(), (1 - y).sum()
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def fit(X, y, l2=3.0, iters=4000, lr=0.3):
    X = np.hstack([np.ones((len(X), 1)), X])
    w = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-X @ w))
        g = X.T @ (p - y) / len(y)
        g[1:] += l2 * w[1:] / len(y)
        w -= lr * g
    return w


def pred(w, X):
    return 1 / (1 + np.exp(-(np.hstack([np.ones((len(X), 1)), X]) @ w)))


FRAGIL = ("rms_speech_db", "peak_db", "crest_db", "rms_noise_db", "noise_std_db",
          "leak_rms_db", "leak_vs_noise", "ag_rms_db", "dc_offset", "clip_frac",
          "zero_frac_sil", "zero_frac_all", "nuniq_sil", "noise_rms_cv", "snr_db",
          "n_samp", "speech0_s", "xcorr", "ag_cen")


def es_fragil(k):
    return k in FRAGIL or k.startswith("sil_") or k.startswith("b_") or \
        k.startswith("roll85") or k.startswith("cen_") or k.startswith("flat_") or \
        k.startswith("zcr_") or k.startswith("flux_") or k == "bw_mean"


def main():
    rows = list(csv.DictReader(open(os.path.join(ROOT, "manifest.csv"))))
    turns_of, dur_of, lab, spl = {}, {}, {}, {}
    for r in rows:
        i = r["anon_id"]
        turns_of[i] = json.load(open(os.path.join(ROOT, "turns", i + ".json")))["turns"]
        dur_of[i] = float(r["duration_s"])
        lab[i] = 1 if r["label"] == "synthetic" else 0
        spl[i] = r["split"]

    # features limpias ya calculadas
    td = {r["anon_id"]: r for r in csv.DictReader(open(os.path.join(ROOT, "analysis", "turn_features.csv")))}
    ad = {r["anon_id"]: r for r in csv.DictReader(open(os.path.join(ROOT, "analysis", "audio_features.csv")))}
    tk = [k for k in next(iter(td.values())) if k not in ("anon_id", "label", "split")]
    ak = [k for k in next(iter(ad.values())) if k not in ("anon_id", "label", "split")]
    keys = tk + ak
    ids = sorted(set(td) & set(ad))

    X = np.array([[float(td[i][k]) for k in tk] + [float(ad[i][k]) for k in ak] for i in ids])
    y = np.array([lab[i] for i in ids])
    sp = np.array([spl[i] for i in ids])
    tr = sp == "train"
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9

    robust_mask = np.array([not es_fragil(k) for k in keys])
    Z = np.clip((X - mu) / sd, -6, 6)
    w_full = fit(Z[tr], y[tr])
    w_rob = fit(Z[tr][:, robust_mask], y[tr])
    w_frag = fit(Z[tr][:, ~robust_mask], y[tr])
    beh_mask = np.array([k in tk for k in keys])
    w_beh = fit(Z[tr][:, beh_mask], y[tr])
    print(f"features: {len(keys)} total, {int(robust_mask.sum())} en el set robusto")

    val = [i for i in ids if spl[i] == "val"]
    print(f"val: {len(val)} llamadas\n")

    # ---- recalcula features de val bajo cada ataque
    variants = {}
    for name in ("limpio", "humano_limpio", "bot_evasivo", "bot_ritmo"):
        feats = {}
        for i in val:
            t = turns_of[i]
            if name == "limpio":
                feats[i] = (dict(zip(tk, [float(td[i][k]) for k in tk])),
                            dict(zip(ak, [float(ad[i][k]) for k in ak])))
                continue
            if name == "bot_ritmo":
                if lab[i] == 0:
                    feats[i] = (dict(zip(tk, [float(td[i][k]) for k in tk])),
                                dict(zip(ak, [float(ad[i][k]) for k in ak])))
                    continue
                t2 = bot_ritmo(t, dur_of[i])
                tf = TP.features(t2, dur_of[i])
                feats[i] = (tf, dict(zip(ak, [float(ad[i][k]) for k in ak])))
                continue
            # ataques de audio
            target_human = (name == "humano_limpio")
            if (lab[i] == 0) != target_human:
                feats[i] = (dict(zip(tk, [float(td[i][k]) for k in tk])),
                            dict(zip(ak, [float(ad[i][k]) for k in ak])))
                continue
            x, _ = sf.read(os.path.join(ROOT, "audio", i + ".wav"), dtype="int16", always_2d=True)
            x2 = humano_limpio(x, t) if target_human else bot_evasivo(x, t)
            af = AP.features_from_array(x2, t)
            feats[i] = (dict(zip(tk, [float(td[i][k]) for k in tk])), af)
        variants[name] = feats
        print(f"  calculado: {name}", flush=True)

    chk = ["b_3k4_4k", "sil_cen_std", "rms_speech_db", "noise_rms_cv", "onset0", "lat_cv"]
    print("\n=== verificacion: el ataque movio las features? (medias en val) ===")
    print(f"{'feature':<16}" + "".join(f"{n[:13]:>14}" for n in variants))
    for k in chk:
        line = f"{k:<16}"
        for n, feats in variants.items():
            src = 0 if k in tk else 1
            tgt = 0 if n == "humano_limpio" else 1
            vals = [feats[i][src].get(k, 0) for i in val if lab[i] == tgt] if n != "limpio" else \
                   [feats[i][src].get(k, 0) for i in val]
            line += f"{np.mean(vals):>14.4g}"
        print(line)

    print(f"\n{'escenario':<18}{'modelo':<10}{'AUC':>8}{'acc':>8}{'FP':>5}{'FN':>5}"
          f"{'FPR':>8}{'FNR':>8}")
    print("-" * 70)
    for name, feats in variants.items():
        Xv = np.array([[feats[i][0].get(k, 0) for k in tk] + [feats[i][1].get(k, 0) for k in ak]
                       for i in val])
        yv = np.array([lab[i] for i in val])
        Zv = np.clip((Xv - mu) / sd, -6, 6)
        for mname, w, m in (("completo", w_full, np.ones(len(keys), bool)),
                            ("robusto", w_rob, robust_mask),
                            ("solo-atajo", w_frag, ~robust_mask),
                            ("solo-conducta", w_beh, beh_mask)):
            p = pred(w, Zv[:, m])
            pr = (p >= 0.5).astype(int)
            fp = int(((pr == 1) & (yv == 0)).sum())
            fn = int(((pr == 0) & (yv == 1)).sum())
            print(f"{name:<18}{mname:<10}{auc(p, yv):>8.4f}{(pr == yv).mean():>8.3f}"
                  f"{fp:>5}{fn:>5}{fp/max(1,(yv==0).sum()):>8.3f}{fn/max(1,(yv==1).sum()):>8.3f}")


if __name__ == "__main__":
    main()
