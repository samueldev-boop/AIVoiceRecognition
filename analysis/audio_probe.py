"""Extrae features acusticas de audio/*.wav y mide su poder discriminativo.

Usa los turns/*.json para separar region de habla vs silencio del canal 0 (caller).
Requiere numpy + soundfile (.venv). AUC solo sobre train.
"""

import csv
import json
import os
import sys

import numpy as np
import soundfile as sf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SR = 8000
EPS = 1e-12


def mask_from_turns(turns, channel, n):
    m = np.zeros(n, dtype=bool)
    for t in turns:
        if t["channel"] == channel:
            a = max(0, int(t["start"] * SR))
            b = min(n, int(t["end"] * SR))
            if b > a:
                m[a:b] = True
    return m


def db(x):
    return 20 * np.log10(max(float(x), EPS))


def frames(x, n=256, hop=128):
    if len(x) < n:
        return np.zeros((0, n))
    k = 1 + (len(x) - n) // hop
    idx = np.arange(n)[None, :] + hop * np.arange(k)[:, None]
    return x[idx]


def spectral_stats(x):
    """Estadisticos espectrales sobre frames de 32 ms."""
    F = frames(x)
    out = {}
    if len(F) == 0:
        return out
    w = np.hanning(F.shape[1])
    S = np.abs(np.fft.rfft(F * w, axis=1)) ** 2
    freqs = np.fft.rfftfreq(F.shape[1], 1 / SR)
    tot = S.sum(axis=1) + EPS
    # centroide y ancho de banda
    cen = (S * freqs).sum(axis=1) / tot
    bw = np.sqrt((S * (freqs[None, :] - cen[:, None]) ** 2).sum(axis=1) / tot)
    # rolloff 85%
    cs = np.cumsum(S, axis=1) / tot[:, None]
    roll = freqs[np.argmax(cs >= 0.85, axis=1)]
    # planitud espectral (geom/arit)
    flat = np.exp(np.mean(np.log(S + EPS), axis=1)) / (S.mean(axis=1) + EPS)
    # energia por banda telefonica
    def band(lo, hi):
        sel = (freqs >= lo) & (freqs < hi)
        return S[:, sel].sum(axis=1) / tot
    out["cen_mean"] = cen.mean()
    out["cen_std"] = cen.std()
    out["bw_mean"] = bw.mean()
    out["roll85_mean"] = roll.mean()
    out["roll85_std"] = roll.std()
    out["flat_mean"] = flat.mean()
    out["flat_std"] = flat.std()
    out["b_0_300"] = band(0, 300).mean()
    out["b_300_1k"] = band(300, 1000).mean()
    out["b_1k_2k"] = band(1000, 2000).mean()
    out["b_2k_3k"] = band(2000, 3000).mean()
    out["b_3k_4k"] = band(3000, 4000).mean()
    out["b_3k4_4k"] = band(3400, 4000).mean()  # sobre el corte telefonico
    # flujo espectral (variabilidad temporal del espectro)
    Sn = S / tot[:, None]
    flux = np.sqrt(((Sn[1:] - Sn[:-1]) ** 2).sum(axis=1))
    out["flux_mean"] = flux.mean() if len(flux) else 0
    out["flux_std"] = flux.std() if len(flux) else 0
    return out


def f0_track(x):
    """F0 por autocorrelacion en frames de 40 ms (50-320 Hz)."""
    n, hop = 320, 160
    F = frames(x, n, hop)
    if len(F) == 0:
        return np.array([]), np.array([])
    F = F - F.mean(axis=1, keepdims=True)
    e = np.sqrt((F ** 2).mean(axis=1))
    lo, hi = int(SR / 320), int(SR / 50)  # 25..160
    nfft = 1024
    S = np.fft.rfft(F, nfft, axis=1)
    ac = np.fft.irfft(S * np.conj(S), nfft, axis=1)[:, :hi + 1]
    ac0 = ac[:, :1] + EPS
    acn = ac / ac0
    lag = lo + np.argmax(acn[:, lo:hi + 1], axis=1)
    peak = acn[np.arange(len(F)), lag]
    f0 = SR / lag
    voiced = (peak > 0.35) & (e > np.percentile(e, 25))
    return f0[voiced], peak[voiced]


def features(path, turns, transform=None):
    x, sr = sf.read(path, dtype="int16", always_2d=True)
    if transform is not None:
        x = transform(x)
    return features_from_array(x, turns)


def features_from_array(x, turns):
    n = len(x)
    f = {"n_samp": n}
    c0 = x[:, 0].astype(np.float64)
    c1 = x[:, 1].astype(np.float64)

    sp = mask_from_turns(turns, 0, n)
    sil = ~sp & ~mask_from_turns(turns, 1, n)  # silencio real de ambos canales

    s0 = c0[sp]
    z0 = c0[sil]
    if len(s0) < SR or len(z0) < SR // 2:
        return None

    # --- niveles y ruido ---
    f["rms_speech_db"] = db(np.sqrt((s0 ** 2).mean()) / 32768)
    f["rms_noise_db"] = db(np.sqrt((z0 ** 2).mean()) / 32768)
    f["snr_db"] = f["rms_speech_db"] - f["rms_noise_db"]
    f["peak_db"] = db(np.abs(s0).max() / 32768)
    f["crest_db"] = f["peak_db"] - f["rms_speech_db"]
    f["dc_offset"] = c0.mean() / 32768
    f["clip_frac"] = float((np.abs(s0) >= 32700).mean())

    # --- silencio digital: TTS/mezcla sintetica deja ceros exactos ---
    f["zero_frac_sil"] = float((z0 == 0).mean())
    f["zero_frac_all"] = float((c0 == 0).mean())
    f["nuniq_sil"] = len(np.unique(z0)) / max(len(z0), 1)
    f["noise_std_db"] = db(z0.std() / 32768)
    zf = frames(z0)
    if len(zf):
        rz = np.sqrt((zf ** 2).mean(axis=1) + EPS)
        f["noise_rms_cv"] = float(rz.std() / (rz.mean() + EPS))  # estabilidad del piso de ruido
    else:
        f["noise_rms_cv"] = 0.0
    f.update({"sil_" + k: v for k, v in spectral_stats(z0).items()})

    # --- espectro del habla ---
    f.update(spectral_stats(s0))

    # --- cruces por cero ---
    sfr = frames(s0)
    if len(sfr):
        zcr = (np.diff(np.sign(sfr), axis=1) != 0).mean(axis=1)
        f["zcr_mean"] = zcr.mean()
        f["zcr_std"] = zcr.std()
        r = np.sqrt((sfr ** 2).mean(axis=1) + EPS)
        f["env_cv"] = float(r.std() / r.mean())
        ld = 20 * np.log10(r / 32768 + EPS)
        f["env_range_db"] = float(np.percentile(ld, 95) - np.percentile(ld, 5))
        # espectro de modulacion de la envolvente (ritmo silabico ~4 Hz)
        env = r - r.mean()
        if len(env) > 128:
            M = np.abs(np.fft.rfft(env * np.hanning(len(env))))
            mf = np.fft.rfftfreq(len(env), 128 / SR)
            tot = M.sum() + EPS
            f["mod_2_6hz"] = float(M[(mf >= 2) & (mf < 6)].sum() / tot)
            f["mod_6_12hz"] = float(M[(mf >= 6) & (mf < 12)].sum() / tot)
            f["mod_peak_hz"] = float(mf[1 + np.argmax(M[1:])])
        else:
            f["mod_2_6hz"] = f["mod_6_12hz"] = f["mod_peak_hz"] = 0.0

    # --- prosodia / F0 ---
    f0, pk = f0_track(s0)
    if len(f0) > 20:
        f["f0_mean"] = f0.mean()
        f["f0_std"] = f0.std()
        f["f0_cv"] = f0.std() / f0.mean()
        f["f0_p5"] = np.percentile(f0, 5)
        f["f0_p95"] = np.percentile(f0, 95)
        f["f0_range"] = f["f0_p95"] - f["f0_p5"]
        d = np.abs(np.diff(f0)) / f0[:-1]
        f["f0_jitter"] = float(np.median(d))          # micro-variacion (humano > TTS)
        f["f0_smooth"] = float((d < 0.02).mean())     # contornos demasiado suaves = TTS
        f["voiced_frac"] = len(f0) / max(1, len(frames(s0, 320, 160)))
        f["ac_peak_mean"] = pk.mean()                 # periodicidad (TTS mas periodico)
        f["ac_peak_std"] = pk.std()
    else:
        for k in ("f0_mean", "f0_std", "f0_cv", "f0_p5", "f0_p95", "f0_range",
                  "f0_jitter", "f0_smooth", "voiced_frac", "ac_peak_mean", "ac_peak_std"):
            f[k] = 0.0

    # --- cruce de canales: fuga / eco del agente en el canal del caller ---
    sp1 = mask_from_turns(turns, 1, n) & ~sp
    if sp1.sum() > SR:
        leak = c0[sp1]
        f["leak_rms_db"] = db(np.sqrt((leak ** 2).mean()) / 32768)
        f["leak_vs_noise"] = f["leak_rms_db"] - f["rms_noise_db"]
        a, b = c0[sp1][:200000], c1[sp1][:200000]
        if a.std() > 0 and b.std() > 0:
            f["xcorr"] = float(np.corrcoef(a, b)[0, 1])
        else:
            f["xcorr"] = 0.0
    else:
        f["leak_rms_db"] = f["leak_vs_noise"] = f["xcorr"] = 0.0

    # --- control: mismas medidas basicas en el canal del agente ---
    s1 = c1[mask_from_turns(turns, 1, n)]
    if len(s1) > SR:
        f["ag_rms_db"] = db(np.sqrt((s1 ** 2).mean()) / 32768)
        f["ag_cen"] = spectral_stats(s1).get("cen_mean", 0)
    else:
        f["ag_rms_db"] = f["ag_cen"] = 0.0
    return f


def auc(pos, neg):
    xs = sorted([(v, 1) for v in pos] + [(v, 0) for v in neg])
    i, ranks = 0, {}
    while i < len(xs):
        j = i
        while j < len(xs) - 1 and xs[j + 1][0] == xs[i][0]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[k] = avg
        i = j + 1
    s = sum(ranks[k] for k in range(len(xs)) if xs[k][1] == 1)
    n1, n0 = len(pos), len(neg)
    return (s - n1 * (n1 + 1) / 2) / (n1 * n0)


def main():
    rows = list(csv.DictReader(open(os.path.join(ROOT, "manifest.csv"))))
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else len(rows)
    data = []
    for i, r in enumerate(rows[:limit]):
        turns = json.load(open(os.path.join(ROOT, "turns", r["anon_id"] + ".json")))["turns"]
        try:
            f = features(os.path.join(ROOT, "audio", r["anon_id"] + ".wav"), turns)
        except Exception as e:
            print("ERR", r["anon_id"], e)
            continue
        if f:
            data.append((r, f))
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{limit}", flush=True)

    keys = [k for k in data[0][1].keys()]
    out = os.path.join(ROOT, "analysis", "audio_features.csv")
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["anon_id", "label", "split"] + keys)
        for r, f in data:
            w.writerow([r["anon_id"], r["label"], r["split"]] + [f.get(k, 0) for k in keys])

    print(f"\n{len(data)} llamadas, {len(keys)} features acusticas -> {out}\n")
    for split in ("train", "val"):
        sub = [(r, f) for r, f in data if r["split"] == split]
        pos = [f for r, f in sub if r["label"] == "synthetic"]
        neg = [f for r, f in sub if r["label"] == "human"]
        if not pos or not neg:
            continue
        res = []
        for k in keys:
            a = auc([f.get(k, 0) for f in pos], [f.get(k, 0) for f in neg])
            res.append((abs(a - 0.5), a, k,
                        float(np.mean([f.get(k, 0) for f in neg])),
                        float(np.mean([f.get(k, 0) for f in pos]))))
        res.sort(reverse=True)
        print(f"=== {split}: {len(pos)} synth vs {len(neg)} human ===")
        print(f"{'feature':<18}{'AUC':>7}{'|sep|':>7}{'human':>12}{'synth':>12}")
        for sep, a, k, mh, ms in res[:30]:
            print(f"{k:<18}{a:>7.3f}{sep:>7.3f}{mh:>12.4g}{ms:>12.4g}")
        print()


if __name__ == "__main__":
    main()
