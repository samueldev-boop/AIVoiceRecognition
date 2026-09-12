"""Extrae features conversacionales de turns/*.json y mide su poder discriminativo.

Solo stdlib. Canal 0 = quien llama (a clasificar), canal 1 = agente.
El AUC se calcula SOLO sobre el split train para no contaminar val.
"""

import csv
import json
import os
from statistics import mean, median, pstdev

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_manifest():
    with open(os.path.join(ROOT, "manifest.csv")) as f:
        return list(csv.DictReader(f))


def merge(segs):
    """Une segmentos solapados de un mismo canal."""
    out = []
    for s, e in sorted(segs):
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out


def overlap_len(a, b):
    """Tiempo total de solape entre dos listas de intervalos (ya mergeadas)."""
    i = j = 0
    tot = 0.0
    while i < len(a) and j < len(b):
        s = max(a[i][0], b[j][0])
        e = min(a[i][1], b[j][1])
        if e > s:
            tot += e - s
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return tot


def safe(fn, xs, default=0.0):
    try:
        return float(fn(xs)) if xs else default
    except Exception:
        return default


def features(turns, duration):
    ch0 = [(t["start"], t["end"]) for t in turns if t["channel"] == 0]
    ch1 = [(t["start"], t["end"]) for t in turns if t["channel"] == 1]
    f = {}
    if not ch0 or not ch1:
        return None

    m0, m1 = merge(ch0), merge(ch1)
    d0 = [e - s for s, e in ch0]
    d1 = [e - s for s, e in ch1]
    sp0 = sum(e - s for s, e in m0)
    sp1 = sum(e - s for s, e in m1)

    # --- volumen de habla ---
    f["dur_call"] = duration
    f["n_seg0"] = len(ch0)
    f["n_seg1"] = len(ch1)
    f["speech0_s"] = sp0
    f["speech_ratio0"] = sp0 / duration
    f["speech_ratio1"] = sp1 / duration
    f["talk_balance"] = sp0 / (sp0 + sp1)
    f["seg_rate0"] = len(ch0) / duration * 60  # segmentos/min

    # --- forma de los segmentos del caller ---
    f["seg0_mean"] = mean(d0)
    f["seg0_med"] = median(d0)
    f["seg0_std"] = safe(pstdev, d0)
    f["seg0_cv"] = f["seg0_std"] / f["seg0_mean"] if f["seg0_mean"] else 0
    f["seg0_max"] = max(d0)
    f["seg0_min"] = min(d0)
    f["seg0_p90"] = sorted(d0)[int(0.9 * (len(d0) - 1))]
    f["frac_short0"] = sum(1 for d in d0 if d < 0.5) / len(d0)   # backchannels
    f["frac_vshort0"] = sum(1 for d in d0 if d < 0.3) / len(d0)
    f["frac_long0"] = sum(1 for d in d0 if d > 8) / len(d0)

    # mismo set para el agente (control: deberia ser ~igual en ambas clases)
    f["seg1_mean"] = mean(d1)
    f["seg1_cv"] = safe(pstdev, d1) / mean(d1) if mean(d1) else 0

    # --- latencia de respuesta: fin de turno del agente -> inicio del caller ---
    lat = []
    for s0, _ in m0:
        prev = [e1 for _, e1 in m1 if e1 <= s0 + 1e-9]
        if prev:
            g = s0 - max(prev)
            if 0 <= g < 15:
                lat.append(g)
    f["n_lat"] = len(lat)
    f["lat_mean"] = safe(mean, lat)
    f["lat_med"] = safe(median, lat)
    f["lat_std"] = safe(pstdev, lat)
    f["lat_cv"] = f["lat_std"] / f["lat_mean"] if f["lat_mean"] else 0
    f["lat_min"] = safe(min, lat)
    f["lat_max"] = safe(max, lat)
    f["lat_iqr"] = (
        sorted(lat)[int(0.75 * (len(lat) - 1))] - sorted(lat)[int(0.25 * (len(lat) - 1))]
        if len(lat) > 3 else 0
    )
    f["frac_lat_fast"] = sum(1 for g in lat if g < 0.35) / len(lat) if lat else 0
    f["frac_lat_slow"] = sum(1 for g in lat if g > 2.0) / len(lat) if lat else 0
    # regularidad: desvio absoluto medio respecto a la mediana (robusto)
    f["lat_mad"] = safe(lambda x: mean([abs(v - f["lat_med"]) for v in x]), lat)

    # --- solapes e interrupciones ---
    ov = overlap_len(m0, m1)
    f["overlap_s"] = ov
    f["overlap_ratio0"] = ov / sp0 if sp0 else 0
    f["overlap_per_min"] = ov / duration * 60
    # caller arranca mientras el agente habla
    f["n_barge0"] = sum(1 for s0, _ in m0 if any(s1 < s0 < e1 for s1, e1 in m1))
    f["barge0_rate"] = f["n_barge0"] / len(m0)
    f["n_barge1"] = sum(1 for s1, _ in m1 if any(s0 < s1 < e0 for s0, e0 in m0))
    f["barge1_rate"] = f["n_barge1"] / len(m1)

    # --- fragmentacion interna del caller (titubeo) ---
    gaps00 = [ch0[i + 1][0] - ch0[i][1] for i in range(len(ch0) - 1)]
    gaps00 = [g for g in gaps00 if 0 < g < 2.0]  # micro-pausas dentro del mismo turno
    f["n_micropause0"] = len(gaps00)
    f["micropause_rate0"] = len(gaps00) / len(ch0)
    f["micropause_mean"] = safe(mean, gaps00)

    # --- silencios globales ---
    allm = merge(ch0 + ch1)
    f["silence_ratio"] = 1 - sum(e - s for s, e in allm) / duration
    f["onset0"] = m0[0][0]          # cuando habla por primera vez el caller
    f["onset1"] = m1[0][0]
    f["tail_silence"] = duration - max(allm[-1][1], 0)

    # --- deriva temporal de la latencia (bots: plana) ---
    if len(lat) >= 6:
        n = len(lat)
        xs = list(range(n))
        mx, my = mean(xs), mean(lat)
        den = sum((x - mx) ** 2 for x in xs)
        f["lat_slope"] = sum((xs[i] - mx) * (lat[i] - my) for i in range(n)) / den if den else 0
        h = n // 2
        f["lat_drift"] = mean(lat[h:]) - mean(lat[:h])
    else:
        f["lat_slope"] = 0
        f["lat_drift"] = 0
    return f


def auc(pos, neg):
    """AUC por rangos (Mann-Whitney). pos = synthetic, neg = human."""
    xs = sorted([(v, 1) for v in pos] + [(v, 0) for v in neg])
    ranks = {}
    i = 0
    r = 0.0
    while i < len(xs):
        j = i
        while j < len(xs) - 1 and xs[j + 1][0] == xs[i][0]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks.setdefault(k, avg)
        i = j + 1
    s = sum(ranks[k] for k in range(len(xs)) if xs[k][1] == 1)
    n1, n0 = len(pos), len(neg)
    return (s - n1 * (n1 + 1) / 2) / (n1 * n0)


def main():
    rows = load_manifest()
    data = []
    for r in rows:
        p = os.path.join(ROOT, "turns", r["anon_id"] + ".json")
        turns = json.load(open(p))["turns"]
        f = features(turns, float(r["duration_s"]))
        if f is None:
            print("SIN TURNOS EN UN CANAL:", r["anon_id"], r["label"], r["split"])
            continue
        data.append((r, f))

    keys = list(data[0][1].keys())
    print(f"{len(data)} llamadas con features, {len(keys)} features\n")

    for split in ("train", "val"):
        sub = [(r, f) for r, f in data if r["split"] == split]
        pos = [f for r, f in sub if r["label"] == "synthetic"]
        neg = [f for r, f in sub if r["label"] == "human"]
        print(f"=== {split}: {len(pos)} synthetic vs {len(neg)} human ===")
        res = []
        for k in keys:
            a = auc([f[k] for f in pos], [f[k] for f in neg])
            res.append((abs(a - 0.5), a, k,
                        mean([f[k] for f in neg]), mean([f[k] for f in pos])))
        res.sort(reverse=True)
        print(f"{'feature':<18}{'AUC':>7}{'|sep|':>7}{'human':>10}{'synth':>10}")
        for sep, a, k, mh, ms in res:
            flag = "  <<<" if sep > 0.15 else ""
            print(f"{k:<18}{a:>7.3f}{sep:>7.3f}{mh:>10.2f}{ms:>10.2f}{flag}")
        print()

    # guarda features para reuso
    out = os.path.join(ROOT, "analysis", "turn_features.csv")
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["anon_id", "label", "split"] + keys)
        for r, f in data:
            w.writerow([r["anon_id"], r["label"], r["split"]] + [f[k] for k in keys])
    print("features ->", out)


if __name__ == "__main__":
    main()
