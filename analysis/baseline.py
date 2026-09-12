"""Baseline: regresion logistica L2 sobre las features ya extraidas.

Entrena en train, evalua en val. Reporta AUC, accuracy y -- lo que mas importa --
la tasa de falsos positivos (humano marcado como sintetico) a distintos umbrales.
"""

import csv
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
A = os.path.join(ROOT, "analysis")


def load(name):
    p = os.path.join(A, name)
    if not os.path.exists(p):
        return {}, []
    rows = list(csv.DictReader(open(p)))
    keys = [k for k in rows[0] if k not in ("anon_id", "label", "split")]
    return {r["anon_id"]: r for r in rows}, keys


def build(sets):
    """Une varios csv de features por anon_id."""
    base = sets[0][0]
    ids = [i for i in base if all(i in s[0] for s, in [(s,) for s in sets])]
    keys = []
    for _, k in sets:
        keys += k
    X, y, sp, kept = [], [], [], []
    for i in ids:
        row = []
        ok = True
        for d, k in sets:
            if i not in d:
                ok = False
                break
            row += [float(d[i][kk]) for kk in k]
        if not ok:
            continue
        X.append(row)
        y.append(1 if base[i]["label"] == "synthetic" else 0)
        sp.append(base[i]["split"])
        kept.append(i)
    return np.array(X), np.array(y), np.array(sp), keys, kept


def auc(score, y):
    o = np.argsort(score)
    r = np.empty(len(score), float)
    r[o] = np.arange(1, len(score) + 1)
    # promedio de rangos en empates
    for v in np.unique(score):
        m = score == v
        if m.sum() > 1:
            r[m] = r[m].mean()
    n1, n0 = y.sum(), (1 - y).sum()
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def fit(X, y, l2=1.0, iters=4000, lr=0.3):
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


def report(name, p, y):
    print(f"\n--- {name} ---")
    print(f"AUC = {auc(p, y):.4f}   (n={len(y)}, {int(y.sum())} synth / {int((1-y).sum())} human)")
    print(f"{'umbral':>7}{'acc':>8}{'FPR':>8}{'FNR':>8}{'recall':>8}  interpretacion")
    for t in (0.3, 0.5, 0.7, 0.9, 0.95):
        pr = (p >= t).astype(int)
        fp = int(((pr == 1) & (y == 0)).sum())
        fn = int(((pr == 0) & (y == 1)).sum())
        acc = (pr == y).mean()
        fpr = fp / max(1, (y == 0).sum())
        fnr = fn / max(1, (y == 1).sum())
        print(f"{t:>7.2f}{acc:>8.3f}{fpr:>8.3f}{fnr:>8.3f}{1-fnr:>8.3f}"
              f"   {fp} humanos marcados bot, {fn} bots colados")


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    turns = load("turn_features.csv")
    audio = load("audio_features.csv")
    sets = {"turns": [turns], "audio": [audio], "both": [turns, audio]}[which]
    sets = [s for s in sets if s[1]]
    X, y, sp, keys, ids = build(sets)
    print(f"conjunto '{which}': {X.shape[0]} llamadas x {X.shape[1]} features")

    tr, va = sp == "train", sp == "val"
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
    Z = (X - mu) / sd
    Z = np.clip(Z, -6, 6)  # recorta outliers

    w = fit(Z[tr], y[tr], l2=3.0)
    report("train", pred(w, Z[tr]), y[tr])
    report("val (nunca visto, hablantes disjuntos)", pred(w, Z[va]), y[va])

    o = np.argsort(-np.abs(w[1:]))
    print("\npesos mas grandes:")
    for i in o[:15]:
        print(f"  {keys[i]:<20}{w[1+i]:+.3f}")

    # cuantas features hacen falta de verdad
    print("\nAUC en val segun numero de features (por |peso|):")
    for k in (3, 5, 8, 12, 20, len(keys)):
        sel = o[:k]
        w2 = fit(Z[tr][:, sel], y[tr], l2=3.0)
        print(f"  top {k:>3}: {auc(pred(w2, Z[va][:, sel]), y[va]):.4f}")


if __name__ == "__main__":
    main()
