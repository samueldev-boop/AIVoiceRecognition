"""Mide el extractor unificado: poder discriminativo por feature, por grupo y por presupuesto.

Sustituye a las sondas separadas de analysis/, que median lo mismo con dos implementaciones.
Calcula las features desde el WAV con nuestro VAD, no desde los turns/*.json del dataset,
porque es lo que va a pasar en produccion.

Guarda un CSV por presupuesto en analysis/ (ignorado por git, se regenera con este script).

uso: python scripts/features_probe.py [n_llamadas]
"""

import collections
import csv
import os
import sys
import time

import numpy as np
import soundfile as sf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from app import features, vad  # noqa: E402

PRESUPUESTOS = ("first_turn", "20s", "full")
SEMILLAS = 5


def auc_una(valores, y):
    """AUC de Mann-Whitney de una sola feature. 0.5 = no separa."""
    return roc_auc_score(y, valores)


def modelo():
    return make_pipeline(StandardScaler(), LogisticRegression(C=0.33, max_iter=2000))


def evaluar(X, y, tr, va):
    lls, aucs = [], []
    for semilla in range(SEMILLAS):
        particion = StratifiedKFold(5, shuffle=True, random_state=semilla)
        for i_tr, i_te in particion.split(X[tr], y[tr]):
            m = modelo().fit(X[tr][i_tr], y[tr][i_tr])
            p = m.predict_proba(X[tr][i_te])[:, 1]
            lls.append(log_loss(y[tr][i_te], p, labels=[0, 1]))
            aucs.append(roc_auc_score(y[tr][i_te], p))
    p_va = modelo().fit(X[tr], y[tr]).predict_proba(X[va])[:, 1]
    return {"cv_logloss": float(np.mean(lls)), "cv_auc": float(np.mean(aucs)),
            "val_auc": float(roc_auc_score(y[va], p_va)),
            "fp": int(((p_va >= 0.5) & (y[va] == 0)).sum()),
            "fn": int(((p_va < 0.5) & (y[va] == 1)).sum())}


def main():
    limite = int(sys.argv[1]) if len(sys.argv) > 1 else 10**9
    filas = list(csv.DictReader(open(os.path.join(ROOT, "manifest.csv"))))[:limite]

    datos = {p: [] for p in PRESUPUESTOS}
    meta = []
    t0 = time.perf_counter()
    for i, r in enumerate(filas):
        ruta = os.path.join(ROOT, "audio", r["anon_id"] + ".wav")
        if not os.path.exists(ruta):
            continue
        x, sr = sf.read(ruta, dtype="int16", always_2d=True)
        turnos = vad.turnos(x, sr)
        extraidas = {p: features.extraer(x, sr, turnos, p) for p in PRESUPUESTOS}
        if not all(extraidas.values()):
            continue
        for p in PRESUPUESTOS:
            datos[p].append(extraidas[p])
        meta.append(r)
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(filas)}  {time.perf_counter()-t0:.0f}s", flush=True)

    y = np.array([1 if r["label"] == "synthetic" else 0 for r in meta])
    tr = np.array([r["split"] == "train" for r in meta])
    va = ~tr
    print(f"\n{len(meta)} llamadas en {time.perf_counter()-t0:.0f}s "
          f"({int(y.sum())} sinteticas / {int((1-y).sum())} humanas)")

    claves = sorted(datos["full"][0])
    grupos = {k: features.grupo_de(k) for k in claves}
    print(f"{len(claves)} features en {len(set(grupos.values()))} grupos: "
          f"{dict(collections.Counter(grupos.values()))}\n")

    for p in PRESUPUESTOS:
        salida = os.path.join(ROOT, "analysis", f"features_{p}.csv")
        with open(salida, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["anon_id", "label", "split"] + claves)
            for r, f in zip(meta, datos[p], strict=True):
                w.writerow([r["anon_id"], r["label"], r["split"]] + [f.get(k, 0.0) for k in claves])

    print(f"{'presupuesto':<12}{'dur media':>11}{'CV-logloss':>12}{'CV-AUC':>9}"
          f"{'val AUC':>10}{'FP':>5}{'FN':>5}")
    matrices = {}
    for p in PRESUPUESTOS:
        X = np.nan_to_num(np.array([[f.get(k, 0.0) for k in claves] for f in datos[p]]),
                          posinf=0.0, neginf=0.0)
        matrices[p] = X
        m = evaluar(X, y, tr, va)
        dur = float(np.mean([f["dur_usada_s"] for f in datos[p]]))
        print(f"{p:<12}{dur:>10.1f}s{m['cv_logloss']:>12.4f}{m['cv_auc']:>9.4f}"
              f"{m['val_auc']:>10.4f}{m['fp']:>5}{m['fn']:>5}")

    print("\n=== por grupo, presupuesto full ===")
    print(f"{'grupo':<14}{'n':>4}{'CV-logloss':>12}{'CV-AUC':>9}{'val AUC':>10}{'FP':>5}{'FN':>5}")
    X = matrices["full"]
    for g in sorted(set(grupos.values())):
        sel = np.array([grupos[k] == g for k in claves])
        m = evaluar(X[:, sel], y, tr, va)
        print(f"{g:<14}{int(sel.sum()):>4}{m['cv_logloss']:>12.4f}{m['cv_auc']:>9.4f}"
              f"{m['val_auc']:>10.4f}{m['fp']:>5}{m['fn']:>5}")
    robusto = np.array([grupos[k] in ("conducta", "prosodia", "razon_canal") for k in claves])
    m = evaluar(X[:, robusto], y, tr, va)
    print(f"{'ROBUSTO':<14}{int(robusto.sum()):>4}{m['cv_logloss']:>12.4f}{m['cv_auc']:>9.4f}"
          f"{m['val_auc']:>10.4f}{m['fp']:>5}{m['fn']:>5}")

    print("\n=== 20 features con mas separacion (train, presupuesto full) ===")
    Xtr, ytr = X[tr], y[tr]
    orden = []
    for j, k in enumerate(claves):
        a = auc_una(Xtr[:, j], ytr)
        orden.append((abs(a - 0.5), a, k, Xtr[ytr == 0, j].mean(), Xtr[ytr == 1, j].mean()))
    orden.sort(reverse=True)
    print(f"{'feature':<22}{'grupo':<13}{'AUC':>7}{'humano':>12}{'sintetico':>12}")
    for _, a, k, mh, ms in orden[:20]:
        print(f"{k:<22}{grupos[k]:<13}{a:>7.3f}{mh:>12.4g}{ms:>12.4g}")

    print("\n=== la nota del jitter: sigue invertido con Praat? ===")
    for k in ("jitter_local", "shimmer_local", "f0_jitter_praat", "f0_smooth", "hnr_db",
              "razon_jitter_local", "razon_shimmer_local"):
        if k not in claves:
            continue
        j = claves.index(k)
        mh, ms = Xtr[ytr == 0, j].mean(), Xtr[ytr == 1, j].mean()
        a = auc_una(Xtr[:, j], ytr)
        direccion = "bot MAYOR" if ms > mh else "bot menor"
        print(f"  {k:<22}humano={mh:>9.4f}  sintetico={ms:>9.4f}  AUC={a:.3f}  ({direccion})")


if __name__ == "__main__":
    main()
