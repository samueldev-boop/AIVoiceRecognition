"""Mide el VAD propio contra los turns/*.json del dataset.

Dos preguntas, en este orden:

  1. Cuanto se parecen nuestras fronteras a las del organizador (IoU por canal).
  2. Lo unico que importa de verdad: cuanto rinden las features de conducta calculadas
     con nuestras fronteras frente a las suyas.

La segunda es el criterio de aceptacion de #3, porque en el set oculto no habra
turns/*.json: el endpoint tendra que derivarlos del WAV.

La seleccion se hace con validacion cruzada repetida SOBRE TRAIN, no con el AUC de val:
con 71 llamadas en val, el AUC no distingue una diferencia de 0.01 del ruido, y usarlo
para elegir quemaria el unico conjunto limpio que hay. El AUC de val se imprime al final
solo para mirarlo.

uso: python scripts/vad_acuerdo.py [n_llamadas]
"""

import csv
import json
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

from app import features, intervalos, vad  # noqa: E402

SEMILLAS = 10


def union_canal(turnos, canal):
    return intervalos.union([(t["start"], t["end"]) for t in turnos if t["channel"] == canal])


def error_fronteras(ref, pred, canal):
    """Diferencia absoluta media entre cada frontera de referencia y la mas cercana nuestra."""
    fr = [v for t in ref if t["channel"] == canal for v in (t["start"], t["end"])]
    fp = [v for t in pred if t["channel"] == canal for v in (t["start"], t["end"])]
    if not fr or not fp:
        return float("nan")
    fp = np.array(fp)
    return float(np.mean([np.min(np.abs(fp - v)) for v in fr]))


def modelo():
    return make_pipeline(StandardScaler(), LogisticRegression(C=0.33, max_iter=2000))


def rendimiento(features, etiquetas, splits):
    """CV repetida sobre train + AUC de val. Devuelve un dict de metricas."""
    claves = list(features[0])
    X = np.nan_to_num(
        np.array([[float(f.get(k, 0.0) or 0.0) for k in claves] for f in features]),
        posinf=0.0, neginf=0.0,
    )
    y = np.array(etiquetas)
    tr = np.array([s == "train" for s in splits])
    va = ~tr

    lls, aucs = [], []
    for semilla in range(SEMILLAS):
        particion = StratifiedKFold(5, shuffle=True, random_state=semilla)
        for i_tr, i_te in particion.split(X[tr], y[tr]):
            m = modelo().fit(X[tr][i_tr], y[tr][i_tr])
            p = m.predict_proba(X[tr][i_te])[:, 1]
            lls.append(log_loss(y[tr][i_te], p, labels=[0, 1]))
            aucs.append(roc_auc_score(y[tr][i_te], p))

    p_va = modelo().fit(X[tr], y[tr]).predict_proba(X[va])[:, 1]
    return {
        "cv_logloss": float(np.mean(lls)), "cv_logloss_sd": float(np.std(lls)),
        "cv_auc": float(np.mean(aucs)), "val_auc": float(roc_auc_score(y[va], p_va)),
        "fp": int(((p_va >= 0.5) & (y[va] == 0)).sum()),
        "fn": int(((p_va < 0.5) & (y[va] == 1)).sum()),
        "n": len(features),
    }


def main():
    limite = int(sys.argv[1]) if len(sys.argv) > 1 else 10**9
    filas = list(csv.DictReader(open(os.path.join(ROOT, "manifest.csv"))))[:limite]

    iou = {0: [], 1: []}
    fronteras = {0: [], 1: []}
    n_seg = {"ref": {0: 0, 1: 0}, "propio": {0: 0, 1: 0}}
    f_ref, f_pro, etiquetas, splits, descartadas = [], [], [], [], []
    audio_s = 0.0
    t0 = time.perf_counter()

    for i, r in enumerate(filas):
        anon = r["anon_id"]
        ruta = os.path.join(ROOT, "audio", anon + ".wav")
        if not os.path.exists(ruta):
            continue
        x, sr = sf.read(ruta, dtype="int16", always_2d=True)
        audio_s += len(x) / sr
        ref = json.load(open(os.path.join(ROOT, "turns", anon + ".json")))["turns"]
        propio = vad.turnos(x, sr)

        for canal in (0, 1):
            iou[canal].append(intervalos.iou(union_canal(ref, canal), union_canal(propio, canal)))
            fronteras[canal].append(error_fronteras(ref, propio, canal))
            n_seg["ref"][canal] += sum(1 for t in ref if t["channel"] == canal)
            n_seg["propio"][canal] += sum(1 for t in propio if t["channel"] == canal)

        dur = float(r["duration_s"])
        a, b = features.conducta(ref, dur), features.conducta(propio, dur)
        if not a or not b:
            descartadas.append((anon, a is None, b is None))
            continue
        f_ref.append(a)
        f_pro.append(b)
        etiquetas.append(1 if r["label"] == "synthetic" else 0)
        splits.append(r["split"])

        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(filas)}", flush=True)

    dt = time.perf_counter() - t0
    print(f"\n{len(f_ref)} llamadas, {audio_s/3600:.2f} h de audio en {dt:.0f}s "
          f"(x{audio_s/dt:.0f} tiempo real, incluye leer el wav y calcular features)")
    if descartadas:
        print(f"descartadas por no tener turnos en algun canal: {len(descartadas)}")
        for anon, falta_ref, falta_pro in descartadas[:5]:
            faltan = (("referencia", falta_ref), ("propio", falta_pro))
            quien = " y ".join(n for n, f in faltan if f)
            print(f"  {anon}: sin turnos en {quien}")

    print("\n=== acuerdo con turns/*.json ===")
    print(f"{'canal':<10}{'IoU medio':>11}{'IoU p10':>9}{'err.frontera':>14}"
          f"{'seg ref':>9}{'seg propios':>13}")
    for canal, nombre in ((0, "llamante"), (1, "agente")):
        a = np.array(iou[canal])
        f = np.array([v for v in fronteras[canal] if not np.isnan(v)])
        print(f"{nombre:<10}{a.mean():>11.3f}{np.percentile(a, 10):>9.3f}"
              f"{f.mean():>13.2f}s{n_seg['ref'][canal]:>9}{n_seg['propio'][canal]:>13}")

    if not splits or "val" not in splits or len(set(etiquetas)) < 2:
        print("\nsin val en este subconjunto: ejecuta sin limite para el criterio de aceptacion")
        return

    print(f"\n=== rendimiento de las features de conducta (CV {SEMILLAS}x5 sobre train) ===")
    print(f"{'fronteras':<24}{'CV-logloss':>12}{'±':>7}{'CV-AUC':>9}"
          f"{'val AUC':>10}{'FP':>5}{'FN':>5}")
    metricas = {}
    for nombre, feats in (("turns/*.json (dataset)", f_ref), ("VAD propio", f_pro)):
        m = rendimiento(feats, etiquetas, splits)
        metricas[nombre] = m
        print(f"{nombre:<24}{m['cv_logloss']:>12.4f}{m['cv_logloss_sd']:>7.3f}"
              f"{m['cv_auc']:>9.4f}{m['val_auc']:>10.4f}{m['fp']:>5}{m['fn']:>5}")

    a = metricas["turns/*.json (dataset)"]["val_auc"]
    b = metricas["VAD propio"]["val_auc"]
    print(f"\ncriterio de #3: el AUC de val no debe caer mas de 0.01 -> "
          f"diferencia {b - a:+.4f} ({'CUMPLE' if b - a >= -0.01 else 'NO CUMPLE'})")


if __name__ == "__main__":
    main()
