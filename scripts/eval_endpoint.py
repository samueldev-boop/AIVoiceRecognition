"""Evalua POST /detect de punta a punta: acierto, FPR, latencia y calibracion.

Recorre un split del manifiesto, manda cada WAV en base64 al endpoint, guarda la respuesta
y la latencia de cada llamada e imprime un informe que cabe en una pantalla:

  - la decision (is_synthetic): acierto, FPR, que son humanos acusados y el error caro, y
    FNR, que son bots sin detectar;
  - AUC, Brier y ECE de 10 bins (la misma de #5) sobre dos puntuaciones. La del contrato,
    P(sintetico) = confidence si is_synthetic y 1 - confidence si no, es lo unico que ve el
    jurado. probability_synthetic es la probabilidad calibrada del servicio. La diferencia
    entre las dos es el efecto de los topes de confianza;
  - latencia p50/p95 medida en el cliente, con subida y red, y la que informa el servidor;
  - que etapa de la cascada decidio cada llamada, con su latencia y sus errores;
  - barrido de umbral 0.3/0.5/0.7/0.9 sobre probability_synthetic con los FP y FN de cada uno.

Las peticiones van de una en una: el servicio corre con un solo worker y, en paralelo, se
mediria la cola y no el endpoint. Una peticion fallida (HTTP distinto de 200, sin respuesta
o sin is_synthetic) no entra en las metricas: se cuenta aparte y el script sale con 1.

Las respuestas quedan en analysis/eval_<split>_<host>.csv, que git ignora y la guardia
bloquea: lleva ids y turnos del dataset.

uso:
    python scripts/eval_endpoint.py --split val --url http://localhost:8000
    python scripts/eval_endpoint.py --split val --url https://detect.midominio.com
"""

import argparse
import base64
import csv
import http.client
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.metrics import brier_score_loss, confusion_matrix, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.issue5_metrics import ece  # noqa: E402

# Limite de respuesta del reto: lo que tarde mas cuenta como fuera de plazo.
PRESUPUESTO_S = 30.0
UMBRALES = (0.3, 0.5, 0.7, 0.9)
CAMPOS = ("anon_id", "label", "status", "latencia_s", "is_synthetic", "confidence",
          "probability_synthetic", "stage", "ms", "error", "respuesta")


def detectar(url: str, wav: Path, timeout: float) -> dict:
    """Un POST /detect. La latencia cubre la peticion y la respuesta, no el base64."""
    cuerpo = json.dumps({"audio": base64.b64encode(wav.read_bytes()).decode()}).encode()
    peticion = urllib.request.Request(f"{url}/detect", data=cuerpo,
                                      headers={"content-type": "application/json"})
    fila = {"status": 0, "error": "", "respuesta": None}
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(peticion, timeout=timeout) as r:
            fila["status"] = r.status
            fila["respuesta"] = json.load(r)
    except urllib.error.HTTPError as e:
        fila["status"] = e.code
        fila["error"] = f"HTTP {e.code}: {e.read().decode(errors='replace')[:200]}"
    except (OSError, http.client.HTTPException) as e:
        fila["error"] = f"sin respuesta: {getattr(e, 'reason', e)}"
    except ValueError as e:
        fila["error"] = f"respuesta invalida: no es JSON ({e})"
    fila["latencia_s"] = time.perf_counter() - t0

    respuesta = fila["respuesta"] if isinstance(fila["respuesta"], dict) else {}
    if not fila["error"] and not isinstance(respuesta.get("is_synthetic"), bool):
        fila["error"] = "respuesta invalida: sin is_synthetic booleano"
    for campo in ("is_synthetic", "confidence", "probability_synthetic", "stage", "ms"):
        fila[campo] = respuesta.get(campo)
    return fila


def p_contrato(fila: dict) -> float:
    """P(sintetico) segun el contrato, que es lo que puede puntuar el jurado.

    confidence es la confianza en la clase elegida, no P(sintetico): cuando la decision es
    humano hay que darle la vuelta. Sin confidence, el contrato solo aporta la decision.
    """
    if fila["confidence"] is None:
        return float(fila["is_synthetic"])
    return fila["confidence"] if fila["is_synthetic"] else 1 - fila["confidence"]


def resumen(filas: list[dict]) -> dict:
    """Los numeros del informe. Solo las respuestas validas entran en las metricas."""
    ok = [f for f in filas if not f["error"]]
    r = {
        "llamadas": len(filas),
        "humanas": sum(f["label"] == "human" for f in filas),
        "respondidas": len(ok),
        # El error empieza por su motivo corto: "HTTP 503", "sin respuesta"...
        "fallidas": Counter(f["error"].split(":")[0] for f in filas if f["error"]),
        "fuera_de_plazo": sum(f["latencia_s"] > PRESUPUESTO_S for f in filas),
    }
    if not ok:
        return r

    y = np.array([f["label"] == "synthetic" for f in ok], dtype=int)
    decision = np.array([f["is_synthetic"] for f in ok], dtype=int)
    tn, fp, fn, tp = confusion_matrix(y, decision, labels=[0, 1]).ravel()
    r |= {"negativos": int(tn + fp), "positivos": int(fn + tp), "fp": int(fp), "fn": int(fn),
          "acierto": float((tn + tp) / len(ok))}

    puntuaciones = {"contrato": np.array([p_contrato(f) for f in ok])}
    if all(f["probability_synthetic"] is not None for f in ok):
        puntuaciones["probability_synthetic"] = np.array(
            [f["probability_synthetic"] for f in ok], dtype=float)
    r["calibracion"] = {
        nombre: {
            "auc": float(roc_auc_score(y, p)) if 0 < y.sum() < len(y) else None,
            "brier": float(brier_score_loss(y, p)),
            "ece": ece(y, p),
        }
        for nombre, p in puntuaciones.items()
    }

    # Misma regla que la cascada (probabilidad >= 0.5), moviendo solo el umbral. Sin
    # probability_synthetic se barre la puntuacion del contrato.
    r["barrido_sobre"] = ("probability_synthetic" if "probability_synthetic" in puntuaciones
                          else "contrato")
    r["barrido"] = []
    for umbral in UMBRALES:
        pred = (puntuaciones[r["barrido_sobre"]] >= umbral).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        r["barrido"].append({"umbral": umbral, "fp": int(fp), "fn": int(fn),
                             "acierto": float((tn + tp) / len(ok))})

    latencias = np.array([f["latencia_s"] for f in ok])
    r["latencia"] = [*np.percentile(latencias, [50, 95]).tolist(), float(latencias.max())]
    ms = [f["ms"] for f in ok if f["ms"] is not None]
    r["latencia_servidor"] = (np.percentile(ms, [50, 95]) / 1000).tolist() if ms else None

    etapas: dict[str, dict] = {}
    for f, real in zip(ok, y, strict=True):
        e = etapas.setdefault(f["stage"] or "-", {"n": 0, "latencias": [], "fp": 0, "fn": 0})
        e["n"] += 1
        e["latencias"].append(f["latencia_s"])
        e["fp"] += int(f["is_synthetic"] and not real)
        e["fn"] += int(real and not f["is_synthetic"])
    # Por latencia, que es el orden en que la cascada recorre las etapas.
    r["etapas"] = sorted(
        ({"etapa": k, "n": v["n"], "p50": float(np.median(v["latencias"])),
          "fp": v["fp"], "fn": v["fn"]} for k, v in etapas.items()),
        key=lambda e: e["p50"],
    )
    return r


def _pct(x: float | None) -> str:
    return "-" if x is None else f"{100 * x:.1f} %"


def _tasa(errores: int, total: int) -> float | None:
    return errores / total if total else None


def informe(r: dict, cabecera: str) -> str:
    """Texto para 80 columnas: una pantalla de terminal, sin desplazarse."""
    lineas = [
        *cabecera.splitlines(),
        f"{r['llamadas']} llamadas ({r['humanas']} humanas, "
        f"{r['llamadas'] - r['humanas']} sinteticas)   respondidas {r['respondidas']}   "
        f"> {PRESUPUESTO_S:.0f} s: {r['fuera_de_plazo']}",
    ]
    if r["fallidas"]:
        detalle = ", ".join(f"{motivo} x{n}" for motivo, n in r["fallidas"].most_common())
        lineas.append(f"FALLIDAS {sum(r['fallidas'].values())}: {detalle} (detalle en el CSV)")
    if not r["respondidas"]:
        return "\n".join(lineas)

    lineas += [
        "",
        f"Decision   acierto {_pct(r['acierto'])}",
        f"  FPR {_pct(_tasa(r['fp'], r['negativos'])):>8}   "
        f"{r['fp']} de {r['negativos']} humanos acusados",
        f"  FNR {_pct(_tasa(r['fn'], r['positivos'])):>8}   "
        f"{r['fn']} de {r['positivos']} bots sin detectar",
        "",
        f"{'Puntuacion':<29}{'AUC':>7}{'Brier':>9}{'ECE':>9}",
    ]
    nombres = {"contrato": "is_synthetic + confidence",
               "probability_synthetic": "probability_synthetic"}
    for nombre, c in r["calibracion"].items():
        auc = "-" if c["auc"] is None else f"{c['auc']:.4f}"
        lineas.append(f"  {nombres[nombre]:<27}{auc:>7}{c['brier']:>9.4f}{c['ece']:>9.4f}")

    p50, p95, maximo = r["latencia"]
    lineas += ["", f"Latencia   cliente   p50 {p50:.2f} s   p95 {p95:.2f} s   max {maximo:.2f} s"]
    if r["latencia_servidor"]:
        s50, s95 = r["latencia_servidor"]
        lineas.append(f"           servidor  p50 {s50:.2f} s   p95 {s95:.2f} s")

    lineas += ["", f"{'Etapa que decidio':<22}{'n':>4}{'%':>9}{'p50':>9}{'FP':>5}{'FN':>5}"]
    for e in r["etapas"]:
        lineas.append(f"  {e['etapa']:<20}{e['n']:>4}{_pct(e['n'] / r['respondidas']):>9}"
                      f"{e['p50']:>7.2f} s{e['fp']:>5}{e['fn']:>5}")

    lineas += ["", f"{'Umbral sobre ' + r['barrido_sobre']:<36}"
                   f"{'FP':>4}{'FN':>5}{'FPR':>9}{'FNR':>9}{'acierto':>9}"]
    for b in r["barrido"]:
        lineas.append(f"  {b['umbral']:<34}{b['fp']:>4}{b['fn']:>5}"
                      f"{_pct(_tasa(b['fp'], r['negativos'])):>9}"
                      f"{_pct(_tasa(b['fn'], r['positivos'])):>9}{_pct(b['acierto']):>9}")
    return "\n".join(lineas)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--split", choices=("train", "val"), default="val")
    p.add_argument("--url", default="http://localhost:8000", help="base del servicio")
    p.add_argument("--raiz", type=Path, default=ROOT,
                   help="carpeta con manifest.csv y audio/ (por defecto la del repositorio)")
    p.add_argument("--salida", type=Path,
                   help="CSV de respuestas (por defecto analysis/eval_<split>_<host>.csv)")
    p.add_argument("--timeout", type=float, default=60.0, metavar="S",
                   help="por peticion, en segundos (por defecto 60; el reto corta en 30)")
    args = p.parse_args(argv)

    url = args.url.rstrip("/")
    try:
        with urllib.request.urlopen(f"{url}/health", timeout=10) as r:
            salud = json.load(r)
            # Caddy con dominio redirige http -> https. GET sigue la redireccion y POST no,
            # asi que se adopta la URL final antes de mandar audio.
            url = r.url.removesuffix("/health")
    except (OSError, http.client.HTTPException, ValueError) as e:
        print(f"{url}/health no responde: {e}", file=sys.stderr)
        return 1
    if not salud.get("model_loaded"):
        print(f"{url} no tiene modelo cargado: /detect responderia 503", file=sys.stderr)
        return 1

    manifiesto = args.raiz / "manifest.csv"
    if not manifiesto.exists():
        print(f"no hay {manifiesto}: descomprime el dataset ahi o usa --raiz", file=sys.stderr)
        return 1
    with manifiesto.open(newline="") as fh:
        llamadas = [f for f in csv.DictReader(fh) if f["split"] == args.split]
    faltan = [f["anon_id"] for f in llamadas
              if not (args.raiz / "audio" / f"{f['anon_id']}.wav").exists()]
    if not llamadas or faltan:
        print(f"split {args.split}: {len(llamadas)} llamadas, {len(faltan)} sin WAV en "
              f"{args.raiz / 'audio'} {faltan[:3]}", file=sys.stderr)
        return 1

    host = re.sub(r"[^A-Za-z0-9.]+", "-", urllib.parse.urlsplit(url).netloc)
    salida = args.salida or ROOT / "analysis" / f"eval_{args.split}_{host}.csv"
    salida.parent.mkdir(parents=True, exist_ok=True)
    progreso = sys.stderr.isatty()

    filas = []
    with salida.open("w", newline="") as fh:
        escritor = csv.DictWriter(fh, fieldnames=CAMPOS)
        escritor.writeheader()
        for i, llamada in enumerate(llamadas, 1):
            wav = args.raiz / "audio" / f"{llamada['anon_id']}.wav"
            fila = {"anon_id": llamada["anon_id"], "label": llamada["label"],
                    **detectar(url, wav, args.timeout)}
            escritor.writerow({**fila, "respuesta": json.dumps(fila["respuesta"])})
            fh.flush()
            filas.append(fila)
            if progreso:
                print(f"\r  {i}/{len(llamadas)}  {fila['latencia_s']:.2f} s ", end="",
                      file=sys.stderr, flush=True)
    if progreso:
        print("\r" + " " * 40 + "\r", end="", file=sys.stderr)

    cabecera = (f"POST {url}/detect   split {args.split}\n"
                f"servicio {salud.get('version')}, modelo {salud.get('model_version')}")
    print(informe(resumen(filas), cabecera))
    if args.split == "train":
        print("\nOjo: el modelo se entreno con train. Estas metricas son optimistas.")
    relativa = os.path.relpath(salida)
    print(f"\nRespuestas y latencias en {salida if relativa.startswith('..') else relativa}")
    return 1 if any(f["error"] for f in filas) else 0


if __name__ == "__main__":
    sys.exit(main())
