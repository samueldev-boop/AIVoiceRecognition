"""Compara sin augmentacion, codecs/sala y RawBoost; bloquea promociones fragiles."""

import hashlib
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import joblib
import numpy as np

from app.learning import grouped_splits
from scripts.augmentation import CODECS
from scripts.issue5_metrics import metrics
from scripts.prepare_issue11 import parser, prepare
from scripts.train_issue5 import fit_calibrated, save_json

ROBUST_LAYERS = ("prosodia", "conducta", "razon_canal")
CANDIDATES = {
    "original": ("none", None),
    "augmentado": ("physical", None),
    "augmentado_robusto": ("physical", ROBUST_LAYERS),
    "rawboost": ("rawboost", None),
    "rawboost_robusto": ("rawboost", ROBUST_LAYERS),
}


def training_samples(data, indices, mode):
    records = [data["clean"][i] for i in indices]
    if mode != "none":
        for i in indices:
            records.extend(data["physical"][i])
            if mode == "rawboost":
                records.extend(data["rawboost"][i])
    if any(r["split"] != "train" for r in records):
        raise ValueError("val no admite ajuste ni augmentacion")
    return records


def fit_model(records, active_layers=None):
    return fit_calibrated(records, {"C": 0.33, "quantile": 0.005, "active_layers": active_layers})


def evaluate_fold(data, fold, train, test, mode, active_layers):
    started = time.perf_counter()
    records = training_samples(data, train, mode)
    if {r["group"] for r in records} & {data["clean"][i]["group"] for i in test}:
        raise ValueError("una augmentacion cruza la frontera de validacion")
    model = fit_model(records, active_layers)
    scenarios = {"clean": data["clean"], **data["scenarios"]}
    probabilities = {
        name: model.predict_proba([samples[i] for i in test])[:, 1]
        for name, samples in scenarios.items()
    }
    return fold, test, probabilities, model, time.perf_counter() - started


def evaluate(data, output, name, mode, active_layers=None, jobs=4):
    key = hashlib.sha256(data["fingerprint"].encode())
    for path in (Path(__file__), Path("app/learning.py"), Path("scripts/train_issue5.py")):
        key.update(path.read_bytes())
    key.update(str((mode, active_layers)).encode())
    path = output / f"cv_{name}.joblib"
    if path.exists():
        cached = joblib.load(path)
        if cached["fingerprint"] == key.hexdigest():
            print(f"{name}: CV reutilizada", flush=True)
            return cached
    y = np.array([r["label"] for r in data["clean"]])
    groups = np.array([r["group"] for r in data["clean"]])
    folds = grouped_splits(y, groups, 5, 5)
    probabilities = {s: np.full(len(y), np.nan) for s in ("clean", *data["scenarios"])}
    models, assignments = {}, np.full(len(y), -1)
    with ProcessPoolExecutor(max_workers=min(jobs, 5)) as pool:
        futures = [
            pool.submit(evaluate_fold, data, fold, train, test, mode, active_layers)
            for fold, (train, test) in enumerate(folds)
        ]
        for future in as_completed(futures):
            fold, test, scores, model, seconds = future.result()
            models[fold] = model
            assignments[test] = fold
            for scenario, p in scores.items():
                probabilities[scenario][test] = p
            print(f"{name}: fold {fold + 1}/5; {seconds:.0f}s", flush=True)
    if any(not np.isfinite(p).all() for p in probabilities.values()):
        raise ValueError("predicciones OOF incompletas")
    result = {
        "fingerprint": key.hexdigest(),
        "probabilities": probabilities,
        "metrics": {s: metrics(y, p) for s, p in probabilities.items()},
        "models": models,
        "fold_id": assignments,
        "folds": folds,
        "mode": mode,
        "active_layers": active_layers,
    }
    joblib.dump(result, path, compress=3)
    return result


def selection_score(result):
    aucs = [m["auc"] for name, m in result["metrics"].items() if name != "clean"]
    return (result["metrics"]["clean"]["auc"] + min(aucs)) / 2


def promotion_gate(result, robust_result, threshold=0.5):
    """Criterios predefinidos de #11, medidos fuera de muestra."""
    probabilities = result["probabilities"]
    threshold_key = str(threshold)
    human_cases = ["humano_limpio", *[f"humano_{c}" for c in CODECS]]
    fps = {
        s: result["metrics"][s]["thresholds"][threshold_key]["confusion_matrix"][0][1]
        for s in human_cases
    }
    fnrs = {
        s: result["metrics"][s]["thresholds"][threshold_key]["fnr"]
        for s in ("bot_evasivo", "bot_sala")
    }
    # Quitar ganancia, codec y silencio implica reentrenar, no llenar columnas con cero.
    drops = {
        s: result["metrics"][s]["auc"] - robust_result["metrics"][s]["auc"] for s in probabilities
    }
    return {
        "passed": max(fps.values()) == 0
        and max(fnrs.values()) < 0.10
        and max(drops.values()) <= 0.02,
        "threshold": threshold,
        "false_positives": fps,
        "false_negative_rates": fnrs,
        "auc_drop_without_fragile_groups": drops,
        "maximum_allowed_auc_drop": 0.02,
    }


def write_report(path, results, selected, gate, data):
    lines = [
        "# Issue #11: augmentacion telefonica",
        "",
        f"{len(data['clean'])} llamadas de train; 5 folds agrupados; "
        f"{data['epochs']} pasadas aleatorias de augmentacion por llamada.",
        "",
        "| Modelo | AUC limpio | Peor AUC de estres | FNR bot sala (0.7) | "
        "Max. FP humano codec (0.7) |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name, r in results.items():
        m = r["metrics"]
        worst = min(v["auc"] for k, v in m.items() if k != "clean")
        fp = max(m[f"humano_{c}"]["thresholds"]["0.7"]["confusion_matrix"][0][1] for c in CODECS)
        lines.append(
            f"| {name} | {m['clean']['auc']:.4f} | {worst:.4f} | "
            f"{m['bot_sala']['thresholds']['0.7']['fnr']:.4f} | {fp} |"
        )
    lines.extend(
        [
            "",
            f"Candidato seleccionado: **{selected}**, por la media del AUC limpio "
            "y el peor AUC bajo estres. Val no interviene en esta decision.",
            "",
            f"Promocion: **{'aprobada' if gate['passed'] else 'bloqueada'}**.",
            "",
            "```json",
            json.dumps(gate, indent=2),
            "```",
            "",
            "El banco incluye los tres ataques de #5, cinco codecs reales, ruido MUSAN "
            "con RIR reales y RawBoost. Se vuelve a ejecutar VAD y extraer features tras "
            "cada transformacion. RawBoost se compara como adicion a la augmentacion fisica.",
            "",
            "Los archivos de ruido y RIR de entrenamiento y estres son disjuntos. "
            "Original y variantes conservan el mismo grupo en todos los folds anidados. "
            "El remuestreador comun es soxr_hq; a 8 kHz no se remuestrea.",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def run(args):
    data = prepare(args)
    results = {
        name: evaluate(data, args.output, name, mode, layers, jobs=args.jobs)
        for name, (mode, layers) in CANDIDATES.items()
    }
    selected = max(
        (n for n in results if n != "original"), key=lambda n: selection_score(results[n])
    )
    robust_name = selected if selected.endswith("_robusto") else selected + "_robusto"
    gate = promotion_gate(results[selected], results[robust_name])
    frozen = {
        "selected": selected,
        "val_used_for_selection": False,
        "gate": gate,
        "resampler": data["resampler"],
        "epochs": data["epochs"],
        "data_fingerprint": data["fingerprint"],
        "metrics": {n: r["metrics"] for n, r in results.items()},
    }
    save_json(args.output / "results.json", frozen)
    write_report(args.output / "reporte.md", results, selected, gate, data)
    mode, active = CANDIDATES[selected]
    model = fit_model(training_samples(data, range(len(data["clean"])), mode), active)
    joblib.dump({"model": model, "selection": frozen}, args.output / "candidate.joblib", compress=3)
    print(f"seleccion: {selected}; promocion: {gate['passed']}", flush=True)
    if args.promote:
        if not gate["passed"]:
            raise SystemExit("promocion bloqueada por el banco de estres; resultados guardados")
        bundle = joblib.load(args.promote)
        bundle["models"]["full"] = model
        bundle["training"]["issue11"] = frozen
        joblib.dump(bundle, args.promote, compress=3)


if __name__ == "__main__":
    p = parser()
    p.add_argument("--promote", type=Path, help="artefacto que se reemplaza solo si supera el gate")
    run(p.parse_args())
