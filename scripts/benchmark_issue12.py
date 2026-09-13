"""Logreg vs LightGBM robusto: mismos folds y estres de #11; umbral por coste de FP."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.pipeline import Pipeline

from app import features
from app.interventions import feature_names
from app.learning import FiniteValues, grouped_splits
from scripts.issue5_metrics import metrics
from scripts.train_issue5 import save_json
from scripts.train_issue11 import ROBUST_LAYERS, selection_score, training_samples


class RobustFeatures(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        indices = [
            i for i, name in enumerate(feature_names()) if features.grupo_de(name) in ROBUST_LAYERS
        ]
        return np.array([r["call_features"][indices] for r in X])


def fit_tree(records):
    from lightgbm import LGBMClassifier

    from scripts.benchmark_issue12 import RobustFeatures

    y = np.array([r["label"] for r in records])
    if any(r["split"] != "train" for r in records):
        raise ValueError("val no puede ajustar el benchmark")
    groups = [r["group"] for r in records]
    model = Pipeline(
        [
            ("features", RobustFeatures()),
            ("finite", FiniteValues()),
            (
                "tree",
                LGBMClassifier(
                    n_estimators=150,
                    learning_rate=0.03,
                    num_leaves=7,
                    max_depth=3,
                    min_child_samples=20,
                    colsample_bytree=0.7,
                    subsample=0.8,
                    subsample_freq=1,
                    reg_lambda=10,
                    reg_alpha=1,
                    random_state=12,
                    n_jobs=1,
                    verbosity=-1,
                ),
            ),
        ]
    )
    return CalibratedClassifierCV(
        model, method="sigmoid", cv=grouped_splits(y, groups, 3, 6), ensemble=False, n_jobs=1
    ).fit(records, y)


def threshold_cost(y, probabilities, threshold, fp_cost=5):
    costs = []
    for p in probabilities.values():
        tn, fp, fn, tp = confusion_matrix(y, p >= threshold, labels=[0, 1]).ravel()
        costs.append((fp_cost * fp + fn) / (tn + fp + fn + tp))
    return float(np.mean(costs))


def choose_threshold(y, probabilities, fp_cost=5):
    if not np.isfinite(fp_cost) or fp_cost <= 0:
        raise ValueError("el coste de FP debe ser positivo")
    # Umbrales derivados solo de scores OOF; los escenarios pesan lo mismo.
    candidates = np.unique(
        np.concatenate([np.array([0.5, 0.7, 1.0]), *list(probabilities.values())])
    )
    costs = np.zeros(len(candidates))
    for p in probabilities.values():
        human, bot = np.sort(p[y == 0]), np.sort(p[y == 1])
        fp = len(human) - np.searchsorted(human, candidates, side="left")
        fn = np.searchsorted(bot, candidates, side="left")
        costs += fp_cost * fp + fn
    return float(candidates[np.flatnonzero(costs == costs.min())[-1]])


def benchmark_tree(data, baseline, output):
    path = output / "cv_lightgbm.joblib"
    key = hashlib.sha256(Path(__file__).read_bytes() + baseline["fingerprint"].encode()).hexdigest()
    if path.exists():
        cached = joblib.load(path)
        if cached["fingerprint"] == key:
            return cached
    y = np.array([r["label"] for r in data["clean"]])
    scenarios = {"clean": data["clean"], **data["scenarios"]}
    probabilities = {name: np.full(len(y), np.nan) for name in scenarios}
    models = {}
    for fold, (train, test) in enumerate(baseline["folds"]):
        model = fit_tree(training_samples(data, train, baseline["mode"]))
        for name, records in scenarios.items():
            probabilities[name][test] = model.predict_proba([records[i] for i in test])[:, 1]
        models[fold] = model
        print(f"LightGBM: fold {fold + 1}/5", flush=True)
    result = {
        "metrics": {s: metrics(y, p) for s, p in probabilities.items()},
        "probabilities": probabilities,
        "models": models,
        "folds": baseline["folds"],
        "data_fingerprint": data["fingerprint"],
        "fingerprint": key,
    }
    joblib.dump(result, path, compress=3)
    return result


def permute_group(records, indices, rng):
    order = rng.permutation(len(records))
    changed = []
    for record, donor_index in zip(records, order, strict=True):
        donor = records[donor_index]
        call = record["call_features"].copy()
        call[indices] = donor["call_features"][indices]
        turns = record["turn_features"].copy()
        if len(turns) and len(donor["turn_features"]):
            donor_rows = np.linspace(0, len(donor["turn_features"]) - 1, len(turns)).astype(int)
            turns[:, indices] = donor["turn_features"][donor_rows][:, indices]
        changed.append({**record, "call_features": call, "turn_features": turns})
    return changed


def permutation_importance(data, evaluation, output):
    """Permuta familias completas en cada fold reservado; no reentrena."""
    names = feature_names()
    groups = sorted({features.grupo_de(n) for n in names})
    y = np.array([r["label"] for r in data["clean"]])
    base_auc = evaluation["metrics"]["clean"]["auc"]
    rows = []
    rng = np.random.default_rng(12)
    for group in groups:
        indices = np.array([i for i, n in enumerate(names) if features.grupo_de(n) == group])
        changes = []
        for _repeat in range(3):
            p = np.empty(len(y))
            for fold, (_, test) in enumerate(evaluation["folds"]):
                records = [data["clean"][i] for i in test]
                p[test] = evaluation["models"][fold].predict_proba(
                    permute_group(records, indices, rng)
                )[:, 1]
            changes.append(base_auc - roc_auc_score(y, p))
        rows.append(
            {
                "group": group,
                "auc_drop_mean": float(np.mean(changes)),
                "auc_drop_std": float(np.std(changes)),
            }
        )
    rows.sort(key=lambda r: r["auc_drop_mean"], reverse=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    return rows


def run(args):
    args.output.mkdir(parents=True, exist_ok=True)
    data = joblib.load(args.input / "dataset.joblib")
    issue11 = json.loads((args.input / "results.json").read_text())
    baseline = joblib.load(args.input / f"cv_{issue11['selected']}.joblib")
    results = {"logreg": baseline, "lightgbm": benchmark_tree(data, baseline, args.output)}
    y = np.array([r["label"] for r in data["clean"]])
    thresholds = {
        name: choose_threshold(y, r["probabilities"], args.fp_cost) for name, r in results.items()
    }
    costs = {
        name: threshold_cost(y, r["probabilities"], thresholds[name], args.fp_cost)
        for name, r in results.items()
    }
    # El coste de acusar a un humano manda; desempate por CV + peor estres y simplicidad.
    selected = min(results, key=lambda n: (costs[n], -selection_score(results[n]), n != "logreg"))
    chosen = results[selected]
    operating_points = {
        name: {
            scenario: metrics(y, p, cutoffs=(thresholds[name],))["thresholds"][
                str(thresholds[name])
            ]
            for scenario, p in result["probabilities"].items()
        }
        for name, result in results.items()
    }
    importance = permutation_importance(data, chosen, args.output / "importance.csv")
    frozen = {
        "selected": selected,
        "threshold": thresholds[selected],
        "thresholds": thresholds,
        "false_positive_cost": args.fp_cost,
        "false_negative_cost": 1,
        "selection_cost": costs,
        "val_used_for_selection": False,
        "augmentation": issue11["selected"],
        "importance": importance,
        "metrics": {name: r["metrics"] for name, r in results.items()},
        "operating_points": operating_points,
    }
    save_json(args.output / "results.json", frozen)
    if selected == "logreg":
        classifier = joblib.load(args.input / "candidate.joblib")["model"]
    else:
        classifier = fit_tree(training_samples(data, range(len(y)), baseline["mode"]))
    joblib.dump(
        {"classifier": classifier, "threshold": thresholds[selected], "selection": frozen},
        args.output / "candidate.joblib",
        compress=3,
    )
    lines = [
        "# Issue #12: benchmark de clasificador",
        "",
        "Logreg y LightGBM usan la misma augmentacion, llamadas y cinco folds de #11. "
        "LightGBM solo recibe prosodia, conducta y razones entre canales.",
        "",
        "| Modelo | AUC agrupada | Peor AUC estres | Umbral | Coste medio |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name, r in results.items():
        worst = min(m["auc"] for s, m in r["metrics"].items() if s != "clean")
        lines.append(
            f"| {name} | {r['metrics']['clean']['auc']:.4f} | {worst:.4f} | "
            f"{thresholds[name]:.4f} | {costs[name]:.4f} |"
        )
    lines.extend(
        [
            "",
            f"Se elige **{selected}**, con umbral **{thresholds[selected]:.4f}**: "
            f"menor coste OOF medio con FP={args.fp_cost:g} y FN=1; en empate, mayor "
            "media de AUC limpio y peor estres. El umbral se ajusta sobre estos scores "
            "OOF; val no se usa para elegir modelo ni umbral.",
            "",
            "| Escenario | AUC logreg | AUC LightGBM |",
            "| --- | ---: | ---: |",
        ]
    )
    for scenario in baseline["metrics"]:
        lines.append(
            f"| {scenario} | {baseline['metrics'][scenario]['auc']:.4f} | "
            f"{results['lightgbm']['metrics'][scenario]['auc']:.4f} |"
        )
    lines.extend(
        [
            "",
            "Importancia por permutacion de familias en las llamadas reservadas "
            "de cada fold (3 repeticiones; caida de AUC):",
            "",
        ]
    )
    lines.extend(
        f"- {r['group']}: {r['auc_drop_mean']:.4f} ± {r['auc_drop_std']:.4f}" for r in importance
    )
    (args.output / "reporte.md").write_text("\n".join(lines) + "\n")
    print(f"seleccion: {selected}; umbral: {thresholds[selected]:.4f}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("analysis/issue11"))
    parser.add_argument("--output", type=Path, default=Path("analysis/issue12"))
    parser.add_argument("--fp-cost", type=float, default=5)
    run(parser.parse_args())
