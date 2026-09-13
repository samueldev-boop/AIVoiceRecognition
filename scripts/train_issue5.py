"""Entrena, selecciona en train agrupado + estres, congela y SOLO entonces informa val.

Uso: OPENBLAS_NUM_THREADS=1 python -m scripts.train_issue5
O:   python -m scripts.train_issue5 --speakers /ruta/mapa_anonimo.csv
"""

import argparse
import csv
import hashlib
import json
import platform
import time
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.calibration import CalibratedClassifierCV

from app import features
from app.interventions import EXTRACTOR_VERSION, LAYER_NAMES, feature_names
from app.learning import LayeredClassifier, grouped_splits
from app.model import BUNDLE_VERSION, available_layers, confidence_policy
from scripts.issue5_data import (
    assign_groups,
    build_dataset,
    export_tables,
    read_manifest,
    speaker_mapping,
    summary,
)
from scripts.issue5_metrics import bootstrap, metrics
from scripts.issue5_stress import SCENARIOS, build_stress
from scripts.issue15_data import load_issue15

SEED = 5
CANDIDATES = {
    "sin_recorte_C033": {"C": 0.33, "quantile": 0.0},
    "limpio_C033": {"C": 0.33, "quantile": 0.005},
    "limpio_C010": {"C": 0.10, "quantile": 0.005},
}


def fit_calibrated(samples, parameters):
    y = np.array([s["label"] for s in samples])
    groups = np.array([s["group"] for s in samples])
    # Se calibran predicciones de TODA la jerarquia fuera de muestra, no scores de
    # capas entrenadas de antemano con las etiquetas de los folds de calibracion.
    folds = grouped_splits(y, groups, 3, SEED + 1)
    model = CalibratedClassifierCV(
        LayeredClassifier(**parameters, seed=SEED + 2),
        method="sigmoid",
        cv=folds,
        ensemble=False,
        n_jobs=1,
    )
    return model.fit(samples, y)


def evaluate_candidate(samples, scenarios, parameters, folds):
    y = np.array([s["label"] for s in samples])
    probabilities = {name: np.full(len(samples), np.nan) for name in ("clean", *SCENARIOS)}
    layers = np.full((len(samples), len(LAYER_NAMES)), np.nan)
    raw = np.full(len(samples), np.nan)
    fold_id = np.full(len(samples), -1)
    fold_metrics, turn_y, turn_p, turn_group = [], [], [], []
    for fold, (train, test) in enumerate(folds):
        start = time.perf_counter()
        model = fit_calibrated([samples[i] for i in train], parameters)
        test_samples = [samples[i] for i in test]
        probabilities["clean"][test] = model.predict_proba(test_samples)[:, 1]
        base = model.calibrated_classifiers_[0].estimator
        layers[test] = base.layer_scores(test_samples)
        raw[test] = base.predict_proba(test_samples)[:, 1]
        fold_id[test] = fold
        for name in SCENARIOS:
            probabilities[name][test] = model.predict_proba([scenarios[name][i] for i in test])[
                :, 1
            ]
        # Diagnostico de discriminacion por turno: promedio de logits de 4 capas,
        # sin atribuir calibracion a este score auxiliar.
        from app.learning import group_indices

        indices = group_indices()
        for s in test_samples:
            rows = s["turn_features"]
            if len(rows):
                logits = np.mean(
                    [
                        base.layers_[name].decision_function(rows[:, indices[name]])
                        for name in LAYER_NAMES[:4]
                    ],
                    axis=0,
                )
                from scipy.special import expit

                turn_y.extend([s["label"]] * len(rows))
                turn_p.extend(expit(logits))
                turn_group.extend([s["group"]] * len(rows))
        fold_metrics.append(metrics(y[test], probabilities["clean"][test]))
        print(
            f"  fold {fold + 1}/5 AUC={fold_metrics[-1]['auc']:.4f}; "
            f"{time.perf_counter() - start:.1f}s",
            flush=True,
        )
    if any(not np.isfinite(p).all() for p in probabilities.values()):
        raise ValueError("evaluacion fuera de muestra incompleta")
    result = {name: metrics(y, p) for name, p in probabilities.items()}
    result["uncalibrated_clean"] = metrics(y, raw)
    result["folds"] = fold_metrics
    result["fold_auc_mean"] = float(np.mean([m["auc"] for m in fold_metrics]))
    result["fold_auc_std"] = float(np.std([m["auc"] for m in fold_metrics], ddof=1))
    result["turn_discrimination_uncalibrated"] = metrics(np.array(turn_y), np.array(turn_p))
    result["selection_score"] = (
        result["clean"]["auc"] + min(result[name]["auc"] for name in SCENARIOS)
    ) / 2
    return {
        "metrics": result,
        "probabilities": probabilities,
        "layers": layers,
        "raw": raw,
        "fold_id": fold_id,
        "turn_y": turn_y,
        "turn_p": turn_p,
        "turn_group": turn_group,
    }


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def run(args):
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    rows, manifest_audit = read_manifest(args.root)
    mapping, grouping_scope = speaker_mapping(rows, args.speakers, args.allow_call_groups)
    data = build_dataset(args.root, [r for r in rows if r["split"] == "train"], output, args.jobs)
    records = data["records"]
    assign_groups(records, mapping)
    print(f"agrupacion: {grouping_scope}", flush=True)
    stress = build_stress(args.root, records, output, data["fingerprint"], args.jobs)
    samples = {budget: [r["samples"][budget] for r in records] for budget in features.Presupuesto}
    y = np.array([s["label"] for s in samples["full"]])
    groups = np.array([s["group"] for s in samples["full"]])
    issue15 = None
    issue15_samples = {budget: [] for budget in features.Presupuesto}
    if args.issue15_dir:
        issue15 = load_issue15(args.issue15_dir, output, args.jobs)
        issue15_samples = {
            budget: [record["samples"][budget] for record in issue15["records"]]
            for budget in features.Presupuesto
        }
        external_groups = {sample["group"] for sample in issue15_samples["full"]}
        if external_groups & set(groups):
            raise ValueError("los grupos de issue15 colisionan con el corpus oficial")
        print(
            f"issue15: {len(issue15_samples['full'])} llamadas incorporadas al ajuste final; "
            f"familias={len(external_groups)}",
            flush=True,
        )
    folds = grouped_splits(y, groups, 5, SEED)
    split_audit = [
        {
            "fold": j,
            "train_calls": [samples["full"][i]["call_id"] for i in tr],
            "test_calls": [samples["full"][i]["call_id"] for i in te],
            "train_groups": sorted(set(groups[tr])),
            "test_groups": sorted(set(groups[te])),
            "group_overlap": sorted(set(groups[tr]) & set(groups[te])),
        }
        for j, (tr, te) in enumerate(folds)
    ]
    save_json(output / "folds.json", split_audit)
    evaluations, selected = {}, {}
    source_digest = hashlib.sha256()
    for path in (Path(__file__), Path("app/learning.py"), Path("app/model.py")):
        source_digest.update(path.read_bytes())
    source_digest.update(data["fingerprint"].encode())
    source_digest.update(json.dumps(groups.tolist()).encode())
    source_digest.update((output / "stress_train.joblib").stat().st_mtime_ns.to_bytes(8, "little"))
    evaluation_key = source_digest.hexdigest()
    for budget in features.Presupuesto:
        evaluations[budget] = {}
        for name, parameters in CANDIDATES.items():
            path = output / f"cv_{budget}_{name}.joblib"
            cached = joblib.load(path) if path.exists() else None
            print(f"CV {budget} / {name}", flush=True)
            if cached and cached["fingerprint"] == evaluation_key:
                evaluation = cached["evaluation"]
            else:
                evaluation = evaluate_candidate(
                    samples[budget], {k: stress[k][budget] for k in SCENARIOS}, parameters, folds
                )
                joblib.dump(
                    {"fingerprint": evaluation_key, "evaluation": evaluation}, path, compress=3
                )
            evaluations[budget][name] = evaluation
        best = max(
            CANDIDATES,
            key=lambda name: (
                evaluations[budget][name]["metrics"]["selection_score"],
                -evaluations[budget][name]["metrics"]["clean"]["log_loss"],
                -CANDIDATES[name]["C"],
            ),
        )
        selected[budget] = best
        print(f"seleccion {budget}: {best}", flush=True)

    frozen = {
        "selected": selected,
        "candidates": CANDIDATES,
        "criterion": "mean(clean OOF AUC, worst stress OOF AUC); tie: logloss then smaller C",
        "frozen_at_utc": datetime.now(UTC).isoformat(),
        "val_used_for_selection": False,
        "grouping_scope": grouping_scope,
        "fingerprint": evaluation_key,
        "issue15": (
            {
                "used_for": "final fit only; excluded from candidate selection and official val",
                "fingerprint": issue15["fingerprint"],
                **issue15["audit"],
            }
            if issue15
            else None
        ),
    }
    save_json(output / "selection_frozen.json", frozen)
    extra = " + issue15" if issue15 else ""
    print(f"Seleccion congelada. Ajustando artefacto con train{extra}.", flush=True)
    models = {
        budget: fit_calibrated(
            samples[budget] + issue15_samples[budget], CANDIDATES[selected[budget]]
        )
        for budget in features.Presupuesto
    }
    bundle = {
        "version": BUNDLE_VERSION,
        "extractor_version": EXTRACTOR_VERSION,
        "features": list(feature_names()),
        "groups": {k: features.grupo_de(k) for k in feature_names()},
        "models": models,
        "sklearn_version": sklearn.__version__,
        "python_version": platform.python_version(),
        "training": frozen,
        "positive_class": "synthetic",
        "calibration": "sigmoid / Platt",
        "aggregation": {
            "mean_logit_sigmoid": 0.6,
            "maximum": 0.2,
            "fraction_suspicious": 0.2,
            "suspicious_threshold": 0.7,
        },
        "confidence_policy": {
            "required_votes": 2,
            "single_vote_cap": 0.65,
            "single_available_layer_cap": 0.9,
        },
    }
    args.model.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, args.model, compress=3)

    print("Sanity check final: abriendo val por primera vez en este flujo.", flush=True)
    validation = build_dataset(
        args.root, [r for r in rows if r["split"] == "val"], output, args.jobs
    )
    all_records = records + validation["records"]
    assign_groups(all_records, mapping)  # tambien comprueba duplicados de PCM entre splits
    report = {
        "version": BUNDLE_VERSION,
        "manifest": manifest_audit,
        "dataset": summary(all_records),
        "issue15": issue15["audit"] if issue15 else None,
        "grouping_scope": grouping_scope,
        "excluded": data["excluded"] + validation["excluded"],
        "selection": frozen,
        "model_sha256": hashlib.sha256(args.model.read_bytes()).hexdigest(),
        "budgets": {},
        "candidates": {
            b: {n: e["metrics"] for n, e in es.items()} for b, es in evaluations.items()
        },
    }
    prediction_rows = []
    for budget in features.Presupuesto:
        evaluation = evaluations[budget][selected[budget]]
        p = evaluation["probabilities"]["clean"]
        val = [r["samples"][budget] for r in validation["records"]]
        y_val = np.array([s["label"] for s in val])
        p_val = models[budget].predict_proba(val)[:, 1]
        val_layers = models[budget].calibrated_classifiers_[0].estimator.layer_scores(val)
        result = {
            "selected": selected[budget],
            "cv": evaluation["metrics"],
            "cv_ci95": bootstrap(y, p, groups),
            "val": metrics(y_val, p_val),
            "val_ci95": bootstrap(y_val, p_val, [s["group"] for s in val]),
        }
        report["budgets"][budget] = result
        for split, current, target, prob, scores in (
            ("train_oof", samples[budget], y, p, evaluation["layers"]),
            ("val", val, y_val, p_val, val_layers),
        ):
            for i, s in enumerate(current):
                layer = dict(zip(LAYER_NAMES, map(float, scores[i]), strict=True))
                prediction_rows.append(
                    {
                        "anon_id": s["call_id"],
                        "group": s["group"],
                        "split": split,
                        "budget": budget,
                        "label": int(target[i]),
                        "p": float(prob[i]),
                        "confidence": confidence_policy(prob[i], layer, available_layers(s)),
                        "audio_used_s": s["audio_used_s"],
                        **layer,
                    }
                )
        print(
            f"{budget}: CV AUC {result['cv']['clean']['auc']:.4f}; "
            f"val AUC {result['val']['auc']:.4f}",
            flush=True,
        )
    save_json(output / "metrics.json", report)
    with (output / "predictions.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(prediction_rows[0]))
        writer.writeheader()
        writer.writerows(prediction_rows)
    export_tables(all_records, output)
    joblib.dump(
        {
            "records": all_records,
            "report": report,
            "evaluations": evaluations,
            "predictions": prediction_rows,
        },
        output / "report_inputs.joblib",
        compress=3,
    )
    print(f"Artefacto: {args.model}; metricas: {output / 'metrics.json'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, default=Path("analysis/issue5"))
    parser.add_argument("--model", type=Path, default=Path("model/model.joblib"))
    parser.add_argument("--speakers", type=Path)
    parser.add_argument("--allow-call-groups", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--issue15-dir",
        type=Path,
        help="WAV y manifest.json del issue #15; solo se usan en el ajuste final",
    )
    parser.add_argument("--jobs", type=int, default=4)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
