"""Train and evaluate isolated telemetry candidates; never replace the API artifact."""

import argparse
import hashlib
import json
import logging
import subprocess
import uuid
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.config import Settings
from src.data_processing import FEATURE_VERSION, validate_training_splits
from src.storage import exclusive_lock, write_json


def metrics(y, probabilities) -> dict:
    tn, fp, fn, tp = confusion_matrix(y, probabilities >= 0.5, labels=[0, 1]).ravel()
    return {
        "auc": float(roc_auc_score(y, probabilities)),
        "brier": float(brier_score_loss(y, probabilities)),
        "balanced_accuracy": float(balanced_accuracy_score(y, probabilities >= 0.5)),
        "fpr": float(fp / (tn + fp)),
        "fnr": float(fn / (tp + fn)),
    }


def promotion_decision(candidate: dict, baseline: dict | None, *, compatible: bool) -> dict:
    """Predeclared validation gate. A completed fit alone never authorizes promotion."""
    reasons = []
    values = [
        v
        for report in (candidate, baseline)
        if report
        for scores in report.values()
        for v in scores.values()
    ]
    if not all(np.isfinite(v) and 0 <= v <= 1 for v in values):
        return {"eligible": False, "reasons": ["invalid_metrics"], "automatic_promotion": False}
    if not compatible:
        reasons.append("incompatible_with_serving_contract")
    if baseline is None:
        reasons.append("missing_comparable_baseline")
    for split in ("validation",):
        scores = candidate[split]
        if scores["auc"] < 0.90 or scores["fpr"] > 0.05:
            reasons.append("absolute_quality_gate")
        if baseline and (
            scores["auc"] < baseline[split]["auc"] + 0.005
            or scores["brier"] > baseline[split]["brier"]
            or scores["fpr"] > baseline[split]["fpr"]
        ):
            reasons.append("no_safe_improvement")
    return {"eligible": not reasons, "reasons": reasons, "automatic_promotion": False}


def git_provenance() -> dict:
    root = Path(__file__).resolve().parents[1]
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            text=True,
            timeout=5,
            stderr=subprocess.DEVNULL,
        ).strip()
        diff = subprocess.check_output(
            ["git", "diff", "HEAD"], cwd=root, timeout=5, stderr=subprocess.DEVNULL
        )
        status = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=root, timeout=5, stderr=subprocess.DEVNULL
        )
        source_digest = hashlib.sha256()
        for folder in ("src", "app"):
            for path in sorted((root / folder).glob("*.py")):
                source_digest.update(str(path.relative_to(root)).encode())
                source_digest.update(path.read_bytes())
        return {
            "commit": commit,
            "dirty": bool(status),
            "tracked_diff_sha256": hashlib.sha256(diff).hexdigest(),
            "source_sha256": source_digest.hexdigest(),
        }
    except (OSError, subprocess.SubprocessError):
        return {"commit": None, "dirty": None}


def execute_retraining(dataset_path: Path, *, baseline_path: Path | None = None, settings=None):
    settings = settings or Settings.from_env()
    dataset_bytes = dataset_path.read_bytes()
    dataset = json.loads(dataset_bytes)
    validate_training_splits(dataset)
    # Keep experiment outputs separate from the production model/model.joblib.
    with exclusive_lock(settings.model_dir / "training.lock"):
        run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:12]
        output = settings.model_dir / run_id
        output.mkdir(parents=True)
        splits = {
            name: (
                np.asarray([r["features"] for r in rows]),
                np.asarray([r["label"] for r in rows]),
            )
            for name, rows in dataset["splits"].items()
        }
        train_x, train_y = splits["train"]
        if len(train_y) < 20:
            raise ValueError("at least 20 training examples are required")
        candidate = make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=1000, random_state=dataset["seed"])
        )
        candidate.fit(train_x, train_y)
        scores = {
            name: metrics(y, candidate.predict_proba(x)[:, 1])
            for name, (x, y) in splits.items()
            if name != "train"
        }
        baseline_scores = None
        if baseline_path:
            # Only load explicitly supplied, trusted local artifacts.
            baseline = joblib.load(baseline_path)
            if baseline.get("feature_version") != FEATURE_VERSION:
                raise ValueError("baseline feature contract is incompatible")
            if baseline.get("dataset_id") != dataset["dataset_id"]:
                raise ValueError("baseline must use the same frozen dataset partitions")
            if baseline.get("feature_names") != dataset["feature_names"]:
                raise ValueError("baseline feature names differ")
            baseline_scores = {
                name: metrics(y, baseline["estimator"].predict_proba(x)[:, 1])
                for name, (x, y) in splits.items()
                if name != "train"
            }
        artifact = output / "candidate.joblib"
        joblib.dump(
            {
                "estimator": candidate,
                "feature_version": FEATURE_VERSION,
                "feature_names": dataset["feature_names"],
                "dataset_id": dataset["dataset_id"],
                "run_id": run_id,
                "sklearn_version": sklearn.__version__,
            },
            artifact,
        )
        report = {
            "run_id": run_id,
            "trained_at": datetime.now(UTC).isoformat(),
            "dataset_id": dataset["dataset_id"],
            "dataset_sha256": hashlib.sha256(dataset_bytes).hexdigest(),
            "artifact_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
            "code": git_provenance(),
            "pipeline_version": settings.pipeline_version,
            "feature_version": FEATURE_VERSION,
            "sklearn_version": sklearn.__version__,
            "hyperparameters": {"max_iter": 1000, "random_state": dataset["seed"], "C": 1.0},
            "distribution": dataset["distribution"],
            "metrics": scores,
            "baseline_metrics": baseline_scores,
            "promotion": promotion_decision(scores, baseline_scores, compatible=False),
        }
        write_json(output / "report.json", report)
        logging.getLogger(__name__).info("event=candidate_evaluated job=%s", run_id)
        return output


def main() -> None:
    from dotenv import load_dotenv

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--baseline", type=Path)
    args = parser.parse_args()
    load_dotenv()
    logging.basicConfig(level=Settings.from_env().log_level)
    execute_retraining(args.dataset, baseline_path=args.baseline)


if __name__ == "__main__":
    main()
