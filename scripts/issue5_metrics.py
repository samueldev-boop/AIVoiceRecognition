"""Metricas sklearn y bootstrap por grupo (no por turno). Positivo = sintetico."""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


def ece(y, p, bins=10):
    indices = np.minimum((np.asarray(p) * bins).astype(int), bins - 1)
    return float(
        sum(
            np.mean(indices == b) * abs(np.mean(p[indices == b]) - np.mean(y[indices == b]))
            for b in range(bins)
            if np.any(indices == b)
        )
    )


def metrics(y, p):
    y, p = np.asarray(y), np.asarray(p)
    thresholds = {}
    for threshold in (0.5, 0.7):
        pred = p >= threshold
        matrix = confusion_matrix(y, pred, labels=[0, 1])
        tn, fp, fn, tp = matrix.ravel()
        thresholds[str(threshold)] = {
            "accuracy": float(accuracy_score(y, pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
            "precision": float(precision_score(y, pred, zero_division=0)),
            "recall": float(recall_score(y, pred, zero_division=0)),
            "f1": float(f1_score(y, pred, zero_division=0)),
            "fpr": float(fp / max(1, tn + fp)),
            "fnr": float(fn / max(1, tp + fn)),
            "confusion_matrix": matrix.tolist(),
        }
    return {
        "n": len(y),
        "human": int((y == 0).sum()),
        "synthetic": int((y == 1).sum()),
        "auc": float(roc_auc_score(y, p)),
        "average_precision": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "ece_10_bins": ece(y, p),
        "thresholds": thresholds,
    }


def bootstrap(y, p, groups, repetitions=1000, seed=5):
    y, p, groups = np.asarray(y), np.asarray(p), np.asarray(groups)
    unique = np.unique(groups)
    members = [np.flatnonzero(groups == group) for group in unique]
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(repetitions):
        index = np.concatenate([members[i] for i in rng.integers(0, len(unique), len(unique))])
        target, score = y[index], p[index]
        if len(np.unique(target)) < 2:
            continue
        pred = score >= 0.5
        estimates.append(
            (
                roc_auc_score(target, score),
                accuracy_score(target, pred),
                precision_score(target, pred, zero_division=0),
                recall_score(target, pred, zero_division=0),
            )
        )
    limits = np.quantile(estimates, [0.025, 0.975], axis=0)
    return {
        "method": "percentile bootstrap by supplied group; 95%; threshold=0.5",
        "repetitions": len(estimates),
        **{
            name: limits[:, j].tolist()
            for j, name in enumerate(("auc", "accuracy", "precision", "recall"))
        },
    }
