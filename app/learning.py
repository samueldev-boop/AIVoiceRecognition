"""Regresiones por capa, agregacion por llamada y fusion sin fuga entre grupos.

El estimador acepta una lista de llamadas, no una matriz de turnos sueltos. Asi la
validacion y CalibratedClassifierCV separan la jerarquia completa por llamada/hablante.
"""

import numpy as np
from scipy.special import expit
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted

from app import features
from app.interventions import LAYER_NAMES, TURN_GROUPS, feature_names


class FiniteValues(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float).copy()
        X[~np.isfinite(X)] = np.nan
        return X


class QuantileClipper(TransformerMixin, BaseEstimator):
    """Winsorizacion ajustada solo en fit; no elimina llamadas ni turnos."""

    def __init__(self, quantile=0.005):
        self.quantile = quantile

    def fit(self, X, y=None):
        self.lower_ = np.quantile(X, self.quantile, axis=0)
        self.upper_ = np.quantile(X, 1 - self.quantile, axis=0)
        return self

    def transform(self, X):
        check_is_fitted(self, "lower_")
        if self.quantile == 0:
            return np.asarray(X)
        return np.clip(X, self.lower_, self.upper_)


def pipeline(C=0.33, quantile=0.005):
    # sklearn 1.9 depreca penalty="l2"; l1_ratio=0 es exactamente L2.
    return Pipeline(
        [
            ("finite", FiniteValues()),
            (
                "imputer",
                SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True),
            ),
            ("clipper", QuantileClipper(quantile)),
            ("scaler", StandardScaler()),
            ("logistic", LogisticRegression(C=C, l1_ratio=0.0, max_iter=1500, random_state=5)),
        ]
    )


def grouped_splits(y, groups, n_splits=3, seed=5):
    """Falla si no es posible evaluar dos clases con grupos realmente separados."""
    y, groups = np.asarray(y), np.asarray(groups)
    if len(np.unique(groups)) < n_splits:
        raise ValueError("no hay suficientes grupos para validacion agrupada")
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = list(cv.split(np.zeros(len(y)), y, groups))
    for train, test in folds:
        if set(groups[train]) & set(groups[test]):
            raise ValueError("fuga de grupos")
        if len(np.unique(y[train])) != 2 or len(np.unique(y[test])) != 2:
            raise ValueError("cada particion agrupada debe contener ambas clases")
    return folds


def group_indices():
    return {
        group: np.array(
            [
                i
                for i, k in enumerate(feature_names())
                if features.grupo_de(k) == group and (group not in TURN_GROUPS or k != "speech0_s")
            ]
        )
        for group in LAYER_NAMES
    }


def aggregate(logits):
    """Tres resumenes de un grupo de turnos -> un score de capa.

    Coeficientes fijados antes de evaluar: 60% media de logits pasada a sigmoide,
    20% maximo, 20% fraccion con p>=0.7. La fusion aprende sobre seis scores.
    """
    if not len(logits):
        return 0.5, (0.5, 0.5, 0.0)
    p = expit(logits)
    stats = (float(expit(np.mean(logits))), float(np.max(p)), float(np.mean(p >= 0.7)))
    return float(np.dot((0.6, 0.2, 0.2), stats)), stats


def fit_layers(records, y, C, quantile, active_layers=None):
    indices = group_indices()
    models = {}
    turns, labels, weights = [], [], []
    for record, label in zip(records, y, strict=True):
        rows = record.get("training_turn_features", record["turn_features"])
        if len(rows):
            turns.extend(rows)
            labels.extend([label] * len(rows))
            weights.extend([1 / len(rows)] * len(rows))
    if len(np.unique(labels)) < 2:
        raise ValueError("faltan turnos de ambas clases para entrenar")
    turns = np.asarray(turns)
    # Cada llamada pesa uno; se normaliza para mantener comparable el parametro C.
    weights = np.asarray(weights) * len(weights) / np.sum(weights)
    for name in active_layers or LAYER_NAMES:
        model = pipeline(C, quantile)
        if name in TURN_GROUPS:
            model.fit(turns[:, indices[name]], labels, logistic__sample_weight=weights)
        else:
            matrix = np.array([r["call_features"][indices[name]] for r in records])
            model.fit(matrix, y)
        models[name] = model
    return models


def layer_scores(models, records):
    indices = group_indices()
    scores = np.empty((len(records), len(LAYER_NAMES)))
    for j, name in enumerate(LAYER_NAMES):
        if name not in models:
            scores[:, j] = 0.5
            continue
        model = models[name]
        if name in TURN_GROUPS:
            lengths = [len(r["turn_features"]) for r in records]
            if sum(lengths):
                rows = np.concatenate([r["turn_features"] for r in records])[:, indices[name]]
                logits = model.decision_function(rows)
            else:
                logits = np.empty(0)
            offset = 0
            for i, length in enumerate(lengths):
                scores[i, j] = aggregate(logits[offset : offset + length])[0]
                offset += length
        else:
            matrix = np.array([r["call_features"][indices[name]] for r in records])
            scores[:, j] = model.predict_proba(matrix)[:, 1]
    return scores


class LayeredClassifier(ClassifierMixin, BaseEstimator):
    """El metamodelo solo ve scores fuera de muestra de sus capas inferiores."""

    def __init__(self, C=0.33, quantile=0.005, inner_splits=3, seed=5, active_layers=None):
        self.C = C
        self.quantile = quantile
        self.inner_splits = inner_splits
        self.seed = seed
        self.active_layers = active_layers

    def fit(self, X, y):
        records, y = list(X), np.asarray(y)
        if any(r.get("split", "train") != "train" for r in records):
            raise ValueError("val no puede usarse para entrenar, fusionar ni calibrar")
        groups = np.array([r["group"] for r in records])
        self.fusion_folds_ = grouped_splits(y, groups, self.inner_splits, self.seed)
        oof = np.full((len(records), len(LAYER_NAMES)), np.nan)
        for train, test in self.fusion_folds_:
            models = fit_layers(
                [records[i] for i in train], y[train], self.C, self.quantile, self.active_layers
            )
            oof[test] = layer_scores(models, [records[i] for i in test])
        if not np.isfinite(oof).all():
            raise ValueError("scores OOF incompletos")
        self.fusion_ = pipeline(self.C, 0).fit(oof, y)
        self.layers_ = fit_layers(records, y, self.C, self.quantile, self.active_layers)
        self.classes_ = np.array([0, 1])
        self.training_groups_ = tuple(sorted(set(groups)))
        return self

    def decision_function(self, X):
        check_is_fitted(self, "fusion_")
        return self.fusion_.decision_function(self.layer_scores(X))

    def predict_proba(self, X):
        p = expit(self.decision_function(X))
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    def layer_scores(self, X):
        check_is_fitted(self, "layers_")
        return layer_scores(self.layers_, list(X))
