"""Carga del artefacto #5, probabilidades calibradas y confianza con acuerdo k de n."""

import os

import joblib
import numpy as np
import sklearn

from app import config
from app.interventions import EXTRACTOR_VERSION, LAYER_NAMES, extract_sample, feature_names
from app.learning import group_indices

BUNDLE_VERSION = "issue5-interventions-v1"


def available_layers(sample):
    indices = group_indices()
    return [
        k
        for k in LAYER_NAMES
        if np.isfinite(
            np.asarray(sample["turn_features"])[:, indices[k]]
            if k in LAYER_NAMES[:4]
            else np.asarray(sample["call_features"])[indices[k]]
        ).any()
    ]


def confidence_policy(probability, scores, available=None):
    """Confianza en la clase elegida; no es P(sintetico).

    P(sintetico) conserva la calibracion Platt. El tope conservador de confianza se
    reporta aparte y no se presenta como una segunda probabilidad calibrada.
    """
    available = list(scores) if available is None else available
    votes = sum(scores[k] >= 0.7 for k in available)
    confidence = max(probability, 1 - probability)
    if len(available) < 2:
        confidence = min(confidence, 0.9)
    if probability >= 0.5 and votes < 2:
        confidence = min(confidence, 0.65)
    return float(confidence)


class Modelo:
    """Envoltorio del artefacto entrenado."""

    def __init__(self, bundle: dict) -> None:
        if bundle.get("version") != BUNDLE_VERSION:
            raise ValueError(
                "version de artefacto incompatible; reentrene con scripts.train_issue5"
            )
        if bundle.get("features") != list(feature_names()):
            raise ValueError("el esquema de features no coincide con el extractor")
        if bundle.get("extractor_version") != EXTRACTOR_VERSION:
            raise ValueError("version del extractor incompatible")
        if bundle.get("sklearn_version") != sklearn.__version__:
            raise ValueError("version de scikit-learn distinta a la del entrenamiento")
        if set(bundle.get("models", {})) != {"first_turn", "20s", "full"}:
            raise ValueError("faltan modelos por presupuesto")
        self.bundle = bundle
        self.version: str = bundle["version"]
        self.features: list[str] = bundle["features"]

    def puntuar(self, features: dict) -> tuple[float, dict[str, float]]:
        """Recibe extract_sample(); devuelve (P(sintetico), scores de las seis capas)."""
        if not {"turn_features", "call_features", "budget"} <= features.keys():
            raise ValueError(
                "se requieren features por intervencion y por llamada: extract_sample()"
            )
        if features["budget"] not in self.bundle["models"]:
            raise ValueError("presupuesto desconocido")
        if np.shape(features["call_features"]) != (len(self.features),):
            raise ValueError("dimension de features de llamada incompatible")
        turns = np.asarray(features["turn_features"])
        if turns.ndim != 2 or turns.shape[1] != len(self.features):
            raise ValueError("dimension de features de intervencion incompatible")
        if not len(turns) or not np.isfinite(turns).any():
            raise ValueError("sin intervenciones utilizables del llamante")
        estimator = self.bundle["models"][features["budget"]]
        probability = float(estimator.predict_proba([features])[0, 1])
        layers = estimator.calibrated_classifiers_[0].estimator.layer_scores([features])[0]
        return probability, dict(zip(LAYER_NAMES, map(float, layers), strict=True))

    def predecir(self, sample: dict) -> dict:
        probability, scores = self.puntuar(sample)
        available = available_layers(sample)
        return {
            "is_synthetic": probability >= 0.5,
            "confidence": confidence_policy(probability, scores, available),
            "probability_synthetic": probability,
            "layer_scores": scores,
            "available_layers": available,
            "audio_used_s": sample.get("audio_used_s"),
            "stage": sample["budget"],
        }

    def puntuar_audio(self, x: np.ndarray, sr: int, presupuesto="first_turn") -> dict:
        return self.predecir(extract_sample(x, sr, presupuesto))


def cargar() -> Modelo | None:
    """Carga el modelo si el artefacto existe. Devuelve None si todavia no hay."""
    ruta = config.MODEL_PATH
    if not os.path.exists(ruta):
        return None
    return Modelo(joblib.load(ruta))
