import logging
import pickle
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from src.config import MODEL_DIR
from src.curation import select_training_batch
from src.db import calls_collection

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [Retrain] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("altur.retrain")

def extract_features(analysis: dict) -> list:
    acoustics = analysis.get("acoustics") or {}
    convo = analysis.get("conversation") or {}
    asr = analysis.get("asr") or {}

    return [
        float(acoustics.get("pitch_mean_hz") or acoustics.get("pitch_f0_mean") or 150.0),
        float(acoustics.get("jitter") or 0.01),
        float(acoustics.get("shimmer") or 0.02),
        float(acoustics.get("spectral_centroid") or 2000.0),
        float(convo.get("caller_response_latency_s") or convo.get("first_response_latency") or 0.3),
        float(convo.get("interruption_count") or convo.get("overlaps") or 0),
        float(asr.get("avg_logprob") or -0.2),
    ]

def execute_retraining():
    batch = select_training_batch(max_per_class=100)
    if not batch or len(batch) < 2:
        logger.info("No hay suficientes muestras balanceadas en estado 'ready'.")
        return

    X = np.array([extract_features(item["analysis"]) for item in batch])
    y = np.array([1 if item["is_synthetic"] else 0 for item in batch])
    call_ids = [item["call_id"] for item in batch]

    pipeline = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, solver="lbfgs"))
    pipeline.fit(X, y)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    version_id = datetime.now(timezone.utc).strftime("%Y_w%U")
    
    version_path = MODEL_DIR / f"classifier_{version_id}.pkl"
    latest_path = MODEL_DIR / "classifier_latest.pkl"

    with open(version_path, "wb") as f:
        pickle.dump(pipeline, f)
    with open(latest_path, "wb") as f:
        pickle.dump(pipeline, f)

    calls_collection.update_many(
        {"call_id": {"$in": call_ids}},
        {"$set": {"status_for_training": f"trained_{version_id}"}}
    )

    logger.info(f"Reentrenamiento completado con {len(call_ids)} muestras curadas.")
    logger.info(f"Artefacto guardado en: {latest_path}")

if __name__ == "__main__":
    execute_retraining()
