import logging
from typing import Any

from src.db import calls_collection

logger = logging.getLogger("altur.curation")


def validate_call_audio(analysis: dict[str, Any]) -> bool:
    if not analysis:
        return False
    turns = analysis.get("turns", [])
    ch0_duration = sum(t["end"] - t["start"] for t in turns if t.get("channel") == 0)
    if ch0_duration < 1.0:
        return False

    acoustics = analysis.get("acoustics") or {}
    pitch = acoustics.get("pitch_mean_hz") or acoustics.get("pitch_f0_mean")
    if not pitch or pitch <= 0:
        return False
    return True


def select_training_batch(max_per_class: int = 100) -> list[dict[str, Any]]:
    query = {
        "status_for_training": "ready",
        "decision.is_synthetic": {"$ne": None},
        "analysis": {"$ne": None},
    }
    candidates = list(calls_collection.find(query))
    if not candidates:
        return []

    bots, humans = [], []
    for doc in candidates:
        dec = doc.get("decision", {})
        analysis = doc.get("analysis", {})

        if not validate_call_audio(analysis):
            continue

        conf = dec.get("confidence", 0.5)
        # Score de incertidumbre: proximidad a 0.5
        uncertainty = 1.0 - (abs(conf - 0.5) * 2)

        entry = {
            "call_id": doc["call_id"],
            "is_synthetic": dec["is_synthetic"],
            "confidence": conf,
            "uncertainty": uncertainty,
            "analysis": analysis,
        }

        if dec["is_synthetic"]:
            bots.append(entry)
        else:
            humans.append(entry)

    bots.sort(key=lambda x: x["uncertainty"], reverse=True)
    humans.sort(key=lambda x: x["uncertainty"], reverse=True)

    target_count = min(len(bots), len(humans), max_per_class)
    if target_count == 0:
        return []

    return bots[:target_count] + humans[:target_count]
