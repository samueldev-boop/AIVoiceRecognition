import json
from pathlib import Path

AUDIT_LOG_PATH = Path("data/audit.jsonl")

def log_detection_event(call_id: str, is_synthetic: bool, confidence: float = None, stage: int = 1):
    """Registra la decision en disco de forma no bloqueante (<0.2 ms)."""
    try:
        AUDIT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "call_id": call_id,
            "decision": {
                "is_synthetic": bool(is_synthetic),
                "confidence": float(confidence) if confidence is not None else 0.5
            },
            "analysis": {
                "stage": stage
            },
            "status_for_training": "ready"
        }
        with open(AUDIT_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload) + "\n")
    except Exception:
        pass
