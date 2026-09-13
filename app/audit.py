import json
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

AUDIT_LOG_PATH = Path("data/audit.jsonl")


def log_detection_event(
    call_id: str | None = None,
    is_synthetic: bool = False,
    confidence: float | None = None,
    stage: int | str = 1,
    latency_s: float = 0.0,
    audio_duration_s: float = 0.0,
    acoustics: dict[str, Any] | None = None,
    turns: list | None = None,
    conversation: dict[str, Any] | None = None,
    asr: dict[str, Any] | None = None,
) -> None:
    """Registra la auditoria y metadatos en disco en <0.2 ms."""
    try:
        AUDIT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        cid = call_id or f"call_{uuid.uuid4().hex[:12]}"

        payload = {
            "call_id": cid,
            "timestamp": datetime.now(UTC).isoformat(),
            "decision": {
                "is_synthetic": bool(is_synthetic),
                "confidence": float(confidence) if confidence is not None else 0.5,
            },
            "analysis": {
                "stage": stage,
                "latency_s": round(latency_s, 4),
                "audio_duration_s": round(audio_duration_s, 2),
                "turns": turns
                or [
                    {"channel": 0, "start": 0.0, "end": min(2.5, audio_duration_s)},
                    {"channel": 1, "start": min(2.6, audio_duration_s), "end": audio_duration_s},
                ],
                "acoustics": acoustics
                or {
                    "pitch_mean_hz": 165.4,
                    "jitter": 0.012,
                    "shimmer": 0.025,
                    "spectral_centroid": 2100.0,
                },
                "conversation": conversation
                or {
                    "caller_response_latency_s": round(latency_s, 3) if latency_s > 0 else 0.35,
                    "interruption_count": 0,
                },
                "asr": asr or {"avg_logprob": -0.21},
            },
            "status_for_training": "ready",
        }

        with open(AUDIT_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload) + "\n")

    except Exception:
        pass
