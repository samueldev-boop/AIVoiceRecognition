"""Durable audit publication, independent of MongoDB availability."""

import hashlib
import hmac
import logging
import uuid
from datetime import UTC, datetime
from typing import Any

from src.config import Settings
from src.schemas import CallAuditRecord
from src.storage import atomic_write

logger = logging.getLogger("altur.audit")


def log_detection_event(
    call_id: str | None = None,
    is_synthetic: bool = False,
    confidence: float | None = None,
    probability_synthetic: float | None = None,
    stage: int | str = 1,
    latency_s: float = 0.0,
    audio_duration_s: float = 0.0,
    turns: list | None = None,
    disagreement: bool = False,
    layer_scores: dict[str, Any] | None = None,
    budget_scores: dict[str, Any] | None = None,
    acoustics: dict[str, Any] | None = None,
    conversation: dict[str, Any] | None = None,
    asr: dict[str, Any] | None = None,
    *,
    input_sha256: str | None = None,
    model_version: str | None = None,
    model_artifact_sha256: str | None = None,
) -> str:
    """Return an event ID only after durable publication; never manufacture telemetry."""
    settings = Settings.from_env()
    event_id = uuid.uuid4().hex
    # Never persist external IDs. Without a private key, omit cross-request correlation.
    key = settings.audit_id_key.get_secret_value()
    opaque_call = (
        hmac.new(key.encode(), call_id.encode(), hashlib.sha256).hexdigest()
        if key and call_id
        else event_id
    )
    record = CallAuditRecord(
        event_id=event_id,
        call_id=opaque_call,
        timestamp=datetime.now(UTC),
        source="detect-api",
        model_version=model_version or settings.model_version,
        model_artifact_sha256=model_artifact_sha256,
        pipeline_version=settings.pipeline_version,
        input_sha256=input_sha256,
        decision={
            "is_synthetic": is_synthetic,
            "confidence": confidence,
            "probability_synthetic": probability_synthetic,
        },
        analysis={
            "stage": stage,
            "latency_s": latency_s,
            "audio_duration_s": audio_duration_s,
            "turns": turns or [],
            "disagreement": disagreement,
            "layer_scores": layer_scores or {},
            "budget_scores": budget_scores or {},
            "acoustics": acoustics,
            "conversation": conversation,
            "asr": asr,
        },
    )
    content = record.model_dump_json().encode("utf-8")
    if len(content) > settings.max_json_bytes:
        raise ValueError("audit event exceeds configured size")
    atomic_write(settings.spool_dir / f"{event_id}.json", content)
    logger.info("event=audit_published document=%s", event_id)
    return event_id
