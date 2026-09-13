"""Referencia duradera a las llamadas inciertas, independiente de la disponibilidad de MongoDB.

Solo se registran las llamadas en las que el modelo no llego a una conclusion clara. De cada
una se guarda una referencia (huella del audio, identificador opaco, decision y como se
llego a ella), nunca el audio: sirve para recuperarla despues en el ciclo de reentrenamiento.
"""

import hashlib
import hmac
import logging
import uuid
from datetime import UTC, datetime
from typing import Any

from app import config
from src.config import Settings
from src.schemas import CallAuditRecord
from src.storage import atomic_write

logger = logging.getLogger("altur.audit")

# Etapas en las que la cascada no pudo decidir y se abstuvo hacia "persona".
ABSTENCIONES = ("sin_habla", "watchdog")


def motivos_de_incertidumbre(resultado: dict[str, Any]) -> list[str]:
    """Por que una respuesta de la cascada es incierta. Lista vacia si la decision es clara.

    - probabilidad_ambigua: la probabilidad final cae dentro de la banda de ambiguedad, la
      misma que hace escalar a la cascada (por defecto 0.10-0.90);
    - desacuerdo: dos presupuestos de audio se contradijeron;
    - abstencion: no hubo habla utilizable, se agoto el tiempo o se corto entre etapas.
    """
    motivos = []
    p = resultado.get("probability_synthetic")
    minimo, maximo = config.AUDIT_BANDA
    if p is not None and minimo <= p <= maximo:
        motivos.append("probabilidad_ambigua")
    if resultado.get("disagreement"):
        motivos.append("desacuerdo")
    etapa = str(resultado.get("stage", ""))
    if etapa in ABSTENCIONES or etapa.endswith("+limite"):
        motivos.append("abstencion")
    return motivos


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
    uncertainty_reasons: list[str] | None = None,
    *,
    input_sha256: str | None = None,
    model_version: str | None = None,
    model_artifact_sha256: str | None = None,
) -> str:
    """Publica el evento y devuelve su ID solo cuando ya esta en disco."""
    settings = Settings.from_env()
    event_id = uuid.uuid4().hex
    # Nunca se guarda un ID externo en claro. Sin clave privada no hay correlacion.
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
            "uncertainty_reasons": uncertainty_reasons or [],
        },
    )
    content = record.model_dump_json().encode("utf-8")
    if len(content) > settings.max_json_bytes:
        raise ValueError("audit event exceeds configured size")
    atomic_write(settings.spool_dir / f"{event_id}.json", content)
    logger.info("event=audit_published document=%s", event_id)
    return event_id
