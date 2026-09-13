"""Contrato versionado de la referencia a una llamada incierta.

No es un dataset: guarda la decision del modelo y como se llego a ella, pero ni audio, ni
transcripcion, ni etiqueta. Sirve para localizar despues la llamada y llevarla al ciclo de
reentrenamiento, que vive fuera de este modulo.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, field_validator, model_validator

Identifier = Annotated[str, Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_.:-]+$")]
Probability = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
Nonnegative = Annotated[float, Field(ge=0, allow_inf_nan=False)]


class Document(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class TurnSegment(Document):
    channel: Literal[0, 1]
    start: Nonnegative
    end: Nonnegative

    @model_validator(mode="after")
    def ordered(self) -> TurnSegment:
        if self.end <= self.start:
            raise ValueError("segment end must exceed start")
        return self


class DecisionPayload(Document):
    is_synthetic: bool = Field(strict=True)
    confidence: Probability | None = None
    probability_synthetic: Probability | None = None


class AnalysisPayload(Document):
    stage: str | int = "unknown"
    latency_s: Nonnegative = 0
    audio_duration_s: Nonnegative = 0
    turns: list[TurnSegment] = Field(default_factory=list, max_length=10000)
    disagreement: bool = False
    layer_scores: dict[str, Probability] = Field(default_factory=dict)
    budget_scores: dict[str, Probability] = Field(default_factory=dict)
    # Por que la llamada se considero incierta: probabilidad_ambigua, desacuerdo, abstencion.
    uncertainty_reasons: list[Identifier] = Field(default_factory=list, max_length=10)

    @model_validator(mode="after")
    def duration_bounds(self) -> AnalysisPayload:
        if any(t.end > self.audio_duration_s for t in self.turns):
            raise ValueError("turn exceeds audio duration")
        return self


class CallAuditRecord(Document):
    schema_version: Literal[1] = 1
    event_id: Identifier
    call_id: Identifier
    timestamp: datetime
    source: Identifier
    model_version: str = Field(default="unknown", min_length=1, max_length=128)
    model_artifact_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    pipeline_version: str = Field(default="audit-v1", min_length=1, max_length=128)
    # Referencia al audio: su huella y, si existe, donde lo guarda quien lo tiene.
    input_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    audio_uri: str | None = Field(default=None, max_length=1024)
    decision: DecisionPayload
    analysis: AnalysisPayload
    errors: list[Identifier] = Field(default_factory=list, max_length=100)
    metadata: dict[str, JsonValue] = Field(default_factory=dict)

    @field_validator("timestamp")
    @classmethod
    def utc_timestamp(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("timestamp requires timezone")
        return value.astimezone(UTC)

    @field_validator("audio_uri")
    @classmethod
    def private_reference(cls, value: str | None) -> str | None:
        if value and ("?" in value or "@" in value or not value.startswith("s3://")):
            raise ValueError("audio_uri must be a private s3 object reference without credentials")
        return value
