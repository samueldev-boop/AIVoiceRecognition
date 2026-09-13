from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class TurnSegment(BaseModel):
    model_config = ConfigDict(extra="allow")
    channel: int  # 0 = llamante (usuario), 1 = agente bancario
    start: float
    end: float


class DecisionPayload(BaseModel):
    model_config = ConfigDict(extra="allow")
    is_synthetic: bool
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)


class AnalysisPayload(BaseModel):
    model_config = ConfigDict(extra="allow")
    turns: list[TurnSegment] = []
    acoustics: dict[str, Any] | None = None
    conversation: dict[str, Any] | None = None
    asr: dict[str, Any] | None = None

    @model_validator(mode="before")
    @classmethod
    def reconcile_analysis_fields(cls, data: Any):
        if isinstance(data, dict):
            if "acoustics" not in data and "acoustic_metrics" in data:
                data["acoustics"] = data["acoustic_metrics"]
            if "conversation" not in data and "conversational_metrics" in data:
                data["conversation"] = data["conversational_metrics"]
            if "asr" not in data and "asr_metrics" in data:
                data["asr"] = data["asr_metrics"]
        return data


class CallAuditRecord(BaseModel):
    model_config = ConfigDict(extra="allow")
    call_id: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    decision: DecisionPayload | None = None
    analysis: AnalysisPayload | None = None
    audio_uri: str | None = None
    status_for_training: str | None = "ready"

    @model_validator(mode="before")
    @classmethod
    def reconcile_root_decision(cls, data: Any):
        if isinstance(data, dict):
            if "decision" not in data and "prediction" in data:
                data["decision"] = data["prediction"]
        return data
