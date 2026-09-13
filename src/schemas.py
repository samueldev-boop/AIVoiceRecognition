from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field, model_validator

class TurnSegment(BaseModel):
    model_config = ConfigDict(extra="allow")
    channel: int  # 0 = llamante (usuario), 1 = agente bancario
    start: float
    end: float

class DecisionPayload(BaseModel):
    model_config = ConfigDict(extra="allow")
    is_synthetic: bool
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)

class AnalysisPayload(BaseModel):
    model_config = ConfigDict(extra="allow")
    turns: List[TurnSegment] = []
    acoustics: Optional[Dict[str, Any]] = None
    conversation: Optional[Dict[str, Any]] = None
    asr: Optional[Dict[str, Any]] = None

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
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    decision: Optional[DecisionPayload] = None
    analysis: Optional[AnalysisPayload] = None
    audio_uri: Optional[str] = None
    status_for_training: Optional[str] = "ready"

    @model_validator(mode="before")
    @classmethod
    def reconcile_root_decision(cls, data: Any):
        if isinstance(data, dict):
            if "decision" not in data and "prediction" in data:
                data["decision"] = data["prediction"]
        return data
