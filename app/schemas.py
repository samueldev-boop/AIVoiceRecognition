"""Contrato del endpoint. Los campos obligatorios los fija el reto; el resto es diagnostico."""

from pydantic import BaseModel, ConfigDict, Field


class DetectRequest(BaseModel):
    # WAV estereo 8 kHz en base64. ch0 = llamante, ch1 = agente.
    audio: str = Field(..., description="WAV estereo 8 kHz codificado en base64")


class DetectResponse(BaseModel):
    is_synthetic: bool
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)

    # Diagnostico: no lo exige el contrato, pero alimenta la demo y el script de evaluacion.
    stage: str | None = None
    probability_synthetic: float | None = None
    budget_scores: dict[str, float] | None = None
    layer_scores: dict[str, float] | None = None
    disagreement: bool | None = None
    audio_used_s: float | None = None
    ms: float | None = None
    # Turnos del VAD, para que la interfaz pueda dibujar sobre que decidio el modelo.
    turns: list[dict] | None = None


class HealthResponse(BaseModel):
    # pydantic reserva el prefijo "model_"; aqui lo usamos a proposito en la respuesta.
    model_config = ConfigDict(protected_namespaces=())

    ok: bool
    version: str
    model_loaded: bool
    model_version: str | None = None
