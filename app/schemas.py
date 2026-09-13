"""Contrato del endpoint. Los campos obligatorios los fija el reto; el resto es diagnostico."""

from pydantic import BaseModel, ConfigDict, Field, model_validator


class DetectRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    # Soporte dual: 'audio_base64' (contrato oficial del jurado) y 'audio' (contrato interno)
    audio: str | None = Field(default=None, description="WAV estereo 8 kHz codificado en base64")
    audio_base64: str | None = Field(
        default=None, description="WAV estereo 8 kHz codificado en base64 (jurado)"
    )
    call_id: str | None = Field(default=None, min_length=1, max_length=128)

    @property
    def raw_audio(self) -> str:
        return self.audio_base64 or self.audio or ""

    @model_validator(mode="after")
    def validate_audio_presence(self):
        if self.audio and self.audio_base64 and self.audio != self.audio_base64:
            raise ValueError("audio y audio_base64 deben coincidir")
        target = self.audio_base64 or self.audio
        if not target:
            raise ValueError("Se requiere el campo 'audio_base64' o 'audio'")
        if not self.audio:
            self.audio = target
        if not self.audio_base64:
            self.audio_base64 = target
        return self


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
