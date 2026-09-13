"""Contrato del endpoint. Los campos obligatorios los fija el reto; el resto es diagnostico."""

from pydantic import AliasChoices, BaseModel, ConfigDict, Field


class DetectRequest(BaseModel):
    # WAV estereo 8 kHz en base64. ch0 = llamante, ch1 = agente.
    audio: str = Field(
        ...,
        validation_alias=AliasChoices("audio_base64", "audio"),
        description="WAV estereo 8 kHz codificado en base64",
    )
    # Identificador del evaluador, opcional. Solo alimenta la auditoria (como HMAC) y el
    # contrato no fija su formato: se acepta texto o numero de cualquier longitud para que
    # nunca convierta en 422 una llamada valida.
    call_id: str | int | None = Field(default=None, description="ID de la llamada, opcional")


class DetectResponse(BaseModel):
    is_synthetic: bool
    confidence: float = Field(..., ge=0.0, le=1.0)


class DetectDiagnosticsResponse(DetectResponse):

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


# /transcribe: solo para la interfaz, fuera del contrato del reto.
class TranscribeRequest(BaseModel):
    audio: str = Field(..., description="El mismo WAV que recibe /detect, en base64")


class TranscriptSegment(BaseModel):
    channel: int
    start: float
    end: float
    text: str


class TranscribeResponse(BaseModel):
    segments: list[TranscriptSegment]
    ms: float | None = None


class HealthResponse(BaseModel):
    # pydantic reserva el prefijo "model_"; aqui lo usamos a proposito en la respuesta.
    model_config = ConfigDict(protected_namespaces=())

    ok: bool
    version: str
    model_loaded: bool
    model_version: str | None = None
