"""Cascada de decision con salida temprana.

El presupuesto de respuesta del reto son 30 s. La cascada existe para responder mucho
antes: se empieza por la primera intervencion del llamante y solo se mira mas audio cuando
la probabilidad calibrada cae en la banda de ambiguedad.

Lo que compra la cascada es LATENCIA, no precision. Medido sobre las predicciones fuera de
muestra de train (282 llamadas), con la banda [0.10, 0.90]:

    solo first_turn siempre          25 errores
    fusionar los tres siempre        24 errores
    cascada con banda [0.10, 0.90]   24 errores, y 223 de 282 (79 %) salen en la etapa 1

Es decir: el mismo acierto, pero cuatro de cada cinco llamadas se contestan con ~0.1 s de
computo en lugar de ~1.3 s. La banda se eligio sobre esas mismas predicciones OOF de train,
nunca sobre val. Bandas mas anchas o mas estrechas dan el mismo total de errores (24-26); se
prefirio [0.10, 0.90] porque deja solo 5 errores fijados en la etapa 1, frente a 8 con
[0.25, 0.75]: los que escalan reciben una segunda y una tercera mirada.

Orden de presupuestos: first_turn, 20s, full. No es el orden de mas a menos audio por
casualidad, es que #5 midio que la primera intervencion es el presupuesto MAS robusto
(CV agrupada por hablante 0.9768 y peor caso de estres 0.9483, frente a 0.9682 y 0.8582 del
completo). Con mas audio el modelo se apoya mas en rasgos de la persona y de su linea.
"""

import time

import numpy as np

from app import config, vad
from app.interventions import extract_sample
from app.model import confidence_policy

# Banda de ambiguedad sobre la probabilidad calibrada: dentro de ella se escala.
BANDA = (0.10, 0.90)
ORDEN = ("first_turn", "20s", "full")

# Cuando dos presupuestos se contradicen, la discrepancia es informacion: baja la confianza
# en lugar de promediarse sin mas.
TOPE_DESACUERDO = 0.65


def _logit(p: float) -> float:
    p = min(max(p, 1e-9), 1 - 1e-9)
    return float(np.log(p / (1 - p)))


def _sigmoide(z: float) -> float:
    return float(1 / (1 + np.exp(-z)))


def decidir(modelo, x: np.ndarray, sr: int, *, limite_s: float | None = None) -> dict:
    """Recorre los presupuestos hasta decidir. Siempre devuelve algo si el primero cabe.

    limite_s acota el tiempo total: se comprueba ENTRE etapas, asi que una etapa ya
    empezada siempre termina. Con las etapas medidas (0.1 s, 0.3 s, 1.3 s) el limite no
    llega a activarse; esta para que un clip patologico no se coma el presupuesto.
    """
    t0 = time.perf_counter()
    limite = config.DETECT_TIMEOUT_S if limite_s is None else limite_s

    # Una llamada larga se analiza solo hasta ANALISIS_MAX_S, para que el computo no crezca
    # con la duracion. Por debajo del tope, que cubre todo el dataset, no cambia nada.
    tope = int(config.ANALISIS_MAX_S * sr)
    if len(x) > tope:
        x = x[:tope]

    # Un solo paso de VAD sobre el audio completo, compartido por first_turn y full. El
    # presupuesto 20s recalcula el suyo a proposito, para no conocer fronteras futuras.
    turnos = vad.turnos(x, sr)
    # Caché de features por intervencion compartida entre presupuestos, igual que en el
    # entrenamiento (scripts/issue5_data.py): asi la fila de un turno se calcula una vez.
    cache: dict = {}

    votos: dict[str, float] = {}
    ultimo: dict | None = None
    etapa = "rechazado"

    for presupuesto in ORDEN:
        if votos and time.perf_counter() - t0 > limite:
            etapa = f"{etapa}+limite"
            break
        if presupuesto == "20s":
            muestra = extract_sample(x, sr, presupuesto, cache=cache)
        else:
            muestra = extract_sample(x, sr, presupuesto, turns=turnos, cache=cache)
        resultado = modelo.predecir(muestra)
        votos[presupuesto] = resultado["probability_synthetic"]
        ultimo = resultado
        etapa = presupuesto
        p = resultado["probability_synthetic"]
        if p < BANDA[0] or p > BANDA[1]:
            break

    if ultimo is None:
        raise ValueError("ningun presupuesto pudo puntuar el clip")

    if len(votos) == 1:
        probabilidad = next(iter(votos.values()))
    else:
        probabilidad = _sigmoide(sum(_logit(v) for v in votos.values()) / len(votos))

    confianza = confidence_policy(probabilidad, ultimo["layer_scores"],
                                  ultimo["available_layers"])
    desacuerdo = len({v >= 0.5 for v in votos.values()}) > 1
    if desacuerdo:
        confianza = min(confianza, TOPE_DESACUERDO)

    return {
        "turns": turnos,
        "is_synthetic": bool(probabilidad >= 0.5),
        "confidence": float(confianza),
        "probability_synthetic": float(probabilidad),
        "stage": etapa,
        "budget_scores": {k: float(v) for k, v in votos.items()},
        "layer_scores": ultimo["layer_scores"],
        "audio_used_s": ultimo["audio_used_s"],
        "disagreement": desacuerdo,
        "ms": (time.perf_counter() - t0) * 1000,
    }


def abstencion(ms: float, motivo: str) -> dict:
    """Respuesta cuando no se pudo decidir dentro del presupuesto.

    Se abstiene hacia la clase que no acusa: marcar a una persona como bot es el error
    caro. La confianza 0.5 lo dice explicitamente en lugar de disfrazarlo.
    """
    return {
        "turns": [],
        "is_synthetic": False,
        "confidence": 0.5,
        "probability_synthetic": 0.5,
        "stage": motivo,
        "budget_scores": {},
        "layer_scores": {},
        "audio_used_s": None,
        "disagreement": False,
        "ms": ms,
    }
