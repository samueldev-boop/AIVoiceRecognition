"""Algebra de intervalos sobre los turnos, con la libreria `portion`.

Antes esto eran merge() y overlap_len() escritos a mano en analysis/turns_probe.py.
`portion` normaliza las uniones (fusiona solapes sola) y da interseccion y diferencia,
que es todo lo que necesitan las features de conducta.

Se eligio `portion` (27 KB, una dependencia) en lugar de pyannote.core, que declara
pandas y son ~60 MB dentro de la imagen para hacer lo mismo.
"""

import portion as P


def union(segmentos: list[tuple[float, float]]) -> P.Interval:
    """Une una lista de (inicio, fin) en un intervalo normalizado, sin solapes."""
    total = P.empty()
    for inicio, fin in segmentos:
        if fin > inicio:
            total |= P.closed(inicio, fin)
    return total


def duracion(intervalo: P.Interval) -> float:
    """Suma de las longitudes de los tramos del intervalo."""
    return float(sum(a.upper - a.lower for a in intervalo if not a.empty))


def tramos(intervalo: P.Interval) -> list[tuple[float, float]]:
    """Tramos atomicos como lista de (inicio, fin)."""
    return [(float(a.lower), float(a.upper)) for a in intervalo if not a.empty]


def solape(a: P.Interval, b: P.Interval) -> float:
    """Tiempo total en que a y b coinciden."""
    return duracion(a & b)


def iou(a: P.Interval, b: P.Interval) -> float:
    """Interseccion sobre union. 1.0 = identicos, 0.0 = disjuntos."""
    u = duracion(a | b)
    return solape(a, b) / u if u > 0 else 0.0
