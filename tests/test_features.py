"""Extractor unificado de features.

Lo que se fija aqui es el contrato: mismas claves en todos los presupuestos, nada de NaN
ni de infinitos, y el recorte por presupuesto de verdad recorta.
"""

import numpy as np
import pytest

from app import config, features

SR = config.SAMPLE_RATE


def llamada(segundos: float = 60.0) -> tuple[np.ndarray, list[dict]]:
    """Llamada sintetica: el agente saluda, el llamante contesta, se alternan."""
    rng = np.random.default_rng(0)
    n = int(SR * segundos)
    x = np.zeros((n, 2), dtype=np.int16)
    turnos = []
    t = 0.5
    canal = 1
    while t + 4.0 < segundos:
        a, b = int(t * SR), int((t + 3.0) * SR)
        x[a:b, canal] = np.clip(rng.normal(0, 3000, b - a), -32768, 32767)
        turnos.append({"channel": canal, "start": round(t, 2), "end": round(t + 3.0, 2)})
        t += 4.0
        canal = 1 - canal
    # un poco de ruido de fondo en el canal del llamante, para que el silencio no sea cero
    x[:, 0] += np.clip(rng.normal(0, 40, n), -32768, 32767).astype(np.int16)
    return x, turnos


def test_mismas_claves_en_todos_los_presupuestos():
    x, turnos = llamada()
    vectores = {p: features.extraer(x, SR, turnos, p) for p in features.Presupuesto}
    claves = [set(v) for v in vectores.values()]
    assert claves[0] == claves[1] == claves[2], "el vector debe ser estable entre presupuestos"
    assert len(claves[0]) > 100


def test_sin_nan_ni_infinitos():
    x, turnos = llamada()
    for p in features.Presupuesto:
        for k, v in features.extraer(x, SR, turnos, p).items():
            assert np.isfinite(v), f"{k} no es finito con presupuesto {p}"


def test_el_presupuesto_recorta_de_verdad():
    x, turnos = llamada(60.0)
    duraciones = {p: features.extraer(x, SR, turnos, p)["dur_usada_s"]
                  for p in features.Presupuesto}
    assert duraciones["first_turn"] < duraciones["20s"] <= 20.0
    assert duraciones["full"] == pytest.approx(60.0)


def test_first_turn_no_mira_mas_alla_de_la_primera_intervencion():
    x, turnos = llamada(60.0)
    primer_fin = min(t["end"] for t in turnos if t["channel"] == 0)
    assert features.extraer(x, SR, turnos, "first_turn")["dur_usada_s"] == pytest.approx(primer_fin)


def test_first_turn_devuelve_algo_util_y_no_todo_ceros():
    x, turnos = llamada()
    f = features.extraer(x, SR, turnos, "first_turn")
    no_cero = [k for k, v in f.items() if v != 0.0]
    assert len(no_cero) > 60, f"solo {len(no_cero)} features distintas de cero"
    for k in ("rms_speech_db", "cen_mean", "onset0", "dur_usada_s"):
        assert f[k] != 0.0


def test_cada_feature_tiene_grupo_conocido():
    x, turnos = llamada()
    validos = {"ganancia", "codec_bw", "silencio", "prosodia", "conducta", "razon_canal"}
    for k in features.extraer(x, SR, turnos, "full"):
        assert features.grupo_de(k) in validos, k


def test_las_razones_comparan_los_dos_canales():
    x, turnos = llamada()
    f = features.extraer(x, SR, turnos, "full")
    razones = {k: v for k, v in f.items() if k.startswith("razon_")}
    assert len(razones) >= 12
    assert any(v != 0.0 for v in razones.values())


def test_rechaza_formatos_que_no_son_los_del_reto():
    x, turnos = llamada(10.0)
    with pytest.raises(ValueError, match="8000 Hz"):
        features.extraer(x, 16000, turnos)
    with pytest.raises(ValueError, match=r"\(n, 2\)"):
        features.extraer(x[:, :1], SR, turnos)
    with pytest.raises(ValueError, match="presupuesto desconocido"):
        features.extraer(x, SR, turnos, "medio_minuto")


def test_claves_es_estable_y_ordenado():
    k1 = features.claves("full")
    k2 = features.claves("full")
    assert k1 == k2 == sorted(k1)
