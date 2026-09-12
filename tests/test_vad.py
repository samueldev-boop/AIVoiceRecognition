"""VAD por canal.

La maquina de estados se prueba con arrays de decisiones hechos a mano, que es
determinista y no depende de como reaccione webrtcvad a una señal sintetica. Ademas hay
dos anclas medidas: el silencio digital nunca da habla y el ruido de banda ancha siempre.
"""

import numpy as np
import pytest

from app import config, vad

SR = config.SAMPLE_RATE


def decisiones(patron: str) -> np.ndarray:
    """'..###..' -> array de bool, cada caracter una trama de 20 ms."""
    return np.array([c == "#" for c in patron], dtype=bool)


def test_segmentar_une_silencios_cortos():
    # 20 tramas de habla, 5 de silencio (100 ms < 300 ms), 20 de habla -> un solo segmento
    d = decisiones("#" * 20 + "." * 5 + "#" * 20)
    segs = vad._segmentar(d, duracion_s=10.0, min_silencio_s=0.30, padding_s=0.0)
    assert len(segs) == 1
    assert segs[0] == (0.0, 0.90)


def test_segmentar_parte_en_silencios_largos():
    # 30 tramas de silencio = 600 ms > 300 ms -> dos segmentos
    d = decisiones("#" * 20 + "." * 30 + "#" * 20)
    segs = vad._segmentar(d, duracion_s=10.0, min_silencio_s=0.30, padding_s=0.0)
    assert len(segs) == 2


def test_segmentar_descarta_lo_mas_corto_que_min_habla():
    d = decisiones("." * 10 + "#" * 5 + "." * 40)  # 100 ms de habla, menos que 200 ms
    assert vad._segmentar(d, duracion_s=5.0, min_habla_s=0.20, padding_s=0.0) == []


def test_segmentar_aplica_padding_sin_salirse_del_audio():
    d = decisiones("#" * 20)
    segs = vad._segmentar(d, duracion_s=0.40, padding_s=0.06)
    assert segs == [(0.0, 0.40)]  # recortado al principio y al final del clip


def test_segmentar_devuelve_tiempos_en_la_rejilla_de_20ms():
    d = decisiones("." * 7 + "#" * 25 + "." * 30)
    for inicio, fin in vad._segmentar(d, duracion_s=5.0, padding_s=0.06):
        for t in (inicio, fin):
            assert abs(round(t / vad.REJILLA_S) * vad.REJILLA_S - t) < 1e-9


def test_silencio_digital_no_produce_turnos():
    x = np.zeros((SR * 5, 2), dtype=np.int16)
    assert vad.turnos(x, SR) == []


def test_ruido_de_banda_ancha_produce_turnos_en_el_canal_correcto():
    rng = np.random.default_rng(0)
    x = np.zeros((SR * 4, 2), dtype=np.int16)
    # solo el canal 1 lleva señal: los turnos deben salir todos de ese canal
    x[SR:SR * 3, 1] = np.clip(rng.normal(0, 3000, SR * 2), -32768, 32767).astype(np.int16)
    turnos = vad.turnos(x, SR)
    assert turnos, "el ruido de banda ancha deberia detectarse como habla"
    assert {t["channel"] for t in turnos} == {1}
    assert turnos[0]["start"] >= 0.0
    assert turnos[-1]["end"] <= 4.0


def test_rechaza_otra_tasa_de_muestreo():
    x = np.zeros((16000, 2), dtype=np.int16)
    with pytest.raises(ValueError, match="8000 Hz"):
        vad.turnos(x, 16000)


def test_rechaza_audio_que_no_sea_bidimensional():
    with pytest.raises(ValueError, match="canales"):
        vad.turnos(np.zeros(SR, dtype=np.int16), SR)


def test_instancia_de_vad_por_canal():
    """webrtcvad es stateful: reutilizar la instancia contamina el segundo canal."""
    rng = np.random.default_rng(1)
    señal = np.clip(rng.normal(0, 3000, SR * 2), -32768, 32767).astype(np.int16)
    a = vad._decisiones(señal, 3)
    b = vad._decisiones(señal, 3)
    assert np.array_equal(a, b), "dos llamadas con la misma señal deben dar lo mismo"
