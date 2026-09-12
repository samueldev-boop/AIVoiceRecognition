"""Exportador JSON de turnos: el entregable de linea de comandos.

Se invoca el script de verdad, con subprocess, porque lo que se esta probando es el
comando y el JSON que escribe, no solo la funcion que hay debajo.
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

RAIZ = Path(__file__).resolve().parent.parent
SCRIPT = RAIZ / "scripts" / "segmentar.py"
SR = 8000


def wav_de_prueba(destino: Path) -> Path:
    """Estereo a 8 kHz: ruido de banda ancha en tramos distintos de cada canal."""
    rng = np.random.default_rng(0)
    x = np.zeros((SR * 12, 2), dtype=np.int16)
    x[SR * 1:SR * 4, 0] = np.clip(rng.normal(0, 3000, SR * 3), -32768, 32767)
    x[SR * 5:SR * 8, 1] = np.clip(rng.normal(0, 3000, SR * 3), -32768, 32767)
    x[SR * 9:SR * 11, 0] = np.clip(rng.normal(0, 3000, SR * 2), -32768, 32767)
    sf.write(destino, x, SR, subtype="PCM_16")
    return destino


def correr(*argumentos):
    return subprocess.run([sys.executable, str(SCRIPT), *map(str, argumentos)],
                          capture_output=True, text=True, cwd=RAIZ)


def test_escribe_json_valido_y_ordenado(tmp_path):
    wav = wav_de_prueba(tmp_path / "llamada.wav")
    salida = tmp_path / "turnos.json"
    r = correr(wav, "-o", salida)
    assert r.returncode == 0, r.stderr

    datos = json.loads(salida.read_text())
    assert "segments" in datos
    segmentos = datos["segments"]
    assert len(segmentos) >= 3, segmentos
    assert {s["channel"] for s in segmentos} == {0, 1}
    for s in segmentos:
        assert set(s) == {"channel", "start", "end"}
        assert 0.0 <= s["start"] < s["end"] <= 12.0
    assert segmentos == sorted(segmentos, key=lambda s: s["start"]), "deben ir en orden"


def test_por_salida_estandar_sin_fichero(tmp_path):
    wav = wav_de_prueba(tmp_path / "llamada.wav")
    r = correr(wav)
    assert r.returncode == 0, r.stderr
    assert json.loads(r.stdout)["segments"]


def test_lote_escribe_un_json_por_llamada(tmp_path):
    a = wav_de_prueba(tmp_path / "a.wav")
    b = wav_de_prueba(tmp_path / "b.wav")
    destino = tmp_path / "turnos"
    r = correr(a, b, "-o", destino)
    assert r.returncode == 0, r.stderr
    assert (destino / "a.json").exists() and (destino / "b.json").exists()


def test_los_parametros_del_vad_cambian_el_resultado(tmp_path):
    wav = wav_de_prueba(tmp_path / "llamada.wav")
    laxo = json.loads(correr(wav, "--min-silencio", "5.0").stdout)["segments"]
    estricto = json.loads(correr(wav, "--min-silencio", "0.1").stdout)["segments"]
    assert len(laxo) < len(estricto), "un silencio minimo alto debe fusionar turnos"


def test_rechaza_audio_que_no_cumple_el_formato(tmp_path):
    mono = tmp_path / "mono.wav"
    sf.write(mono, np.zeros(SR * 10, dtype=np.int16), SR, subtype="PCM_16")
    r = correr(mono)
    assert r.returncode == 1
    assert "canales" in r.stderr
