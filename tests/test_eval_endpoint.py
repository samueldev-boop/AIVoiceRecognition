"""Script de evaluacion contra el endpoint (#8).

Las cuentas se fijan con respuestas inventadas, donde se sabe a mano cuanto tiene que salir.
El comando se prueba de verdad, como el de segmentar: el servicio real escuchando en un
puerto, el script en un subproceso y un dataset de juguete en disco, porque lo que se
entrega es el comando y lo que imprime.
"""

import base64
import csv
import io
import json
import socket
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import uvicorn

from scripts.eval_endpoint import resumen
from tests.test_cascade import wav_base64

RAIZ = Path(__file__).resolve().parent.parent
SCRIPT = RAIZ / "scripts" / "eval_endpoint.py"


def respuesta(label, p, confidence=None, stage="first_turn", latencia=0.3):
    """Fila como la que guarda el script para un 200 de la cascada."""
    return {"anon_id": "x", "label": label, "status": 200, "latencia_s": latencia,
            "is_synthetic": p >= 0.5,
            "confidence": max(p, 1 - p) if confidence is None else confidence,
            "probability_synthetic": p, "stage": stage, "ms": 900 * latencia, "error": "",
            "respuesta": None}


# Tres humanos y tres bots, con el coste de cada umbral calculable a mano.
FILAS = [
    respuesta("human", 0.05),
    respuesta("human", 0.60),                                   # FP hasta 0.5
    respuesta("human", 0.20),
    respuesta("synthetic", 0.95, confidence=0.65),              # acierto con confianza topada
    respuesta("synthetic", 0.40, stage="full", latencia=2.0),   # FN desde 0.5
    respuesta("synthetic", 0.80, stage="20s", latencia=0.9),    # FN solo a 0.9
]


def test_decision_y_coste_de_cada_umbral():
    r = resumen(FILAS)
    assert (r["fp"], r["fn"], r["negativos"], r["positivos"]) == (1, 1, 3, 3)
    assert r["acierto"] == pytest.approx(4 / 6)
    assert r["barrido_sobre"] == "probability_synthetic"
    coste = {b["umbral"]: (b["fp"], b["fn"]) for b in r["barrido"]}
    assert coste == {0.3: (1, 0), 0.5: (1, 1), 0.7: (0, 1), 0.9: (0, 2)}


def test_confidence_se_lee_como_confianza_en_la_clase_elegida():
    """Un humano con confidence 0.8 es P(sintetico) 0.2. El tope de confianza se paga en Brier."""
    r = resumen(FILAS)
    # 0.05² + 0.6² + 0.2² + 0.35² + 0.6² + 0.2²: el bot topado a 0.65 cuenta 0.35², no 0.05²
    assert r["calibracion"]["contrato"]["brier"] == pytest.approx(0.925 / 6)
    assert r["calibracion"]["probability_synthetic"]["brier"] == pytest.approx(0.805 / 6)

    # Un endpoint que solo cumple el contrato se evalua igual, con su puntuacion.
    solo_contrato = resumen([{**f, "probability_synthetic": None} for f in FILAS])
    assert list(solo_contrato["calibracion"]) == ["contrato"]
    assert solo_contrato["barrido_sobre"] == "contrato"


def test_las_fallidas_se_cuentan_aparte_y_no_entran_en_las_metricas():
    vacia = dict.fromkeys(("is_synthetic", "confidence", "probability_synthetic", "stage", "ms"))
    rechazada = {**FILAS[0], **vacia, "status": 503, "error": "HTTP 503: sin modelo"}
    agotada = {**FILAS[0], **vacia, "status": 0, "error": "sin respuesta: timed out",
               "latencia_s": 60.0}
    r = resumen([*FILAS, rechazada, agotada])

    assert (r["llamadas"], r["respondidas"]) == (8, 6)
    assert r["fallidas"] == {"HTTP 503": 1, "sin respuesta": 1}
    assert r["fuera_de_plazo"] == 1
    assert (r["fp"], r["fn"]) == (1, 1)
    etapas = [(e["etapa"], e["n"], e["fp"], e["fn"]) for e in r["etapas"]]
    assert etapas == [("first_turn", 4, 1, 0), ("20s", 1, 0, 0), ("full", 1, 0, 1)]


def correr(*argumentos):
    return subprocess.run([sys.executable, str(SCRIPT), *map(str, argumentos)],
                          capture_output=True, text=True, cwd=RAIZ, timeout=120)


def dataset(raiz: Path, llamadas: list[tuple[str, str, str, bytes]]) -> Path:
    """manifest.csv y audio/ de juguete a partir de (anon_id, label, split, wav)."""
    (raiz / "audio").mkdir()
    with (raiz / "manifest.csv").open("w", newline="") as fh:
        escritor = csv.writer(fh)
        escritor.writerow(["anon_id", "label", "split", "duration_s"])
        for anon_id, label, split, wav in llamadas:
            escritor.writerow([anon_id, label, split, 12])
            (raiz / "audio" / f"{anon_id}.wav").write_bytes(wav)
    return raiz


@pytest.fixture(scope="module")
def servicio():
    """El servicio real en un puerto libre, servido desde un hilo del propio pytest."""
    from app.main import app

    servidor = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=0, log_config=None))
    hilo = threading.Thread(target=servidor.run, daemon=True)
    hilo.start()
    limite = time.monotonic() + 30
    while not servidor.started:
        assert hilo.is_alive() and time.monotonic() < limite, "el servicio no arranco"
        time.sleep(0.05)
    url = f"http://127.0.0.1:{servidor.servers[0].sockets[0].getsockname()[1]}"
    with urllib.request.urlopen(f"{url}/health", timeout=10) as r:
        cargado = json.load(r)["model_loaded"]
    try:
        if not cargado:
            pytest.skip("sin artefacto entrenado en model/model.joblib")
        yield url
    finally:
        servidor.should_exit = True
        hilo.join(timeout=10)


def test_el_comando_evalua_el_split_y_cabe_en_una_pantalla(servicio, tmp_path):
    wav = base64.b64decode(wav_base64())
    raiz = dataset(tmp_path, [("call_h", "human", "val", wav),
                              ("call_s", "synthetic", "val", wav),
                              ("call_t", "synthetic", "train", wav)])
    salida = tmp_path / "eval.csv"

    r = correr("--split", "val", "--url", servicio, "--raiz", raiz, "--salida", salida)

    assert r.returncode == 0, r.stdout + r.stderr
    lineas = r.stdout.splitlines()
    # La ultima linea es la ruta del CSV, que la elige quien llama: el informe es lo que cabe.
    assert lineas[-1].endswith("eval.csv")
    assert len(lineas) <= 30 and max(map(len, lineas[:-1])) <= 80, r.stdout
    for dato in ("acierto", "FPR", "FNR", "AUC", "Brier", "ECE", "p50", "p95",
                 "Etapa que decidio", "Umbral", "0.3", "0.9"):
        assert dato in r.stdout, f"falta {dato} en el informe"

    guardadas = list(csv.DictReader(salida.open()))
    assert [g["anon_id"] for g in guardadas] == ["call_h", "call_s"], "solo el split pedido"
    for g in guardadas:
        assert g["status"] == "200" and float(g["latencia_s"]) > 0
        assert isinstance(json.loads(g["respuesta"])["is_synthetic"], bool)


def test_una_peticion_rechazada_se_informa_y_el_comando_sale_con_error(servicio, tmp_path):
    mono = (1000 * np.sin(np.arange(8000 * 10) / 20.0)).astype(np.int16)
    buf = io.BytesIO()
    sf.write(buf, np.stack([mono, mono], axis=1), 8000, subtype="PCM_16", format="WAV")
    raiz = dataset(tmp_path, [("call_ok", "human", "val", base64.b64decode(wav_base64())),
                              ("call_mono", "human", "val", buf.getvalue())])
    salida = tmp_path / "eval.csv"

    r = correr("--url", servicio, "--raiz", raiz, "--salida", salida)

    assert r.returncode == 1
    assert "FALLIDAS 1: HTTP 422 x1" in r.stdout
    rechazada = next(g for g in csv.DictReader(salida.open()) if g["anon_id"] == "call_mono")
    assert rechazada["status"] == "422" and "identicos" in rechazada["error"]


def test_sin_servicio_falla_rapido_y_dice_por_que(tmp_path):
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        puerto = s.getsockname()[1]
    r = correr("--url", f"http://127.0.0.1:{puerto}", "--raiz", tmp_path)
    assert r.returncode == 1
    assert "/health no responde" in r.stderr
