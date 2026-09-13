"""La interfaz la sirve el mismo FastAPI que el endpoint: no hay build ni segundo despliegue.

Lo que se fija aqui es que los tres archivos se sirven con el tipo correcto y que la
respuesta de /detect/details trae lo que la pagina necesita para dibujar.
"""

import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import app

ESTATICOS = Path(__file__).resolve().parent.parent / "static"


@pytest.mark.parametrize(
    ("ruta", "tipo"),
    [("/", "text/html"), ("/app.js", "javascript"), ("/tailwind.js", "javascript")],
)
def test_los_estaticos_se_sirven(ruta, tipo):
    with TestClient(app) as c:
        r = c.get(ruta)
        assert r.status_code == 200, ruta
        assert tipo in r.headers["content-type"], r.headers["content-type"]
        assert len(r.content) > 200


def test_la_interfaz_se_revalida_en_cada_carga():
    """Tras un despliegue el navegador no puede quedarse con el app.js anterior de su cache."""
    with TestClient(app) as c:
        for ruta in ("/", "/app.js", "/tailwind.js"):
            r = c.get(ruta)
            assert r.headers["cache-control"] == "no-cache", ruta
            # no-cache no obliga a descargar otra vez: si no cambio, basta un 304 sin cuerpo.
            revalidada = c.get(ruta, headers={"if-none-match": r.headers["etag"]})
            assert revalidada.status_code == 304, ruta
            assert revalidada.headers["cache-control"] == "no-cache", ruta


def test_la_pagina_referencia_sus_archivos():
    html = (ESTATICOS / "index.html").read_text()
    for recurso in ("tailwind.js", "app.js"):
        assert recurso in html, f"index.html no carga {recurso}"
    # Rutas relativas: la pagina tiene que funcionar igual detras de Caddy.
    assert 'src="/' not in html and 'href="/' not in html, "no usar rutas absolutas"


def test_no_hay_dependencias_externas():
    """Tailwind va vendorizado: la demo funciona aunque la red del sitio bloquee los CDN."""
    for nombre in ("app.js", "index.html"):
        fuente = (ESTATICOS / nombre).read_text()
        assert not re.search(r"https?://(?!127\.0\.0\.1)", fuente), f"url externa en {nombre}"


def test_bordes_rectos_y_sin_degradados():
    """Decision de diseno del equipo, aqui fijada para que no se cuele de vuelta."""
    for nombre in ("app.js", "index.html"):
        fuente = (ESTATICOS / nombre).read_text()
        assert "rounded-" not in fuente, f"{nombre} usa esquinas redondeadas"
        assert "gradient" not in fuente, f"{nombre} usa un degradado"


def test_las_capas_que_pinta_existen_en_el_modelo():
    """Si alguien renombra una capa, el test lo dice en lugar de dejar barras vacias."""
    from app.interventions import LAYER_NAMES

    js = (ESTATICOS / "app.js").read_text()
    declaradas = re.search(r"const CAPAS = \[(.*?)\]", js, re.S).group(1)
    pintadas = set(re.findall(r'"([a-z_]+)"', declaradas))
    esperadas = set(LAYER_NAMES)
    assert pintadas == esperadas, f"la interfaz pinta {pintadas}, el modelo da {esperadas}"


def test_detect_devuelve_lo_que_la_pagina_dibuja():
    from tests.test_cascade import wav_base64

    with TestClient(app) as c:
        if not c.get("/health").json()["model_loaded"]:
            pytest.skip("sin artefacto entrenado")
        js = (ESTATICOS / "app.js").read_text()
        assert 'fetch("detect/details"' in js
        cuerpo = c.post("/detect/details", json={"audio": wav_base64()}).json()
        for campo in ("is_synthetic", "confidence", "stage", "ms", "layer_scores",
                      "budget_scores", "audio_used_s", "turns"):
            assert campo in cuerpo, f"falta {campo} en la respuesta"
        assert isinstance(cuerpo["turns"], list)


def test_todos_los_ids_que_busca_el_js_existen_en_el_html():
    """El fallo tipico de una interfaz: $("algo") devuelve null y la pagina se queda muda."""
    js = (ESTATICOS / "app.js").read_text()
    html = (ESTATICOS / "index.html").read_text()
    buscados = set(re.findall(r'\$\("([^"]+)"\)', js))
    declarados = set(re.findall(r'id="([^"]+)"', html))
    assert buscados, "el test no encontro ningun $(...) que comprobar"
    assert buscados <= declarados, f"el js busca ids que no existen: {buscados - declarados}"
