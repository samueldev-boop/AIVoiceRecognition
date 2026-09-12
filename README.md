# AIVoiceRecognition

Detector de llamantes sintéticos para atención telefónica bancaria. Dada una llamada
grabada, el sistema decide si quien llama es una persona o un agente autónomo
(reconocimiento de voz + modelo de lenguaje + voz sintética).

**Altur Challenge · HackMTY 2026**

---

## Contexto

Un agente de atención al cliente automatizado recibe llamadas en español mexicano. En unas
llama una persona real; en otras, un sistema autónomo marcando el mismo número. Ambos
recorren el mismo guion: el agente pide repetir datos, pregunta por productos que no
existen, interrumpe y se queda callado a propósito.

La tarea es clasificar al llamante. El jurado evalúa sobre llamadas de voces y hablantes que
no aparecen en ningún split de entrenamiento, así que **generalizar importa más que acertar
en validación**.

### Forma de los datos

| | |
| --- | --- |
| Audio | WAV estéreo, 8 kHz, 16-bit PCM. **Canal 0 = llamante** (el que se clasifica), **canal 1 = agente** |
| Volumen | 353 llamadas etiquetadas, ~14.5 h, duración media 148 s |
| Segmentación | Los tiempos de habla por canal vienen derivados automáticamente; son un punto de partida, no verdad absoluta |
| Splits | `train` y `val` disjuntos por hablante |

El dataset **no está en este repositorio** y no se redistribuye: se obtiene de la
distribución oficial del reto y se descomprime en la raíz para que el audio quede en
`audio/`. `manifest.csv`, `turns/` y las salidas de `analysis/` están en `.gitignore`.

### Contrato del endpoint

```
POST /detect
  <- WAV estéreo 8 kHz en base64 (ch0 = llamante, ch1 = agente)
  -> {"is_synthetic": true, "confidence": 0.87}
```

`is_synthetic` es obligatorio. `confidence` es opcional y puntúa la calibración.

---

## Stack

| Capa | Elección | Por qué |
| --- | --- | --- |
| Lenguaje | **Python 3.14** | El modelo, el ASR y todo el procesado de señal son Python. Un segundo runtime duplicaría la superficie de despliegue sin aportar nada a la evaluación. Verificado que todas las dependencias tienen wheel en 3.14, así que el contenedor usa la misma versión que el entorno local |
| API | **FastAPI + uvicorn** | Async, validación con pydantic y OpenAPI gratis. Sirve también el frontend estático: un solo despliegue, sin CORS |
| Frontend | **HTML + JS + Tailwind por CDN** | Sin Node y sin build. La evaluación es sobre `/detect`, no sobre la interfaz |
| Audio / DSP | `numpy`, `scipy`, `soundfile`, `praat-parselmouth` | Praat es el patrón oro para prosodia; el resto cubre espectro y niveles |
| VAD | **Silero VAD vía `onnxruntime`** | Tiene modo 8 kHz nativo: no hay que remuestrear. Milisegundos por minuto de audio |
| ASR | **faster-whisper (CTranslate2, int8)** | Devuelve `avg_logprob` y probabilidad por palabra, que es la señal más fuerte medida hasta ahora. No arrastra torch |
| Modelo | **scikit-learn** (regresión logística por capas + fusión calibrada) | Con 353 llamadas el cuello de botella no es la capacidad del modelo, es la robustez de las features |
| Despliegue | **Vultr**, 1 VM 4 vCPU / 8 GB, Docker Compose + Caddy | Sin GPU: el servicio es libre de torch y cabe en el presupuesto de latencia |

**El servicio no incluye torch a propósito.** La imagen construida pesa **1.02 GB** y arranca
hasta responder `/health` en **2.5 s** (ambos medidos); meter torch la multiplicaría sin que el
servicio use nada de él. Cualquier modelo que lo requiera se entrena aparte (Colab) y se
exporta a ONNX, o se queda fuera.

La imagen no instala ningún paquete de `apt`: las wheels traen sus librerías nativas
(`libgomp` dentro de `ctranslate2`, `libsndfile` dentro de `soundfile`).

No se usa base de datos. Si hace falta auditar predicciones, es una línea JSONL en disco.

---

## Arquitectura de decisión

El presupuesto de respuesta es de 30 s, pero el objetivo es decidir con la **primera
intervención del llamante** (~1–3 s de habla, normalmente dentro de los primeros 10 s de
llamada). Con la primera *palabra* literal no alcanza: no se puede estimar piso de ruido ni
espectro con 0.3 s.

| Etapa | Qué usa | Coste | Salida |
| --- | --- | --- | --- |
| 0 | Formato, 2 canales, ≥1 s de habla en ch0 | <10 ms | rechaza o degrada |
| 1 | 1ª intervención + primeros 20 s: canal, prosodia, latencia de entrada | ~0.3 s | si la probabilidad calibrada sale de [0.15, 0.85], **responde** |
| 2 | ASR sobre ch0 y ch1 recortados | 5–10 s | confianza por canal, razón entre canales, artefactos de lectura |
| 3 | Conducta con la llamada completa | ~0.5 s | fusión final |

Watchdog de 25 s: si una etapa se agota, se responde con lo que ya votó. `confidence` refleja
**qué etapas alcanzaron a votar**; una sola capa nunca devuelve más de 0.9.

### Principio de diseño

Ninguna capa aislada sobrevive a un ataque dirigido, y está medido: un modelo que sólo lee el
canal deja escapar el 58.8 % de los bots si se les añade ruido de sala, y uno que sólo lee los
tiempos deja escapar el 38.2 % si se reescriben las latencias. El conjunto aguanta porque los
dos ataques son incompatibles: limpiar el audio no arregla el ritmo, y arreglar el ritmo no
añade una habitación.

Por eso el sistema puntúa por capas y exige acuerdo entre ellas, en vez de promediar un único
score. Y por eso acusar a una persona real de ser un bot se trata como el error caro: hay
reglas que sólo pueden **bajar** la sospecha, nunca subirla.

---

## Estructura

```
app/         servicio
  main.py      FastAPI: /health, /detect y los estáticos
  audio.py     decodifica base64 y valida el clip (etapa 0)
  vad.py       actividad de voz por canal        -> #3
  features.py  extractor único con presupuesto   -> #4
  model.py     carga del artefacto y scoring     -> #5
  schemas.py   contrato de entrada y salida
  config.py    variables de entorno
model/       artefactos entrenados
scripts/     entrenamiento y evaluación (no se importan desde app/)
static/      frontend
tests/       criterio de aceptación automatizado
deploy/      configuración de la instancia (cloud-init) y guía de despliegue
analysis/    exploración: sondas de features y banco de estrés
docs/        informe de exploración
```

`vad.py`, `features.py` y `model.py` son la estructura con la firma ya fijada; cada uno
lanza `NotImplementedError` apuntando a su issue. Mientras no haya modelo entrenado,
`/detect` valida el clip y responde 503 con el motivo en lugar de adivinar.

`analysis/` es exploratorio y se conserva como registro de lo medido: `turns_probe` y
`audio_probe` extraen features y miden su poder discriminativo, `baseline` y `ablation`
separan señal robusta de atajos, `stress_test` ataca el modelo con audio y tiempos
modificados, y `transcribe` / `asr_confidence` cubren la capa de texto.

## Puesta en marcha

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
uvicorn app.main:app --reload            # http://localhost:8000
```

Comprobar que está bien: `pytest -q` y `ruff check app tests`.
Para trabajar con el dataset, descomprimir el zip oficial en la raíz para que el audio
quede en `audio/`.

Con Docker, igual que en la instancia:

```bash
cp .env.example .env
docker compose up -d --build
curl -s localhost/health
```

Configuración de la instancia y despliegue: **[deploy/README.md](deploy/README.md)**.

## Contribuir

Ramas, prefijos de commit, flujo de PR y reglas sobre configuración y secretos:
**[CONTRIBUTING.md](CONTRIBUTING.md)**. Instala los hooks antes del primer commit:

```bash
git config core.hooksPath .githooks
```

## Hoja de ruta

El trabajo está organizado en milestones; cada issue lleva tareas concretas y criterio de
aceptación numérico.

- **M1 · Base servible** — endpoint desplegado y medible antes de añadir capas
- **M2 · Precisión y robustez** — capa de texto a escala, augmentación contra atajos, selección de clasificador
- **M3 · Demo** — frontend mínimo
- **M4 · Extras condicionales** — sólo si aportan una ganancia medida
