# AIVoiceRecognition

Detector de llamantes sintéticos para atención telefónica bancaria. Dada una llamada
grabada, el sistema decide si quien llama es una persona o un agente autónomo
(reconocimiento de voz + modelo de lenguaje + voz sintética).

**Altur Challenge · HackMTY 2026**

La guía de arquitectura, algoritmo, features, endpoint y despliegue está en
[docs/documentacion-repositorio.md](docs/documentacion-repositorio.md). Es el
documento preparado también para la Wiki del proyecto.

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
| Intervalos | `portion` | Álgebra de turnos (unión, solape, IoU) en 27 KB. `pyannote.core` haría lo mismo arrastrando pandas |
| VAD | **webrtcvad** | Nativo a 8 kHz con tramas de 20 ms, que es exactamente la rejilla en la que caen los tiempos de `turns/*.json`. Medido contra Silero: IoU 0.84/0.94 frente a 0.62/0.88, y mejor rendimiento aguas abajo. Procesa el dataset a 5300× tiempo real |
| ASR | **faster-whisper (CTranslate2, int8)** | Devuelve `avg_logprob` y probabilidad por palabra, que es la señal más fuerte medida hasta ahora. No arrastra torch |
| Modelo | **scikit-learn** (regresión logística por capas + fusión calibrada) | Con 353 llamadas el cuello de botella no es la capacidad del modelo, es la robustez de las features |
| Despliegue | **Vultr**, 1 VM 4 vCPU / 8 GB, Docker Compose + Caddy | Sin GPU: el servicio es libre de torch y cabe en el presupuesto de latencia |

**El servicio no incluye torch a propósito.** La imagen construida pesa **1.04 GB** y arranca
hasta responder `/health` en **2.5 s** (ambos medidos); meter torch la multiplicaría sin que el
servicio use nada de él. Cualquier modelo que lo requiera se entrena aparte (Colab) y se
exporta a ONNX, o se queda fuera.

El build es en dos etapas. La imagen final no instala ningún paquete de `apt` —las wheels
traen sus librerías nativas (`libgomp` dentro de `ctranslate2`, `libsndfile` dentro de
`soundfile`)— y el `gcc` que necesita compilar `webrtcvad` en Python 3.14 se queda en el
builder.

No se usa base de datos. Si hace falta auditar predicciones, es una línea JSONL en disco.

---

## Arquitectura de decisión

El presupuesto de respuesta es de 30 s, pero el objetivo es decidir con la **primera
intervención del llamante** (~1–3 s de habla, normalmente dentro de los primeros 10 s de
llamada). Con la primera *palabra* literal no alcanza: no se puede estimar piso de ruido ni
espectro con 0.3 s.

| Etapa | Qué usa | Audio | CV por hablante | Peor caso de estrés |
| --- | --- | --- | --- | --- |
| 0 | Formato, 2 canales, ≥1 s de habla en ch0 | — | rechaza o degrada | — |
| 1 | 1ª intervención del llamante | 10.5 s de media | **0.9768** | **0.9483** |
| 2 | Primeros 20 s | 20 s | 0.9739 | 0.9287 |
| 3 | Llamada completa | 148 s de media | 0.9682 | 0.8582 |

Se sale en la primera etapa cuya probabilidad calibrada queda fuera de la banda
**[0.10, 0.90]**; si se escala, se fusiona por media de logits y, cuando dos presupuestos se
contradicen, la confianza se topa en 0.65 en lugar de promediarse sin más.

**El orden no es casual: la primera intervención es el presupuesto más fuerte y el más
robusto al ataque.** Agrupando la validación por hablante, mirar más audio empeora la
generalización, porque el modelo se apoya en rasgos de la persona y de su línea que no
transfieren a hablantes nuevos. El juicio final usa voces que no están en ningún split.

Lo que la cascada compra es **latencia, no precisión**: sobre las predicciones fuera de
muestra de train, la cascada comete los mismos 24 errores que fusionar los tres presupuestos
siempre, pero contesta el 79 % de las llamadas con una etapa.

Medido contra el endpoint real, las 71 llamadas de val:

| | |
| --- | --- |
| Latencia | p50 **0.27 s** · p95 **2.47 s** · máx 3.40 s |
| Por encima de 12 s / 30 s | 0 / 0 |
| Etapa que decidió | 59 en la 1ª (0.27 s) · 4 en los 20 s · 8 con la llamada completa (2.45 s) |
| Resultado | AUC 0.9921 · acierto 97.2 % · 1 FP · 1 FN · Brier 0.0237 |

Y el subconjunto que sobrevive a un cambio de equipo —conducta, prosodia y razones entre
canales— es el que domina en la etapa 1: conducta 0.9472 y razones 0.9270, frente al silencio
de la sala, que solo manda con la llamada completa (0.9862). El atajo, visto de frente.

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
  vad.py       actividad de voz por canal con webrtcvad
  intervalos.py  unión, solape e IoU de turnos
  features.py  extractor único: 139 features en 6 grupos, con presupuesto de audio
  model.py     carga del artefacto y scoring     -> #5
  schemas.py   contrato de entrada y salida
  config.py    variables de entorno
model/       artefactos entrenados
scripts/     entrenamiento y evaluación (no se importan desde app/)
static/      interfaz: sube un WAV y ve el veredicto, los scores por capa
             y los turnos del VAD sobre la forma de onda
tests/       criterio de aceptación automatizado
deploy/      configuración de la instancia (cloud-init) y guía de despliegue
analysis/    exploración: sondas de features y banco de estrés
docs/        informe de exploración
```

`model.py` carga el artefacto de #5 y permite puntuar audio con `puntuar_audio()` o muestras
extraídas por `app/interventions.py`. La conexión del modelo con la cascada HTTP de
`/detect` corresponde a #6; el endpoint informa 503 mientras esa cascada no esté integrada.

`app/features.py` y `app/interventions.py` son la ruta compartida de extracción para
entrenamiento e inferencia. `analysis/baseline.py` y `analysis/stress_test.py` invocan el
entrenamiento sklearn; `analysis/ablation.py` consulta los resultados fuera de muestra.

Las sondas `turns_probe` / `audio_probe` conservan la exploración histórica;
`transcribe` / `asr_confidence` cubren la capa de texto.

## Entrenamiento y reporte del issue #5

El modelo entrena cuatro regresiones por intervención del llamante. Agrega sus scores por
llamada y añade dos capas de conducta y razones entre canales; una regresión de fusión y
calibración Platt producen la probabilidad final. Hay modelos para primera intervención,
20 segundos y llamada completa. La evaluación usa cinco folds agrupados por llamada, con
fusión y calibración fuera de muestra dentro de cada fold.

```bash
pip install -r requirements-analysis.txt
OPENBLAS_NUM_THREADS=1 python -m scripts.train_issue5
python -m scripts.report_issue5
python -m analysis.ablation
```

Para reforzar el modelo con las conversaciones autorizadas del issue #15, se
mantiene la seleccion y la validacion oficial libres de ese material y solo se
anade al ajuste final:

```bash
OPENBLAS_NUM_THREADS=1 python -m scripts.train_issue5 \
  --issue15-dir generated_issue15_jeff
```

Las variantes de una misma llamada base se agrupan juntas; no se interpretan
como hablantes independientes. El manifiesto y el artefacto registran el hash,
las etiquetas y las familias utilizadas.

Se generan `model/model.joblib` y `analysis/issue5/`, con el informe HTML/PDF/Markdown,
mapas de calor Pearson y Spearman, distribuciones, grafo de correlaciones, nulos, atípicos,
ROC, calibración, matrices de confusión a 0.5/0.7 y métricas bajo estrés. El modelo se elige
con `train` y estrés; la configuración se congela antes de informar `val`.

Los artefactos derivados quedan locales y están ignorados por git. El dataset original
se conserva intacto. Detalles del formato, comandos y verificación:
[docs/issue5-modelo.md](docs/issue5-modelo.md).

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

## Evaluar el endpoint

Mide el servicio entero por HTTP, como lo verá el jurado. Complementa la validación agrupada
y el banco de estrés, no los sustituye: `val` está saturado. Con el servicio levantado y el
dataset en la raíz:

```bash
python scripts/eval_endpoint.py --split val --url http://localhost:8000
```

Manda las llamadas del split a `/detect` de una en una e imprime, en una pantalla: acierto,
FPR (humanos acusados) y FNR; AUC, Brier y ECE; latencia p50/p95; qué etapa decidió cada
llamada, y el coste en FP y FN de los umbrales 0.3, 0.5, 0.7 y 0.9. La calibración se mide
dos veces: con lo que ve el jurado (`is_synthetic` + `confidence`) y con
`probability_synthetic`. Cada respuesta queda con su latencia en
`analysis/eval_<split>_<host>.csv`, ignorado por git. Contra el despliegue es el mismo
comando con su URL. Si alguna petición falla, el script sale con código 1.

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
