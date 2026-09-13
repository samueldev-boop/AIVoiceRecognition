# AIVoiceRecognition: documentación integral

## 1. El problema de Altur Challenge

Este proyecto responde al reto de **Altur Challenge · HackMTY 2026**: durante
una llamada de atención telefónica bancaria, determinar si la persona que está
del otro lado del teléfono es un humano o un agente autónomo que escucha,
razona y responde con voz sintética. La salida requerida es una decisión binaria
(`is_synthetic`) y una confianza asociada.

No basta con reconocer si una voz “suena robótica”. Los sistemas modernos de
texto a voz pueden imitar pausas, entonación, ruido de línea y calidad telefónica.
Además, tanto humanos como agentes siguen el mismo guion de atención, hablan por
un canal telefónico limitado y pueden sonar distintos según su micrófono, red y
habitación. El detector debe decidir sobre una voz que no conoció al entrenar,
sin confundir la identidad o el teléfono concreto con la clase humano/IA.

### Por qué es difícil

| Dificultad | Impacto en la detección | Respuesta del proyecto |
| --- | --- | --- |
| Voces TTS muy naturales | La prosodia aislada ya no separa siempre a una IA de una persona. | Se combinan prosodia, comportamiento conversacional y relación entre canales. |
| Audio telefónico a 8 kHz | Codecs, ruido, pérdidas y ganancia alteran las señales acústicas. | Se entrena y prueba con codecs reales, ruido, reverberación y RawBoost. |
| Hablantes desconocidos | Una métrica por llamada puede filtrar la misma persona entre folds y parecer mejor de lo que es. | La validación se agrupa por hablante/voz aproximada y por huella PCM. |
| Mismo guion para ambas clases | El contenido literal de la conversación no es una pista fiable. | El modelo actual se centra en cómo se habla y cómo se responde, no en palabras. |
| Falso positivo costoso | Acusar a una persona real de ser IA es peor que pedir evidencia adicional. | La confianza se limita cuando las capas discrepan y la abstención favorece la clase humana. |
| Límite de tiempo del reto | El endpoint debe responder antes de 30 segundos. | Una cascada intenta decidir con la primera intervención y sólo analiza más audio si hay ambigüedad. |

La pregunta operativa no es “¿quién es la persona?”, sino:

> Dado un WAV estéreo de llamada, ¿la evidencia acústica y conversacional del
> canal del llamante es más consistente con una voz sintética que con una voz
> humana?

## 2. Datos y alcance

El corpus oficial no se distribuye en este repositorio. Al colocarlo en la raíz,
debe contener `manifest.csv`, `audio/` y, si está disponible, `turns/`.
La documentación del proyecto describe 353 llamadas etiquetadas, aproximadamente
14.5 horas de audio y una duración media de 148 segundos.

| Elemento | Contrato |
| --- | --- |
| Formato | WAV PCM de 16 bits, estéreo, 8 kHz. |
| Canal 0 | Llamante: es el canal que se clasifica. |
| Canal 1 | Agente u operador: proporciona el contexto de conversación. |
| Etiquetas | `human` y `synthetic`. |
| Particiones | `train` y `val`, separadas por hablante según el reto. |
| Duración aceptada por API | Entre 5 y 600 segundos. |

El directorio `generated_issue15_jeff/` contiene material adicional autorizado:
49 conversaciones de clon sintético y una conversación humana real. Sus 45
variantes telefónicas provienen de sólo cuatro conversaciones base; por eso el
flujo las conserva como cuatro familias sintéticas, no como 49 hablantes nuevos.
Se usan sólo para el ajuste final opcional, nunca para seleccionar hiperparámetros
ni para medir `val`. Más detalles: [entrenamiento del issue #15](issue15-entrenamiento.md).

## 3. Arquitectura de decisión

```text
WAV estéreo en Base64
        |
        v
Validación de formato, canales, duración y habla del llamante
        |
        v
Remuestreo uniforme a 8 kHz (SoXR HQ) y VAD WebRTC
        |
        v
Extracción de features por intervención y por llamada
        |
        +--> ganancia | codec/ancho de banda | silencio | prosodia
        +--> conducta conversacional | razón entre canales
        |
        v
Seis capas logísticas -> fusión -> calibración Platt
        |
        v
Cascada: primera intervención -> 20 s -> llamada completa
        |
        v
JSON: is_synthetic, confidence, probability_synthetic y diagnóstico
```

El entrenamiento y el endpoint comparten el extractor de
`app/interventions.py`; esto evita que el modelo aprenda con una definición de
feature distinta a la usada en producción.

## 4. Mapa del repositorio

| Directorio o archivo | Responsabilidad |
| --- | --- |
| `app/` | Servicio FastAPI, lectura de audio, VAD, extracción de features, modelo y cascada. |
| `scripts/` | Preparación de datos, entrenamiento, augmentación, validación, benchmark y reportes. |
| `analysis/` | Experimentos reproducibles: análisis de features, ASR, estrés y ablaciones. |
| `model/model.joblib` | Artefacto servido: extractores esperados, capas, fusión y calibración. |
| `tests/` | Contratos de audio, extracción, entrenamiento, cascada, endpoint y robustez. |
| `static/` | Interfaz web para cargar un WAV y visualizar la decisión y los turnos VAD. |
| `deploy/`, `Dockerfile`, `docker-compose.yml`, `Caddyfile` | Contenedores, proxy inverso, TLS y guía de despliegue. |
| `docs/` | Informes técnicos por issue y esta guía. |

## 5. Flujo de trabajo empleado

1. **Auditar el corpus.** `scripts/issue5_data.py` valida el manifiesto, formato
   PCM, duración, canales, hashes, clipping, silencios y duplicados exactos.
2. **Localizar los turnos de habla.** Se aplica WebRTC VAD en ambos canales; los
   turnos son la unidad para analizar una intervención real del llamante.
3. **Extraer señales consistentes.** `app/features.py` y
   `app/interventions.py` producen el mismo esquema de features para `train`,
   validación y endpoint, en tres presupuestos de audio.
4. **Evitar fuga de identidad.** Se agrupa por llamada, mapa de hablante/voz si
   existe y hash exacto del PCM del llamante. `speaker_groups.py` puede derivar
   grupos aproximados mediante MFCC y clustering de Ward.
5. **Seleccionar sin usar validación final.** Se comparan configuraciones
   predefinidas con cinco folds `StratifiedGroupKFold`, usando AUC limpia y el
   peor AUC ante estrés. La selección se congela antes de abrir `val`.
6. **Entrenar por capas y calibrar.** Las capas se entrenan con `train`; la
   fusión y Platt usan predicciones fuera de muestra en folds internos.
7. **Atacar el modelo antes de promoverlo.** Se simulan codecs, ganancia, ruido,
   sala y cambios de conducta. El gate de `scripts/train_issue11.py` marca como
   bloqueado un candidato que acuse humanos a 0.5 o dependa de grupos frágiles.
   El candidato con RawBoost quedó bloqueado por 7 falsos positivos fuera de
   muestra en `humano_limpio`, y aun así se promovió a la etapa `full` por
   decisión explícita (#33): mejora el peor AUC de estrés de 0.8451 a 0.9726 y,
   en `val`, pasa de 1 a 0 humanos acusados.
8. **Servir y medir de extremo a extremo.** La API ejecuta una cascada de salida
   temprana y `scripts/eval_endpoint.py` mide precisión, calibración y latencia
   mediante HTTP, como lo haría el jurado.

## 6. Presupuestos de audio

No todas las decisiones necesitan una llamada completa. Se entrenan tres modelos
con el mismo contrato de features:

| Presupuesto | Audio que puede consultar | Motivo |
| --- | --- | --- |
| `first_turn` | Hasta el final de la primera intervención del llamante. | Es la primera decisión y reduce latencia y dependencia de la identidad. |
| `20s` | Los primeros 20 segundos de la llamada. | Añade contexto cuando el primer turno es ambiguo. |
| `full` | La llamada completa. | Último recurso para casos aún inciertos. |

La cascada se detiene cuando una probabilidad calibrada queda fuera de la banda
de ambigüedad `[0.10, 0.90]`. Si necesita más de un presupuesto, combina sus
probabilidades como media de *logits*. Si los presupuestos se contradicen, no
disimula el desacuerdo: limita la confianza a 0.65.

## 7. Features de referencia

El extractor genera más de cien señales. En vez de depender de una sola, las
organiza en seis grupos con significado físico. Esta separación permite saber
qué evidencia está sosteniendo una decisión y probar si el modelo sobrevive al
daño de una clase de señales.

| Grupo | Ejemplos | Qué intenta capturar | Robustez esperada |
| --- | --- | --- | --- |
| `ganancia` | RMS de habla y ruido, SNR, pico, *crest factor*, clipping, offset DC y fuga. | Nivel de grabación, relación señal/ruido y efectos de línea. | Frágil: cambia con teléfono, micrófono y volumen. |
| `codec_bw` | Energía por bandas, centroide, ancho de banda, roll-off, planitud, ZCR y flujo espectral. | Huellas de codec, recorte telefónico y cadena de audio. | Frágil: el codec puede simularse o cambiar. |
| `silencio` | Estadísticos del silencio, ceros, ruido de fondo y correlación cruzada. | Sala, ruido ambiental y actividad fuera de habla. | Frágil: un agente puede añadir ruido de sala. |
| `prosodia` | F0, jitter, shimmer, HNR, fracción sonora, envolvente y modulación de 2–6 y 6–12 Hz. | Entonación, estabilidad vocal, ritmo silábico y microvariaciones de la voz. | Más robusta que una firma de codec, aunque no infalible. |
| `conducta` | Número y duración de turnos, latencias de respuesta, solapes, pausas y su tendencia. | Cómo el llamante reacciona al operador e interactúa con el guion. | Robusta: es difícil modificarla sin cambiar el comportamiento del agente. |
| `razon_canal` | Razones de energía/espectro entre llamante y operador, fuga y relación intercanal. | Diferencia entre la cadena del llamante y el canal del agente. | Robusta por construcción: compara el contexto de la misma llamada. |

La ausencia de una medición también se representa de forma explícita como dato
faltante: por ejemplo, no se inventa F0 cuando un turno es demasiado corto. Los
imputadores del pipeline aprenden sus valores únicamente en el fold de
entrenamiento, nunca con ejemplos de prueba.

## 8. Algoritmos y para qué se usan

### Procesamiento de señal y segmentación

- **SoXR HQ:** remuestrea cualquier entrada a 8 kHz de manera idéntica en disco
  y en HTTP. Así la distribución de entrenamiento y producción coincide.
- **WebRTC VAD:** separa tramas de voz y no voz a 8 kHz. Permite concentrar las
  medidas de voz en el canal del llamante y calcular turnos y latencias entre
  ambos canales.
- **FFT sobre ventanas de 32 ms:** calcula energía por banda, centroide, ancho,
  roll-off, planitud y flujo espectral.
- **Praat/Parselmouth:** estima F0, jitter, shimmer y HNR para la capa prosódica.
- **Álgebra de intervalos:** une turnos, calcula silencios, solapes y tiempos de
  respuesta sin tratar cada trama como una conversación independiente.

### Clasificador por capas

Cada una de las cuatro capas por turno (`ganancia`, `codec_bw`, `silencio` y
`prosodia`) usa una **regresión logística L2**. Los turnos de una llamada se
ponderan para que una llamada larga no tenga más peso que una corta. Cada capa
se resume con tres valores fijados antes de la evaluación:

- 60 %: sigmoide de la media de logits de sus turnos;
- 20 %: probabilidad máxima;
- 20 %: fracción de turnos con probabilidad de al menos 0.7.

Las capas de `conducta` y `razon_canal` se entrenan directamente por llamada.
Una sexta regresión logística de **fusión** recibe los seis scores, no las
features originales. Para no filtrar etiquetas, sus entradas se construyen con
predicciones fuera de muestra de folds internos.

Antes de cada regresión se aplica esta secuencia:

```text
valores no finitos -> NaN -> imputación mediana + indicador de ausencia
-> winsorización por cuantiles -> StandardScaler -> regresión logística L2
```

Finalmente, `CalibratedClassifierCV` ajusta **calibración Platt** con folds
agrupados. La probabilidad final expresa `P(synthetic)`; `confidence` es otra
cantidad, una confianza conservadora en la clase elegida que baja si hay pocas
capas disponibles o falta acuerdo entre ellas.

### Validación, robustez y candidatos alternativos

- **StratifiedGroupKFold:** conserva ambas clases y evita que una voz, llamada o
  PCM repetido aparezca simultáneamente en entrenamiento y prueba.
- **MFCC + Ward:** genera grupos aproximados de hablante/voz cuando el corpus no
  trae identidad. Se elimina el coeficiente de energía para no agrupar por volumen.
- **Augmentación física:** codecs GSM, AMR-NB, G.722, A-law y μ-law; ganancia,
  clipping, filtros, ruido MUSAN y RIR reales. Las variantes se vuelven a pasar
  por VAD y extracción, pues el ataque cambia las señales observables.
- **RawBoost:** añade degradaciones de anti-spoofing (LnL + ISD) adaptadas a 8 kHz
  para evaluar y entrenar ante manipulaciones de audio adicionales.
- **LightGBM:** se comparó como candidato sobre prosodia, conducta y razones de
  canal, con el mismo protocolo agrupado y de estrés (#12). Quedó descartado:
  su AUC agrupada es 0.9833, pero cae a 0.8995 cuando se ataca el ritmo, frente
  a 0.9726 de la regresión logística. TabPFN también se descartó.
- **ASR con Faster-Whisper:** se emplea en análisis exploratorio para estudiar la
  confianza de transcripción. No es una dependencia decisoria del artefacto
  principal de features, por lo que una caída de ASR no bloquea `/detect`.

## 9. Herramientas utilizadas

| Herramienta | Uso en el proyecto |
| --- | --- |
| Python | Lenguaje de procesamiento de señal, entrenamiento y API. |
| NumPy y SciPy | Vectores, FFT, estadística y transformaciones de señal. |
| SoundFile | Lectura y validación de WAV PCM. |
| SoXR | Remuestreo de alta calidad a 8 kHz. |
| WebRTC VAD | Detección de voz y construcción de turnos. |
| Praat-Parselmouth | Medidas prosódicas y MFCC. |
| scikit-learn y joblib | Pipelines, regresiones, CV agrupada, calibración y serialización. |
| FFmpeg | Codificación/decodificación real de codecs durante augmentación. |
| audiomentations, MUSAN y OpenSLR RIR | Ruido, filtros, ganancia, clipping y reverberación de entrenamiento/estrés. |
| RawBoost | Degradaciones de audio anti-spoofing. |
| Faster-Whisper / CTranslate2 | Experimentos de ASR en CPU `int8`. |
| FastAPI y Uvicorn | API HTTP, OpenAPI y servicio web. |
| Docker Compose y Caddy | Contenedores, proxy inverso, compresión y TLS automático. |
| pytest y Ruff | Pruebas de contrato y revisión estática. |

## 10. Endpoint y dominio

### API

El servicio ofrece las siguientes rutas:

| Ruta | Método | Propósito |
| --- | --- | --- |
| `/health` | `GET` | Informa si el servicio y el artefacto del modelo están cargados. |
| `/detect` | `POST` | Clasifica una llamada WAV estéreo enviada en Base64. |
| `/docs` | `GET` | Documentación OpenAPI interactiva generada por FastAPI. |
| `/` | `GET` | Interfaz web estática para la demostración. |

Contrato mínimo de `POST /detect`. El campo del jurado es `audio_base64`; `audio`
se mantiene como alias para la interfaz web y clientes anteriores. Los campos
adicionales, como `call_id`, se ignoran:

```json
{
  "audio_base64": "<WAV-estéreo-8-kHz-codificado-en-Base64>"
}
```

Respuesta típica (los dos primeros campos son el contrato; el resto es
diagnóstico):

```json
{
  "is_synthetic": true,
  "confidence": 0.9458,
  "stage": "first_turn",
  "probability_synthetic": 0.9458,
  "budget_scores": {"first_turn": 0.9458},
  "layer_scores": {"ganancia": 0.71, "codec_bw": 0.88, "silencio": 0.93,
                   "prosodia": 0.64, "conducta": 0.97, "razon_canal": 0.81},
  "disagreement": false,
  "audio_used_s": 10.4,
  "ms": 96.1,
  "turns": [{"channel": 0, "start": 1.2, "end": 3.4}]
}
```

El endpoint responde `422` si el audio no cumple el contrato y `503` si no hay
un artefacto entrenado disponible. El watchdog de cómputo se establece en 25 s
para respetar el límite de 30 s del reto y dejar margen para transporte.

### Dominio y publicación

El repositorio **no incluye un dominio público real configurado**, por lo que no
se debe inventar una URL de producción. El dominio se define fuera del control
de versiones mediante la variable `DOMINIO` del archivo `.env`:

```env
DOMINIO=detect.midominio.com
```

Con esa configuración, Caddy obtiene y renueva TLS automáticamente y la URL
pública es:

```text
https://detect.midominio.com/detect
```

Sin `DOMINIO`, Compose configura Caddy como `:80`; sirve para probar por IP en
HTTP, por ejemplo `http://<IP-DEL-SERVIDOR>/detect`. También se puede usar un
hostname temporal que resuelva a la IP, como
`<ip-con-guiones>.sslip.io`, para disponer de TLS. El puerto interno `8000` sólo
lo ve Caddy dentro de Docker; al exterior se publican 80 y 443.

## 11. Entrenar, verificar y desplegar

Entrenamiento base con el corpus oficial:

```bash
pip install -r requirements-analysis.txt
OPENBLAS_NUM_THREADS=1 python -m scripts.train_issue5
python -m scripts.report_issue5
```

Entrenamiento que incorpora el material autorizado del issue #15 sin contaminar
la selección ni `val`:

```bash
OPENBLAS_NUM_THREADS=1 python -m scripts.train_issue5 \
  --issue15-dir generated_issue15_jeff
```

Pruebas locales y servidor de desarrollo:

```bash
pip install -r requirements-dev.txt
pytest -q
ruff check app tests
uvicorn app.main:app --reload
```

Despliegue con contenedores:

```bash
cp .env.example .env
docker compose up -d --build
curl http://localhost/health
```

Para medir el sistema completo contra el split de validación:

```bash
python scripts/eval_endpoint.py --split val --url http://localhost:8000
```

## 12. Límites y decisiones responsables

El sistema estima una clase a partir de evidencia de audio: no identifica a una
persona, no verifica su identidad ni sustituye una revisión humana en una acción
de alto impacto. Una voz humana con condiciones atípicas o una IA diseñada para
imitar la conversación puede producir una señal ambigua. Por eso se reportan
probabilidad, confianza, capa y etapa; el diseño prefiere abstenerse o reducir
la confianza antes que acusar con certeza injustificada a una persona real.

Para ampliar la información técnica se pueden consultar:

- [Modelo y validación del issue #5](issue5-modelo.md)
- [Augmentación telefónica del issue #11](issue11-augmentacion.md)
- [Benchmark del issue #12](issue12-benchmark.md)
- [Guía de despliegue](../deploy/README.md)
