# Modelo por intervención y análisis del dataset

El entrenamiento de #5 usa las intervenciones del llamante obtenidas por `app.vad`,
con el mismo extractor en entrenamiento e inferencia. Se conservan las llamadas y audios
originales y se generan tablas auditables en `analysis/issue5/`.

## Reproducir

Desde la raíz del repositorio, con `manifest.csv`, `audio/` y `turns/` disponibles:

```bash
.venv/bin/python -m pip install -r requirements-analysis.txt
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m scripts.speaker_groups
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m scripts.train_issue5 \
  --speakers speaker_groups.csv
.venv/bin/python -m scripts.report_issue5
.venv/bin/python -m analysis.ablation
.venv/bin/ruff check app tests scripts
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m pytest -q
```

`--root` permite leer el dataset desde otra carpeta; `--output` cambia el directorio de
resultados, `--model` el destino del artefacto y `--jobs` los procesos de extracción.
Los resultados de extracción y evaluación tienen huellas del código y entradas para
invalidar cachés después de cambios. No se usa ASR ni una API remota en este flujo.

## Limpieza y unidad de observación

Se validan las columnas del manifiesto, etiquetas, splits, duplicados, WAV PCM de 16 bits
a 8 kHz, canales, duraciones reales, clipping y fronteras de segmentos. Los duplicados
exactos se detectan mediante SHA-256 del PCM. Las llamadas excluidas y sus motivos se
registran; los originales no se modifican.

Una intervención contiene su audio y hasta 300 ms de contexto anterior, sin leer el turno
siguiente. `first_turn` requiere detectar el final del primer turno con el lookahead del
VAD; `20s` calcula el VAD sobre el prefijo truncado. Las fronteras y features se exportan.

Las mediciones no disponibles se representan con NaN. Cada pipeline ajusta conversión
de infinitos a faltantes, mediana con indicadores de ausencia, recorte opcional por
cuantiles y StandardScaler. Las columnas constantes no aportan señal tras el escalado.
Se retiran dos alias exactos del extractor: `f0_jitter_praat` y `dur_usada_s`.

## Arquitectura y evaluación

1. Regresiones L2 por turno para ganancia, codec/ancho de banda, silencio y prosodia.
   Cada llamada aporta el mismo peso total a la pérdida de las regresiones acústicas.
2. Agregación por capa: 0.6 de sigmoide(media de logits), 0.2 del máximo y 0.2 de la
   fracción de turnos cuyo score llega a 0.7. Coeficientes fijados antes de evaluar.
3. Dos regresiones adicionales sobre conducta de llamada y razones entre canales.
4. Regresión L2 de fusión sobre los seis scores. Sus datos de entrenamiento son scores
   fuera de muestra de las capas, obtenidos con tres folds internos agrupados.
5. CalibratedClassifierCV con Platt y tres folds agrupados vuelve a entrenar la
   jerarquía completa en cada partición de calibración. No recibe scores precalculados
   con acceso a las etiquetas del fold que calibra.

## Agrupación por hablante

El dataset no da identidades de hablante, así que agrupar la validación cruzada solo por
llamada deja que la misma persona aparezca en varios folds. `scripts/speaker_groups.py`
deriva identidades aproximadas: huella MFCC de Praat sobre el habla del llamante (sin el
coeficiente de energía, que es ganancia), agrupamiento de Ward dentro de cada split y por
clase, y umbral elegido por silueta. Resultado: 27 personas para 113 llamadas humanas de
train y 27 voces para 169 sintéticas. La silueta de las voces TTS (0.43) es bastante mejor
que la de las personas (0.25), que es lo esperable y sirve de comprobación de plausibilidad.

El límite hay que decirlo: **no es reconocimiento de hablante**. Se intentó calibrar el
umbral con la restricción que el dataset garantiza —train y val son disjuntos por
hablante— y no fue posible: incluso agrupando fino quedaban grupos mezclando llamadas de
los dos splits, que por construcción son personas distintas. A 8 kHz esta huella agrupa
de más. Para el propósito sirve: agrupar de más hace la validación más pesimista, nunca
más optimista.

Lo que cuesta agrupar bien, medido:

| Presupuesto | CV por llamada | CV por hablante | Δ | Peor estrés llamada | Peor estrés hablante |
| --- | --- | --- | --- | --- | --- |
| Primera intervención | 0.9936 | 0.9768 | −0.0168 | 0.9783 | 0.9483 |
| Primeros 20 s | 0.9991 | 0.9739 | −0.0252 | 0.9823 | 0.9287 |
| Llamada completa | 0.9998 | 0.9682 | −0.0317 | 0.9669 | 0.8582 |

La CV por llamada era optimista, y **cuanto más audio se mira, más lo era**. Con agrupación
por hablante la primera intervención pasa a ser el presupuesto más fuerte y el más robusto
al ataque, por delante de la llamada completa. La lectura es la del resto del proyecto: con
más audio el modelo se apoya más en rasgos de la persona y de su línea, que no transfieren a
hablantes nuevos. Importa porque el juicio final usa voces que no están en ningún split.

Los tres presupuestos eligen ahora `limpio_C010`, la configuración más regularizada: folds
más difíciles premian más regularización.

| Capa (AUC fuera de muestra) | Primera intervención | Llamada completa |
| --- | --- | --- |
| conducta | 0.9472 | 0.9775 |
| razon_canal | 0.9270 | 0.9408 |
| codec_bw | 0.8947 | 0.9701 |
| prosodia | 0.8770 | 0.8214 |
| silencio | 0.8580 | 0.9862 |
| ganancia | 0.7988 | 0.8447 |

En el presupuesto corto mandan la conducta y las razones entre canales, que son robustas.
En el completo manda el silencio, que describe la habitación detrás del micrófono: el atajo.

## Validación

La validación de desarrollo usa cinco folds de StratifiedGroupKFold sobre componentes
conexas de llamada, identidad e igualdad exacta del PCM del llamante.
La selección compara tres configuraciones predefinidas: C=0.33 sin recorte, C=0.33 con
recorte P0.5/P99.5 y C=0.10 con ese recorte. El criterio es el promedio entre AUC limpia
y peor AUC de estrés; los desempates usan log-loss y menor C. Se guarda la selección,
se ajusta el artefacto con train y luego se informa val.

Los escenarios del banco de estrés se aplican a las llamadas reservadas de cada fold:
filtrado y puerta de ruido en humanos, ruido rosa y cambios de ganancia en bots, y
modificación de conducta en bots. Los ataques de audio vuelven a pasar por el VAD.
El ataque de ritmo es una ablación de metadatos: no retemporiza el WAV.

Las métricas distinguen ROC-AUC, precisión, exactitud, sensibilidad, F1, FPR/FNR,
Brier, log-loss y ECE. Se incluyen matrices a umbrales 0.5/0.7 e intervalos bootstrap
por grupo. El informe muestra por separado cada presupuesto y cada escenario.

## Uso del artefacto

```python
from app.audio import leer_wav
from app.model import cargar

modelo = cargar()
x, sr = leer_wav("ruta/a/llamada.wav")
resultado = modelo.puntuar_audio(x, sr, "first_turn")
```

`probability_synthetic` es la probabilidad calibrada. `confidence` es confianza en la
clase predicha con una regla conservadora: si se predice sintético y no hay dos votos
de capas disponibles con score≥0.7, su máximo es 0.65. Con una sola capa disponible
no supera 0.9. La regla no modifica la probabilidad usada en las matrices de confusión.

La carga verifica versión de artefacto, extractor, scikit-learn y esquema de features.
La integración de este modelo en la cascada HTTP se desarrolla en #6.

## Entregables locales

| Archivo | Contenido |
| --- | --- |
| `model/model.joblib` | Pipelines, fusión, calibración y metadatos por presupuesto |
| `analysis/issue5/reporte.html` | Informe autónomo con imágenes integradas |
| `analysis/issue5/reporte.pdf` | Texto, tablas y figuras para consulta o exportación |
| `analysis/issue5/reporte.md` | Informe editable con vínculos a las figuras |
| `analysis/issue5/metrics.json` | Métricas, matrices, selección y SHA-256 del artefacto |
| `analysis/issue5/selection_frozen.json` | Parámetros fijados antes de informar val |
| `analysis/issue5/folds.json` | Asignación auditable de llamadas a train/prueba |
| `analysis/issue5/interventions.csv` | Features y fronteras de cada intervención |
| `analysis/issue5/calls_*.csv` | Features de llamada por presupuesto |
| `analysis/issue5/audio_audit.csv` | Formato, duración, hashes, ceros y clipping |
| `analysis/issue5/feature_audit.csv` | Faltantes, cuantiles, constantes y atípicos IQR |
| `analysis/issue5/correlation_*.csv` | Matrices completas Pearson y Spearman |
| `analysis/issue5/feature_label_correlations.csv` | Asociación por llamada y corrección BH |
| `analysis/issue5/strong_correlation_pairs.csv` | Correlaciones globales y por clase |
| `analysis/issue5/predictions.csv`, `errors.csv` | Predicciones y errores por presupuesto |
| `analysis/issue5/figures/` | Gráficas PNG y SVG |

## Referencias

- [Contrato oficial del dataset](https://github.com/alturio/hackmty26).
- [Validación agrupada de sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html).
- [Calibración de probabilidades](https://scikit-learn.org/stable/modules/calibration.html).
- [Prevención de fuga en el preprocesamiento](https://scikit-learn.org/stable/common_pitfalls.html).
