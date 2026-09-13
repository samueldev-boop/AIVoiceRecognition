# Artefactos del modelo

## model.joblib

Se genera con `python -m scripts.train_issue5` usando el dataset local. El archivo está
ignorado por git; se copia como artefacto local al preparar el servicio.

Contiene versión del modelo y extractor, orden de features, grupos, versiones de Python
y scikit-learn, configuración seleccionada y tres clasificadores calibrados (`first_turn`,
`20s`, `full`). Cada clasificador conserva sus imputadores, recortes, escaladores,
regresiones por capa, regresión de fusión y calibrador Platt. Su SHA-256 queda en
`analysis/issue5/metrics.json`.

`app.model.cargar()` comprueba versión y esquema. `Modelo.puntuar_audio(x, sr, presupuesto)`
ejecuta el mismo VAD y extractor del entrenamiento y devuelve `probability_synthetic`,
`is_synthetic`, `confidence`, scores y capas disponibles. `confidence` limita la confianza
cuando las capas no concuerdan; los umbrales de clasificación se aplican sobre la
probabilidad calibrada. El endpoint y su cascada se integran en #6.

El análisis completo se regenera con `python -m scripts.report_issue5`:
`analysis/issue5/reporte.html`, `reporte.pdf` y `reporte.md`.

## Sobre el VAD

No hay artefacto de VAD que versionar. Se usa **webrtcvad**, que es una extension en C con el
modelo dentro y no necesita ningun archivo aparte.

Se probo tambien Silero VAD por ONNX y se descarto con datos (ver #3): funciona a 8 kHz
nativos, pero su acuerdo con `turns/*.json` es bastante peor (IoU 0.618 en el canal del
llamante, contra 0.729 de webrtcvad) y las features de conducta derivadas de sus fronteras
rinden menos. La pista estaba en el propio dataset: los tiempos de `turns/*.json` caen al
100 % en una rejilla de 20 ms con duracion minima de 0.2 s, que es exactamente como trabaja
webrtcvad y no como trabaja Silero (ventanas de 32 ms a 8 kHz).
