# Artefactos del modelo

## model.joblib

El modelo entrenado (ver #5). Todavia no existe: mientras falte, `/detect` valida el clip y
responde 503 con el motivo.

## Sobre el VAD

No hay artefacto de VAD que versionar. Se usa **webrtcvad**, que es una extension en C con el
modelo dentro y no necesita ningun archivo aparte.

Se probo tambien Silero VAD por ONNX y se descarto con datos (ver #3): funciona a 8 kHz
nativos, pero su acuerdo con `turns/*.json` es bastante peor (IoU 0.618 en el canal del
llamante, contra 0.729 de webrtcvad) y las features de conducta derivadas de sus fronteras
rinden menos. La pista estaba en el propio dataset: los tiempos de `turns/*.json` caen al
100 % en una rejilla de 20 ms con duracion minima de 0.2 s, que es exactamente como trabaja
webrtcvad y no como trabaja Silero (ventanas de 32 ms a 8 kHz).
