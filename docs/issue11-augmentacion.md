# Augmentacion telefonica (#11)

El entrenamiento compara la fusion logistica existente sin augmentacion, con codecs y
sala, y con RawBoost adicional. Cada alternativa augmentada tambien se reentrena sin
ganancia, codec y silencio para medir su dependencia de estos grupos. Se evalua la
llamada completa con los mismos cinco folds y el mapa local `speaker_groups.csv` cuando
esta presente. `val` no participa en la seleccion.

Se generan dos variantes aleatorias por llamada, con una semilla distinta en cada pasada:
GSM 06.10, AMR-NB, G.722, A-law y mu-law para humanos; ruido MUSAN y RIR reales de
OpenSLR 28 para bots. `audiomentations` aplica ganancia, clipping y filtros. RawBoost
usa el algoritmo 5 (LnL + ISD), con frecuencias adaptadas a 8 kHz. Las variantes mantienen
el grupo del original dentro de todos los folds de entrenamiento, fusion y calibracion.

Los codecs se codifican y decodifican con FFmpeg, sin sustituirlos por filtros.
Se conserva el canal del agente y se recalculan VAD y features del llamante transformado.
Los archivos de ruido y RIR del banco de estres son distintos de los de entrenamiento.
`app.audio.remuestrear()` fija `soxr_hq` en disco, HTTP y augmentacion; el audio nativo
de 8 kHz pasa intacto.

## Ejecucion

FFmpeg necesita los encoders `libgsm`, `libopencore_amrnb`, `g722`, `pcm_alaw` y `pcm_mulaw`.
Son herramientas de entrenamiento; el servicio solo agrega la dependencia `soxr`.

```bash
pip install -r requirements-analysis.txt
mkdir -p analysis/issue11/resources
curl --fail --location --retry 2 \
  --output analysis/issue11/resources/rirs_noises.zip \
  https://www.openslr.org/resources/28/rirs_noises.zip
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m scripts.train_issue11 --jobs 6
```

Las salidas locales son `analysis/issue11/reporte.md`, `results.json`, las predicciones
OOF en `cv_*.joblib` y `candidate.joblib`. Los caches evitan repetir extraccion y ajustes.
`--epochs` controla las pasadas aleatorias, no iteraciones del optimizador logistico.

`--promote model/model.joblib` reemplaza el modelo completo solo si supera el banco:
cero FP en humanos con codecs y en `humano_limpio`, FNR menor que 0.10 en `bot_evasivo`
y `bot_sala`, y sin una caida de AUC superior a 0.02 al reentrenar sin grupos fragiles.
Los errores se miden a 0.5; el informe incluye tambien 0.7. Si falla, se conserva el
modelo vigente y se guardan los resultados del candidato.

## Fuentes

- [OpenSLR 28: ruido MUSAN y RIR, Apache 2.0](https://www.openslr.org/28)
- [Codecs de FFmpeg](https://ffmpeg.org/ffmpeg-codecs.html)
- [audiomentations](https://iver56.github.io/audiomentations/)
- [RawBoost oficial, MIT](https://github.com/TakHemlata/RawBoost-antispoofing)
- [RawBoost: articulo](https://arxiv.org/abs/2111.04433)
- [Python SoXR](https://python-soxr.readthedocs.io/en/latest/soxr.html)
