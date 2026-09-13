# Entrenamiento con las conversaciones del issue #15

`generated_issue15_jeff/` contiene 49 conversaciones de clon sintetico y una
grabacion humana real. Las variantes acusticas son derivadas de cuatro
conversaciones sinteticas base; por tanto, no aportan 49 identidades nuevas.

El flag `--issue15-dir` de `scripts.train_issue5` las incorpora despues de
congelar los hiperparametros con el corpus oficial de `train`. La seleccion de
candidatos, los folds fuera de muestra, el banco de estres y `val` excluyen por
completo el issue #15. Solo el ajuste calibrado final ve esas llamadas.

Cada familia derivada conserva un grupo `issue15:clone:<llamada-base>` y la
grabacion humana usa su propio grupo. Asi ninguna variante de una misma llamada
puede cruzar los folds internos de calibracion ni aparentar una voz independiente.

## Ejecucion

Con el corpus oficial en la raiz (`manifest.csv`, `audio/` y opcionalmente
`turns/`):

```bash
OPENBLAS_NUM_THREADS=1 python -m scripts.train_issue5 \
  --issue15-dir generated_issue15_jeff
```

El manifiesto #15 se valida antes de extraer: debe listar exactamente todos los
WAV, con audio estereo PCM_16 a 8 kHz y duracion coherente. Las features se
cachean en `analysis/issue5/issue15_features.joblib`; el cache y los reportes
locales no se versionan. `selection_frozen.json`, `metrics.json` y
`model/model.joblib` registran su fingerprint y la composicion de familias.

No se debe promocionar un modelo entrenado solo con este directorio: tiene una
sola voz humana y cuatro conversaciones sinteticas de origen. El corpus oficial
aporta la diversidad de personas y voces que hace posible generalizar.
