# Benchmark de clasificador (#12)

Compara la fusion logistica elegida en #11 con LightGBM. Ambos usan las mismas llamadas,
variantes de entrenamiento, cinco folds agrupados y escenarios de estres de #11.
No se consulta `val` para elegir modelo ni umbral.

TabPFN, el tercer candidato del issue, queda descartado: su papel era servir de oraculo de
precision en Colab, y el equipo ya no usa Colab para ninguna herramienta. En este servicio
tampoco tiene cabida, porque arrastra torch y una licencia propia, contra la decision de un
servicio ligero.

LightGBM recibe solo conducta, prosodia y razones entre canales. Usa 7 hojas, profundidad
3, un minimo de 20 filas por hoja, 70% de columnas, 80% de filas y regularizacion L1/L2.
No incorpora grupos de ganancia, codec ni silencio. Ambos candidatos tienen calibracion
sigmoide ajustada en folds agrupados internos.

Se elige el umbral con menor coste medio sobre los escenarios OOF: un falso positivo
cuesta 5 y un falso negativo 1. `--fp-cost` permite cambiar ese supuesto. Se selecciona
el modelo con menor coste; el desempate usa la media del AUC limpio y el peor AUC de
estres, y finalmente la regresion logistica. Las metricas del umbral elegido describen
esta seleccion sobre OOF; las tablas tambien conservan los umbrales fijos 0.5 y 0.7.

La importancia se calcula permutando familias de variables en las llamadas reservadas
de cada fold, con tres repeticiones. Se informa la caida de AUC, sin reajustar el modelo.

```bash
pip install -r requirements-analysis.txt
# Requiere primero las salidas de scripts.train_issue11.
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m scripts.benchmark_issue12
```

`analysis/issue12/reporte.md` entrega la tabla y justificacion; `results.json` conserva
las metricas; `importance.csv` entrega las importancias; `candidate.joblib` contiene
el clasificador elegido y su umbral. Son salidas locales regenerables. El benchmark
no sustituye automaticamente el artefacto del servicio.

La PR de #12 depende de la PR de #11; deben integrarse en ese orden.

- [Parametros oficiales de LightGBM](https://lightgbm.readthedocs.io/en/stable/Parameters.html)
- [Importancia por permutacion](https://scikit-learn.org/stable/modules/permutation_importance.html)
