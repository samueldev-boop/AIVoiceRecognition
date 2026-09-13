# Referencias de llamadas inciertas

## Qué hace y qué no

Cuando el modelo **no llega a una conclusión clara**, `/detect` guarda una **referencia** a la
llamada. Un worker la sube a MongoDB Atlas. Nada más.

- **Solo llamadas inciertas.** Las decisiones claras no dejan rastro.
- **Solo una referencia, nunca el audio.** Ni el WAV, ni el base64, ni transcripciones, ni
  el `call_id` en claro.
- **No entrena ni cambia el modelo.** `model/model.joblib` no se toca.

La referencia existe para que esas llamadas puedan **recuperarse después** y entrar en el
**ciclo de reentrenamiento y actualización continua** del modelo. Ese ciclo no está
implementado aquí (ver [más abajo](#el-ciclo-de-reentrenamiento)).

```
/detect ──decisión clara──▶ responde (no se guarda nada)
   │
   └──incierta──▶ referencia JSON en data/raw ──worker──▶ MongoDB Atlas (colección calls)
                  (y responde igual)
```

## Cuándo una llamada es incierta

`app/audit.py::motivos_de_incertidumbre` devuelve los motivos; si la lista está vacía no se
guarda nada. Se registran en `analysis.uncertainty_reasons`.

| Motivo | Condición |
| --- | --- |
| `probabilidad_ambigua` | La probabilidad final está dentro de la banda `AUDIT_BANDA_MIN`–`AUDIT_BANDA_MAX` (por defecto 0.10–0.90, la misma que hace escalar a la cascada) |
| `desacuerdo` | Dos presupuestos de audio se contradijeron |
| `abstencion` | Sin habla utilizable (`sin_habla`), tiempo agotado (`watchdog`) o corte entre etapas (`+limite`) |

Medido con las 71 llamadas de `val` y el modelo actual: se guardan **4 (5.6 %)**, las 4 por
desacuerdo y 3 de ellas también con probabilidad ambigua.

El filtro captura incertidumbre, no errores. En `val`, el único error del modelo fue una
decisión segura (probabilidad 0.07) y no se guardó. Una decisión segura y equivocada solo
puede detectarse con una etiqueta externa.

## Qué se guarda: la referencia

| Campo | Para qué |
| --- | --- |
| `event_id` | Identificador único del evento (`_id` en MongoDB) |
| `call_id` | HMAC-SHA256 del `call_id` recibido si hay `AUDIT_ID_KEY`; si no, el propio `event_id` |
| `input_sha256` | Huella del audio decodificado: permite reencontrar la llamada en la fuente que la conserve y detectar duplicados |
| `audio_uri` | Opcional: referencia privada `s3://` donde otro sistema guarde el audio. La API no la rellena |
| `timestamp`, `source` | Cuándo y desde dónde |
| `model_version`, `model_artifact_sha256` | Qué modelo decidió exactamente |
| `decision` | `is_synthetic`, `confidence`, `probability_synthetic` |
| `analysis` | Etapa, latencia, duración, turnos del VAD, desacuerdo, scores por capa y por presupuesto, `uncertainty_reasons` |

El contrato está en `src/schemas.py` (`schema_version: 1`) y rechaza campos no declarados,
NaN, probabilidades fuera de rango y turnos invertidos o fuera de la duración.

## Módulos

| Módulo | Responsabilidad |
| --- | --- |
| `app/audit.py` | Filtro de incertidumbre y publicación atómica de la referencia |
| `src/schemas.py` | Contrato del documento |
| `src/storage.py` | Escritura atómica con fsync y bloqueo entre procesos |
| `src/worker.py` | Lee la bandeja, valida, reintenta, guarda en MongoDB y deja comprobantes |
| `src/repositories.py` | Índices y escritura idempotente |
| `src/db.py` | Cliente de MongoDB perezoso y con timeouts |
| `src/config.py` | Variables de entorno tipadas |
| `src/migrate_audit.py` | Migración explícita del JSONL del auditor anterior |

## Configuración

Copiar `.env.example` a `.env`. Importar un módulo no abre conexión con MongoDB.

| Variable | Por defecto | Para qué |
| --- | --- | --- |
| `MONGO_URI` | vacía | URI de Atlas (`mongodb+srv://…`). Nunca en el repositorio |
| `MONGO_DB` | `altur_defense` | Base de datos |
| `COLLECTION_CALLS` | `calls_v1` | Colección; en producción, `calls` |
| `AUDIT_BANDA_MIN`, `AUDIT_BANDA_MAX` | `0.10`, `0.90` | Banda de probabilidad incierta |
| `AUDIT_REQUIRED` | `false` | `true`: `/detect` responde 503 si no puede guardar la referencia |
| `AUDIT_ID_KEY` | vacía | Clave del HMAC de `call_id`; vacía, sin correlación entre peticiones |
| `AUDIT_SPOOL_DIR` | `data/raw` | Bandeja donde la API publica |
| `PROCESSED_DIR` | `data/processed` | Comprobantes y rechazos del worker |
| `AUDIT_LOG_PATH` | `data/audit.jsonl` | JSONL del auditor anterior; solo lo lee la migración |
| `MONGODB_TIMEOUT_MS` | `5000` | Timeouts de conexión y operación |
| `WORKER_POLL_SECONDS`, `WORKER_RETRY_ATTEMPTS`, `WORKER_RETRY_SECONDS`, `WORKER_BATCH_SIZE` | `2`, `3`, `1`, `100` | Ritmo y reintentos del worker |
| `MAX_JSON_BYTES` | `1048576` | Tamaño máximo de una referencia |

También se aceptan `MONGODB_URI`, `MONGODB_DATABASE` y `MONGODB_COLLECTION`, que prevalecen
si existen, y `DATABASE_NAME`/`COLLECTION_NAME`.

```bash
python -m src.worker          # bucle
python -m src.worker --once   # un lote; sale con 1 si falla la infraestructura
python -m src.migrate_audit   # migra AUDIT_LOG_PATH a la bandeja, sin tocar el original
```

## Garantías

- **La detección no depende de esto.** Si no se puede escribir la referencia (disco lleno,
  `data/` sin permisos), se registra `event=audit_failed` y `/detect` responde igual, salvo
  con `AUDIT_REQUIRED=true`. Tampoco depende de MongoDB: la API solo escribe en disco.
- **Al menos una vez.** La API publica con escritura temporal, fsync y rename atómico. El
  worker solo lee `*.json` completos, guarda con `$setOnInsert` sobre `_id = event_id` y
  deja comprobante en `data/processed/receipts/`. Repetir es idempotente; un mismo
  `event_id` con otro contenido va a `data/processed/rejected/`.
- **Caídas de Atlas.** Se reintenta con backoff; lo pendiente se queda en `data/raw` para el
  siguiente ciclo. Los originales nunca se borran ni se editan.
- **Un solo consumidor** por bandeja, con bloqueo del sistema operativo. Solo disco local.

## MongoDB Atlas

- La IP de la instancia debe estar en **Network Access** de Atlas.
- Índices que crea el worker: `event_id` único y parcial (solo documentos con `event_id`),
  `call_id`, `timestamp` e `input_sha256`. Si ya existe un índice con el mismo nombre y otras
  opciones, lo respeta y registra `event=index_kept`.
- La colección `calls` puede contener documentos del auditor anterior; no se modifican.
- Si se vuelve a crear un índice **único** sobre `call_id`, con `AUDIT_ID_KEY` activa las
  llamadas repetidas se rechazarían por conflicto.

## Despliegue

`docker compose` levanta `api`, `worker` y `caddy`, que comparten `./data`. API y worker
corren con uid 10001: `deploy/arrancar.sh` crea `data/raw` y `data/processed` con ese dueño
antes de levantar el compose. No hay retención automática de `data/raw`: vigilar el disco.

## El ciclo de reentrenamiento

Este pipeline es solo el primer tramo. Para cerrar el ciclo de actualización continua
faltan, fuera de este cambio:

1. **Recuperar el audio** de cada referencia, por `input_sha256` o `audio_uri`, desde la
   fuente autorizada que lo conserve. La API no guarda audio.
2. **Etiquetar** las llamadas con revisión humana: la predicción del modelo nunca es verdad.
3. **Reentrenar con el pipeline existente** (`scripts/train_issue5.py`,
   `scripts/train_issue11.py`), con validación agrupada por hablante y banco de estrés.
4. **Promover** solo si mejora sin acusar a más humanos, por PR a `stage` y después a `main`.

## Pruebas

```bash
ruff check src app tests scripts analysis
pytest -q tests/test_pipeline.py tests/test_pipeline_api.py
MONGODB_TEST_URI=mongodb://… pytest -q -m integration tests/test_mongo_integration.py
```

La prueba de integración crea y borra una base aleatoria `altur_test_*`: usar solo un
servidor desechable. Los tests unitarios usan fakes y directorios temporales.
