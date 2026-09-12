# Cómo trabajamos

## Ramas

| Rama | Para qué | Protección |
| --- | --- | --- |
| `main` | Estable. Lo que se demuestra y lo que se despliega | Protegida. Sólo entra por PR desde `stage`, con 1 aprobación |
| `stage` | Integración. **Todo el trabajo sale de aquí y vuelve aquí** | Protegida. Sólo entra por PR. No requiere aprobación |
| `<tipo>/<issue>-<slug>` | Tu trabajo | — |

Nadie hace `push` directo a `main` ni a `stage`. No se hace force-push ni se borran esas dos ramas.

### Nombre de rama

```
<tipo>/<numero-de-issue>-<descripcion-corta-con-guiones>
```

`<tipo>` es uno de:

| Tipo | Cuándo |
| --- | --- |
| `feat` | Funcionalidad nueva |
| `fix` | Corrección de un fallo |
| `model` | Features, entrenamiento, evaluación |
| `infra` | Docker, despliegue, configuración |
| `docs` | Documentación |
| `chore` | Mantenimiento, dependencias, limpieza |

Ejemplos:

```
model/3-vad-silero
feat/6-endpoint-detect
infra/7-despliegue-vultr
```

## Commits

El prefijo **sale de la rama**: mismo tipo, mismo número de issue.

```
<tipo>(#<issue>): mensaje en imperativo y en minúscula
```

```
model(#3): sustituir turns/*.json por silero vad a 8 khz
feat(#6): anadir salida temprana en la etapa 1
infra(#7): fijar cpu_threads a 4 en el contenedor
```

El hook `prepare-commit-msg` **inserta el prefijo automáticamente** leyendo el nombre de la
rama, y `commit-msg` rechaza el commit si el formato no cuadra. Instálalos una vez:

```bash
git config core.hooksPath .githooks
```

Eso activa tres hooks: `prepare-commit-msg` (inserta el prefijo), `commit-msg` (valida el
formato) y `pre-commit` (bloquea secretos y archivos prohibidos).

## Pull requests

- **Destino `stage`**, siempre. `stage → main` sólo cuando la integración está estable.
- **Título del PR con el mismo prefijo que los commits**: `model(#3): vad propio con silero`.
- Un PR por issue. Si el PR crece más allá del issue, se parte.
- En la descripción: qué cambia, cómo se probó y `Closes #<issue>`.
- El PR se puede fusionar con *squash*; el mensaje resultante mantiene el prefijo.

## Guardia de configuración y secretos

Hay dos niveles, y la diferencia importa:

| Nivel | Qué cubre | Cómo se desbloquea |
| --- | --- | --- |
| **Prohibido** | Cualquier `.env`, derivados del dataset (`audio/`, `turns/`, `manifest.csv`, `DATASET.md`, `*.wav`, `*.zip`, salidas de `analysis/`) y cadenas con aspecto de credencial | **No se desbloquea.** Hay que sacarlo del cambio; si la credencial era real, rotarla |
| **Vigilado** | `Dockerfile`, `.dockerignore`, `docker-compose.yml`, `Caddyfile`, `.gitignore`, `requirements*.txt`, `pyproject.toml`, cualquier `*.yml` / `*.yaml`, y todo `.github/`, `.githooks/` y `deploy/` | Etiqueta **`config-revisada`** en el PR. Al ponerla, el check se repite solo y pasa a verde |

La idea del nivel vigilado no es impedir el cambio: es que **no pueda ocurrir por accidente**.
Si tocas la configuración a propósito, pones la etiqueta y queda registrado; si aparece en un PR
que iba de otra cosa, el check lo para.

Dónde se aplica:

- **Al hacer commit**, el hook `pre-commit` corre las reglas prohibidas en local.
- **En cada PR**, `.github/workflows/guardia.yml` las corre otra vez, más el nivel vigilado.
  Esta es la que manda: `--no-verify` salta el hook, no el CI.
- **En `main`**, las rutas de `.github/CODEOWNERS` exigen además revisión del propietario.
- **En el propio GitHub**, el escaneo de secretos con protección de push está activado: si
  intentas empujar una credencial de un proveedor conocido, el push se rechaza.

En la instancia, el `.env` real vive en `/opt/servicio/.env` con permisos `600`, y el
contenedor no monta el repositorio: el servicio no puede leerlo ni modificarlo, sólo recibe
las variables ya resueltas.

## Reglas que no se negocian

1. **No se commitea audio ni derivados del dataset.** `audio/`, `manifest.csv`, `turns/`,
   `DATASET.md` y las salidas de `analysis/` están en `.gitignore` porque el reto prohíbe
   redistribuir el dataset. Si algo de eso aparece en un `git status`, es un error, no una excepción.
2. **Ningún secreto en el repositorio.** Claves de API y URIs por variable de entorno o `.env`
   (ignorado). Si una clave se filtra en un commit, se rota; no se borra el commit y ya está.
3. **Nada entra sin número medido.** Un cambio en el modelo se acompaña del resultado en
   validación agrupada y en el banco de estrés. `val` está saturado y no sirve para decidir.
4. **No sobreingeniería.** Antes de escribir una función, comprobar si ya existe en una
   librería del stack. El repositorio ya arrastra regresión logística, AUC y algebra de
   intervalos hechas a mano: eso se está retirando, no se amplía.

## Excepción de emergencia

Las protecciones no aplican a administradores, a propósito: si a una hora del cierre hay que
empujar un arreglo directo a `main`, se puede. Se avisa en el canal del equipo y se abre el
issue después. Es una excepción, no una vía.
