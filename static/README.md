# Interfaz

Estatica, sin paso de build, servida por el mismo FastAPI que el endpoint.

## tailwind.js

Build de navegador de Tailwind CSS v4: compila las utilidades en tiempo de ejecucion leyendo
el DOM, asi que no hace falta compilar nada en el repositorio.

| | |
| --- | --- |
| Origen | https://cdn.jsdelivr.net/npm/@tailwindcss/browser@4 |
| Licencia | MIT |
| Tamano | 276 KB (69 KB con gzip, y Caddy comprime) |
| SHA256 | `6d8c473ef2f8ad63feafc0bd76502dda31501a6c135dc4c6173f6268cde595be` |

Se versiona en lugar de cargarlo del CDN por dos razones: la demo tiene que funcionar aunque
la red del sitio bloquee jsdelivr, y hay un test que falla si aparece una URL externa en el
HTML o en el JS.
