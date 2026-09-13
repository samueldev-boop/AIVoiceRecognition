#!/usr/bin/env bash
# Guardia de configuracion. Dos tipos de regla:
#
#   PROHIBIDO  -> no hay forma de saltarlo. Secretos y derivados del dataset.
#   VIGILADO   -> requiere la etiqueta "config-revisada" en el PR. Configuracion critica.
#
# uso: guardia.sh <rango-git> [etiquetas] [integracion]
#      guardia.sh origin/stage...HEAD "config-revisada,backend"
#      guardia.sh origin/main...HEAD "" integracion
#
# En modo integracion (una PR stage -> main) se salta el nivel VIGILADO: esos cambios ya se
# revisaron uno a uno en su PR original y volver a pedir la etiqueta anade friccion sin
# anadir seguridad. El nivel PROHIBIDO no se exime nunca.

set -eu

RANGO="${1:?falta el rango git, por ejemplo origin/stage...HEAD}"
ETIQUETAS="${2:-}"
INTEGRACION="${3:-}"
ETIQUETA_PERMISO="config-revisada"

archivos=$(git diff --name-only --diff-filter=ACMR "$RANGO")
if [ -z "$archivos" ]; then
  echo "Sin archivos modificados."
  exit 0
fi

fallo=0
aviso=0

# ---------------------------------------------------------------- PROHIBIDO: rutas
for f in $archivos; do
  case "$f" in
    .env.example) : ;;
    .env|.env.*|*/.env|*/.env.*)
      echo "::error file=$f::PROHIBIDO: un archivo .env nunca entra en el repositorio. Usa .env.example con valores vacios."
      fallo=1
      ;;
    audio/*|turns/*|manifest.csv|DATASET.md|*.wav|*.zip|analysis/*.csv|analysis/transcripts/*)
      echo "::error file=$f::PROHIBIDO: derivado del dataset. El reto prohibe redistribuirlo."
      fallo=1
      ;;
  esac
done

# ------------------------------------------------------- PROHIBIDO: secretos anadidos
# Solo se revisan las lineas ANADIDAS, para no marcar lo que ya estaba.
anadidas=$(git diff -U0 --diff-filter=ACMR "$RANGO" -- . ':(exclude).env.example' \
           | grep '^+' | grep -v '^+++' || true)

buscar() {
  patron="$1"; motivo="$2"
  if printf '%s\n' "$anadidas" | grep -qiE -- "$patron"; then
    echo "::error::PROHIBIDO: $motivo. Rota la credencial y quitala del diff."
    fallo=1
  fi
}

buscar 'sk_[0-9a-f]{32,}'                                    'clave con formato sk_ (ElevenLabs y similares)'
buscar 'xi-api-key[[:space:]]*[:=][[:space:]]*[^[:space:]"]{8,}' 'cabecera xi-api-key con valor'
buscar 'AKIA[0-9A-Z]{16}'                                    'clave de acceso AWS'
buscar '-----BEGIN [A-Z ]*PRIVATE KEY-----'                  'clave privada'
buscar '(mongodb(\+srv)?|postgres(ql)?|mysql|redis)://[^[:space:]:]+:[^[:space:]@]+@' 'URI de conexion con contrasena'
buscar '(api[_-]?key|secret|token|passwd|password)[[:space:]]*[:=][[:space:]]*.?[A-Za-z0-9/_+-]{20,}' \
       'valor con aspecto de credencial'

# ------------------------------------------------------------------ VIGILADO: config
vigilado=""
for f in $archivos; do
  case "$f" in
    Dockerfile|.dockerignore|docker-compose.yml|Caddyfile|.gitignore|\
    requirements.txt|requirements-dev.txt|pyproject.toml|\
    .github/*|.githooks/*|deploy/*|\
    *.yml|*.yaml)
      vigilado="$vigilado $f"
      aviso=1
      ;;
  esac
done

if [ "$aviso" = "1" ] && [ "$INTEGRACION" = "integracion" ]; then
  echo "PR de integracion: la configuracion critica que toca ya se reviso en sus PR de origen."
  for f in $vigilado; do echo "  - $f"; done
elif [ "$aviso" = "1" ]; then
  echo "Configuracion critica tocada por este PR:"
  for f in $vigilado; do echo "  - $f"; done
  case ",$ETIQUETAS," in
    *",$ETIQUETA_PERMISO,"*)
      echo "Etiqueta '$ETIQUETA_PERMISO' presente: cambio deliberado, se permite."
      ;;
    *)
      echo "::error::Este PR cambia configuracion critica sin la etiqueta '$ETIQUETA_PERMISO'."
      echo "::error::Si el cambio es intencionado, anade la etiqueta al PR. Si no lo es, revierte esos archivos."
      fallo=1
      ;;
  esac
fi

[ "$fallo" = "0" ] && echo "Guardia: sin hallazgos."
exit "$fallo"
