#!/bin/sh
# Comprobacion de un despliegue ya en pie. Sin dependencias: curl y base64.
#
#   ./deploy/verificar.sh http://<ip>
#   ./deploy/verificar.sh https://detect.midominio.com audio/una_llamada.wav
#
# Sin el WAV comprueba solo que el servicio responde y tiene modelo. Con el WAV hace una
# peticion real a /detect y ensena la respuesta completa.

set -eu

URL="${1:?uso: verificar.sh <url-base> [llamada.wav]}"
WAV="${2:-}"
fallo=0

paso() { printf '  %-34s %s\n' "$1" "$2"; }

salud=$(curl -fsS --max-time 10 "$URL/health" 2>/dev/null || echo "")
if [ -z "$salud" ]; then
  paso "/health" "SIN RESPUESTA"
  exit 1
fi
paso "/health" "$salud"

case "$salud" in
  *'"model_loaded":true'*) paso "modelo cargado" "si" ;;
  *) paso "modelo cargado" "NO: /detect respondera 503"; fallo=1 ;;
esac

codigo=$(curl -fsS -o /dev/null -w '%{http_code}' --max-time 10 "$URL/" 2>/dev/null || echo "000")
paso "frontend en /" "HTTP $codigo"

codigo=$(curl -fsS -o /dev/null -w '%{http_code}' --max-time 10 "$URL/docs" 2>/dev/null || echo "000")
paso "contrato en /docs" "HTTP $codigo"

# Un cuerpo invalido debe dar 422 y no 500: la validacion de entrada esta en pie.
codigo=$(curl -s -o /dev/null -w '%{http_code}' --max-time 15 -X POST "$URL/detect" \
  -H 'content-type: application/json' -d '{"audio":"no-es-base64!!"}' || echo "000")
if [ "$codigo" = "422" ]; then
  paso "/detect con base64 invalido" "HTTP 422, correcto"
else
  paso "/detect con base64 invalido" "HTTP $codigo, se esperaba 422"; fallo=1
fi

if [ -n "$WAV" ]; then
  [ -f "$WAV" ] || { paso "wav" "no existe: $WAV"; exit 1; }
  cuerpo=$(mktemp)
  printf '{"audio":"%s"}' "$(base64 -w0 "$WAV")" > "$cuerpo"
  inicio=$(date +%s)
  respuesta=$(curl -fsS --max-time 40 -X POST "$URL/detect" \
    -H 'content-type: application/json' -d @"$cuerpo" || echo "")
  fin=$(date +%s)
  rm -f "$cuerpo"
  if [ -z "$respuesta" ]; then
    paso "/detect con $WAV" "SIN RESPUESTA"; fallo=1
  else
    paso "/detect con $WAV" "$(( fin - inicio ))s"
    printf '    %s\n' "$respuesta"
  fi
fi

[ "$fallo" = "0" ] && echo "  despliegue correcto" || echo "  hay fallos arriba"
exit "$fallo"
