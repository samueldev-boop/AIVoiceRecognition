#!/bin/sh
# Arranque idempotente del servicio en una instancia limpia de Ubuntu 24.04.
# Se puede volver a ejecutar para desplegar una version nueva: es la misma ruta.
#
#   sudo sh deploy/arrancar.sh
#   sudo RAMA=main sh deploy/arrancar.sh     # desplegar otra rama
#
# Hace lo minimo: instala docker si falta, pone el repositorio en la rama pedida, crea el
# .env si no existe, levanta el compose y verifica. No toca el firewall: eso se configura
# en Vultr, fuera de la maquina, porque docker publica puertos saltandose ufw.

set -eu

REPO="${REPO:-https://github.com/samueldev-boop/AIVoiceRecognition.git}"
DESTINO="${DESTINO:-/opt/servicio}"
RAMA="${RAMA:-stage}"

[ "$(id -u)" = "0" ] || { echo "hace falta root: sudo sh deploy/arrancar.sh" >&2; exit 1; }

paso() { printf '\n== %s\n' "$1"; }

paso "docker"
if command -v docker >/dev/null 2>&1; then
  echo "  ya instalado: $(docker --version)"
else
  export DEBIAN_FRONTEND=noninteractive
  apt-get update -qq
  apt-get install -y -qq --no-install-recommends docker.io docker-compose-v2 git
  echo "  instalado: $(docker --version)"
fi
systemctl enable --now docker

paso "repositorio en $DESTINO ($RAMA)"
if [ -d "$DESTINO/.git" ]; then
  git -C "$DESTINO" fetch -q origin
  git -C "$DESTINO" checkout -q "$RAMA"
  git -C "$DESTINO" reset -q --hard "origin/$RAMA"
  echo "  actualizado a $(git -C "$DESTINO" rev-parse --short HEAD)"
else
  git clone -q --branch "$RAMA" "$REPO" "$DESTINO"
  echo "  clonado en $(git -C "$DESTINO" rev-parse --short HEAD)"
fi

paso ".env"
if [ -f "$DESTINO/.env" ]; then
  echo "  ya existe, no se toca"
else
  cp "$DESTINO/.env.example" "$DESTINO/.env"
  chmod 600 "$DESTINO/.env"
  echo "  creado desde .env.example con permisos 600"
fi

paso "datos de auditoria"
# La API y el worker corren con uid 10001 y montan ./data. Si docker crea el directorio,
# lo crea como root y ninguno de los dos podria escribir.
mkdir -p "$DESTINO/data/raw" "$DESTINO/data/processed"
chown -R 10001:10001 "$DESTINO/data"
echo "  $DESTINO/data escribible por el uid 10001"

paso "levantando el servicio"
cd "$DESTINO"
docker compose up -d --build

paso "verificacion"
sh deploy/verificar.sh http://localhost
