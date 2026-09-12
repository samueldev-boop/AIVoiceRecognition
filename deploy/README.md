# Despliegue

Una instancia, dos contenedores (servicio + Caddy). El aprovisionamiento y el primer
despliegue son el issue #7; aqui está la configuración que consume.

## Instancia de Vultr

### Recomendada

| Campo | Valor | Por qué |
| --- | --- | --- |
| Tipo | **Optimized Cloud Compute — General Purpose** (vCPU dedicado) | La etapa de ASR mantiene los 4 hilos al 100 % durante 5–10 s por llamada. En planes de vCPU compartido eso sufre *steal time* y la latencia se vuelve impredecible, que es justo lo que se está midiendo |
| vCPU / RAM | **4 vCPU dedicados / 16 GB** | Los 4 vCPU son el requisito real (4 hilos de ASR). Los 16 GB vienen con el escalón, no porque hagan falta: el servicio consume ~1.5 GB |
| Disco | El del plan (NVMe) | La imagen ronda 600 MB y el modelo de ASR 250 MB |
| Región | **Ciudad de México** | Menor latencia para la demostración. Vultr la tiene entre sus regiones |
| Sistema | **Ubuntu 24.04 LTS** | Lo que espera el `cloud-init.yaml` |
| IPv6 | Activado | Gratis y no molesta |
| Auto-backups | **Desactivado** | Cuesta un porcentaje del plan y no aporta nada en un fin de semana |
| Firewall | Grupo nuevo (ver abajo) | |
| SSH Keys | Las del equipo | Evita contraseñas por correo |
| Startup Script | `deploy/cloud-init.yaml`, tipo **Boot** | Deja Docker y el repositorio listos al primer arranque |

### Alternativa económica

**Cloud Compute — High Performance, 4 vCPU / 8 GB** (vCPU compartido, AMD + NVMe). Cuesta
bastante menos y funciona; el riesgo es variabilidad de latencia bajo carga sostenida de ASR.
Si se elige esta, conviene medir p95 con `scripts/eval_endpoint.py` (#8) antes de confiar en ella.

### Mínimo para una demo

**2 vCPU / 4 GB**, con `CPU_THREADS=2` y la etapa de ASR limitada a las llamadas ambiguas.
La etapa 1 (canal, prosodia, latencia de entrada) resuelve la mayoría de casos en ~0.3 s y no
necesita más máquina.

> Las tarifas exactas de cada escalón conviene confirmarlas en la consola: la facturación es
> por horas, así que un fin de semana de hackathon cuesta una fracción del precio mensual.
> Con GPU no hace falta: el servicio es libre de torch a propósito.

## Grupo de firewall

Se configura en Vultr (fuera de la máquina), no con `ufw`: Docker publica sus puertos
saltándose las reglas de `ufw`, y el firewall de Vultr no tiene ese problema.

| Protocolo | Puerto | Origen |
| --- | --- | --- |
| TCP | 22 | Solo las IP del equipo |
| TCP | 80 | Cualquiera |
| TCP | 443 | Cualquiera |

Todo lo demás, denegado. El puerto 8000 **no** se abre: solo lo ve Caddy por la red interna
de Docker.

## Primer despliegue

```bash
ssh despliegue@<ip>
cd /opt/servicio
git checkout stage
cp .env.example .env          # rellenar lo necesario
chmod 600 .env                # solo el usuario de despliegue puede leerlo
docker compose up -d --build
curl -s localhost/health
```

El `.env` vive únicamente en la instancia, nunca en el repositorio. Los contenedores no montan
el directorio del proyecto —sólo `Caddyfile` en modo lectura— así que el servicio recibe las
variables ya resueltas y no puede leer ni reescribir el archivo. Y corre como usuario sin
privilegios (uid 10001), no como root.

## Despliegues siguientes

```bash
cd /opt/servicio && git pull && docker compose up -d --build
```

## TLS

Caddy sin la variable `DOMINIO` escucha en HTTP por el puerto 80, que basta para probar con
la IP. Para HTTPS hacen falta un nombre y una `A` apuntando a la instancia:

```bash
echo "DOMINIO=detect.midominio.com" >> .env && docker compose up -d
```

Sin dominio propio sirve un hostname de DNS comodín, que sí admite certificado real —
por ejemplo `DOMINIO=203-0-113-7.sslip.io` para la IP `203.0.113.7`.

## Antes de la demostración

Tomar un snapshot de la instancia cuando el servicio ya responda. Restaurarlo es más rápido
que depurar a contrarreloj.
