"use strict";

// Interfaz sobre POST /detect y, para los WAV, POST /transcribe. La sirve el mismo FastAPI
// que el endpoint, asi que no hay CORS ni un segundo despliegue. Tailwind va vendorizado en
// static/tailwind.js: sin paso de build y sin depender de un CDN durante la demo.

// Orden a proposito: primero las capas robustas, despues las que describen el equipo.
const CAPAS = ["conducta", "razon_canal", "prosodia", "codec_bw", "silencio", "ganancia"];
const ROBUSTAS = new Set(["conducta", "razon_canal", "prosodia"]);

const NOMBRE_ETAPA = {
  first_turn: "1ª intervención",
  "20s": "primeros 20 s",
  full: "llamada completa",
};

const NOMBRE_CANAL = ["llamante", "agente"];

const $ = (id) => document.getElementById(id);
const soltar = $("soltar");
const archivo = $("archivo");
const estado = $("estado");

// Cada archivo abre un analisis nuevo: las respuestas de uno anterior ya no se pintan.
let analisisActual = 0;
// Se apaga para el resto de la sesion si el servidor no tiene clave de ElevenLabs.
let transcripcionDisponible = true;

soltar.addEventListener("click", () => archivo.click());
soltar.addEventListener("keydown", (e) => {
  if (e.key === "Enter" || e.key === " ") { e.preventDefault(); archivo.click(); }
});
archivo.addEventListener("change", () => archivo.files[0] && analizar(archivo.files[0]));

for (const evento of ["dragenter", "dragover"]) {
  soltar.addEventListener(evento, (e) => {
    e.preventDefault();
    soltar.classList.add("border-zinc-900", "bg-zinc-50");
  });
}
for (const evento of ["dragleave", "drop"]) {
  soltar.addEventListener(evento, (e) => {
    e.preventDefault();
    soltar.classList.remove("border-zinc-900", "bg-zinc-50");
  });
}
soltar.addEventListener("drop", (e) => {
  const f = e.dataTransfer?.files?.[0];
  if (f) analizar(f);
});

function mensaje(texto, esError) {
  estado.textContent = texto;
  estado.className = esError
    ? "mt-4 min-h-5 text-sm text-red-700"
    : "mt-4 min-h-5 text-sm text-zinc-600";
}

async function analizar(fichero) {
  const id = ++analisisActual;
  $("resultado").hidden = true;
  mensaje(`Analizando ${fichero.name}…`, false);

  const bytes = await fichero.arrayBuffer();
  if (id !== analisisActual) return;
  const peticion = JSON.stringify({ audio: aBase64(bytes) });
  const deteccion = fetch("detect", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: peticion,
  });

  // La transcripcion va en paralelo y por su cuenta: el veredicto no la espera.
  const conTranscripcion = transcripcionDisponible && esWav(fichero, bytes);
  $("transcripcion").hidden = !conTranscripcion;
  if (conTranscripcion) transcribir(peticion, id);

  let respuesta;
  try {
    respuesta = await deteccion;
  } catch (e) {
    if (id !== analisisActual) return;
    return mensaje(`No se pudo contactar con el servicio: ${e.message}`, true);
  }

  const cuerpo = await respuesta.json().catch(() => null);
  if (id !== analisisActual) return;
  if (!respuesta.ok) {
    return mensaje(`Grabación rechazada: ${cuerpo?.detail ?? `HTTP ${respuesta.status}`}`, true);
  }

  mensaje("", false);
  pintar(cuerpo);
  $("resultado").hidden = false;
  dibujarOnda(bytes, cuerpo).catch((e) => mensaje(`No pude dibujar la onda: ${e.message}`, true));
}

// Base64 por trozos: una llamada de dos minutos son ~7 MB y pasar el array entero a
// String.fromCharCode revienta la pila de argumentos.
function aBase64(buffer) {
  const bytes = new Uint8Array(buffer);
  const trozo = 0x8000;
  let texto = "";
  for (let i = 0; i < bytes.length; i += trozo) {
    texto += String.fromCharCode.apply(null, bytes.subarray(i, i + trozo));
  }
  return btoa(texto);
}

// Solo se transcriben los WAV: extension .wav y cabecera RIFF/WAVE.
function esWav(fichero, buffer) {
  if (!/\.wav$/i.test(fichero.name) || buffer.byteLength < 12) return false;
  const cabecera = String.fromCharCode(...new Uint8Array(buffer, 0, 12));
  return cabecera.startsWith("RIFF") && cabecera.endsWith("WAVE");
}

async function transcribir(peticion, id) {
  estadoTranscripcion("Transcribiendo la llamada…");
  let respuesta;
  try {
    respuesta = await fetch("transcribe", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: peticion,
    });
  } catch (e) {
    if (id === analisisActual) estadoTranscripcion(`No se pudo transcribir: ${e.message}`);
    return;
  }

  const cuerpo = await respuesta.json().catch(() => null);
  if (id !== analisisActual) return;
  // 503: el servidor no tiene clave. 422: grabacion rechazada, y el estado general ya lo dice.
  if (respuesta.status === 503 || respuesta.status === 422) {
    if (respuesta.status === 503) transcripcionDisponible = false;
    $("transcripcion").hidden = true;
    return;
  }
  if (!respuesta.ok) {
    estadoTranscripcion(`No se pudo transcribir: ${cuerpo?.detail ?? `HTTP ${respuesta.status}`}`);
    return;
  }
  pintarTranscripcion(cuerpo.segments);
}

function estadoTranscripcion(texto) {
  $("transcripcion-estado").textContent = texto;
  $("transcripcion-estado").hidden = false;
  $("conversacion").hidden = true;
}

function pintarTranscripcion(segmentos) {
  if (!segmentos.length) {
    estadoTranscripcion("No se transcribió habla en ningún canal.");
    return;
  }
  $("dialogo").replaceChildren(...segmentos.map(intervencion));
  $("transcripcion-estado").hidden = true;
  $("conversacion").hidden = false;
  $("conversacion").scrollTop = 0;
}

// Llamante a la izquierda y agente a la derecha, con los colores de la leyenda de turnos. En
// pantallas estrechas queda una sola columna y cada intervencion lleva el nombre del canal.
function intervencion(s) {
  const esLlamante = s.channel === 0;
  const fila = document.createElement("li");
  fila.className = "grid gap-x-6 sm:grid-cols-2";

  const bloque = document.createElement("div");
  bloque.className = esLlamante
    ? "border-l-4 border-blue-700 bg-blue-50 px-3 py-2"
    : "border-l-4 border-zinc-400 bg-zinc-100 px-3 py-2 sm:col-start-2";

  const cabecera = document.createElement("p");
  cabecera.className = "font-mono text-[11px] text-zinc-500";
  const canal = document.createElement("span");
  canal.className = "sm:hidden";
  canal.textContent = ` · ${NOMBRE_CANAL[s.channel] ?? `canal ${s.channel}`}`;
  cabecera.append(minutos(s.start), canal);

  const texto = document.createElement("p");
  texto.className = `mt-0.5 text-[14px] leading-snug ${esLlamante ? "text-zinc-900" : "text-zinc-700"}`;
  texto.textContent = s.text;

  bloque.append(cabecera, texto);
  fila.append(bloque);
  return fila;
}

function minutos(segundos) {
  const s = Math.floor(segundos);
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
}

function pintar(d) {
  const esBot = d.is_synthetic;
  $("acento").className = `w-1 shrink-0 ${esBot ? "bg-red-700" : "bg-teal-700"}`;
  $("etiqueta").className = `text-2xl font-semibold tracking-tight ${
    esBot ? "text-red-700" : "text-teal-700"}`;
  $("etiqueta").textContent = esBot ? "Agente autónomo" : "Persona";
  $("probabilidad").textContent = d.probability_synthetic.toFixed(3);
  $("confianza").textContent = `${(d.confidence * 100).toFixed(1)} %`;
  $("segundos").textContent = d.audio_used_s == null ? "—" : `${d.audio_used_s.toFixed(1)} s`;
  $("etapa").textContent = NOMBRE_ETAPA[d.stage] ?? d.stage;
  $("ms").textContent = `${Math.round(d.ms)} ms`;

  const aviso = $("aviso");
  const textoAviso = avisoDe(d);
  aviso.hidden = !textoAviso;
  aviso.textContent = textoAviso ?? "";

  $("capas").replaceChildren(
    ...CAPAS.filter((c) => d.layer_scores?.[c] != null)
      .map((c) => barra(c, d.layer_scores[c], ROBUSTAS.has(c) ? "bg-zinc-900" : "bg-zinc-400")),
  );
  $("presupuestos").replaceChildren(
    ...Object.entries(d.budget_scores ?? {})
      .map(([k, v]) => barra(NOMBRE_ETAPA[k] ?? k, v, "bg-zinc-900")),
  );
}

function avisoDe(d) {
  if (d.stage === "sin_habla") return "Sin habla utilizable del llamante. Se abstiene.";
  if (d.stage === "watchdog") return "Tiempo agotado. Se abstiene.";
  if (d.disagreement) return "Los presupuestos se contradicen; la confianza baja a propósito.";
  return null;
}

function barra(nombre, valor, colorRelleno) {
  const fila = document.createElement("div");
  fila.className = "grid grid-cols-[7rem_1fr_3rem] items-center gap-3 text-[13px]";

  const etiqueta = document.createElement("span");
  etiqueta.className = "truncate text-zinc-700";
  etiqueta.textContent = nombre;

  const pista = document.createElement("span");
  pista.className = "h-2 bg-zinc-200";
  const relleno = document.createElement("span");
  relleno.className = `block h-full ${colorRelleno}`;
  relleno.style.width = `${Math.max(1, Math.min(100, valor * 100))}%`;
  pista.append(relleno);

  const numero = document.createElement("span");
  numero.className = "text-right font-mono text-zinc-500";
  numero.textContent = valor.toFixed(3);

  fila.append(etiqueta, pista, numero);
  return fila;
}

// La onda se decodifica en el navegador; el servidor no devuelve audio.
async function dibujarOnda(buffer, d) {
  const ctx = new (window.AudioContext || window.webkitAudioContext)();
  const audio = await ctx.decodeAudioData(buffer.slice(0));
  await ctx.close();

  const lienzo = $("onda");
  const g = lienzo.getContext("2d");
  const { width: ancho, height: alto } = lienzo;
  const medio = alto / 2;
  g.clearRect(0, 0, ancho, alto);

  // Solo si es una franja parcial: cuando escala a la llamada completa cubriria todo el
  // panel y se leeria como un fondo, no como informacion.
  const fraccionUsada = (d.audio_used_s ?? 0) / audio.duration;
  if (fraccionUsada > 0 && fraccionUsada < 0.98) {
    g.fillStyle = "#dbeafe";
    g.fillRect(0, 0, fraccionUsada * ancho, alto);
    g.strokeStyle = "#60a5fa";
    g.beginPath();
    g.moveTo(fraccionUsada * ancho + 0.5, 0);
    g.lineTo(fraccionUsada * ancho + 0.5, alto);
    g.stroke();
  }

  for (let canal = 0; canal < Math.min(2, audio.numberOfChannels); canal++) {
    const datos = audio.getChannelData(canal);
    const base = canal === 0 ? medio * 0.5 : medio * 1.5;
    const escala = medio * 0.4;
    g.strokeStyle = canal === 0 ? "#1d4ed8" : "#a1a1aa";
    g.lineWidth = 1;
    g.beginPath();
    const porPixel = Math.max(1, Math.floor(datos.length / ancho));
    for (let x = 0; x < ancho; x++) {
      let pico = 0;
      const desde = x * porPixel;
      for (let i = desde; i < desde + porPixel && i < datos.length; i++) {
        const v = Math.abs(datos[i]);
        if (v > pico) pico = v;
      }
      g.moveTo(x + 0.5, base - pico * escala);
      g.lineTo(x + 0.5, base + pico * escala);
    }
    g.stroke();
  }

  // Turnos del VAD: barra maciza bajo cada canal.
  for (const t of d.turns ?? []) {
    const x0 = (t.start / audio.duration) * ancho;
    const x1 = (t.end / audio.duration) * ancho;
    g.fillStyle = t.channel === 0 ? "#1d4ed8" : "#a1a1aa";
    g.fillRect(x0, t.channel === 0 ? medio - 7 : alto - 7, Math.max(1, x1 - x0), 5);
  }

  g.strokeStyle = "#e4e4e7";
  g.beginPath();
  g.moveTo(0, medio + 0.5);
  g.lineTo(ancho, medio + 0.5);
  g.stroke();
}
