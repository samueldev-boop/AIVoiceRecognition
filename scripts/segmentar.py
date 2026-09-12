"""Exporta los turnos de habla de un WAV estereo a JSON.

Es la version de linea de comandos de app/vad.py, para usar el VAD fuera del endpoint:
preparar datos, inspeccionar una llamada concreta o regenerar los turnos de un lote.

    python scripts/segmentar.py llamada.wav                    # JSON por salida estandar
    python scripts/segmentar.py llamada.wav -o turnos.json
    python scripts/segmentar.py audio/*.wav -o turnos/         # un JSON por llamada
    python scripts/segmentar.py llamada.wav --agresividad 2 --min-silencio 0.5

Formato de salida:

    {"segments": [{"channel": 0, "start": 0.5, "end": 5.3}, ...]}

Los segmentos van ordenados cronologicamente y los tiempos en segundos sobre la rejilla
de 20 ms. channel 0 = quien llama, channel 1 = el agente.
"""

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from app import audio, vad  # noqa: E402


def segmentar(ruta: str, **parametros) -> dict:
    """Devuelve {"segments": [...]} para un WAV del disco."""
    x, sr = audio.leer_wav(ruta)
    audio.validar(x, sr)
    return {"segments": vad.turnos(x, sr, **parametros)}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("wav", nargs="+", help="uno o varios WAV estereo a 8 kHz")
    p.add_argument("-o", "--salida",
                   help="archivo de salida, o directorio si hay varios WAV. "
                        "Sin esto, escribe por salida estandar")
    p.add_argument("--agresividad", type=int, choices=range(4), default=vad.AGRESIVIDAD,
                   help=f"0 permisivo, 3 estricto (por defecto {vad.AGRESIVIDAD})")
    p.add_argument("--min-silencio", type=float, default=vad.MIN_SILENCIO_S, metavar="S",
                   help=f"silencio que cierra un turno (por defecto {vad.MIN_SILENCIO_S})")
    p.add_argument("--min-habla", type=float, default=vad.MIN_HABLA_S, metavar="S",
                   help=f"turno mas corto que esto se descarta (por defecto {vad.MIN_HABLA_S})")
    p.add_argument("--padding", type=float, default=vad.PADDING_S, metavar="S",
                   help=f"margen a cada lado del turno (por defecto {vad.PADDING_S})")
    args = p.parse_args(argv)

    parametros = {
        "agresividad": args.agresividad,
        "min_silencio_s": args.min_silencio,
        "min_habla_s": args.min_habla,
        "padding_s": args.padding,
    }

    varios = len(args.wav) > 1
    if varios and not args.salida:
        p.error("con varios WAV hace falta -o con un directorio")
    if varios:
        os.makedirs(args.salida, exist_ok=True)

    fallos = 0
    for ruta in args.wav:
        try:
            resultado = segmentar(ruta, **parametros)
        except audio.AudioInvalido as e:
            print(f"{ruta}: {e}", file=sys.stderr)
            fallos += 1
            continue

        if not args.salida:
            json.dump(resultado, sys.stdout, ensure_ascii=False, indent=1)
            sys.stdout.write("\n")
            continue

        destino = (os.path.join(args.salida, os.path.splitext(os.path.basename(ruta))[0] + ".json")
                   if varios else args.salida)
        with open(destino, "w") as fh:
            json.dump(resultado, fh, ensure_ascii=False, indent=1)
        print(f"{ruta} -> {destino}  ({len(resultado['segments'])} segmentos)", file=sys.stderr)

    return 1 if fallos else 0


if __name__ == "__main__":
    sys.exit(main())
