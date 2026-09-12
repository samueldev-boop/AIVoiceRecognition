"""Transcribe canal 0 (caller) y canal 1 (agente) por separado con faster-whisper.

Cada canal va aparte: el dataset ya viene diarizado por construccion, asi que no
hace falta diarizacion. Guarda un JSON por llamada con palabras y timestamps,
que es lo que despues alimenta las features linguisticas.

uso: transcribe.py [n_por_clase] [modelo]
"""

import csv
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "analysis", "transcripts")
TMP = os.environ.get("TMPDIR", "/tmp")


def extract(path, ch, dst):
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-i", path,
         "-filter_complex", f"[0:a]pan=mono|c0=c{ch}[a]", "-map", "[a]",
         "-ar", "16000", "-ac", "1", dst],
        check=True)


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    size = sys.argv[2] if len(sys.argv) > 2 else "small"
    from faster_whisper import WhisperModel

    os.makedirs(OUT, exist_ok=True)
    rows = list(csv.DictReader(open(os.path.join(ROOT, "manifest.csv"))))
    pick = []
    for lab in ("human", "synthetic"):
        pick += [r for r in rows if r["label"] == lab and r["split"] == "val"][:n]

    print(f"cargando modelo {size} (CPU int8)...", flush=True)
    model = WhisperModel(size, device="cpu", compute_type="int8", cpu_threads=10)

    for r in pick:
        i = r["anon_id"]
        dst = os.path.join(OUT, i + ".json")
        if os.path.exists(dst):
            print("ya existe", i)
            continue
        rec = {"anon_id": i, "label": r["label"], "split": r["split"],
               "duration_s": float(r["duration_s"]), "channels": {}}
        for ch, who in ((0, "caller"), (1, "agent")):
            wav = os.path.join(TMP, f"{i}_c{ch}.wav")
            extract(os.path.join(ROOT, "audio", i + ".wav"), ch, wav)
            t0 = time.time()
            segs, info = model.transcribe(
                wav, language="es", word_timestamps=True,
                vad_filter=True, beam_size=5,
                condition_on_previous_text=False)
            out = []
            for s in segs:
                out.append({
                    "start": round(s.start, 2), "end": round(s.end, 2),
                    "text": s.text.strip(),
                    "avg_logprob": round(s.avg_logprob, 3),
                    "no_speech_prob": round(s.no_speech_prob, 3),
                    "words": [{"w": w.word, "s": round(w.start, 2),
                               "e": round(w.end, 2), "p": round(w.probability, 3)}
                              for w in (s.words or [])],
                })
            rec["channels"][who] = out
            os.remove(wav)
            print(f"  {i} ch{ch} ({who}): {len(out)} segmentos en {time.time()-t0:.0f}s", flush=True)
        json.dump(rec, open(dst, "w"), ensure_ascii=False, indent=1)
        print(" ->", dst, flush=True)


if __name__ == "__main__":
    main()
