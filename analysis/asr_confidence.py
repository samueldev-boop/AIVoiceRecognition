"""Mide a escala la confianza del ASR en el canal del caller.

Hipotesis: el habla sintetica (TTS, sin ruido, prosodia canonica) se transcribe
con MAS confianza que el habla humana por telefono. Si aguanta, es una feature
casi gratis y muy dificil de falsificar sin degradar la propia voz del bot.

Transcribe solo el canal 0 para que sea barato. Guarda un CSV incremental.
"""

import csv
import json
import os
import random
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "analysis", "asr_caller.csv")
TMP = os.environ.get("TMPDIR", "/tmp")
FILLERS = ("eh", "este", "mmm", "mm", "ah", "o sea", "pues", "digo", "perdon",
           "perdón", "no se", "no sé", "bueno")


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    from faster_whisper import WhisperModel

    rows = list(csv.DictReader(open(os.path.join(ROOT, "manifest.csv"))))
    random.seed(11)
    pick = []
    for lab in ("human", "synthetic"):
        sub = [r for r in rows if r["label"] == lab]
        random.shuffle(sub)
        pick += sub[:n]
    random.shuffle(pick)

    done = set()
    if os.path.exists(OUT):
        done = {r["anon_id"] for r in csv.DictReader(open(OUT))}
    fh = open(OUT, "a", newline="")
    w = csv.writer(fh)
    if not done:
        w.writerow(["anon_id", "label", "split", "n_seg", "n_words", "speech_s",
                    "logprob_mean", "logprob_std", "wordp_mean", "wordp_p10",
                    "no_speech_mean", "wpm", "filler_rate", "pause_mean",
                    "pause_p90", "uniq_ratio", "digit_rate", "guion"])

    model = WhisperModel("small", device="cpu", compute_type="int8", cpu_threads=10)
    t00 = time.time()
    for k, r in enumerate(pick):
        i = r["anon_id"]
        if i in done:
            continue
        wav = os.path.join(TMP, f"asr_{i}.wav")
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i",
                        os.path.join(ROOT, "audio", i + ".wav"),
                        "-filter_complex", "[0:a]pan=mono|c0=c0[a]", "-map", "[a]",
                        "-ar", "16000", "-ac", "1", wav], check=True)
        segs, _ = model.transcribe(wav, language="es", word_timestamps=True,
                                   vad_filter=True, beam_size=5,
                                   condition_on_previous_text=False)
        segs = list(segs)
        os.remove(wav)
        words = [x for s in segs for x in (s.words or [])]
        if len(words) < 8:
            continue
        import statistics as st
        lp = [s.avg_logprob for s in segs]
        ns = [s.no_speech_prob for s in segs]
        wp = sorted(x.probability for x in words)
        speech = sum(s.end - s.start for s in segs)
        text = " ".join(s.text for s in segs).lower()
        toks = [t.strip(".,¿?¡!;:") for t in text.split()]
        pauses = [words[j + 1].start - words[j].end for j in range(len(words) - 1)]
        pauses = [p for p in pauses if 0 <= p < 3]
        w.writerow([i, r["label"], r["split"], len(segs), len(words), round(speech, 1),
                    round(st.mean(lp), 4), round(st.pstdev(lp), 4) if len(lp) > 1 else 0,
                    round(sum(wp) / len(wp), 4), round(wp[len(wp) // 10], 4),
                    round(st.mean(ns), 4), round(len(words) / speech * 60, 1),
                    round(sum(1 for t in toks if t in FILLERS) / len(toks), 4),
                    round(st.mean(pauses), 4) if pauses else 0,
                    round(sorted(pauses)[int(.9 * (len(pauses) - 1))], 4) if len(pauses) > 3 else 0,
                    round(len(set(toks)) / len(toks), 4),
                    round(sum(1 for t in toks if any(c.isdigit() for c in t)) / len(toks), 4),
                    int("guion" in text or "guión" in text)])
        fh.flush()
        print(f"  [{k+1}/{len(pick)}] {i} {r['label']:<10} "
              f"logprob={st.mean(lp):+.3f} palabras={len(words)} "
              f"({time.time()-t00:.0f}s)", flush=True)
    fh.close()


if __name__ == "__main__":
    main()
