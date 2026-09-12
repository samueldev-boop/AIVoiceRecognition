"""Regenera escenarios de estres exclusivamente sobre las llamadas de train."""

import hashlib
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import joblib
import numpy as np

from analysis.stress_test import bot_evasivo, bot_ritmo, humano_limpio
from app import audio, features, vad
from app.interventions import extract_sample, feature_names

SCENARIOS = ("humano_limpio", "bot_evasivo", "bot_ritmo")


def stress_one(root, record):
    meta = record["meta"]
    if meta["split"] != "train":
        raise ValueError("el banco de seleccion no puede leer val")
    seed = int(hashlib.sha256(meta["anon_id"].encode()).hexdigest()[:8], 16)
    x, sr = audio.leer_wav(str(Path(root) / "audio" / (meta["anon_id"] + ".wav")))
    turns = record["vad_turns"]
    is_bot = meta["label"] == "synthetic"
    variant = "bot_evasivo" if is_bot else "humano_limpio"
    attacked = bot_evasivo(x, turns, seed=seed) if is_bot else humano_limpio(x, turns)
    new_turns = vad.turnos(attacked, sr)
    output = {name: dict(record["samples"]) for name in SCENARIOS}
    cache = {}
    for budget in features.Presupuesto:
        original = record["samples"][budget]
        changed = extract_sample(attacked, sr, budget, turns=new_turns, cache=cache)
        for key in ("call_id", "split", "group", "label"):
            changed[key] = original[key]
        output[variant][budget] = changed
        if is_bot:
            duration = original["audio_used_s"]
            prefix_turns = [
                {**t, "end": min(t["end"], duration)} for t in turns if t["start"] < duration
            ]
            edited = bot_ritmo(prefix_turns, duration, seed=seed)
            behavior = features.conducta(edited, duration)
            call_vector = original["call_features"].copy()
            for j, name in enumerate(feature_names()):
                if features.grupo_de(name) == "conducta":
                    call_vector[j] = behavior.get(name, np.nan)
            output["bot_ritmo"][budget] = {**original, "call_features": call_vector}
    return output


def build_stress(root, records, output, data_fingerprint, jobs=4):
    digest = hashlib.sha256(data_fingerprint.encode())
    for path in (Path(__file__), Path(__file__).resolve().parents[1] / "analysis/stress_test.py"):
        digest.update(path.read_bytes())
    key = digest.hexdigest()
    cache_path = Path(output) / "stress_train.joblib"
    if cache_path.exists():
        cached = joblib.load(cache_path)
        if cached["fingerprint"] == key:
            print("cache de estres verificada", flush=True)
            return cached["scenarios"]
    start = time.perf_counter()
    extracted = {}
    with ProcessPoolExecutor(max_workers=jobs) as executor:
        futures = {executor.submit(stress_one, root, record): i for i, record in enumerate(records)}
        for n, future in enumerate(as_completed(futures), 1):
            extracted[futures[future]] = future.result()
            if n % 20 == 0 or n == len(records):
                print(f"estres {n}/{len(records)}; {time.perf_counter() - start:.0f}s", flush=True)
    scenarios = {
        name: {
            budget: [extracted[i][name][budget] for i in range(len(records))]
            for budget in features.Presupuesto
        }
        for name in SCENARIOS
    }
    joblib.dump({"fingerprint": key, "scenarios": scenarios}, cache_path, compress=3)
    return scenarios
