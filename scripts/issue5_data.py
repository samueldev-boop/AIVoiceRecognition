"""Auditoria y extraccion reproducible. Nunca modifica el dataset distribuido."""

import csv
import hashlib
import json
import re
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import joblib
import numpy as np
import soundfile as sf
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from app import audio, features, vad
from app.interventions import EXTRACTOR_VERSION, extract_sample, feature_names


def read_manifest(root):
    path = Path(root) / "manifest.csv"
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"anon_id", "label", "split", "duration_s"}
        if not required <= set(reader.fieldnames or []):
            raise ValueError(f"manifest.csv necesita {sorted(required)}")
        raw = list(reader)
    unique = {}
    duplicates = 0
    for row in raw:
        if not re.fullmatch(r"[A-Za-z0-9_-]+", row["anon_id"]):
            raise ValueError("anon_id vacio o inseguro")
        if row["label"] not in ("human", "synthetic") or row["split"] not in ("train", "val"):
            raise ValueError("label/split nulo o fuera del contrato")
        if not np.isfinite(float(row["duration_s"])) or float(row["duration_s"]) <= 0:
            raise ValueError("duracion nula o invalida")
        if row["anon_id"] in unique:
            if unique[row["anon_id"]] != row:
                raise ValueError("anon_id duplicado con metadatos contradictorios")
            duplicates += 1
        unique[row["anon_id"]] = row
    return list(unique.values()), {
        "manifest_rows": len(raw),
        "exact_manifest_duplicates": duplicates,
        "manifest_nulls": sum(not v for r in raw for v in r.values()),
        "manifest_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def speaker_mapping(rows, path=None, allow_call_groups=True):
    if path is None:
        return {}, "call"
    with Path(path).open(newline="") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
        entity_fields = sorted(fields & {"speaker_id", "voice_id"})
        if "anon_id" not in fields or not entity_fields:
            raise ValueError("el mapa necesita anon_id y speaker_id y/o voice_id")
        mapping = {}
        for row in reader:
            if row["anon_id"] in mapping:
                raise ValueError("anon_id duplicado en el mapa de hablantes")
            entities = tuple(f"{k}:{row[k]}" for k in entity_fields if row.get(k))
            if not entities:
                raise ValueError("identidad de grupo vacia; no se inventan hablantes")
            mapping[row["anon_id"]] = entities
    if any(r["anon_id"] not in mapping for r in rows):
        raise ValueError("el mapa de hablantes no cubre todas las llamadas")
    owners = {}
    for row in rows:
        for entity in mapping[row["anon_id"]]:
            if entity in owners and owners[entity] != row["split"]:
                raise ValueError("un hablante/voz aparece en train y val")
            owners[entity] = row["split"]
    return mapping, "connected components of call, speaker_id/voice_id and exact caller PCM"


def assign_groups(records, mapping):
    """Componentes conexas: una llamada o una voz compartida nunca cruza un fold."""
    owners, a, b = {}, [], []
    for i, record in enumerate(records):
        entities = (
            f"pcm:{record['audit']['caller_sha256']}",
            *mapping.get(record["meta"]["anon_id"], ()),
        )
        for entity in entities:
            if entity in owners:
                a.extend([i, owners[entity]])
                b.extend([owners[entity], i])
            else:
                owners[entity] = i
    graph = coo_matrix((np.ones(len(a)), (a, b)), shape=(len(records), len(records))).tocsr()
    _, components = connected_components(graph, directed=False)
    for component in set(components):
        members = np.flatnonzero(components == component)
        splits = {records[i]["meta"]["split"] for i in members}
        if len(splits) > 1:
            raise ValueError("audio/voz duplicada cruza train y val")
        key = min(records[i]["meta"]["anon_id"] for i in members)
        for i in members:
            for sample in records[i]["samples"].values():
                sample["group"] = key


def fingerprint(root, rows):
    root = Path(root)
    digest = hashlib.sha256(EXTRACTOR_VERSION.encode())
    for name in ("features.py", "interventions.py", "vad.py", "intervalos.py"):
        digest.update((Path(__file__).resolve().parents[1] / "app" / name).read_bytes())
    digest.update(Path(__file__).read_bytes())
    digest.update(json.dumps(rows, sort_keys=True).encode())
    for row in rows:
        for folder, suffix in (("audio", ".wav"), ("turns", ".json")):
            path = root / folder / (row["anon_id"] + suffix)
            if path.exists():
                stat = path.stat()
                digest.update(f"{path}:{stat.st_size}:{stat.st_mtime_ns}".encode())
    return digest.hexdigest()


def extract_one(root, meta):
    root = Path(root)
    path = root / "audio" / (meta["anon_id"] + ".wav")
    info = sf.info(path)
    if info.format != "WAV" or info.subtype != "PCM_16":
        raise ValueError("WAV PCM_16 requerido")
    x, sr = audio.leer_wav(str(path))
    duration = audio.validar(x, sr)
    turns = vad.turnos(x, sr)
    reference_path = root / "turns" / (meta["anon_id"] + ".json")
    reference = json.loads(reference_path.read_text())["turns"] if reference_path.exists() else []
    invalid = [
        t
        for t in reference
        if t.get("channel") not in (0, 1)
        or not np.isfinite(t.get("start", np.nan))
        or not np.isfinite(t.get("end", np.nan))
        or not 0 <= t["start"] < t["end"] <= duration + 0.02
    ]
    c0 = x[:, 0].astype(np.int32)
    audit = {
        "duration_s": duration,
        "declared_duration_s": float(meta["duration_s"]),
        "duration_error_s": duration - float(meta["duration_s"]),
        "samplerate": sr,
        "channels": x.shape[1],
        "subtype": info.subtype,
        "pcm_sha256": hashlib.sha256(x.tobytes()).hexdigest(),
        "caller_sha256": hashlib.sha256(x[:, 0].tobytes()).hexdigest(),
        "zero_fraction": float(np.mean(c0 == 0)),
        "clipping_fraction": float(np.mean(np.abs(c0) >= 32700)),
        "peak_dbfs": float(20 * np.log10(max(1, np.max(np.abs(c0))) / 32768)),
        "reference_missing": not reference_path.exists(),
        "reference_invalid_turns": len(invalid),
        "reference_caller_durations": [
            t["end"] - t["start"] for t in reference if t["channel"] == 0 and t not in invalid
        ],
        "vad_caller_durations": [t["end"] - t["start"] for t in turns if t["channel"] == 0],
        "vad_agent_turns": sum(t["channel"] == 1 for t in turns),
    }
    cache = {}
    full = extract_sample(x, sr, "full", turns=turns, cache=cache)
    samples = {"full": full}
    for budget in ("first_turn", "20s"):
        samples[budget] = extract_sample(x, sr, budget, turns=turns, cache=cache)
    for sample in samples.values():
        sample.update(
            {
                "call_id": meta["anon_id"],
                "split": meta["split"],
                "label": int(meta["label"] == "synthetic"),
                "training_turn_features": full["turn_features"],
            }
        )
    return {"meta": meta, "audit": audit, "samples": samples, "vad_turns": turns}


def build_dataset(root, rows, output, jobs=4, refresh=False):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    key = fingerprint(root, rows)
    cache_path = output / f"features_{rows[0]['split']}.joblib"
    if cache_path.exists() and not refresh:
        cached = joblib.load(cache_path)
        if cached["fingerprint"] == key:
            print(f"cache verificada: {len(cached['records'])} llamadas", flush=True)
            return cached
    records, excluded = [], []
    start = time.perf_counter()
    with ProcessPoolExecutor(max_workers=jobs) as executor:
        futures = {executor.submit(extract_one, root, row): row for row in rows}
        for n, future in enumerate(as_completed(futures), 1):
            row = futures[future]
            try:
                records.append(future.result())
            except (ValueError, OSError, KeyError, json.JSONDecodeError) as error:
                excluded.append({**row, "reason": str(error)})
            if n % 20 == 0 or n == len(rows):
                print(
                    f"extraccion {n}/{len(rows)}; {time.perf_counter() - start:.0f}s; "
                    f"excluidas={len(excluded)}",
                    flush=True,
                )
    records.sort(key=lambda r: r["meta"]["anon_id"])
    seen, deduplicated = {}, []
    for record in records:
        digest = record["audit"]["pcm_sha256"]
        if digest in seen:
            if seen[digest]["label"] != record["meta"]["label"]:
                raise ValueError("audio identico con etiquetas contradictorias")
            excluded.append({**record["meta"], "reason": "exact PCM duplicate"})
        else:
            seen[digest] = record["meta"]
            deduplicated.append(record)
    result = {
        "fingerprint": key,
        "records": deduplicated,
        "excluded": excluded,
        "extraction_seconds": time.perf_counter() - start,
    }
    if not deduplicated:
        raise ValueError(f"no hay llamadas utilizables: {excluded[:3]}")
    joblib.dump(result, cache_path, compress=3)
    return result


def export_tables(records, output):
    output = Path(output)
    names = feature_names()
    with (output / "interventions.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["anon_id", "label", "split", "group", "turn_index", "start_s", "end_s", *names]
        )
        for record in records:
            sample = record["samples"]["full"]
            for i, (bounds, row) in enumerate(
                zip(sample["turn_bounds"], sample["turn_features"], strict=True)
            ):
                writer.writerow(
                    [
                        sample["call_id"],
                        sample["label"],
                        sample["split"],
                        sample["group"],
                        i,
                        *bounds,
                        *[v if np.isfinite(v) else "" for v in row],
                    ]
                )
    for budget in features.Presupuesto:
        with (output / f"calls_{budget}.csv").open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["anon_id", "label", "split", "group", *names])
            for record in records:
                s = record["samples"][budget]
                writer.writerow(
                    [
                        s["call_id"],
                        s["label"],
                        s["split"],
                        s["group"],
                        *[v if np.isfinite(v) else "" for v in s["call_features"]],
                    ]
                )
    with (output / "audio_audit.csv").open("w", newline="") as handle:
        keys = [k for k in records[0]["audit"] if not k.endswith("durations")]
        writer = csv.writer(handle)
        writer.writerow(["anon_id", "label", "split", *keys])
        for record in records:
            writer.writerow(
                [record["meta"][k] for k in ("anon_id", "label", "split")]
                + [record["audit"][k] for k in keys]
            )


def summary(records):
    counts = Counter((r["meta"]["split"], r["meta"]["label"]) for r in records)
    return {
        "calls": len(records),
        "counts": {f"{s}/{y}": n for (s, y), n in counts.items()},
        "hours": sum(r["audit"]["duration_s"] for r in records) / 3600,
        "reference_turns": sum(len(r["audit"]["reference_caller_durations"]) for r in records),
        "vad_turns": sum(len(r["audit"]["vad_caller_durations"]) for r in records),
        "no_caller_calls": sum(not r["audit"]["vad_caller_durations"] for r in records),
        "reference_invalid_turns": sum(r["audit"]["reference_invalid_turns"] for r in records),
        "reference_missing": sum(r["audit"]["reference_missing"] for r in records),
        "duration_mismatches_gt_20ms": sum(
            abs(r["audit"]["duration_error_s"]) > 0.020001 for r in records
        ),
    }
