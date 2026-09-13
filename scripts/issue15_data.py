"""Carga supervisada de las conversaciones generadas para el issue #15.

El corpus de #15 no sustituye al corpus oficial: sus variaciones se incorporan
solamente al ajuste final. Las variaciones que salen de una misma conversacion
base comparten grupo para que nunca aparenten ser hablantes independientes.
"""

import hashlib
import json
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import soundfile as sf

from app import audio, vad
from app.interventions import EXTRACTOR_VERSION, extract_sample

SYNTHETIC_LABEL = "synthetic_voice_clone"
HUMAN_LABEL = "real_human_recording"
LABELS = {SYNTHETIC_LABEL: 1, HUMAN_LABEL: 0}


def _safe_file(directory: Path, value: str) -> Path:
    """Devuelve un WAV del manifiesto sin permitir rutas fuera del directorio."""
    if not isinstance(value, str) or Path(value).name != value or not value.endswith(".wav"):
        raise ValueError(f"nombre de audio inseguro o invalido: {value!r}")
    path = directory / value
    if not path.is_file():
        raise ValueError(f"falta el audio declarado: {value}")
    return path


def read_manifest(directory: Path) -> tuple[list[dict], dict]:
    """Valida el manifiesto #15 y asigna una familia a cada variacion."""
    directory = Path(directory)
    path = directory / "manifest.json"
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ValueError(f"falta {path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"manifest.json invalido: {error}") from error
    if not isinstance(raw, list) or not raw:
        raise ValueError("manifest.json debe contener una lista no vacia")

    by_file = {}
    for entry in raw:
        if not isinstance(entry, dict):
            raise ValueError("cada entrada del manifiesto debe ser un objeto")
        file = entry.get("file")
        _safe_file(directory, file)
        if file in by_file:
            raise ValueError(f"archivo repetido en manifest.json: {file}")
        if entry.get("label") not in LABELS:
            raise ValueError(f"etiqueta #15 fuera del contrato: {entry.get('label')!r}")
        if entry["label"] == HUMAN_LABEL and entry.get("base_file"):
            raise ValueError("una grabacion humana no puede derivarse de un clon")
        by_file[file] = dict(entry)

    actual = {p.name for p in directory.glob("*.wav")}
    declared = set(by_file)
    if actual != declared:
        missing, extra = sorted(declared - actual), sorted(actual - declared)
        raise ValueError(f"WAV/manifiesto no coinciden; faltan={missing}, no declarados={extra}")

    def family(entry: dict, seen: set[str] | None = None) -> str:
        if entry["label"] == HUMAN_LABEL:
            return f"human:{Path(entry['file']).stem}"
        base = entry.get("base_file")
        if not base:
            return f"clone:{Path(entry['file']).stem}"
        if base not in by_file or by_file[base]["label"] != SYNTHETIC_LABEL:
            raise ValueError(f"base_file sintetico invalido: {base!r}")
        seen = set() if seen is None else seen
        if entry["file"] in seen:
            raise ValueError("ciclo en base_file de manifest.json")
        seen.add(entry["file"])
        return family(by_file[base], seen)

    records = []
    for entry in raw:
        item = dict(entry)
        item["family"] = family(item)
        records.append(item)
    records.sort(key=lambda item: item["file"])
    return records, {
        "files": len(records),
        "labels": dict(Counter(item["label"] for item in records)),
        "families": dict(Counter(item["family"] for item in records)),
        "manifest_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _extract_one(directory: Path, entry: dict) -> dict:
    path = _safe_file(directory, entry["file"])
    info = sf.info(path)
    if info.format != "WAV" or info.subtype != "PCM_16":
        raise ValueError(f"{path.name}: se requiere WAV PCM_16")
    x, sr = audio.leer_wav(str(path))
    duration = audio.validar(x, sr)
    declared_duration = entry.get("duration_s")
    if declared_duration is not None and not np.isclose(
        duration, float(declared_duration), atol=0.020001
    ):
        raise ValueError(
            f"{path.name}: duracion declarada {declared_duration} no coincide con {duration:.3f}"
        )
    turns = vad.turnos(x, sr)
    cache = {}
    full = extract_sample(x, sr, "full", turns=turns, cache=cache)
    samples = {"full": full}
    for budget in ("first_turn", "20s"):
        samples[budget] = extract_sample(x, sr, budget, turns=turns, cache=cache)
    call_id = f"issue15:{Path(entry['file']).stem}"
    group = f"issue15:{entry['family']}"
    label = LABELS[entry["label"]]
    for sample in samples.values():
        sample.update(
            {
                "call_id": call_id,
                "group": group,
                "label": label,
                "split": "train",
                "training_turn_features": full["turn_features"],
                "source": "issue15",
                "source_family": entry["family"],
            }
        )
    return {
        "entry": entry,
        "audit": {
            "duration_s": duration,
            "samplerate": sr,
            "channels": x.shape[1],
            "subtype": info.subtype,
            "caller_sha256": hashlib.sha256(x[:, 0].tobytes()).hexdigest(),
            "vad_caller_turns": sum(turn["channel"] == 0 for turn in turns),
        },
        "samples": samples,
    }


def _fingerprint(directory: Path, records: list[dict]) -> str:
    digest = hashlib.sha256(EXTRACTOR_VERSION.encode())
    for path in (
        Path(__file__),
        Path("app/interventions.py"),
        Path("app/features.py"),
        Path("app/vad.py"),
    ):
        digest.update(path.read_bytes())
    for entry in records:
        path = _safe_file(directory, entry["file"])
        stat = path.stat()
        digest.update(f"{path.name}:{stat.st_size}:{stat.st_mtime_ns}".encode())
    return digest.hexdigest()


def load_issue15(directory: Path, output: Path | None = None, jobs: int = 4) -> dict:
    """Extrae los tres presupuestos y conserva los grupos de procedencia.

    ``output`` es opcional; si se indica, cachea las features para que reentrenar
    el modelo no vuelva a recorrer las ~dos horas de audio.
    """
    directory = Path(directory)
    entries, audit = read_manifest(directory)
    key = _fingerprint(directory, entries)
    cache_path = Path(output) / "issue15_features.joblib" if output else None
    if cache_path and cache_path.exists():
        import joblib

        cached = joblib.load(cache_path)
        if cached.get("fingerprint") == key:
            print(f"issue15: cache verificada: {len(cached['records'])} llamadas", flush=True)
            return cached

    records, excluded = [], []
    with ProcessPoolExecutor(max_workers=max(1, jobs)) as executor:
        futures = {executor.submit(_extract_one, directory, entry): entry for entry in entries}
        for future in as_completed(futures):
            entry = futures[future]
            try:
                records.append(future.result())
            except (OSError, ValueError, KeyError) as error:
                excluded.append({"file": entry["file"], "reason": str(error)})
    if excluded:
        raise ValueError(f"issue15 contiene audios no utilizables: {excluded}")
    records.sort(key=lambda record: record["entry"]["file"])
    result = {
        "fingerprint": key,
        "records": records,
        "audit": {**audit, "excluded": excluded},
    }
    if cache_path:
        import joblib

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(result, cache_path, compress=3)
    return result
