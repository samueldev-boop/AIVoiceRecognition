"""Prepara variantes por epoca y estres real; reutiliza la extraccion de #5."""

import argparse
import hashlib
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import joblib

from app import audio
from app.interventions import extract_sample
from scripts.augmentation import CODECS, augment, resource_files, seed_for
from scripts.issue5_data import assign_groups, build_dataset, read_manifest, speaker_mapping
from scripts.issue5_stress import build_stress


def unpack_resources(resources):
    if (resources / "RIRS_NOISES").is_dir():
        return
    with zipfile.ZipFile(resources / "rirs_noises.zip") as archive:
        for member in archive.infolist():
            if member.filename.startswith(
                ("RIRS_NOISES/pointsource_noises/", "RIRS_NOISES/real_rirs_isotropic_noises/")
            ):
                archive.extract(member, resources)


def prepare_one(root, record, resources, epochs):
    original = record["samples"]["full"]
    if original["split"] != "train":
        raise ValueError("solo train admite augmentacion y seleccion")
    x, sr = audio.leer_wav(str(root / "audio" / f"{original['call_id']}.wav"))
    label = original["label"]

    def extract(purpose, epoch=0, **options):
        changed = augment(
            x, sr, label, resources, seed_for(original["call_id"], purpose, epoch), **options
        )
        sample = extract_sample(changed, sr, "full")
        sample.update({k: original[k] for k in ("call_id", "group", "label", "split")})
        sample["variant"] = f"{purpose}:{epoch}"
        return sample

    training = [extract("epoch", epoch) for epoch in range(epochs)]
    raw = extract("rawboost", use_rawboost=True)
    scenarios = {}
    if label == 0:
        for codec in CODECS:
            scenarios[f"humano_{codec}"] = extract(f"stress:{codec}", codec=codec, heldout=True)
    else:
        scenarios["bot_sala"] = extract("stress:sala", heldout=True)
    scenarios["rawboost"] = extract("stress:rawboost", use_rawboost=True, heldout=True)
    return {"physical": training, "rawboost": [raw], "stress": scenarios}


def prepare(args):
    args.output.mkdir(parents=True, exist_ok=True)
    unpack_resources(args.resources)
    rows, _ = read_manifest(args.root)
    mapping, scope = speaker_mapping(rows, args.speakers)
    data = build_dataset(
        args.root,
        [r for r in rows if r["split"] == "train"],
        args.root / "analysis/issue5",
        args.jobs,
    )
    records = data["records"]
    assign_groups(records, mapping)
    fingerprint = hashlib.sha256(data["fingerprint"].encode())
    for path in (
        Path(__file__),
        Path("scripts/augmentation.py"),
        Path("scripts/rawboost.py"),
        Path("app/audio.py"),
    ):
        fingerprint.update(path.read_bytes())
    fingerprint.update(str(args.epochs).encode())
    fingerprint.update(str([r["samples"]["full"]["group"] for r in records]).encode())
    resources = {}
    for kind in ("noise", "rir"):
        for heldout in (False, True):
            paths = resource_files(args.resources, kind, heldout)
            resources[f"{kind}/{'test' if heldout else 'train'}"] = [p.name for p in paths]
            for path in paths:
                fingerprint.update(f"{path.name}:{path.stat().st_size}".encode())
    key = fingerprint.hexdigest()
    destination = args.output / "dataset.joblib"
    if destination.exists():
        cached = joblib.load(destination)
        if cached["fingerprint"] == key:
            print("cache de augmentacion verificada", flush=True)
            return cached
    legacy = build_stress(
        args.root, records, args.root / "analysis/issue5", data["fingerprint"], args.jobs
    )
    clean = [r["samples"]["full"] for r in records]
    variants, started = {}, time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        futures = {
            pool.submit(prepare_one, args.root, r, args.resources, args.epochs): i
            for i, r in enumerate(records)
        }
        for n, future in enumerate(as_completed(futures), 1):
            variants[futures[future]] = future.result()
            if n % 20 == 0 or n == len(records):
                print(
                    f"augmentacion {n}/{len(records)}; {time.perf_counter() - started:.0f}s",
                    flush=True,
                )
    scenarios = {name: value["full"] for name, value in legacy.items()}
    for name in (*[f"humano_{c}" for c in CODECS], "bot_sala", "rawboost"):
        scenarios[name] = [variants[i]["stress"].get(name, s) for i, s in enumerate(clean)]
    result = {
        "fingerprint": key,
        "clean": clean,
        "scenarios": scenarios,
        "grouping": scope,
        "physical": [variants[i]["physical"] for i in range(len(clean))],
        "rawboost": [variants[i]["rawboost"] for i in range(len(clean))],
        "epochs": args.epochs,
        "resources": resources,
        "resampler": audio.RESAMPLER,
    }
    joblib.dump(result, destination, compress=3)
    return result


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path.cwd())
    p.add_argument("--output", type=Path, default=Path("analysis/issue11"))
    p.add_argument("--resources", type=Path, default=Path("analysis/issue11/resources"))
    p.add_argument(
        "--speakers",
        type=Path,
        default=Path("speaker_groups.csv") if Path("speaker_groups.csv").exists() else None,
    )
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--jobs", type=int, default=4)
    return p


if __name__ == "__main__":
    prepare(parser().parse_args())
