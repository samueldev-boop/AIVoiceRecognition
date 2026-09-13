"""Reproducible dataset preparation with transitive grouping and explicit labels."""

import hashlib
import json
import math
from collections import Counter
from collections.abc import Iterable

from src.schemas import CallAuditRecord

FEATURE_VERSION = "measured-telemetry-v1"
FEATURE_PATHS = (
    ("acoustics", "pitch_mean_hz"),
    ("acoustics", "jitter"),
    ("acoustics", "shimmer"),
    ("acoustics", "spectral_centroid"),
    ("conversation", "caller_response_latency_s"),
    ("conversation", "interruption_count"),
    ("asr", "avg_logprob"),
)


def extract_features(analysis: dict) -> list[float]:
    """Require measured values; zero is valid and missing values are never invented."""
    values = []
    for section, key in FEATURE_PATHS:
        value = (analysis.get(section) or {}).get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"missing measured feature: {section}.{key}")
        if not math.isfinite(value):
            raise ValueError("non-finite measured feature")
        values.append(float(value))
    return values


def prepare_dataset(
    documents: Iterable[dict], seed: int = 57, *, split_registry: dict | None = None
) -> dict:
    """Group BEFORE deduplication, preserving every shared-identity edge.

    Stable hash partitions (70/15/15) avoid random sample splitting. Group membership
    must be recomputed over historical data each run; split changes require review.
    """
    records = sorted(
        (CallAuditRecord.model_validate(d) for d in documents), key=lambda r: r.event_id
    )
    if len({r.event_id for r in records}) != len(records):
        raise ValueError("duplicate event_id in dataset input")
    parents = list(range(len(records)))

    def root(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    owners = {}
    for i, record in enumerate(records):
        entities = [f"call:{record.source}:{record.call_id}"]
        entities.extend(f"group:{g}" for g in record.group_ids)
        if record.input_sha256:
            entities.append(f"audio:{record.input_sha256}")
        for entity in entities:
            if entity in owners:
                parents[root(i)] = root(owners[entity])
            else:
                owners[entity] = i
    component_keys = {}
    component_entities = {}
    for entity, owner in owners.items():
        key = root(owner)
        component_keys[key] = min(component_keys.get(key, entity), entity)
        component_entities.setdefault(key, []).append(hashlib.sha256(entity.encode()).hexdigest())

    registry = dict(split_registry or {})
    assignments = {}
    for component, entities in component_entities.items():
        previous = {registry[e] for e in entities if e in registry}
        if len(previous) > 1:
            raise ValueError("new identity link crosses frozen partitions; review required")
        if previous and not previous <= {"train", "validation", "test"}:
            raise ValueError("invalid split registry")
        group = hashlib.sha256(component_keys[component].encode()).hexdigest()
        bucket = int(hashlib.sha256(f"{seed}:{group}".encode()).hexdigest(), 16) % 100
        split = (
            next(iter(previous))
            if previous
            else ("train" if bucket < 70 else "validation" if bucket < 85 else "test")
        )
        assignments[component] = split
        registry.update(dict.fromkeys(entities, split))

    partitions = {k: [] for k in ("train", "validation", "test")}
    excluded = Counter()
    seen_audio = {}
    for i, record in enumerate(records):
        if record.status_for_training != "ready" or record.errors:
            excluded["not_ready"] += 1
            continue
        if not record.group_ids or not record.input_sha256:
            excluded["missing_identity_or_fingerprint"] += 1
            continue
        label = int(record.label.value == "synthetic")
        if record.input_sha256 in seen_audio:
            if seen_audio[record.input_sha256] != label:
                raise ValueError("conflicting verified labels for identical audio")
            excluded["duplicate_audio"] += 1
            continue
        try:
            values = extract_features(record.analysis.model_dump())
        except ValueError:
            excluded["missing_features"] += 1
            continue
        if record.analysis.audio_duration_s <= 0 or not any(
            turn.channel == 0 and turn.end - turn.start >= 1 for turn in record.analysis.turns
        ):
            excluded["missing_speech"] += 1
            continue
        seen_audio[record.input_sha256] = label
        group = hashlib.sha256(component_keys[root(i)].encode()).hexdigest()
        split = assignments[root(i)]
        partitions[split].append(
            {
                "event_id": record.event_id,
                "group": group,
                "input_sha256": record.input_sha256,
                "label": label,
                "features": values,
            }
        )
    content = {
        "feature_version": FEATURE_VERSION,
        "feature_names": [".".join(p) for p in FEATURE_PATHS],
        "seed": seed,
        "splits": partitions,
    }
    digest = hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()
    return {
        **content,
        "dataset_id": digest,
        "excluded": dict(excluded),
        "distribution": {
            split: dict(Counter(str(r["label"]) for r in rows))
            for split, rows in partitions.items()
        },
        "source_events": len(records),
        "split_registry": registry,
    }


def validate_training_splits(dataset: dict) -> None:
    """Fail closed if an externally supplied dataset leaks groups/audio or lacks classes."""
    if dataset["feature_version"] != FEATURE_VERSION:
        raise ValueError("incompatible feature version")
    if dataset["feature_names"] != [".".join(p) for p in FEATURE_PATHS]:
        raise ValueError("incompatible feature names")
    content = {k: dataset[k] for k in ("feature_version", "feature_names", "seed", "splits")}
    digest = hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()
    if digest != dataset["dataset_id"]:
        raise ValueError("dataset checksum mismatch")
    groups, audio, events = set(), set(), set()
    for split in ("train", "validation", "test"):
        rows = dataset["splits"][split]
        if {r["label"] for r in rows} != {0, 1}:
            raise ValueError(f"{split} requires both verified classes")
        current_groups = {r["group"] for r in rows}
        current_audio = {r["input_sha256"] for r in rows}
        current_events = {r["event_id"] for r in rows}
        if not all(current_groups) or not all(current_audio) or not all(current_events):
            raise ValueError("missing sample identity")
        if groups & current_groups or audio & current_audio or events & current_events:
            raise ValueError("data leakage between partitions")
        if len(current_audio) != len(rows) or len(current_events) != len(rows):
            raise ValueError("duplicates within partition")
        for row in rows:
            if len(row["features"]) != len(FEATURE_PATHS) or not all(
                isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
                for v in row["features"]
            ):
                raise ValueError("invalid feature vector")
        groups.update(current_groups)
        audio.update(current_audio)
        events.update(current_events)
