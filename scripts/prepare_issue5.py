"""Solo prepara train; val se extrae al final de train_issue5.py."""

import argparse
from pathlib import Path

from scripts.issue5_data import (
    assign_groups,
    build_dataset,
    read_manifest,
    speaker_mapping,
    summary,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, default=Path("analysis/issue5"))
    parser.add_argument("--speakers", type=Path)
    parser.add_argument("--allow-call-groups", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()
    rows, _ = read_manifest(args.root)
    mapping, scope = speaker_mapping(rows, args.speakers, args.allow_call_groups)
    data = build_dataset(
        args.root, [r for r in rows if r["split"] == "train"], args.output, args.jobs
    )
    assign_groups(data["records"], mapping)
    print(scope)
    print(summary(data["records"]))


if __name__ == "__main__":
    main()
