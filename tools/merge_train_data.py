#!/usr/bin/env python3
"""Merge all JSON/JSONL files under a directory into one JSON list."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def load_file(path: Path):
    if path.suffix == ".jsonl":
        rows = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    return [data]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", required=True, type=Path)
    parser.add_argument("--output_file", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shuffle", action="store_true")
    args = parser.parse_args()

    files = sorted(
        list(args.input_dir.glob("*.json")) + list(args.input_dir.glob("*.jsonl"))
    )
    merged = []
    for fp in files:
        try:
            merged.extend(load_file(fp))
            print(f"[OK] {fp.name}: loaded")
        except Exception as exc:  # noqa: BLE001
            print(f"[SKIP] {fp.name}: {exc}")

    if args.shuffle:
        random.seed(args.seed)
        random.shuffle(merged)

    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    with args.output_file.open("w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    print(f"[DONE] {len(files)} files -> {len(merged)} samples -> {args.output_file}")


if __name__ == "__main__":
    main()
