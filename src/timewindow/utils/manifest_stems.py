#!/usr/bin/env python3
"""Read train/test stems from a manifest and materialize legacy list files."""
from __future__ import annotations

import argparse
import json
import os
from collections.abc import Iterable


def normalize_stems(values: Iterable[object], field: str) -> list[str]:
    stems: list[str] = []
    for value in values:
        if not isinstance(value, str) or not value:
            raise ValueError(f"{field} must contain non-empty strings")
        if any(character.isspace() for character in value):
            raise ValueError(
                f"{field} stem contains whitespace unsupported by csh: {value!r}"
            )
        stems.append(value)
    return stems


def write_stem_lists(
    output_dir: str,
    train_stems: Iterable[object],
    test_stems: Iterable[object],
) -> tuple[str, str]:
    """Write the legacy train_stems.txt/test_stems.txt compatibility files."""
    train = normalize_stems(train_stems, "train_stems")
    test = normalize_stems(test_stems, "test_stems")
    if not train:
        raise ValueError("train_stems must not be empty")
    if not test:
        raise ValueError("test_stems must not be empty")
    os.makedirs(output_dir, exist_ok=True)
    paths = (
        os.path.join(output_dir, "train_stems.txt"),
        os.path.join(output_dir, "test_stems.txt"),
    )
    for path, stems in zip(paths, (train, test)):
        with open(path, "w", encoding="utf-8") as handle:
            handle.write("\n".join(stems) + "\n")
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create legacy *_stems.txt files from a Step1 manifest"
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    with open(args.manifest, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    train_path, test_path = write_stem_lists(
        args.output_dir,
        manifest.get("train_stems", []),
        manifest.get("test_stems", []),
    )
    print(f"wrote {train_path} and {test_path}")


if __name__ == "__main__":
    main()
