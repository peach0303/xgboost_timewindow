#!/usr/bin/env python3
"""Build a stable output-directory tag from selected time-window combinations."""
from __future__ import annotations

import argparse
import json
import re
from collections import OrderedDict

from filter_timewindow_manifests import combo_keys


COMBO_PATTERN = re.compile(r"^(.+)_offset(-?\d+)$")
SAFE_TAG_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def unique(values):
    return list(OrderedDict.fromkeys(values))


def selected_combo_keys(manifest: dict, offsets: str, data_types: str, combos: str):
    selected = combo_keys(offsets, data_types, combos)
    if not selected:
        if offsets.strip() or data_types.strip():
            raise ValueError("offset/type filtering requires both --offsets and --data-types")
        selected = list(manifest.get("combinations", {}))
    selected = unique(selected)
    if not selected:
        raise ValueError("manifest has no combinations")
    missing = [key for key in selected if key not in manifest.get("combinations", {})]
    if missing:
        raise ValueError(f"manifest misses combinations: {missing}")
    return selected


def build_tag(keys: list[str]) -> str:
    parsed = []
    for key in keys:
        match = COMBO_PATTERN.fullmatch(key)
        if not match:
            raise ValueError(f"invalid combination key: {key!r}")
        parsed.append((match.group(1), int(match.group(2))))
    data_types = unique(dtype for dtype, _ in parsed)
    canonical = [dtype for dtype in ("tog", "level") if dtype in data_types]
    canonical.extend(sorted(dtype for dtype in data_types if dtype not in {"tog", "level"}))
    safe_types = [re.sub(r"[^A-Za-z0-9.-]+", "-", dtype).strip("-.") for dtype in canonical]
    if any(not value for value in safe_types):
        raise ValueError(f"data types cannot form a safe tag: {data_types}")
    offsets = unique(offset for _, offset in parsed)
    offset_parts = [str(value) if value >= 0 else f"m{abs(value)}" for value in offsets]
    return "_".join(safe_types + offset_parts)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Step3-Step6 output tag")
    parser.add_argument("--tw-manifest", required=True)
    parser.add_argument("--offsets", default="")
    parser.add_argument("--data-types", default="")
    parser.add_argument("--include-combos", default="")
    parser.add_argument("--override", default="")
    args = parser.parse_args()
    if args.override:
        if not SAFE_TAG_PATTERN.fullmatch(args.override) or ".." in args.override:
            parser.error("--override must be one safe directory name without '/' or '..'")
        print(args.override)
        return
    with open(args.tw_manifest, encoding="utf-8") as handle:
        manifest = json.load(handle, object_pairs_hook=OrderedDict)
    try:
        keys = selected_combo_keys(
            manifest, args.offsets, args.data_types, args.include_combos
        )
        print(build_tag(keys))
    except ValueError as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
