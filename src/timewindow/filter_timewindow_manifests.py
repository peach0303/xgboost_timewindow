#!/usr/bin/env python3
"""Filter Step1/Step2 time-window manifests to a subset of combinations."""
from __future__ import annotations

import argparse
import json
import os
from collections import OrderedDict


def parse_csv(text: str) -> list[str]:
    return [x.strip() for x in text.split(',') if x.strip()]


def combo_keys(offsets: str, data_types: str, combos: str) -> list[str]:
    explicit = parse_csv(combos)
    if explicit:
        return explicit
    offs = [int(x) for x in parse_csv(offsets)]
    dtypes = parse_csv(data_types)
    return [f"{dtype}_offset{offset}" for offset in offs for dtype in dtypes]


def filter_manifest(manifest: OrderedDict, keep: list[str], source_path: str) -> OrderedDict:
    combos = manifest.get('combinations', {})
    missing = [key for key in keep if key not in combos]
    if missing:
        raise SystemExit(f"ERROR: missing combinations in {source_path}: {missing}")

    out = OrderedDict((k, v) for k, v in manifest.items() if k != 'combinations')
    if 'offsets' in out:
        out['offsets'] = sorted({int(key.rsplit('_offset', 1)[1]) for key in keep}, reverse=True)
    if 'data_types' in out:
        seen = []
        for key in keep:
            dtype = key.split('_offset', 1)[0]
            if dtype not in seen:
                seen.append(dtype)
        out['data_types'] = seen
    out['combinations'] = OrderedDict((key, combos[key]) for key in keep)
    out['filtered_from'] = os.path.abspath(source_path)
    out['filtered_combinations'] = keep
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description='Filter Step1/Step2 manifests by offset/type or explicit combo list')
    ap.add_argument('--tw-manifest-in', required=True)
    ap.add_argument('--mcp-manifest-in', required=True)
    ap.add_argument('--tw-manifest-out', required=True)
    ap.add_argument('--mcp-manifest-out', required=True)
    ap.add_argument('--offsets', default='')
    ap.add_argument('--data-types', default='')
    ap.add_argument('--include-combos', default='')
    args = ap.parse_args()

    keep = combo_keys(args.offsets, args.data_types, args.include_combos)
    if not keep:
        raise SystemExit('ERROR: empty subset; set --include-combos or both --offsets/--data-types')

    with open(args.tw_manifest_in, 'r', encoding='utf-8') as f:
        tw = json.load(f, object_pairs_hook=OrderedDict)
    with open(args.mcp_manifest_in, 'r', encoding='utf-8') as f:
        mcp = json.load(f, object_pairs_hook=OrderedDict)

    tw_out = filter_manifest(tw, keep, args.tw_manifest_in)
    mcp_out = filter_manifest(mcp, keep, args.mcp_manifest_in)

    os.makedirs(os.path.dirname(os.path.abspath(args.tw_manifest_out)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.mcp_manifest_out)), exist_ok=True)
    with open(args.tw_manifest_out, 'w', encoding='utf-8') as f:
        json.dump(tw_out, f, indent=2, ensure_ascii=False)
    with open(args.mcp_manifest_out, 'w', encoding='utf-8') as f:
        json.dump(mcp_out, f, indent=2, ensure_ascii=False)

    print(f"filtered {len(keep)} combinations")
    print(f"  tw : {args.tw_manifest_out}")
    print(f"  mcp: {args.mcp_manifest_out}")


if __name__ == '__main__':
    main()
