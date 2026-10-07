#!/usr/bin/env python3
"""Concatenate block-split fsdb_to_tog_cache manifests for one case."""
from __future__ import annotations

import argparse
import json
import shutil
from collections import OrderedDict
from pathlib import Path

from merge_tog_cache_manifests import load_json, read_csc_ref_stats, read_uint8_npy, mkdirp


def one_key(d: dict) -> str:
    if not d:
        raise ValueError("empty keyed dict")
    return next(iter(d.keys()))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Concatenate block-split TOG cache manifests for one case.")
    p.add_argument("--manifest", action="append", required=True, help="Block manifest. Repeat for each block.")
    p.add_argument("--output-dir", required=True, help="Output directory for concatenated case cache.")
    p.add_argument("--dataset-name", required=True, help="Case/dataset name for final marker.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    manifest_paths = [Path(p).resolve() for p in args.manifest]
    manifests = [load_json(p) for p in manifest_paths]
    if not manifests:
        raise SystemExit("no manifests")

    first = manifests[0]
    output_dir = Path(args.output_dir).resolve()
    work_dir = output_dir / "_work"
    marker_dir = work_dir / "source_markers"
    mkdirp(marker_dir)

    # The exporter writes the full expanded layout/header even for --only-block-id.
    shutil.copyfile(manifest_paths[0].parent / "expanded_layout.jsonl", work_dir / "expanded_layout.jsonl")
    headers = manifest_paths[0].parent.parent / "original_headers.txt"
    if headers.is_file():
        shutil.copyfile(headers, output_dir / "original_headers.txt")

    marker = marker_dir / f"{args.dataset_name}.fsdbtog"
    marker_s = str(marker)
    marker.write_text(
        "source_fsdb={}\n".format(first.get("source_fsdb", ""))
        + "dataset_name={}\n".format(args.dataset_name)
        + "block_split_manifests={}\n".format(len(manifest_paths)),
        encoding="utf-8",
    )

    blocks = []
    clean_names_after_xz = []
    xz_removed_count = 0
    block_split_records = []
    for path, manifest in sorted(zip(manifest_paths, manifests), key=lambda pm: int(pm[1]["blocks"][0]["block_id"])):
        if len(manifest.get("blocks", [])) != 1:
            raise ValueError(f"{path}: expected exactly one block")
        for key in ("sample_start", "sample_end", "sample_step", "drop_first_tog_row", "output_rows", "tog_mode"):
            if manifest.get(key) != first.get(key):
                raise ValueError(f"{path}: {key} mismatch")
        src_block = manifest["blocks"][0]
        old_key = one_key(src_block.get("files", {}))
        out_block = OrderedDict()
        out_block["block_id"] = int(src_block["block_id"])
        out_block["raw_indices"] = list(src_block.get("raw_indices", []))
        out_block["names"] = list(src_block.get("names", []))
        out_block["removed_xz_names"] = list(src_block.get("removed_xz_names", []))
        out_block["files"] = OrderedDict()
        out_block["initial_level_files"] = OrderedDict()
        out_block["n_rows"] = OrderedDict()
        out_block["level_n_rows"] = OrderedDict()
        out_block["nnz"] = OrderedDict()
        out_block["initial_level_nnz"] = OrderedDict()

        file_ref = src_block["files"][old_key]
        rows, cols, nnz = read_csc_ref_stats(file_ref)
        if cols != len(out_block["names"]):
            raise ValueError(f"{path}: block cols={cols}, names={len(out_block['names'])}")
        out_block["files"][marker_s] = file_ref
        out_block["n_rows"][marker_s] = int(rows)
        out_block["nnz"][marker_s] = int(nnz)

        if old_key in src_block.get("initial_level_files", {}):
            init_ref = src_block["initial_level_files"][old_key]
            init_values = read_uint8_npy(Path(init_ref))
            if len(init_values) != len(out_block["names"]):
                raise ValueError(f"{path}: initial vector cols={len(init_values)}, names={len(out_block['names'])}")
            out_block["initial_level_files"][marker_s] = init_ref
            out_block["level_n_rows"][marker_s] = int(src_block.get("level_n_rows", {}).get(old_key, rows + 1))
            out_block["initial_level_nnz"][marker_s] = int(src_block.get("initial_level_nnz", {}).get(old_key, sum(init_values)))

        blocks.append(out_block)
        clean_names_after_xz.extend(out_block["names"])
        xz_removed_count += len(out_block["removed_xz_names"])
        block_split_records.append(OrderedDict([
            ("block_id", int(out_block["block_id"])),
            ("manifest", str(path)),
            ("raw_indices", list(out_block["raw_indices"])),
            ("names", len(out_block["names"])),
            ("removed_xz", len(out_block["removed_xz_names"])),
        ]))

    merged = OrderedDict()
    for key in (
        "cache_format", "input_data_type", "tog_diff_applied", "tog_note",
        "level_cache_available", "level_cache_mode", "level_note", "xz_policy",
        "source_fsdb", "sample_start", "sample_end", "sample_step",
        "sample_end_inclusive", "drop_first_tog_row", "sample_rows", "level_rows",
        "output_rows", "tog_mode",
    ):
        if key in first:
            merged[key] = first[key]
    merged["tog_note"] = "Concatenated block-split FSDB TOG cache; each block was exported by an independent job."
    merged["block_split"] = OrderedDict([("blocks", block_split_records)])
    merged["blocks"] = blocks
    merged["csv_paths"] = [marker_s]
    merged["clean_names_after_xz"] = clean_names_after_xz
    merged["xz_removed_count"] = int(xz_removed_count)
    merged["num_blocks"] = len(blocks)
    merged["raw_signal_count"] = int(first.get("raw_signal_count", 0))

    out_manifest = work_dir / "tog_cache_manifest.json"
    out_manifest.write_text(json.dumps(merged, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"concatenated block manifest: {out_manifest}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
