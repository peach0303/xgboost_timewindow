#!/usr/bin/env python3
"""Concatenate fsdb_to_tog_cache manifests split by time windows for one case."""
from __future__ import annotations

import argparse
import json
import shutil
from collections import OrderedDict
from pathlib import Path

from merge_tog_cache_manifests import (
    load_json,
    load_layout,
    ordered_names_for_block,
    mkdirp,
    read_csc_npz_maybe_sharded,
    subset_csc_and_write,
    read_csc_ref_stats,
    read_uint8_npy,
    write_initial_level_npy,
)


def flatten_ref(ref):
    if isinstance(ref, list):
        out = []
        for item in ref:
            if isinstance(item, list):
                out.extend(item)
            else:
                out.append(item)
        return out
    return [ref]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Concatenate time-window-split TOG cache manifests for one case.")
    p.add_argument("--manifest", action="append", required=True, help="Chunk manifest in time order. Repeat for each chunk.")
    p.add_argument("--output-dir", required=True, help="Output directory for concatenated case cache.")
    p.add_argument("--dataset-name", required=True, help="Case/dataset name for marker and output files.")
    p.add_argument("--max-mem", default="0", help="Reserved for interface compatibility; currently not enforced here.")
    p.add_argument("--max-cycles-per-file", type=int, default=0, help="Row shard output chunks when needed; 0 = auto only.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    manifest_paths = [Path(p).resolve() for p in args.manifest]
    if not manifest_paths:
        raise SystemExit("no manifests")

    manifests = [load_json(p) for p in manifest_paths]
    layouts = [load_layout(p) for p in manifest_paths]
    first = manifests[0]
    n_blocks = len(first.get("blocks", []))
    if any(len(m.get("blocks", [])) != n_blocks for m in manifests):
        raise ValueError("chunk manifests have different block counts")
    if any(m.get("sample_step") != first.get("sample_step") for m in manifests):
        raise ValueError("chunk manifests have different sample_step")
    if any(m.get("drop_first_tog_row") is not True for m in manifests):
        raise ValueError("time-window concat requires chunk manifests generated with drop_first_tog_row=true")

    output_dir = Path(args.output_dir).resolve()
    work_dir = output_dir / "_work"
    cache_dir = work_dir / "tog_cache"
    marker_dir = work_dir / "source_markers"
    mkdirp(cache_dir)
    mkdirp(marker_dir)

    # Preserve the full original layout/header from the first chunk.
    shutil.copyfile(manifest_paths[0].parent / "expanded_layout.jsonl", work_dir / "expanded_layout.jsonl")
    headers = manifest_paths[0].parent.parent / "original_headers.txt"
    if headers.is_file():
        shutil.copyfile(headers, output_dir / "original_headers.txt")

    marker = marker_dir / f"{args.dataset_name}.fsdbtog"
    marker_s = str(marker)
    marker.write_text(
        "source_fsdb={}\n".format(first.get("source_fsdb", ""))
        + "dataset_name={}\n".format(args.dataset_name)
        + "time_window_chunks={}\n".format(len(manifest_paths)),
        encoding="utf-8",
    )

    out_blocks = []
    total_output_rows = 0
    chunk_records = []
    for ci, m in enumerate(manifests):
        rows = int(m.get("output_rows", 0))
        total_output_rows += rows
        chunk_records.append(OrderedDict([
            ("chunk_index", ci),
            ("manifest", str(manifest_paths[ci])),
            ("sample_start", int(m.get("sample_start", 0))),
            ("sample_end", int(m.get("sample_end", 0))),
            ("sample_rows", int(m.get("sample_rows", 0))),
            ("output_rows", rows),
        ]))

    for bi in range(n_blocks):
        base_block = first["blocks"][bi]
        block_id = int(base_block["block_id"])
        raw_order = ordered_names_for_block(base_block, layouts[0])
        clean_sets = []
        removed = set()
        for m in manifests:
            b = m["blocks"][bi]
            clean_sets.append(set(b.get("names", [])))
            removed.update(b.get("removed_xz_names", []))
        clean_names = [name for name in raw_order if all(name in s for s in clean_sets)]
        removed.update(name for name in raw_order if name not in set(clean_names))

        out_block = OrderedDict()
        out_block["block_id"] = block_id
        out_block["raw_indices"] = list(base_block["raw_indices"])
        out_block["names"] = clean_names
        out_block["removed_xz_names"] = [name for name in raw_order if name in removed]
        out_block["files"] = OrderedDict()
        out_block["initial_level_files"] = OrderedDict()
        out_block["n_rows"] = OrderedDict()
        out_block["level_n_rows"] = OrderedDict()
        out_block["nnz"] = OrderedDict()
        out_block["initial_level_nnz"] = OrderedDict()

        chunk_paths = []
        chunk_nnz = 0
        for ci, m in enumerate(manifests):
            b = m["blocks"][bi]
            csv_path = m["csv_paths"][0]
            src_ref = b["files"][csv_path]
            src_names = list(b["names"])
            matrix = read_csc_npz_maybe_sharded(src_ref)
            out_npz = cache_dir / f"{args.dataset_name}.chunk{ci:05d}.block{block_id:05d}.tog.cache.csc.npz"
            paths, nnz = subset_csc_and_write(
                matrix,
                src_names,
                clean_names,
                out_npz,
                max_cycles_per_file=args.max_cycles_per_file,
            )
            chunk_paths.extend(paths)
            chunk_nnz += int(nnz)

        rows, cols, nnz = read_csc_ref_stats(chunk_paths)
        if cols != len(clean_names):
            raise ValueError(f"block {block_id}: concatenated cols={cols}, expected={len(clean_names)}")
        out_block["files"][marker_s] = chunk_paths if len(chunk_paths) > 1 else chunk_paths[0]
        out_block["n_rows"][marker_s] = int(rows)
        out_block["nnz"][marker_s] = int(nnz)

        first_block = first["blocks"][bi]
        first_csv = first["csv_paths"][0]
        if first_csv in first_block.get("initial_level_files", {}):
            init_values = read_uint8_npy(Path(first_block["initial_level_files"][first_csv]))
            init_path = cache_dir / f"{args.dataset_name}.block{block_id:05d}.initial_level.cache.uint8.npy"
            written, init_nnz = write_initial_level_npy(init_values, list(first_block["names"]), clean_names, init_path)
            out_block["initial_level_files"][marker_s] = written
            out_block["level_n_rows"][marker_s] = int(rows) + 1
            out_block["initial_level_nnz"][marker_s] = int(init_nnz)

        out_blocks.append(out_block)

    merged = OrderedDict()
    merged["cache_format"] = "csc"
    merged["input_data_type"] = "tog"
    merged["tog_diff_applied"] = False
    merged["tog_note"] = "Concatenated time-window-split FSDB TOG cache; chunk files are row-stacked in time order."
    merged["level_cache_available"] = bool(first.get("level_cache_available", False))
    merged["level_cache_mode"] = "initial_plus_tog"
    merged["level_note"] = "Initial level comes from the first time chunk; full level is restored from initial level plus concatenated TOG."
    merged["xz_policy"] = first.get("xz_policy", "")
    merged["source_fsdb"] = first.get("source_fsdb", "")
    merged["sample_start"] = int(first.get("sample_start", 0))
    merged["sample_end"] = int(manifests[-1].get("sample_end", first.get("sample_end", 0)))
    merged["sample_step"] = int(first.get("sample_step", 0))
    merged["sample_end_inclusive"] = False
    merged["drop_first_tog_row"] = True
    merged["sample_rows"] = int(total_output_rows) + 1
    merged["level_rows"] = int(total_output_rows) + 1
    merged["output_rows"] = int(total_output_rows)
    merged["tog_mode"] = first.get("tog_mode", "sample-xor")
    merged["time_window_split"] = OrderedDict([
        ("chunk_output_rows", [int(m.get("output_rows", 0)) for m in manifests]),
        ("chunks", chunk_records),
    ])
    merged["blocks"] = out_blocks
    merged["csv_paths"] = [marker_s]
    merged["num_blocks"] = len(out_blocks)

    out_manifest = work_dir / "tog_cache_manifest.json"
    out_manifest.write_text(json.dumps(merged, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"concatenated manifest: {out_manifest}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
