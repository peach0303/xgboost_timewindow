#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Inspect stage NPZ files produced by dimentionality_reduction_csv.py.

The script loads one sparse NPZ at a time and prints a small dense preview
for data validation. It can scan an output directory for all saved stages, or
inspect exactly one file with --file.
"""

import argparse
import os
import re
from pathlib import Path

import numpy as np
from scipy.sparse import load_npz


STAGES = ("after_xz_clean", "after_zero", "after_one", "after_duplicate", "final")
STAGE_RE = re.compile(
    r"\.(after_xz_clean|after_zero|after_one|after_duplicate|final)"
    r"(?:\.row(\d+))?\.tog\.(csr|csc)\.npz$"
)


def parse_args():
    ap = argparse.ArgumentParser(
        description="Print 5-column previews from reduction stage sparse NPZ files."
    )
    ap.add_argument(
        "path",
        nargs="?",
        default=None,
        help="output directory or one .npz file. Equivalent to --output-dir/--file by path type.",
    )
    ap.add_argument("--output-dir", default=None, help="directory containing stage NPZ outputs")
    ap.add_argument("--file", default=None, help="only read this single stage .npz file")
    ap.add_argument(
        "--stage",
        choices=STAGES,
        action="append",
        help="stage to inspect. Can be repeated. Default: all stages found.",
    )
    ap.add_argument("--cols", type=int, default=5, help="number of columns to print, default: 5")
    ap.add_argument("--col-start", type=int, default=0, help="0-based first column index, default: 0")
    ap.add_argument("--rows", type=int, default=20, help="number of rows to print, default: 20")
    ap.add_argument("--row-start", type=int, default=0, help="0-based first row index, default: 0")
    ap.add_argument(
        "--all-rows",
        action="store_true",
        help="print all rows from row-start instead of limiting by --rows",
    )
    ap.add_argument(
        "--no-names",
        action="store_true",
        help="do not try to read <stage>.namefile.txt for column names",
    )
    return ap.parse_args()


def infer_stage(path):
    m = STAGE_RE.search(Path(path).name)
    if not m:
        return None, None, None
    stage, row_id, fmt = m.groups()
    return stage, row_id, fmt


def resolve_inputs(args):
    if args.file and args.output_dir:
        raise SystemExit("ERROR: --file and --output-dir cannot be used together")

    path = Path(args.path).expanduser() if args.path else None
    if args.file:
        return "file", Path(args.file).expanduser()
    if args.output_dir:
        return "dir", Path(args.output_dir).expanduser()
    if path:
        return ("dir", path) if path.is_dir() else ("file", path)
    raise SystemExit("ERROR: provide an output directory or one .npz file")


def discover_stage_files(output_dir, selected_stages):
    selected = set(selected_stages or STAGES)
    out = []
    for path in sorted(Path(output_dir).rglob("*.npz")):
        if "_work" in path.parts:
            continue
        stage, row_id, fmt = infer_stage(path)
        if stage in selected:
            out.append((stage, row_id, fmt, path))
    return out


def read_column_names(npz_path, stage, no_names):
    if no_names or not stage:
        return None
    namefile = Path(npz_path).parent / f"{stage}.namefile.txt"
    if not namefile.is_file():
        return None
    with open(namefile, "r", encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]


def clip_window(shape, row_start, rows, all_rows, col_start, cols):
    n_rows, n_cols = shape
    if row_start < 0 or col_start < 0:
        raise SystemExit("ERROR: --row-start and --col-start must be non-negative")
    if row_start >= n_rows:
        raise SystemExit(f"ERROR: row_start={row_start} is outside matrix rows={n_rows}")
    if col_start >= n_cols:
        raise SystemExit(f"ERROR: col_start={col_start} is outside matrix cols={n_cols}")

    row_end = n_rows if all_rows else min(n_rows, row_start + max(0, rows))
    col_end = min(n_cols, col_start + max(0, cols))
    return row_end, col_end


def format_cell(value):
    if isinstance(value, np.generic):
        value = value.item()
    return str(value)


def print_preview(path, args):
    stage, row_id, fmt = infer_stage(path)
    if args.stage and stage not in set(args.stage):
        return

    mat = load_npz(path)
    if mat.shape[0] == 0 or mat.shape[1] == 0:
        print("=" * 100)
        print(f"file: {path}")
        print(f"stage: {stage or 'unknown'}  row_shard: {row_id or '-'}  format: {fmt or mat.getformat()}")
        print(f"shape: {mat.shape}  nnz: {mat.nnz}")
        print("EMPTY matrix, no data to print")
        return

    row_end, col_end = clip_window(
        mat.shape, args.row_start, args.rows, args.all_rows, args.col_start, args.cols
    )
    names = read_column_names(path, stage, args.no_names)
    if names:
        headers = names[args.col_start:col_end]
    else:
        headers = [f"col_{i}" for i in range(args.col_start, col_end)]

    preview = mat[args.row_start:row_end, args.col_start:col_end].toarray()
    print("=" * 100)
    print(f"file: {path}")
    print(f"stage: {stage or 'unknown'}  row_shard: {row_id or '-'}  format: {fmt or mat.getformat()}")
    print(f"shape: {mat.shape}  nnz: {mat.nnz}")
    print(f"preview rows: {args.row_start}:{row_end}  cols: {args.col_start}:{col_end}")
    print(",".join(["row"] + headers))
    for i, row in enumerate(preview):
        # Row-shard filenames do not encode the global row offset, so this is
        # always the row index inside the loaded NPZ file.
        row_no = args.row_start + i
        print(",".join([str(row_no)] + [format_cell(v) for v in row]))


def main():
    args = parse_args()
    mode, target = resolve_inputs(args)
    if args.cols <= 0:
        raise SystemExit("ERROR: --cols must be positive")
    if args.rows < 0:
        raise SystemExit("ERROR: --rows must be non-negative")

    if mode == "file":
        if not target.is_file():
            raise SystemExit(f"ERROR: file not found: {target}")
        print_preview(target, args)
        return

    if not target.is_dir():
        raise SystemExit(f"ERROR: output directory not found: {target}")
    files = discover_stage_files(target, args.stage)
    if not files:
        stages = ",".join(args.stage or STAGES)
        raise SystemExit(f"ERROR: no stage NPZ files found under {target} for stage(s): {stages}")
    for _stage, _row_id, _fmt, path in files:
        print_preview(path, args)


if __name__ == "__main__":
    main()
