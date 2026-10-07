#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Verify FSDB-generated level/TOG sparse cache manifests.

Checks:
  1. Manifest references exist and NPZ matrices are internally consistent.
  2. Per-block/case shape, row count, column count, and nnz match manifest.
  3. When full level cache is available, verify:
       TOG[row] == Level[row + 1] XOR Level[row]
     for sample-xor/drop-first cache, or the non-drop-first variant if needed.

This script validates the cache files themselves; it does not reopen the FSDB.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.sparse import load_npz


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Verify FSDB -> TOG cache manifest correctness.")
    ap.add_argument("--manifest", required=True, help="Path to _work/tog_cache_manifest.json")
    ap.add_argument("--max-blocks", type=int, default=3, help="Max blocks to verify; 0 means all blocks")
    ap.add_argument("--max-cases", type=int, default=3, help="Max cases/files per block; 0 means all cases")
    ap.add_argument("--random", action="store_true", help="Randomly sample blocks/cases instead of taking the first N")
    ap.add_argument("--seed", type=int, default=20260812)
    ap.add_argument(
        "--check",
        choices=["structure", "xor"],
        default="xor",
        help="structure checks only metadata/NPZ consistency; xor also checks level-to-TOG relation",
    )
    ap.add_argument(
        "--allow-any-change",
        action="store_true",
        help="Do not fail XOR mismatch when manifest tog_mode is any-change; structure checks still run.",
    )
    ap.add_argument("--max-mismatches", type=int, default=20, help="Print at most this many mismatch coordinates")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument(
        "--print-tog-column",
        action="store_true",
        help="Print one TOG cache column instead of running the verifier.",
    )
    ap.add_argument("--block-id", type=int, default=None, help="Block id for --print-tog-column; default: first block")
    ap.add_argument(
        "--case",
        default=None,
        help="Case key for --print-tog-column. Exact match first; otherwise substring match. Default: first case in block.",
    )
    ap.add_argument("--column-index", type=int, default=None, help="0-based column index inside the selected block")
    ap.add_argument("--column-name", default=None, help="Column/signal name inside the selected block")
    ap.add_argument("--row-start", type=int, default=0, help="First row to print for --print-tog-column")
    ap.add_argument("--rows", type=int, default=50, help="Rows to print for --print-tog-column; default: 50")
    ap.add_argument("--all-rows", action="store_true", help="Print all rows for --print-tog-column")
    return ap.parse_args()


def as_list(file_ref) -> list[str]:
    if file_ref is None:
        return []
    if isinstance(file_ref, list):
        return [str(x) for x in file_ref]
    return [str(file_ref)]


def choose(items: list, limit: int, use_random: bool, rng: random.Random) -> list:
    if limit == 0 or limit >= len(items):
        return list(items)
    if use_random:
        idx = sorted(rng.sample(range(len(items)), limit))
        return [items[i] for i in idx]
    return list(items[:limit])


def load_initial_level_ref(file_ref, expected_cols: int):
    paths = as_list(file_ref)
    if len(paths) != 1:
        raise ValueError("initial_level_files must point to one .npy/.npz file, not row shards")
    path = Path(paths[0])
    if not path.is_file():
        raise FileNotFoundError(path)
    if str(path).endswith(".npy"):
        arr = np.load(path).astype(np.uint8, copy=False).reshape(1, -1)
        bad = arr[(arr != 0) & (arr != 1)]
        if bad.size:
            raise ValueError(f"{path}: initial level contains values other than 0/1, sample={bad[:5]}")
        if arr.shape[1] != expected_cols:
            raise ValueError(f"{path}: initial level cols={arr.shape[1]} != names={expected_cols}")
        return sparse.csr_matrix(arr, dtype=np.uint8)
    mat = load_sparse_ref(path)
    if mat.shape[1] != expected_cols:
        raise ValueError(f"{path}: initial level cols={mat.shape[1]} != names={expected_cols}")
    if mat.shape[0] != 1:
        raise ValueError(f"{path}: initial_level_files should have one row, got {mat.shape[0]}")
    return mat.tocsr()


def load_sparse_ref(file_ref):
    paths = [Path(p) for p in as_list(file_ref)]
    if not paths:
        raise ValueError("empty file reference")
    for p in paths:
        if not p.is_file():
            raise FileNotFoundError(p)

    mats = [load_npz(p) for p in paths]
    for p, mat in zip(paths, mats):
        if not sparse.issparse(mat):
            raise ValueError(f"{p}: not a scipy sparse matrix")
        if mat.dtype != np.uint8:
            mat = mat.astype(np.uint8)
        bad = mat.data[(mat.data != 0) & (mat.data != 1)]
        if bad.size:
            raise ValueError(f"{p}: sparse data contains values other than 0/1, sample={bad[:5]}")

    if len(mats) == 1:
        return mats[0].tocsc()
    cols = mats[0].shape[1]
    for p, mat in zip(paths[1:], mats[1:]):
        if mat.shape[1] != cols:
            raise ValueError(f"{p}: row-shard column count mismatch: {mat.shape[1]} != {cols}")
    return sparse.vstack([m.tocsr() for m in mats], format="csc", dtype=np.uint8)


def expected_tog_from_level(level, tog_rows: int):
    level = level.tocsr()
    if level.shape[0] == tog_rows + 1:
        # Normal --drop-first-tog-row mode: TOG rows are transitions 0->1, 1->2, ...
        return ((level[1:] != level[:-1]).astype(np.uint8)).tocsc()
    if level.shape[0] == tog_rows:
        # Non-drop-first mode: exporter keeps row 0 as a dummy all-zero TOG row.
        if level.shape[0] == 0:
            return sparse.csc_matrix(level.shape, dtype=np.uint8)
        diff = (level[1:] != level[:-1]).astype(np.uint8)
        first = sparse.csr_matrix((1, level.shape[1]), dtype=np.uint8)
        return sparse.vstack([first, diff], format="csc", dtype=np.uint8)
    raise ValueError(
        f"cannot align level rows to TOG rows: level_rows={level.shape[0]}, tog_rows={tog_rows}; "
        "expected level_rows == tog_rows + 1 for drop-first, or == tog_rows for dummy-first"
    )


def matrix_diff_report(expected, actual, max_mismatches: int):
    delta = (expected != actual).tocoo()
    rows = delta.row[:max_mismatches].tolist()
    cols = delta.col[:max_mismatches].tolist()
    return int(delta.nnz), list(zip(rows, cols))


def verify_block_case(manifest, block, case_key: str, args) -> tuple[int, int]:
    names = block.get("names", [])
    tog_ref = block.get("files", {}).get(case_key)
    if tog_ref is None:
        raise ValueError(f"block {block.get('block_id')} case missing TOG file: {case_key}")

    tog = load_sparse_ref(tog_ref)
    errors = 0
    xor_checked = 0
    bid = block.get("block_id", "?")

    expected_cols = len(names)
    if tog.shape[1] != expected_cols:
        raise ValueError(f"block {bid} {case_key}: TOG cols={tog.shape[1]} != names={expected_cols}")
    if case_key in block.get("n_rows", {}) and tog.shape[0] != int(block["n_rows"][case_key]):
        raise ValueError(f"block {bid} {case_key}: TOG rows={tog.shape[0]} != manifest n_rows={block['n_rows'][case_key]}")
    if case_key in block.get("nnz", {}) and tog.nnz != int(block["nnz"][case_key]):
        raise ValueError(f"block {bid} {case_key}: TOG nnz={tog.nnz} != manifest nnz={block['nnz'][case_key]}")

    level_ref = None
    level_kind = None
    if case_key in block.get("level_files", {}):
        level_ref = block["level_files"][case_key]
        level_kind = "level_files"
    elif case_key in block.get("initial_level_files", {}):
        level_ref = block["initial_level_files"][case_key]
        level_kind = "initial_level_files"

    if level_ref is not None:
        if level_kind == "initial_level_files":
            level = load_initial_level_ref(level_ref, expected_cols)
            if case_key in block.get("initial_level_nnz", {}) and level.nnz != int(block["initial_level_nnz"][case_key]):
                raise ValueError(
                    f"block {bid} {case_key}: initial level nnz={level.nnz} "
                    f"!= manifest initial_level_nnz={block['initial_level_nnz'][case_key]}"
                )
        else:
            level = load_sparse_ref(level_ref)
            if level.shape[1] != expected_cols:
                raise ValueError(f"block {bid} {case_key}: level cols={level.shape[1]} != names={expected_cols}")
            if case_key in block.get("level_n_rows", {}) and level.shape[0] != int(block["level_n_rows"][case_key]):
                raise ValueError(
                    f"block {bid} {case_key}: level rows={level.shape[0]} != manifest level_n_rows={block['level_n_rows'][case_key]}"
                )
            if case_key in block.get("level_nnz", {}) and level.nnz != int(block["level_nnz"][case_key]):
                raise ValueError(f"block {bid} {case_key}: level nnz={level.nnz} != manifest level_nnz={block['level_nnz'][case_key]}")

        if args.check == "xor" and level_kind == "level_files":
            expected = expected_tog_from_level(level, tog.shape[0])
            if expected.shape != tog.shape:
                raise ValueError(f"block {bid} {case_key}: expected TOG shape={expected.shape} != actual={tog.shape}")
            mismatch_count, coords = matrix_diff_report(expected, tog, args.max_mismatches)
            xor_checked = 1
            if mismatch_count:
                errors += 1
                print(
                    f"[FAIL] block={bid} case={Path(case_key).name} XOR mismatch_count={mismatch_count} "
                    f"sample_coords={coords}",
                    flush=True,
                )
        elif args.check == "xor" and level_kind == "initial_level_files":
            print(
                f"[SKIP-XOR] block={bid} case={Path(case_key).name}: only initial_level_files available; "
                "full level cache is required for direct XOR verification",
                flush=True,
            )
    elif args.check == "xor":
        print(
            f"[SKIP-XOR] block={bid} case={Path(case_key).name}: manifest has no level_files/initial_level_files",
            flush=True,
        )

    if args.verbose or errors == 0:
        print(
            f"[OK] block={bid} case={Path(case_key).name} tog_shape={tog.shape} tog_nnz={tog.nnz} "
            f"level={'yes' if level_ref else 'no'} xor_checked={bool(xor_checked)}",
            flush=True,
        )
    return errors, xor_checked



def find_block(blocks: list[dict], block_id: int | None) -> dict:
    if not blocks:
        raise ValueError("manifest has no blocks")
    if block_id is None:
        return blocks[0]
    for block in blocks:
        if int(block.get("block_id", -1)) == int(block_id):
            return block
    raise ValueError(f"block_id not found: {block_id}")


def find_case_key(block: dict, case_selector: str | None) -> str:
    keys = list(block.get("files", {}).keys())
    if not keys:
        raise ValueError(f"block {block.get('block_id')} has no TOG files")
    if case_selector is None:
        return keys[0]
    if case_selector in block.get("files", {}):
        return case_selector
    matches = [k for k in keys if case_selector in k or case_selector in Path(k).name]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise ValueError(f"case not found in block {block.get('block_id')}: {case_selector}")
    shown = "\n  ".join(matches[:10])
    raise ValueError(f"case selector is ambiguous: {case_selector}\n  {shown}")


def find_column(block: dict, column_index: int | None, column_name: str | None) -> tuple[int, str]:
    names = list(block.get("names", []))
    if column_index is not None and column_name is not None:
        raise ValueError("use only one of --column-index or --column-name")
    if column_name is not None:
        if column_name in names:
            idx = names.index(column_name)
            return idx, names[idx]
        matches = [i for i, name in enumerate(names) if column_name in name]
        if len(matches) == 1:
            idx = matches[0]
            return idx, names[idx]
        if not matches:
            raise ValueError(f"column name not found in block {block.get('block_id')}: {column_name}")
        shown = "\n  ".join(f"{i}: {names[i]}" for i in matches[:20])
        raise ValueError(f"column name selector is ambiguous: {column_name}\n  {shown}")
    if column_index is None:
        raise ValueError("--print-tog-column requires --column-index or --column-name")
    if column_index < 0 or column_index >= len(names):
        raise ValueError(f"column-index out of range: {column_index}; block columns={len(names)}")
    return int(column_index), names[int(column_index)]


def print_tog_column(manifest: dict, args) -> int:
    block = find_block(manifest.get("blocks", []), args.block_id)
    case_key = find_case_key(block, args.case)
    col_idx, col_name = find_column(block, args.column_index, args.column_name)

    tog_ref = block.get("files", {}).get(case_key)
    tog = load_sparse_ref(tog_ref).tocsc()
    col = tog[:, col_idx].tocoo()
    one_rows = col.row.tolist()

    if args.row_start < 0:
        raise ValueError("--row-start must be non-negative")
    if args.rows < 0:
        raise ValueError("--rows must be non-negative")
    if args.row_start >= tog.shape[0] and tog.shape[0] > 0:
        raise ValueError(f"--row-start={args.row_start} outside row count {tog.shape[0]}")

    row_end = tog.shape[0] if args.all_rows else min(tog.shape[0], args.row_start + args.rows)
    one_set = set(r for r in one_rows if args.row_start <= r < row_end)

    print(f"manifest_case: {Path(case_key).name}")
    print(f"block_id: {block.get('block_id')}")
    print(f"tog_ref: {tog_ref}")
    print(f"tog_shape: {tog.shape}  tog_nnz: {tog.nnz}")
    print(f"column_index: {col_idx}")
    print(f"column_name: {col_name}")
    print(f"column_nnz: {len(one_rows)}")
    print(f"one_rows_sample: {one_rows[:200]}")
    print(f"preview rows: {args.row_start}:{row_end}")
    print("row,value")
    for row in range(args.row_start, row_end):
        print(f"{row},{1 if row in one_set else 0}")
    return 0


def main() -> int:
    args = parse_args()
    manifest_path = Path(args.manifest).resolve()
    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)

    if manifest.get("tog_mode") == "any-change" and args.check == "xor" and not args.allow_any_change:
        raise SystemExit(
            "ERROR: manifest tog_mode=any-change, which is not expected to equal level sample XOR. "
            "Use --check structure or --allow-any-change if this is intentional."
        )

    blocks = manifest.get("blocks", [])
    if not blocks:
        raise SystemExit(f"ERROR: no blocks in manifest: {manifest_path}")

    if args.print_tog_column:
        return print_tog_column(manifest, args)

    rng = random.Random(args.seed)
    selected_blocks = choose(blocks, args.max_blocks, args.random, rng)
    total_cases = 0
    total_errors = 0
    total_xor = 0

    print(f"manifest: {manifest_path}")
    print(f"blocks: selected={len(selected_blocks)} total={len(blocks)} cache_format={manifest.get('cache_format')}")
    print(f"level_cache_available: {manifest.get('level_cache_available', False)}")

    for block in selected_blocks:
        files = block.get("files", {})
        case_keys = choose(list(files.keys()), args.max_cases, args.random, rng)
        for case_key in case_keys:
            total_cases += 1
            errors, xor_checked = verify_block_case(manifest, block, case_key, args)
            total_errors += errors
            total_xor += xor_checked

    print("=" * 80)
    print(f"checked block/case pairs: {total_cases}")
    print(f"xor_checked pairs: {total_xor}")
    print(f"errors: {total_errors}")
    if total_errors:
        return 1
    if args.check == "xor" and total_xor == 0:
        print("WARNING: no full level cache was available, so XOR relation was not checked.")
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
