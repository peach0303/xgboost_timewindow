#!/usr/bin/env python3
"""
Step 1: Time-Window Data Preparation.

Reads reduced LEVEL and TOG sparse NPZ files, constructs time-shifted
feature/label pairs for each (offset, data_type) combination.

Key alignment: LEVEL[t+1] ↔ TOG[t] ↔ label[t] (same time point)
For label[i] at offset k: TOG_row = i-k, LEVEL_row = i+1-k
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from collections import OrderedDict, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from scipy.sparse import load_npz, save_npz, vstack as sparse_vstack

from utils.sampling_policy import (
    assign_joint_bins,
    choose_2d_sample_indices,
    delta_power,
    quantile_edges,
    recommend_joint_weights,
)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def mkdirp(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def write_filelist(path: str, items: list[str]) -> None:
    mkdirp(os.path.dirname(os.path.abspath(path)))
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(item + "\n")


def choose_power_sample_indices(
    y: np.ndarray,
    max_rows: int,
    bins: int,
    min_per_bin: int,
    seed: int,
) -> np.ndarray:
    """Compatibility wrapper: select in joint (power, delta-power) space."""
    n = int(len(y))
    if max_rows <= 0 or max_rows >= n:
        return np.arange(n, dtype=np.int64)
    if n == 0:
        return np.array([], dtype=np.int64)

    delta = delta_power(y)
    selected, _ = choose_2d_sample_indices(
        y, delta, max_rows, quantile_edges(y, bins), quantile_edges(delta, bins),
        max(1, min_per_bin), seed,
    )
    return selected


def build_shifted_data(
    level_dir: str,
    tog_dir: str,
    label_dir: str,
    output_dir: str,
    offsets: list[int],
    data_types: list[str],
    boundary_discard: int,
    test_count: int,
    stage: str = "after_one",
    sample_rows_per_file: int = 0,
    sample_bins: int = 20,
    sample_min_per_bin: int = 1,
    sample_seed: int = 108,
    jobs: int = 1,
    include_stems: list[str] | None = None,
    sample_index_manifest: str | None = None,
    compress_output: bool = True,
    skip_existing: bool = True,
) -> dict:
    """
    For each source file (matched by stem), construct shifted feature/label
    pairs for every (offset, data_type). Returns a manifest dict.
    """
    level_dir = os.path.abspath(level_dir)
    tog_dir = os.path.abspath(tog_dir)
    label_dir = os.path.abspath(label_dir)

    # ---- Row-shard-aware file discovery ----
    # Supports:
    #   single:  {stem}.{stage}.tog.csr.npz
    #   sharded: {stem}.{stage}.row00000.tog.csr.npz
    _single_pat = re.compile(rf"^(.+)\.{re.escape(stage)}\.tog\.csr\.npz$")
    _shard_pat = re.compile(rf"^(.+)\.{re.escape(stage)}\.row(\d+)\.tog\.csr\.npz$")
    _merge_prefix_pat = re.compile(r"^m\d{3}_")  # m000_, m001_, etc.

    # group files by stem: stem -> {level|tog: [paths]}
    stem_files: dict[str, dict[str, list[str]]] = defaultdict(lambda: {"level": [], "tog": []})

    for dir_key, scan_dir in [("level", level_dir), ("tog", tog_dir)]:
        if not os.path.isdir(scan_dir):
            continue
        for fname in os.listdir(scan_dir):
            m_single = _single_pat.match(fname)
            m_shard = _shard_pat.match(fname)
            if m_single:
                stem = m_single.group(1)
                stem_files[stem][dir_key].append(os.path.join(scan_dir, fname))
            elif m_shard:
                stem = m_shard.group(1)
                stem_files[stem][dir_key].append(os.path.join(scan_dir, fname))

    # Sort shards within each stem so vstack order is deterministic
    for stem in stem_files:
        for dk in ("level", "tog"):
            stem_files[stem][dk].sort()

    stems_all = sorted(stem_files.keys())
    log(f"Found {len(stems_all)} source stems (row-shard aware)")

    # ---- Power label matching with m{NNN}_ prefix fallback ----
    def _find_label(stem: str) -> str | None:
        """Try {stem}.npy; if missing, strip mNNN_ prefix and retry."""
        direct = os.path.join(label_dir, f"{stem}.npy")
        if os.path.exists(direct):
            return direct
        stripped = _merge_prefix_pat.sub("", stem)
        if stripped != stem:
            fallback = os.path.join(label_dir, f"{stripped}.npy")
            if os.path.exists(fallback):
                log(f"  Label fallback: {stem} -> {stripped}.npy")
                return fallback
        return None

    valid_stems = []
    for stem in stems_all:
        # Need both TOG files and a label
        if stem_files[stem]["tog"] and _find_label(stem):
            valid_stems.append(stem)
    log(f"Valid stems (TOG+label): {len(valid_stems)}")

    if len(valid_stems) <= test_count:
        raise ValueError(f"Need > {test_count} stems, got {len(valid_stems)}")

    train_stems = valid_stems[:-test_count]
    test_stems = valid_stems[-test_count:]
    log(f"Train: {len(train_stems)}, Test: {len(test_stems)}")

    include_set = set(include_stems or [])
    if include_set:
        missing = sorted(include_set.difference(valid_stems))
        if missing:
            raise ValueError(f"--include-stems not found in valid stems: {missing}")
        train_stems = [stem for stem in train_stems if stem in include_set]
        test_stems = [stem for stem in test_stems if stem in include_set]
        log(f"Filtered stems: train={len(train_stems)}, test={len(test_stems)}")

    sample_index_info = None
    if sample_index_manifest:
        with open(sample_index_manifest, "r", encoding="utf-8") as f:
            sample_index_info = json.load(f, object_pairs_hook=OrderedDict)
        log(f"Using sample index manifest: {sample_index_manifest}")

    manifest = OrderedDict([
        ("offsets", offsets),
        ("data_types", data_types),
        ("boundary_discard", boundary_discard),
        ("test_count", test_count),
        ("stage", stage),
        ("sample_rows_per_file", sample_rows_per_file),
        ("sample_bins", sample_bins),
        ("sample_min_per_bin", sample_min_per_bin),
        ("sample_seed", sample_seed),
        ("sample_index_manifest", os.path.abspath(sample_index_manifest) if sample_index_manifest else ""),
        ("compress_output", bool(compress_output)),
        ("skip_existing", bool(skip_existing)),
        ("jobs", jobs),
        ("train_stems", train_stems),
        ("test_stems", test_stems),
        ("combinations", OrderedDict()),
    ])

    sample_cache: dict[str, np.ndarray] = {}
    weight_cache: dict[str, np.ndarray] = {}
    if sample_index_info is not None:
        stems_info = sample_index_info.get("stems", {})
        for stem in train_stems + test_stems:
            info = stems_info.get(stem)
            if not info:
                raise ValueError(f"No sample index entry for stem: {stem}")
            index_file = info.get("index_file", "")
            if not index_file or not os.path.exists(index_file):
                raise FileNotFoundError(f"Missing sample index file for {stem}: {index_file}")
            sample_cache[stem] = np.load(index_file).astype(np.int64)
            weight_file = info.get("weight_file", "")
            if weight_file and os.path.exists(weight_file):
                weight_cache[stem] = np.load(weight_file).reshape(-1).astype(np.float64)
            log(f"  Sample index {stem}: {len(sample_cache[stem])} rows")
    elif sample_rows_per_file > 0:
        for si, stem in enumerate(train_stems + test_stems):
            y = np.load(_find_label(stem)).reshape(-1).astype(np.float64)
            yv = y[boundary_discard:]
            sample_cache[stem] = choose_power_sample_indices(
                yv,
                max_rows=sample_rows_per_file,
                bins=sample_bins,
                min_per_bin=sample_min_per_bin,
                seed=sample_seed + si,
            )
            selected = sample_cache[stem]
            delta = delta_power(yv)
            pe, de = quantile_edges(yv, sample_bins), quantile_edges(delta, sample_bins)
            _, _, joint = assign_joint_bins(yv[selected], delta[selected], pe, de)
            table = recommend_joint_weights(joint, (len(pe) - 1) * (len(de) - 1))
            table[table == 0] = 1.0
            weight_cache[stem] = table[joint]
            log(f"  Sample {stem}: {len(sample_cache[stem])}/{len(yv)} rows by power")

    def _sample_indices_for_stem(stem: str) -> np.ndarray | None:
        if stem in sample_cache:
            return sample_cache[stem]
        return None

    def _empty_like_first_shard(src_list: list[str]):
        first = load_npz(src_list[0]).tocsr()
        shape = (0, first.shape[1])
        dtype = first.dtype
        del first
        from scipy.sparse import csr_matrix
        return csr_matrix(shape, dtype=dtype)

    def _load_row_window(src_list: list[str], r0: int, r1: int):
        if len(src_list) == 1:
            return load_npz(src_list[0]).tocsr()[r0:r1].tocsr()

        pieces = []
        cursor = 0
        for src in src_list:
            shard = load_npz(src).tocsr()
            shard_start = cursor
            shard_end = cursor + shard.shape[0]
            cursor = shard_end
            take0 = max(r0, shard_start)
            take1 = min(r1, shard_end)
            if take0 < take1:
                pieces.append(shard[take0 - shard_start:take1 - shard_start].tocsr())
            del shard
        if not pieces:
            return _empty_like_first_shard(src_list)
        if len(pieces) == 1:
            return pieces[0]
        return sparse_vstack(pieces, format="csr")

    def _load_absolute_rows(src_list: list[str], rows: np.ndarray):
        rows = np.asarray(rows, dtype=np.int64)
        if len(rows) == 0:
            return _empty_like_first_shard(src_list)
        if np.any(rows < 0):
            raise ValueError(f"Negative row requested: min={rows.min()}")
        if len(src_list) == 1:
            return load_npz(src_list[0]).tocsr()[rows].tocsr()

        pieces = []
        cursor = 0
        pos0 = 0
        for src in src_list:
            shard = load_npz(src).tocsr()
            shard_start = cursor
            shard_end = cursor + shard.shape[0]
            cursor = shard_end
            pos1 = int(np.searchsorted(rows, shard_end, side="left"))
            if pos1 > pos0:
                local_rows = rows[pos0:pos1] - shard_start
                if np.any(local_rows < 0) or np.any(local_rows >= shard.shape[0]):
                    raise ValueError(f"Sample row outside shard range for {src}")
                pieces.append(shard[local_rows].tocsr())
            pos0 = pos1
            del shard
            if pos0 >= len(rows):
                break
        if pos0 < len(rows):
            raise ValueError(f"Requested row {int(rows[pos0])} beyond available shard rows")
        if not pieces:
            return _empty_like_first_shard(src_list)
        if len(pieces) == 1:
            return pieces[0]
        return sparse_vstack(pieces, format="csr")

    combo_tasks = [(offset_k, dtype) for offset_k in offsets for dtype in data_types]

    def _build_combo(offset_k: int, dtype: str) -> tuple[str, OrderedDict]:
        combo_key = f"{dtype}_offset{offset_k}"
        combo_dir = os.path.join(output_dir, combo_key)
        mkdirp(combo_dir)

        train_feat, train_lbl, train_weight = [], [], []
        test_feat, test_lbl, test_weight = [], [], []
        rows_per_file = 0

        for split, stem_list in [("train", train_stems), ("test", test_stems)]:
            for stem in stem_list:
                src_list = stem_files[stem].get(dtype, [])
                if not src_list:
                    raise FileNotFoundError(
                        f"No {dtype} file for stem '{stem}' (stage={stage})"
                    )

                of = os.path.join(combo_dir, f"{stem}.feature.npz")
                ol = os.path.join(combo_dir, f"{stem}.label.npy")
                ow = os.path.join(combo_dir, f"{stem}.weight.npy")
                if skip_existing and os.path.exists(of) and os.path.exists(ol) and os.path.exists(ow):
                    yv_existing = np.load(ol).reshape(-1)
                    rows_per_file = len(yv_existing)
                    log(f"  Reuse existing {stem}/{combo_key}: {rows_per_file} rows")
                    if split == "train":
                        train_feat.append(os.path.abspath(of))
                        train_lbl.append(os.path.abspath(ol))
                        train_weight.append(os.path.abspath(ow))
                    else:
                        test_feat.append(os.path.abspath(of))
                        test_lbl.append(os.path.abspath(ol))
                        test_weight.append(os.path.abspath(ow))
                    continue

                y = np.load(_find_label(stem)).reshape(-1).astype(np.float64)

                # Row slicing with TOG/LEVEL alignment
                if dtype == "tog":
                    r0 = boundary_discard - offset_k
                else:
                    r0 = boundary_discard + 1 - offset_k
                yv = y[boundary_discard:]
                r1 = r0 + len(yv)

                sample_idx = _sample_indices_for_stem(stem)
                if sample_idx is not None:
                    sample_idx = np.asarray(sample_idx, dtype=np.int64)
                    if len(sample_idx) and (sample_idx[0] < 0 or sample_idx[-1] >= len(yv)):
                        raise ValueError(
                            f"Sample index out of range {stem}/{combo_key}: max={sample_idx[-1]} y={len(yv)}"
                        )
                    Xs = _load_absolute_rows(src_list, r0 + sample_idx)
                    yv = yv[sample_idx]
                else:
                    Xs = _load_row_window(src_list, r0, r1)

                    if Xs.shape[0] == len(yv) + 1:
                        log(f"  Trim last X row for {stem}/{combo_key}: X={Xs.shape[0]} y={len(yv)}")
                        Xs = Xs[:-1].tocsr()

                if Xs.shape[0] != len(yv):
                    raise ValueError(
                        f"Row mismatch {stem}/{combo_key}: X={Xs.shape[0]} y={len(yv)}"
                    )

                save_npz(of, Xs, compressed=compress_output)
                np.save(ol, yv)
                weights = weight_cache.get(stem)
                if weights is None:
                    weights = np.ones(len(yv), dtype=np.float64)
                if len(weights) != len(yv):
                    raise ValueError(f"Weight mismatch {stem}: weights={len(weights)} labels={len(yv)}")
                np.save(ow, weights)
                rows_per_file = Xs.shape[0]

                if split == "train":
                    train_feat.append(os.path.abspath(of))
                    train_lbl.append(os.path.abspath(ol))
                    train_weight.append(os.path.abspath(ow))
                else:
                    test_feat.append(os.path.abspath(of))
                    test_lbl.append(os.path.abspath(ol))
                    test_weight.append(os.path.abspath(ow))

        write_filelist(os.path.join(combo_dir, "train_feature.f"), train_feat)
        write_filelist(os.path.join(combo_dir, "train_label.f"), train_lbl)
        write_filelist(os.path.join(combo_dir, "test_feature.f"), test_feat)
        write_filelist(os.path.join(combo_dir, "test_label.f"), test_lbl)
        write_filelist(os.path.join(combo_dir, "train_weight.f"), train_weight)
        write_filelist(os.path.join(combo_dir, "test_weight.f"), test_weight)

        result = OrderedDict([
            ("offset", offset_k),
            ("data_type", dtype),
            ("train_files", len(train_feat)),
            ("test_files", len(test_feat)),
            ("valid_rows_per_file", rows_per_file),
            ("sampled", sample_rows_per_file > 0 or bool(sample_index_manifest)),
            ("filelist_dir", os.path.abspath(combo_dir)),
        ])
        log(f"  {combo_key}: {len(train_feat)}T+{len(test_feat)}E, {rows_per_file} rows/file")
        return combo_key, result

    jobs = max(1, int(jobs))
    combo_results: dict[str, OrderedDict] = {}
    if jobs == 1 or len(combo_tasks) <= 1:
        for offset_k, dtype in combo_tasks:
            combo_key, result = _build_combo(offset_k, dtype)
            combo_results[combo_key] = result
    else:
        workers = min(jobs, len(combo_tasks))
        log(f"Parallel Step1 combos: jobs={workers}, combos={len(combo_tasks)}")
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(_build_combo, offset_k, dtype): (offset_k, dtype) for offset_k, dtype in combo_tasks}
            for fut in as_completed(futs):
                combo_key, result = fut.result()
                combo_results[combo_key] = result

    for offset_k, dtype in combo_tasks:
        combo_key = f"{dtype}_offset{offset_k}"
        manifest["combinations"][combo_key] = combo_results[combo_key]

    return manifest


def main():
    ap = argparse.ArgumentParser(description="Time-window data preparation")
    ap.add_argument("--level-dir", required=True)
    ap.add_argument("--tog-dir", required=True)
    ap.add_argument("--label-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--offsets", default="12,11,10,9,8,7,6")
    ap.add_argument("--data-types", default="tog,level")
    ap.add_argument("--boundary-discard", type=int, default=None)
    ap.add_argument("--test-count", type=int, default=1)
    ap.add_argument("--stage", default="after_one")
    ap.add_argument("--sample-rows-per-file", type=int, default=0,
                    help="Power-stratified rows to keep per source file; 0 keeps all")
    ap.add_argument("--sample-bins", type=int, default=20,
                    help="Number of power quantile bins for sampling")
    ap.add_argument("--sample-min-per-bin", type=int, default=1,
                    help="Minimum random rows per non-empty power bin")
    ap.add_argument("--sample-seed", type=int, default=108)
    ap.add_argument("--jobs", type=int, default=1,
                    help="Parallel workers across (offset, data_type) combinations")
    ap.add_argument("--include-stems", default="",
                    help="Comma-separated source stems to process; empty processes all")
    ap.add_argument("--sample-index-manifest", default="",
                    help="Manifest from step1_make_global_sample_index.py")
    ap.add_argument("--compress-output", type=int, default=1,
                    help="1 writes compressed npz; 0 writes uncompressed npz for faster/lower-memory Step1")
    ap.add_argument("--skip-existing", type=int, default=1,
                    help="Reuse existing feature.npz/label.npy inside the output directory")
    args = ap.parse_args()

    offsets = [int(x.strip()) for x in args.offsets.split(",")]
    data_types = [x.strip() for x in args.data_types.split(",")]
    boundary_discard = args.boundary_discard or max(offsets)
    include_stems = [x.strip() for x in args.include_stems.split(",") if x.strip()]

    log(f"Offsets={offsets}, types={data_types}, discard={boundary_discard}, test={args.test_count}")

    t0 = time.time()
    manifest = build_shifted_data(
        level_dir=args.level_dir,
        tog_dir=args.tog_dir,
        label_dir=args.label_dir,
        output_dir=args.output_dir,
        offsets=offsets,
        data_types=data_types,
        boundary_discard=boundary_discard,
        test_count=args.test_count,
        stage=args.stage,
        sample_rows_per_file=args.sample_rows_per_file,
        sample_bins=args.sample_bins,
        sample_min_per_bin=args.sample_min_per_bin,
        sample_seed=args.sample_seed,
        jobs=args.jobs,
        include_stems=include_stems,
        sample_index_manifest=args.sample_index_manifest or None,
        compress_output=bool(args.compress_output),
        skip_existing=bool(args.skip_existing),
    )

    mp = os.path.join(args.output_dir, "timewindow_manifest.json")
    with open(mp, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    log(f"Done in {time.time() - t0:.1f}s. Manifest: {mp}")


if __name__ == "__main__":
    main()
