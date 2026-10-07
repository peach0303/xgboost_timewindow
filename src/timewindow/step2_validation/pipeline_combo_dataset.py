#!/usr/bin/env python3
"""Build legacy-XGBoost cache chunks directly from Step2 combo proxies."""

from __future__ import print_function

import json
import os
import shutil
from collections import OrderedDict

import numpy as np
from scipy.sparse import hstack, load_npz, save_npz


def read_filelist(path):
    with open(path, "r") as f:
        return [line.strip() for line in f if line.strip() and not line.lstrip().startswith("#")]


def resolve_path(path, manifest_dir):
    if os.path.isabs(path):
        return path
    beside_manifest = os.path.abspath(os.path.join(manifest_dir, path))
    if os.path.exists(beside_manifest):
        return beside_manifest
    return os.path.abspath(path)


def stem_from_feature(path):
    name = os.path.basename(path)
    suffix = ".feature.npz"
    if not name.endswith(suffix):
        raise ValueError("unexpected Step1 feature filename: %s" % path)
    return name[:-len(suffix)]


def build_split_index(combo_info, split):
    filelist_dir = combo_info["filelist_dir"]
    feature_files = read_filelist(os.path.join(filelist_dir, "%s_feature.f" % split))
    label_files = read_filelist(os.path.join(filelist_dir, "%s_label.f" % split))
    if len(feature_files) != len(label_files):
        raise ValueError("%s feature/label filelist length mismatch in %s" % (split, filelist_dir))
    indexed = OrderedDict()
    for feature_file, label_file in zip(feature_files, label_files):
        stem = stem_from_feature(feature_file)
        if stem in indexed:
            raise ValueError("duplicate stem %s in %s" % (stem, filelist_dir))
        indexed[stem] = (feature_file, label_file)
    return indexed


def load_indices(path):
    with open(path, "r") as f:
        raw = f.read().replace(",", " ").replace("\n", " ")
    values = [int(value) for value in raw.split() if value]
    if not values:
        raise ValueError("empty selected indices: %s" % path)
    return np.asarray(values, dtype=np.int64)


def load_signal_names(namefile):
    if not namefile:
        return []
    with open(namefile, "r") as f:
        return [line.strip() for line in f]


def load_sample_index_info(tw, tw_manifest_dir):
    path = tw.get("sample_index_manifest", "")
    if not path:
        return None
    path = resolve_path(path, tw_manifest_dir)
    with open(path, "r") as f:
        return json.load(f)


def cycle_indices_for_stem(tw, sample_info, stem, row_count):
    boundary = int(tw.get("boundary_discard", 0))
    if sample_info and sample_info.get("sampled", False):
        stem_info = sample_info.get("stems", {}).get(stem, {})
        index_file = stem_info.get("index_file", "")
        if not index_file:
            raise ValueError("sampled dataset lacks index_file for stem %s" % stem)
        indices = np.load(index_file).reshape(-1).astype(np.int64)
        if indices.size != int(row_count):
            raise ValueError("sample index rows mismatch for %s: %d != %d" % (
                stem, indices.size, row_count))
        return boundary + indices

    if int(tw.get("sample_rows_per_file", 0)) > 0:
        # Older sequential Step1 runs did not persist sampled row indices. Use a
        # non-consecutive synthetic axis so point metrics stay valid and jump
        # evaluation does not invent adjacency between sampled points.
        return boundary + np.arange(row_count, dtype=np.int64) * 2
    return boundary + np.arange(row_count, dtype=np.int64)


def build_pipeline_chunks(tw_manifest, proxy_manifest, cache_dir, label_scale,
                          data_dtype, namefile=None, block_size=200,
                          combo_keys=None):
    with open(tw_manifest, "r") as f:
        tw = json.load(f, object_pairs_hook=OrderedDict)
    with open(proxy_manifest, "r") as f:
        proxy = json.load(f, object_pairs_hook=OrderedDict)

    tw_dir = os.path.dirname(os.path.abspath(tw_manifest))
    proxy_dir = os.path.dirname(os.path.abspath(proxy_manifest))
    all_combos = tw.get("combinations", OrderedDict())
    proxy_combos = proxy.get("combinations", OrderedDict())
    if not all_combos:
        raise ValueError("Step1 manifest has no combinations")
    if combo_keys:
        requested = [str(key).strip() for key in combo_keys if str(key).strip()]
        if len(requested) != len(set(requested)):
            raise ValueError("--combos contains duplicate combo keys")
        missing_step1 = [key for key in requested if key not in all_combos]
        if missing_step1:
            raise ValueError("Step1 manifest lacks requested combos: %s; available: %s" % (
                ",".join(missing_step1), ",".join(all_combos.keys())))
        combos = OrderedDict((key, all_combos[key]) for key in requested)
    else:
        combos = all_combos
    missing = [key for key in combos if key not in proxy_combos]
    if missing:
        raise ValueError("Step2 manifest lacks combos: %s" % ",".join(missing))

    selected_by_combo = OrderedDict()
    feature_names = []
    all_signal_names = load_signal_names(namefile)
    for combo_key, combo_info in combos.items():
        selected_file = resolve_path(proxy_combos[combo_key]["selected_indices_file"], proxy_dir)
        selected = load_indices(selected_file)
        selected_by_combo[combo_key] = selected
        dtype = str(combo_info["data_type"])
        offset = int(combo_info["offset"])
        for column in selected:
            signal = all_signal_names[column] if column < len(all_signal_names) else "idx%d" % column
            feature_names.append("%s_o%d_%s" % (dtype, offset, signal))

    split_indices = {
        split: OrderedDict((key, build_split_index(info, split)) for key, info in combos.items())
        for split in ("train", "test")
    }
    train_stems = list(tw.get("train_stems", []))
    test_stems = list(tw.get("test_stems", []))
    stems = [("train", stem) for stem in train_stems] + [("test", stem) for stem in test_stems]
    if not train_stems or not test_stems:
        raise ValueError("pipeline validation requires non-empty train_stems and test_stems")

    chunk_dir = os.path.join(cache_dir, "chunks")
    staging_root = os.path.join(cache_dir, "_combo_block_staging")
    os.makedirs(chunk_dir, exist_ok=True)
    shutil.rmtree(staging_root, ignore_errors=True)
    os.makedirs(staging_root, exist_ok=True)
    for name in os.listdir(chunk_dir):
        if name.startswith("part_") and name.endswith(("_X.npz", "_y.npy", "_m.npy")):
            os.remove(os.path.join(chunk_dir, name))

    sample_info = load_sample_index_info(tw, tw_dir)
    dtype = np.float64 if data_dtype == "float64" else np.float32
    chunk_x_files, chunk_y_files = [], []
    file_records = []
    total_rows = 0
    chunk_id = 0

    try:
        for file_idx, (split, stem) in enumerate(stems):
            label = None
            label_path_used = None
            case_staging = os.path.join(staging_root, "file_%06d" % file_idx)
            os.makedirs(case_staging, exist_ok=True)
            staged_by_block = None

            # Load one full combo CSR at a time, select proxy columns, and spill
            # row blocks to disk. Full-case proxy matrices from different combos
            # therefore never coexist in memory.
            for combo_idx, combo_key in enumerate(combos):
                try:
                    feature_path, label_path = split_indices[split][combo_key][stem]
                except KeyError:
                    raise FileNotFoundError("missing %s/%s/%s in Step1 filelists" % (
                        split, combo_key, stem))
                X = load_npz(feature_path).tocsr()
                selected = selected_by_combo[combo_key]
                if selected.max() >= X.shape[1] or selected.min() < 0:
                    raise ValueError("selected index outside %s columns for %s" % (
                        X.shape[1], combo_key))
                current_label = np.load(label_path).reshape(-1).astype(np.float64)
                if X.shape[0] != current_label.size:
                    raise ValueError("feature/label rows mismatch: %s" % feature_path)
                if label is None:
                    label = current_label
                    label_path_used = label_path
                    effective_block_size = int(block_size)
                    if effective_block_size <= 0:
                        effective_block_size = int(label.size)
                    block_ranges = [
                        (begin, min(begin + effective_block_size, int(label.size)))
                        for begin in range(0, int(label.size), effective_block_size)
                    ]
                    staged_by_block = [[] for _ in block_ranges]
                elif label.size != current_label.size or not np.array_equal(label, current_label):
                    raise ValueError("labels differ across combos for stem %s" % stem)

                for block_idx, (begin, end) in enumerate(block_ranges):
                    staged_path = os.path.join(
                        case_staging,
                        "combo_%04d_block_%06d.npz" % (combo_idx, block_idx))
                    selected_block = X[begin:end, selected].tocsr()
                    if selected_block.data.dtype != dtype:
                        selected_block.data = selected_block.data.astype(dtype, copy=False)
                    save_npz(staged_path, selected_block, compressed=False)
                    staged_by_block[block_idx].append(staged_path)
                    del selected_block
                del X, current_label

            y_case = (label / float(label_scale)).astype(np.float32).reshape(-1, 1)
            cycles = cycle_indices_for_stem(tw, sample_info, stem, y_case.shape[0])
            label_indices = cycles - int(tw.get("boundary_discard", 0))

            for block_idx, (begin, end) in enumerate(block_ranges):
                row_parts = [
                    load_npz(path).tocsr() for path in staged_by_block[block_idx]
                ]
                X_block = hstack(row_parts, format="csr", dtype=dtype)
                y_block = y_case[begin:end]
                meta_block = np.column_stack([
                    np.full(end - begin, file_idx, dtype=np.int64),
                    label_indices[begin:end].astype(np.int64),
                    cycles[begin:end].astype(np.int64),
                ])
                x_path = os.path.join(chunk_dir, "part_%06d_X.npz" % chunk_id)
                y_path = os.path.join(chunk_dir, "part_%06d_y.npy" % chunk_id)
                m_path = os.path.join(chunk_dir, "part_%06d_m.npy" % chunk_id)
                save_npz(x_path, X_block, compressed=True)
                np.save(y_path, y_block)
                np.save(m_path, meta_block)
                chunk_x_files.append(x_path)
                chunk_y_files.append(y_path)
                total_rows += int(end - begin)
                print("[PIPELINE-DATA] %s/%s chunk=%06d rows=[%d,%d) X=%s" % (
                    split, stem, chunk_id, begin, end, str(X_block.shape)))
                chunk_id += 1
                del row_parts, X_block, y_block, meta_block

            shutil.rmtree(case_staging)
            file_records.append(OrderedDict([
                ("file_idx", file_idx), ("stem", stem), ("pipeline_split", split),
                ("rows", int(y_case.shape[0])), ("chunks", len(block_ranges)),
                ("label_file", label_path_used),
            ]))
            del label, y_case, cycles, label_indices, staged_by_block
    finally:
        shutil.rmtree(staging_root, ignore_errors=True)

    dataset_meta = OrderedDict([
        ("mode", "step2_combo_selected_features"),
        ("step1_manifest", os.path.abspath(tw_manifest)),
        ("step2_manifest", os.path.abspath(proxy_manifest)),
        ("combo_order", list(combos.keys())),
        ("selected_columns_per_combo", OrderedDict(
            (key, selected_by_combo[key].tolist()) for key in combos)),
        ("feature_names", feature_names),
        ("total_columns", len(feature_names)),
        ("total_rows", total_rows),
        ("block_size", int(block_size)),
        ("num_chunks", len(chunk_x_files)),
        ("train_file_indices", list(range(len(train_stems)))),
        ("test_file_indices", list(range(len(train_stems), len(stems)))),
        ("files", file_records),
        ("label_scale", float(label_scale)),
    ])
    with open(os.path.join(cache_dir, "dataset.meta.json"), "w") as f:
        json.dump(dataset_meta, f, indent=2, ensure_ascii=False)
    return chunk_x_files, chunk_y_files, dataset_meta
