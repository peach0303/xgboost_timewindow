#!/usr/bin/env python3
"""
Step 3: Feature Concatenation.

Horizontally concatenates all MCP-selected proxy features from every
(offset, data_type) combination into a single sparse matrix, then saves
as XGBoost DMatrix binary buffers for train/test.

Feature naming: "{data_type}_o{offset}_{original_signal_name}"
"""
from __future__ import annotations

import argparse
import json
import os
import time
from collections import OrderedDict

import numpy as np
from utils.training_policy import shuffle_xyw
from scipy.sparse import load_npz, hstack, vstack, csr_matrix, save_npz

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def mkdirp(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def read_filelist(path: str) -> list[str]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s and not s.startswith("#"):
                out.append(s)
    return out


def resolve_manifest_path(path: str, manifest_dir: str) -> str:
    if os.path.isabs(path):
        return path
    cwd_path = os.path.abspath(path)
    if os.path.exists(cwd_path):
        return cwd_path
    return os.path.abspath(os.path.join(manifest_dir, path))


def stem_from_feature_path(path: str) -> str:
    base = os.path.basename(path)
    suffix = ".feature.npz"
    if not base.endswith(suffix):
        raise ValueError(f"Unexpected Step1 feature filename: {path}")
    return base[:-len(suffix)]


def build_split_index(combo_info: dict, split: str) -> dict[str, tuple[str, str]]:
    filelist_dir = combo_info["filelist_dir"]
    feat_files = read_filelist(os.path.join(filelist_dir, f"{split}_feature.f"))
    label_files = read_filelist(os.path.join(filelist_dir, f"{split}_label.f"))
    if len(feat_files) != len(label_files):
        raise ValueError(
            f"{split} filelist length mismatch in {filelist_dir}: "
            f"features={len(feat_files)} labels={len(label_files)}"
        )
    indexed = OrderedDict()
    for feat_path, label_path in zip(feat_files, label_files):
        stem = stem_from_feature_path(feat_path)
        if stem in indexed:
            raise ValueError(f"Duplicate stem '{stem}' in {split} filelist: {filelist_dir}")
        indexed[stem] = (feat_path, label_path)
    return indexed


def load_selected_indices(path: str) -> np.ndarray:
    """Load selected indices from text file (space/comma separated)."""
    with open(path) as f:
        raw = f.read().strip().replace(",", " ").replace("\n", " ")
    return np.array([int(x) for x in raw.split() if x], dtype=np.int64)


def main():
    ap = argparse.ArgumentParser(description="Feature concatenation across offsets/types")
    ap.add_argument("--tw-manifest", required=True, help="timewindow_manifest.json from step1")
    ap.add_argument("--mcp-manifest", required=True, help="mcp_per_offset_manifest.json from step2")
    ap.add_argument("--namefile", default=None, help="Signal name file for traceability")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--shuffle-seed", type=int, default=108)
    args = ap.parse_args()

    with open(args.tw_manifest) as f:
        tw = json.load(f, object_pairs_hook=OrderedDict)
    with open(args.mcp_manifest) as f:
        mcp = json.load(f, object_pairs_hook=OrderedDict)
    mcp_manifest_dir = os.path.dirname(os.path.abspath(args.mcp_manifest))

    train_stems = tw["train_stems"]
    test_stems = tw["test_stems"]
    combos = tw["combinations"]
    mcp_combos = mcp["combinations"]

    feature_names = []
    concat_meta_blocks = []
    col_offset = 0

    for combo_key, combo_info in combos.items():
        dtype = combo_info["data_type"]
        offset_k = combo_info["offset"]
        mcp_res = mcp_combos[combo_key]
        sel_path = resolve_manifest_path(mcp_res["selected_indices_file"], mcp_manifest_dir)
        sel_indices = load_selected_indices(sel_path)
        n_sel = len(sel_indices)
        log(f"  {combo_key}: {n_sel} selected features")

        # Get feature names for this combination
        combo_names = []
        if args.namefile and os.path.exists(args.namefile):
            with open(args.namefile) as f:
                all_names = [line.strip() for line in f]
            for idx in sel_indices:
                if idx < len(all_names):
                    combo_names.append(f"{dtype}_o{offset_k}_{all_names[idx]}")
                else:
                    combo_names.append(f"{dtype}_o{offset_k}_idx{idx}")
        else:
            combo_names = [f"{dtype}_o{offset_k}_f{i}" for i in range(n_sel)]

        feature_names.extend(combo_names)
        concat_meta_blocks.append(OrderedDict([
            ("offset", offset_k),
            ("data_type", dtype),
            ("n_selected", n_sel),
            ("start_col", col_offset),
            ("end_col", col_offset + n_sel),
            ("feature_names", combo_names),
        ]))
        col_offset += n_sel

    # Sanitize feature names for xgboost (no [, ], or < allowed)
    feature_names = [n.replace("[", "_").replace("]", "_").replace("<", "_") for n in feature_names]

    total_cols = col_offset
    log(f"Total concatenated columns: {total_cols}")

    # Build concatenated matrices for train and test
    train_X_parts, train_y_list, train_w_list = [], [], []
    test_X_parts, test_y_list, test_w_list = [], [], []
    split_file_index = {
        split: {
            combo_key: build_split_index(combo_info, split)
            for combo_key, combo_info in combos.items()
        }
        for split in ("train", "test")
    }

    for split, stem_list in [("train", train_stems), ("test", test_stems)]:
        for stem in stem_list:
            row_parts = []
            # Load label once per stem (same for all combinations)
            first_combo = list(combos.keys())[0]
            try:
                _feat_path, lbl_path = split_file_index[split][first_combo][stem]
            except KeyError as exc:
                raise FileNotFoundError(
                    f"Missing {split} stem '{stem}' in Step1 filelist for {first_combo}"
                ) from exc
            y_val = np.load(lbl_path).reshape(-1).astype(np.float64)
            weight_path = lbl_path.replace(".label.npy", ".weight.npy")
            w_val = np.load(weight_path).reshape(-1).astype(np.float64) if os.path.exists(weight_path) else np.ones(len(y_val))

            for combo_key, combo_info in combos.items():
                mcp_res = mcp_combos[combo_key]
                sel_path = resolve_manifest_path(mcp_res["selected_indices_file"], mcp_manifest_dir)
                sel_indices = load_selected_indices(sel_path)
                try:
                    npz_path, _lbl_path = split_file_index[split][combo_key][stem]
                except KeyError as exc:
                    raise FileNotFoundError(
                        f"Missing {split} stem '{stem}' in Step1 filelist for {combo_key}"
                    ) from exc
                X = load_npz(npz_path).tocsr()
                X_sel = X[:, sel_indices].tocsr()
                row_parts.append(X_sel)

            row_mat = hstack(row_parts, format="csr")
            if split == "train":
                train_X_parts.append(row_mat)
                train_y_list.append(y_val)
                train_w_list.append(w_val)
            else:
                test_X_parts.append(row_mat)
                test_y_list.append(y_val)
                test_w_list.append(w_val)

    # Vertically stack rows within train/test
    X_train = vstack(train_X_parts, format="csr") if len(train_X_parts) > 1 else train_X_parts[0]
    X_test = vstack(test_X_parts, format="csr") if len(test_X_parts) > 1 else test_X_parts[0]
    y_train = np.concatenate(train_y_list)
    y_test = np.concatenate(test_y_list)
    w_train = np.concatenate(train_w_list)
    w_test = np.concatenate(test_w_list)
    X_train, y_train, w_train, _ = shuffle_xyw(X_train, y_train, w_train, args.shuffle_seed)

    log(f"X_train: {X_train.shape}, nnz={X_train.nnz}")
    log(f"X_test:  {X_test.shape}, nnz={X_test.nnz}")

    # Save as XGBoost DMatrix binary buffers (or NPZ fallback)
    mkdirp(args.output_dir)
    train_buf = os.path.join(args.output_dir, "train_tw.buffer")
    test_buf = os.path.join(args.output_dir, "test_tw.buffer")

    if HAS_XGB:
        dtrain = xgb.DMatrix(X_train, label=y_train, weight=w_train, feature_names=feature_names)
        dtest = xgb.DMatrix(X_test, label=y_test, weight=w_test, feature_names=feature_names)
        dtrain.save_binary(train_buf)
        dtest.save_binary(test_buf)
        log(f"Saved XGBoost buffers: {train_buf}, {test_buf}")
    else:
        # Fallback: save as NPZ + NPY
        train_npz = os.path.join(args.output_dir, "train_tw.npz")
        test_npz = os.path.join(args.output_dir, "test_tw.npz")
        train_npy = os.path.join(args.output_dir, "train_label.npy")
        test_npy = os.path.join(args.output_dir, "test_label.npy")
        save_npz(train_npz, X_train, compressed=True)
        save_npz(test_npz, X_test, compressed=True)
        np.save(train_npy, y_train)
        np.save(test_npy, y_test)
        np.save(os.path.join(args.output_dir, "train_weight.npy"), w_train)
        np.save(os.path.join(args.output_dir, "test_weight.npy"), w_test)
        log(f"WARNING: xgboost not available. Saved NPZ+NPY instead of DMatrix buffers.")
        log(f"  Train: {train_npz} + {train_npy}")
        log(f"  Test:  {test_npz} + {test_npy}")

    # Save feature names and meta
    with open(os.path.join(args.output_dir, "feature_names.txt"), "w") as f:
        for n in feature_names:
            f.write(n + "\n")
    with open(os.path.join(args.output_dir, "concat_meta.json"), "w") as f:
        json.dump(OrderedDict([
            ("total_columns", total_cols),
            ("train_shape", list(X_train.shape)),
            ("test_shape", list(X_test.shape)),
            ("train_rows", int(X_train.shape[0])),
            ("test_rows", int(X_test.shape[0])),
            ("weighted", True),
            ("shuffle_seed", args.shuffle_seed),
            ("blocks", concat_meta_blocks),
        ]), f, indent=2, ensure_ascii=False)

    log(f"Buffers: {train_buf}, {test_buf}")
    log("Done.")


if __name__ == "__main__":
    main()
