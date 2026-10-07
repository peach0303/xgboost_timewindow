#!/usr/bin/env python3
"""
Step 2: Per-Offset Proxy Selection.

For each (data_type, offset) combination, selects proxy signals.
Uses MCPRegression (skglm) if available, falls back to variance-based
selection for fast testing.

Version: 2.1.1-batch-union (2026-08-24)
Changes:
  * Train MCP once on every batch in one pass.
  * Clamp batch size to the available file count when batch_size > file_num.
  * Union proxies selected by all batches; never choose a best batch.
  * Allow signed MCP coefficients (positive=False).
  * Re-read all shards with unioned columns, fit one Ridge validation model,
    evaluate once, and save selected train/test XGBoost data buffers.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
from scipy.sparse import load_npz, vstack, csr_matrix, issparse
from sklearn.linear_model import Ridge
import joblib
import xgboost as xgb
from utils.indexed_split import (
    index_row_batch_size,
    indexed_split_paths,
    iter_indexed_combo_batches,
)

try:
    from skglm import MCPRegression
    HAS_SKGLM = True
except ImportError:
    HAS_SKGLM = False


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


def ensure_sparse_int32_indices(X):
    """Return CSR sparse matrix with int32 indices for sklearn/skglm solvers."""
    if not issparse(X):
        return X
    X = X.tocsr(copy=False)
    if X.indices.dtype == np.int32 and X.indptr.dtype == np.int32:
        return X

    int32_max = np.iinfo(np.int32).max
    if X.nnz > int32_max or (X.shape[1] - 1) > int32_max:
        raise ValueError(
            "Sparse matrix is too large for 32-bit sparse indices required by "
            "sklearn/skglm. Reduce batch size or split the matrix before fit."
        )

    X = X.copy()
    X.indices = X.indices.astype(np.int32, copy=False)
    X.indptr = X.indptr.astype(np.int32, copy=False)
    return X


def load_dataset(filelist_path: str):
    files = read_filelist(filelist_path)
    mats = [load_npz(fp).tocsr() for fp in files]
    if not mats:
        return csr_matrix((0, 0), dtype=np.float64)
    return ensure_sparse_int32_indices(vstack(mats, format="csr"))


def load_labels(filelist_path: str) -> np.ndarray:
    files = read_filelist(filelist_path)
    return np.concatenate([np.load(fp).reshape(-1).astype(np.float64) for fp in files])


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> OrderedDict:
    denom = np.where(np.abs(y_true) > 1e-12, np.abs(y_true), 1e-12)
    abs_err = np.abs(y_true - y_pred)
    rel_err = abs_err / denom
    mse = float(np.mean((y_true - y_pred) ** 2))
    rmse = float(np.sqrt(mse))
    y_mean = float(np.mean(y_true))
    y_var = float(np.var(y_true))
    return OrderedDict([
        ("n", int(y_true.shape[0])),
        ("label_mean", y_mean),
        ("label_std", float(np.std(y_true))),
        ("label_min", float(np.min(y_true))),
        ("label_max", float(np.max(y_true))),
        ("MAE", float(np.mean(abs_err))),
        ("MAPE", float(np.mean(rel_err) * 100.0)),
        ("MaxAPE", float(np.max(rel_err) * 100.0)),
        ("MSE", mse),
        ("RMSE", rmse),
        ("NRMSE", float(rmse / y_mean) if abs(y_mean) > 1e-12 else float("nan")),
        ("R2", float(1.0 - mse / y_var) if y_var > 1e-12 else float("nan")),
    ])


def save_eval_outputs(output_dir: str, metrics: OrderedDict, pred: np.ndarray | None) -> None:
    eval_dir = os.path.join(output_dir, "eval")
    mkdirp(eval_dir)
    with open(os.path.join(eval_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    pd.DataFrame.from_dict(metrics, orient="index", columns=["value"]).to_csv(
        os.path.join(eval_dir, "metrics.csv")
    )
    if pred is not None:
        np.savetxt(os.path.join(eval_dir, "predictions.txt"), pred)


def select_by_variance(X, top_k=200):
    """Fast feature selection by variance on sparse data."""
    if issparse(X):
        sq = X.multiply(X).mean(axis=0)
        mean = X.mean(axis=0)
        var = np.asarray(sq).flatten() - np.asarray(mean).flatten() ** 2
    else:
        var = np.var(X, axis=0)
    idx = np.argsort(var)[-top_k:]
    return np.sort(idx).astype(np.int64)


def run_proxy_selection(
    train_feat_f: str,
    train_label_f: str,
    test_feat_f: str,
    test_label_f: str,
    output_dir: str,
    alpha: float,
    gamma: float,
    alpha_l2: float,
    batch_size: int,
    namefile_path: str | None,
    top_k: int,
    indexed_combo: dict | None = None,
    split_indices: dict[str, str] | None = None,
    indexed_row_batch_size: int = 200_000,
):
    """Union one-pass per-batch proxies, then validate once on all rows."""
    mkdirp(output_dir)
    started = time.time()

    if split_indices is not None:
        if indexed_combo is None:
            raise ValueError("indexed split requires combo metadata")
        train_feat_files = ["<indexed-train>"]
        train_label_files = ["<indexed-train>"]
        test_feat_files = ["<indexed-test>"]
        test_label_files = ["<indexed-test>"]
        log("  Using virtual index split backed by original Step1 shards")
    else:
        train_feat_files = read_filelist(train_feat_f)
        train_label_files = read_filelist(train_label_f)
        test_feat_files = read_filelist(test_feat_f)
        test_label_files = read_filelist(test_label_f)
    if not train_feat_files:
        raise ValueError(f"Empty train feature filelist: {train_feat_f}")
    if len(train_feat_files) != len(train_label_files):
        raise ValueError(
            f"Train feature/label file count mismatch: "
            f"{len(train_feat_files)} != {len(train_label_files)}"
        )
    if not test_feat_files:
        raise ValueError(f"Empty test feature filelist: {test_feat_f}")
    if len(test_feat_files) != len(test_label_files):
        raise ValueError(
            f"Test feature/label file count mismatch: "
            f"{len(test_feat_files)} != {len(test_label_files)}"
        )

    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    train_pairs = list(zip(train_feat_files, train_label_files))
    batch_unit = "files"
    requested_batch_size = int(batch_size)
    effective_batch_size = min(requested_batch_size, len(train_pairs))
    indexed_batch_iter = None
    if split_indices is not None:
        batch_unit = "rows"
        requested_batch_size = int(indexed_row_batch_size)
        indexed_rows = int(np.load(split_indices["train"], mmap_mode="r").size)
        effective_batch_size = min(requested_batch_size, indexed_rows)
        indexed_batch_iter = iter_indexed_combo_batches(
            indexed_combo, split_indices["train"], requested_batch_size
        )
        batches = range(int(np.ceil(indexed_rows / requested_batch_size)))
    elif requested_batch_size > len(train_pairs):
        log(
            f"  Requested batch_size={requested_batch_size} exceeds "
            f"train_file_num={len(train_pairs)}; using one batch with "
            f"effective_batch_size={effective_batch_size}"
        )
    test_pairs = list(zip(test_feat_files, test_label_files))
    if split_indices is None:
        batches = [
            train_pairs[i:i + effective_batch_size]
            for i in range(0, len(train_pairs), effective_batch_size)
        ]

    if HAS_SKGLM:
        selector_class = "mcp"
        log(
            f"  Signed MCP batch pass: batches={len(batches)}, "
            f"alpha={alpha}, gamma={gamma}"
        )
    else:
        selector_class = "variance"
        log(f"  Variance batch pass: batches={len(batches)}, top_k={top_k}")

    batch_records = []
    selected_parts = []
    n_features = None
    for batch_index, batch in enumerate(batches, 1):
        batch_started = time.time()
        if indexed_batch_iter is not None:
            X_batch, y_batch, _weight = next(indexed_batch_iter)
            X_batch = ensure_sparse_int32_indices(X_batch)
            y_batch = y_batch.astype(np.float64, copy=False)
            if n_features is None:
                n_features = int(X_batch.shape[1])
        else:
            X_parts = []
            y_parts = []
            for feature_file, label_file in batch:
                X_part = load_npz(feature_file).tocsr()
                y_part = np.load(label_file).reshape(-1).astype(np.float64, copy=False)
                if X_part.shape[0] != y_part.shape[0]:
                    raise ValueError(
                        f"Feature/label row mismatch: {X_part.shape[0]} != "
                        f"{y_part.shape[0]} ({feature_file}, {label_file})"
                    )
                if n_features is None:
                    n_features = int(X_part.shape[1])
                elif X_part.shape[1] != n_features:
                    raise ValueError(
                        f"Inconsistent feature count: {X_part.shape[1]} != "
                        f"{n_features} in {feature_file}"
                    )
                X_parts.append(X_part)
                y_parts.append(y_part)
            X_batch = ensure_sparse_int32_indices(vstack(X_parts, format="csr"))
            y_batch = np.concatenate(y_parts)
        if HAS_SKGLM:
            selector = MCPRegression(
                alpha=alpha,
                gamma=gamma,
                warm_start=False,
                positive=False,
                verbose=0,
                max_iter=100,
            )
            selector.fit(X_batch, y_batch)
            batch_selected = np.flatnonzero(selector.coef_).astype(np.int64)
            n_positive = int(np.count_nonzero(selector.coef_ > 0))
            n_negative = int(np.count_nonzero(selector.coef_ < 0))
        else:
            batch_selected = select_by_variance(X_batch, top_k=top_k)
            n_positive = 0
            n_negative = 0

        selected_parts.append(batch_selected)
        batch_records.append(OrderedDict([
            ("batch", batch_index),
            ("n_files", 0 if indexed_batch_iter is not None else len(batch)),
            ("n_rows", int(X_batch.shape[0])),
            ("n_selected", int(len(batch_selected))),
            ("mcp_positive_coefficients", n_positive),
            ("mcp_negative_coefficients", n_negative),
            ("selected_indices", batch_selected.tolist()),
        ]))
        log(
            f"  Batch {batch_index}/{len(batches)}: rows={X_batch.shape[0]}, "
            f"selected={len(batch_selected)}, positive={n_positive}, "
            f"negative={n_negative}, t={time.time()-batch_started:.1f}s"
        )
        if indexed_batch_iter is None:
            del X_parts, y_parts
        del X_batch, y_batch

    nonempty_parts = [part for part in selected_parts if len(part) > 0]
    if not nonempty_parts:
        raise RuntimeError(f"No features were selected for {output_dir}")
    selected = np.unique(np.concatenate(nonempty_parts)).astype(np.int64)
    log(
        f"  Unioned proxies: {len(selected)} unique features from "
        f"{len(batches)} batches"
    )

    def load_union_data(file_pairs, split_name):
        if split_indices is not None:
            indexed_parts = list(iter_indexed_combo_batches(
                indexed_combo, split_indices[split_name], indexed_row_batch_size,
                columns=selected,
            ))
            return (
                ensure_sparse_int32_indices(vstack(
                    [item[0] for item in indexed_parts], format="csr"
                )),
                np.concatenate([item[1] for item in indexed_parts]).astype(
                    np.float64, copy=False
                ),
            )
        selected_X_parts = []
        labels = []
        for file_index, (feature_file, label_file) in enumerate(file_pairs, 1):
            part_started = time.time()
            X_full = load_npz(feature_file).tocsr()
            y_part = np.load(label_file).reshape(-1).astype(np.float64, copy=False)
            if X_full.shape[0] != y_part.shape[0]:
                raise ValueError(
                    f"Feature/label row mismatch: {X_full.shape[0]} != "
                    f"{y_part.shape[0]} ({feature_file}, {label_file})"
                )
            if X_full.shape[1] != n_features:
                raise ValueError(
                    f"Inconsistent feature count: {X_full.shape[1]} != "
                    f"{n_features} in {feature_file}"
                )
            X_selected = ensure_sparse_int32_indices(X_full[:, selected].tocsr())
            selected_X_parts.append(X_selected)
            labels.append(y_part)
            log(
                f"  UnionData[{split_name}] {file_index}/{len(file_pairs)}: "
                f"shape={X_selected.shape}, nnz={X_selected.nnz}, "
                f"t={time.time()-part_started:.1f}s"
            )
            del X_full
        return (
            ensure_sparse_int32_indices(vstack(selected_X_parts, format="csr")),
            np.concatenate(labels),
        )

    X_train_selected, y_train = load_union_data(train_pairs, "train")
    X_test_selected, y_test = load_union_data(test_pairs, "test")

    ridge = Ridge(alpha=alpha_l2, solver="sparse_cg", fit_intercept=True)
    ridge.fit(X_train_selected, y_train)
    y_pred = ridge.predict(X_test_selected)
    metrics = evaluate_predictions(y_test, y_pred)
    score = float(metrics["MAPE"])
    log(
        f"  Union Ridge: selected={len(selected)}, MAPE={score:.3f}%, "
        f"RMSE={metrics['RMSE']:.6g}, NRMSE={metrics['NRMSE']:.6g}, "
        f"R2={metrics['R2']:.6g}, total_t={time.time()-started:.1f}s"
    )

    selected_file = os.path.join(output_dir, "selected_indices.txt")
    batch_file = os.path.join(output_dir, "batch_selected_indices.json")
    ridge_file = os.path.join(output_dir, "ridge_model.joblib")
    train_buffer = os.path.join(output_dir, "train_selected.buffer")
    test_buffer = os.path.join(output_dir, "test_selected.buffer")
    np.savetxt(selected_file, selected, fmt="%d")
    with open(batch_file, "w", encoding="utf-8") as f:
        json.dump(batch_records, f, indent=2, ensure_ascii=False)
    joblib.dump(ridge, ridge_file)
    save_eval_outputs(output_dir, metrics, y_pred)

    xgb.DMatrix(X_train_selected, label=y_train).save_binary(train_buffer)
    xgb.DMatrix(X_test_selected, label=y_test).save_binary(test_buffer)
    log(f"  Saved selected data buffers: {train_buffer}, {test_buffer}")

    if namefile_path and os.path.exists(namefile_path):
        with open(namefile_path) as f:
            names = [line.strip() for line in f]
        selected_names = [names[i] for i in selected if i < len(names)]
        with open(os.path.join(output_dir, "selected_feature_names.txt"), "w") as f:
            for name in selected_names:
                f.write(name + "\n")

    return OrderedDict([
        ("selector", selector_class),
        ("version", "2.1.1-batch-union"),
        ("training_mode", "batch_mcp_union_ridge"),
        ("signed_mcp_coefficients", True),
        ("requested_batch_size", requested_batch_size),
        ("effective_batch_size", effective_batch_size),
        ("batch_unit", batch_unit),
        ("n_batches", len(batches)),
        ("n_train", int(X_train_selected.shape[0])),
        ("n_features_total", int(n_features)),
        ("n_selected", int(len(selected))),
        ("best_mape", score),
        ("test_rmse", float(metrics.get("RMSE", float("nan")))),
        ("test_nrmse", float(metrics.get("NRMSE", float("nan")))),
        ("test_r2", float(metrics.get("R2", float("nan")))),
        ("eval_metrics_file", os.path.join(output_dir, "eval", "metrics.json")),
        ("batch_selected_indices_file", batch_file),
        ("selected_indices_file", selected_file),
        ("ridge_model_file", ridge_file),
        ("train_buffer", train_buffer),
        ("test_buffer", test_buffer),
    ])


def main():
    ap = argparse.ArgumentParser(description="Per-offset proxy selection")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--namefile", default=None)
    ap.add_argument("--alpha", type=float, default=50.0)
    ap.add_argument("--gamma", type=float, default=3.0)
    ap.add_argument("--alpha-l2", type=float, default=8.0)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--top-k", type=int, default=200,
                    help="Variance mode: top-K features to select")
    ap.add_argument("--jobs", type=int, default=1,
                    help="Parallel workers across (offset, data_type) combinations")
    ap.add_argument("--include-combos", default="",
                    help="Comma-separated combo keys to process, e.g. tog_offset12")
    args = ap.parse_args()

    if args.batch <= 0:
        raise SystemExit("ERROR: --batch must be positive")

    with open(args.manifest) as f:
        manifest = json.load(f, object_pairs_hook=OrderedDict)
    split_indices = indexed_split_paths(manifest)

    combos = manifest.get("combinations", {})
    log(f"Processing {len(combos)} combinations (selector={'MCP' if HAS_SKGLM else 'variance'})")

    mcp_manifest = OrderedDict([
        ("alpha", args.alpha), ("gamma", args.gamma), ("alpha_l2", args.alpha_l2),
        ("selector", "mcp" if HAS_SKGLM else "variance"),
        ("version", "2.1.1-batch-union"),
        ("training_mode", "batch_mcp_union_ridge"),
        ("signed_mcp_coefficients", True),
        ("jobs", args.jobs),
        ("combinations", OrderedDict()),
    ])

    combo_items = list(combos.items())
    if args.include_combos.strip():
        include_set = {x.strip() for x in args.include_combos.split(",") if x.strip()}
        missing = sorted(include_set.difference(combos.keys()))
        if missing:
            raise SystemExit(f"ERROR: --include-combos not in manifest: {missing}")
        combo_items = [(k, v) for k, v in combo_items if k in include_set]
        log(f"Filtered combinations: {len(combo_items)} selected by --include-combos")

    def _run_combo(combo_key: str, combo_info: dict) -> tuple[str, OrderedDict]:
        log(f"\n=== {combo_key} ===")
        fd = combo_info["filelist_dir"]
        out_dir = os.path.join(args.output_dir, combo_key)
        result = run_proxy_selection(
            train_feat_f=os.path.join(fd, "train_feature.f"),
            train_label_f=os.path.join(fd, "train_label.f"),
            test_feat_f=os.path.join(fd, "test_feature.f"),
            test_label_f=os.path.join(fd, "test_label.f"),
            output_dir=out_dir,
            alpha=args.alpha, gamma=args.gamma, alpha_l2=args.alpha_l2,
            batch_size=args.batch, namefile_path=args.namefile,
            top_k=args.top_k,
            indexed_combo=combo_info, split_indices=split_indices,
            indexed_row_batch_size=index_row_batch_size(manifest, combo_info),
        )
        return combo_key, result

    jobs = max(1, int(args.jobs))
    combo_results: dict[str, OrderedDict] = {}
    if jobs == 1 or len(combo_items) <= 1:
        for combo_key, combo_info in combo_items:
            key, result = _run_combo(combo_key, combo_info)
            combo_results[key] = result
    else:
        workers = min(jobs, len(combo_items))
        log(f"Parallel Step2 combos: jobs={workers}, combos={len(combo_items)}")
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(_run_combo, combo_key, combo_info): combo_key for combo_key, combo_info in combo_items}
            for fut in as_completed(futs):
                key, result = fut.result()
                combo_results[key] = result

    for combo_key, _combo_info in combo_items:
        mcp_manifest["combinations"][combo_key] = combo_results[combo_key]

    mp = os.path.join(args.output_dir, "mcp_per_offset_manifest.json")
    with open(mp, "w", encoding="utf-8") as f:
        json.dump(mcp_manifest, f, indent=2, ensure_ascii=False)
    log(f"\nDone. Manifest: {mp}")


if __name__ == "__main__":
    main()
