#!/usr/bin/env python3
"""
Step 2: Per-Offset XGBoost Proxy Selection.

For each (data_type, offset) combination, train a lightweight XGBoost
regressor and export the top-K feature columns by model importance. The output
manifest intentionally keeps the Step2/Step3 contract used by
step2_mcp_per_offset.py so downstream concatenation and training are unchanged.

Version: 2.1-data-mode (2026-08-24)
Changes:
  * Stream every train/test shard through xgboost.DataIter.
  * Add --data-mode iter|in_memory; both modes train one model on all files.
  * Build one QuantileDMatrix and train one model per combination.
  * Keep --batch only as a deprecated, ignored compatibility option.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from scipy.sparse import load_npz, vstack, csr_matrix
import xgboost as xgb
from utils.indexed_split import (
    index_row_batch_size,
    indexed_split_paths,
    iter_indexed_combo_batches,
    load_indexed_combo_labels,
)


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


def load_dataset(filelist_path: str):
    files = read_filelist(filelist_path)
    mats = [load_npz(fp).tocsr() for fp in files]
    if not mats:
        return csr_matrix((0, 0), dtype=np.float64)
    return vstack(mats, format="csr")

def paired_files(feature_filelist: str, label_filelist: str) -> list[tuple[str, str]]:
    """Read and validate corresponding sparse-feature/label file lists."""
    feature_files = read_filelist(feature_filelist)
    label_files = read_filelist(label_filelist)
    if not feature_files:
        raise ValueError(f"Empty feature filelist: {feature_filelist}")
    if len(feature_files) != len(label_files):
        raise ValueError(
            f"Feature/label file count mismatch: {len(feature_files)} != "
            f"{len(label_files)} ({feature_filelist}, {label_filelist})"
        )
    return list(zip(feature_files, label_files))


class CSRFileDataIter(xgb.DataIter):
    """Stream one CSR/NumPy file pair at a time into XGBoost."""

    def __init__(self, file_pairs: list[tuple[str, str]], name: str):
        self.file_pairs = list(file_pairs)
        self.name = name
        self._cursor = 0
        self.n_features: int | None = None
        self.n_rows = 0
        self._pass_rows = 0
        try:
            super().__init__()
        except TypeError:
            # Compatibility with older DataIter constructor signatures.
            pass

    def reset(self) -> None:
        self._cursor = 0
        self._pass_rows = 0

    def next(self, input_data) -> int:
        if self._cursor >= len(self.file_pairs):
            return 0
        feature_file, label_file = self.file_pairs[self._cursor]
        self._cursor += 1
        started = time.time()
        X = load_npz(feature_file).tocsr()
        y = np.load(label_file).reshape(-1).astype(np.float32, copy=False)
        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"Feature/label row mismatch: {X.shape[0]} != {y.shape[0]} "
                f"({feature_file}, {label_file})"
            )
        if self.n_features is None:
            self.n_features = int(X.shape[1])
        elif X.shape[1] != self.n_features:
            raise ValueError(
                f"Inconsistent feature count: {X.shape[1]} != {self.n_features} "
                f"in {feature_file}"
            )
        self._pass_rows += int(X.shape[0])
        self.n_rows = max(self.n_rows, self._pass_rows)
        log(
            f"  DataIter[{self.name}] {self._cursor}/{len(self.file_pairs)}: "
            f"shape={X.shape}, nnz={X.nnz}, t={time.time()-started:.1f}s"
        )
        input_data(data=X, label=y)
        return 1


class IndexedCSRDataIter(xgb.DataIter):
    """Stream one virtual index split in bounded row batches."""

    def __init__(self, combo: dict, indices_file: str, row_batch_size: int, name: str):
        self.combo = combo
        self.indices_file = indices_file
        self.row_batch_size = int(row_batch_size)
        self.name = name
        self._iterator = None
        self._batch = 0
        self.n_features: int | None = None
        self.n_rows = int(np.load(indices_file, mmap_mode="r").size)
        try:
            super().__init__()
        except TypeError:
            pass

    def reset(self) -> None:
        self._iterator = iter_indexed_combo_batches(
            self.combo, self.indices_file, self.row_batch_size
        )
        self._batch = 0

    def next(self, input_data) -> int:
        if self._iterator is None:
            self.reset()
        try:
            matrix, labels, _weights = next(self._iterator)
        except StopIteration:
            return 0
        self._batch += 1
        if self.n_features is None:
            self.n_features = int(matrix.shape[1])
        elif matrix.shape[1] != self.n_features:
            raise ValueError(
                f"Inconsistent indexed feature count: {matrix.shape[1]} != {self.n_features}"
            )
        log(
            f"  IndexedDataIter[{self.name}] batch={self._batch}: "
            f"shape={matrix.shape}, nnz={matrix.nnz}"
        )
        input_data(data=matrix, label=labels)
        return 1


def make_indexed_quantile_dmatrix(
    combo: dict,
    indices_file: str,
    row_batch_size: int,
    name: str,
    max_bin: int,
    ref=None,
):
    data_iter = IndexedCSRDataIter(combo, indices_file, row_batch_size, name)
    kwargs = {"max_bin": int(max_bin)}
    if ref is not None:
        kwargs["ref"] = ref
    try:
        matrix = xgb.QuantileDMatrix(data_iter, **kwargs)
    except TypeError:
        matrix = xgb.QuantileDMatrix(data_iter)
    if data_iter.n_features is None or data_iter.n_rows <= 0:
        raise ValueError(f"IndexedDataIter produced no rows for {name}")
    return matrix, data_iter.n_features, data_iter.n_rows


def make_quantile_dmatrix(
    feature_filelist: str,
    label_filelist: str,
    name: str,
    max_bin: int,
    ref=None,
):
    """Build a QuantileDMatrix without vertically stacking source files."""
    data_iter = CSRFileDataIter(paired_files(feature_filelist, label_filelist), name)
    kwargs = {"max_bin": int(max_bin)}
    if ref is not None:
        kwargs["ref"] = ref
    try:
        matrix = xgb.QuantileDMatrix(data_iter, **kwargs)
    except TypeError:
        matrix = xgb.QuantileDMatrix(data_iter)
    if data_iter.n_features is None or data_iter.n_rows <= 0:
        raise ValueError(f"DataIter produced no rows for {name}")
    return matrix, data_iter.n_features, data_iter.n_rows


def load_labels(filelist_path: str) -> np.ndarray:
    files = read_filelist(filelist_path)
    return np.concatenate([np.load(fp).reshape(-1).astype(np.float64) for fp in files])


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.where(np.abs(y_true) > 1e-12, np.abs(y_true), 1e-12)
    return float(np.average(np.abs(y_true - y_pred) / denom) * 100.0)


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
    with open(os.path.join(eval_dir, "metrics.csv"), "w", encoding="utf-8") as f:
        f.write("metric,value\n")
        for key, value in metrics.items():
            f.write(f"{key},{value}\n")
    if pred is not None:
        np.savetxt(os.path.join(eval_dir, "predictions.txt"), pred)


def parse_feature_id(fid: str) -> int | None:
    if fid.startswith("f") and fid[1:].isdigit():
        return int(fid[1:])
    if fid.isdigit():
        return int(fid)
    return None


def select_topk_from_model(model: xgb.Booster, n_features: int, top_k: int) -> np.ndarray:
    scores = model.get_score(importance_type="gain")
    importance_type = "gain"
    if not scores:
        scores = model.get_score(importance_type="weight")
        importance_type = "weight"

    ranked: list[tuple[int, float]] = []
    for fid, score in scores.items():
        idx = parse_feature_id(fid)
        if idx is not None and 0 <= idx < n_features:
            ranked.append((idx, float(score)))

    ranked.sort(key=lambda item: (-item[1], item[0]))
    selected = [idx for idx, _score in ranked[:top_k]]
    if len(selected) < min(top_k, n_features):
        used = set(selected)
        for idx in range(n_features):
            if idx not in used:
                selected.append(idx)
                used.add(idx)
            if len(selected) >= min(top_k, n_features):
                break
    log(f"  Importance type={importance_type}, nonzero importance columns={len(ranked)}")
    return np.array(sorted(set(selected)), dtype=np.int64)


def save_selected_names(namefile_path: str | None, selected: np.ndarray, output_dir: str) -> None:
    if not namefile_path or not os.path.exists(namefile_path):
        return
    with open(namefile_path) as f:
        names = [line.strip() for line in f]
    sel_names = [names[i] for i in selected if i < len(names)]
    with open(os.path.join(output_dir, "selected_feature_names.txt"), "w") as f:
        for name in sel_names:
            f.write(name + "\n")


def run_proxy_selection(
    train_feat_f: str,
    train_label_f: str,
    test_feat_f: str,
    test_label_f: str,
    output_dir: str,
    namefile_path: str | None,
    top_k: int,
    rounds: int,
    params: dict,
    max_bin: int,
    data_mode: str,
    indexed_combo: dict | None = None,
    split_indices: dict[str, str] | None = None,
    indexed_row_batch_size: int = 200_000,
) -> OrderedDict:
    """Train one model using either streamed or in-memory input."""
    mkdirp(output_dir)
    started = time.time()

    effective_data_mode = data_mode
    if split_indices is not None:
        if indexed_combo is None:
            raise ValueError("indexed split requires combo metadata")
        log("  Streaming virtual index split from original Step1 shards...")
        dtrain, n_features, n_train = make_indexed_quantile_dmatrix(
            indexed_combo, split_indices["train"], indexed_row_batch_size,
            "train", max_bin=max_bin,
        )
        dtest, n_test_features, n_test = make_indexed_quantile_dmatrix(
            indexed_combo, split_indices["test"], indexed_row_batch_size,
            "test", max_bin=max_bin, ref=dtrain,
        )
        if n_features != n_test_features:
            raise ValueError(
                f"Train/test feature count mismatch: {n_features} != {n_test_features}"
            )
        y_test, _test_weight = load_indexed_combo_labels(
            indexed_combo, split_indices["test"]
        )
        if y_test.size != n_test:
            raise ValueError(f"Test label count mismatch: {y_test.size} != {n_test}")
        effective_data_mode = "indexed_iter"
    elif data_mode == "iter":
        log("  Building training QuantileDMatrix from DataIter...")
        dtrain, n_features, n_train = make_quantile_dmatrix(
            train_feat_f, train_label_f, "train", max_bin=max_bin
        )
        log("  Building test QuantileDMatrix from DataIter...")
        dtest, n_test_features, n_test = make_quantile_dmatrix(
            test_feat_f, test_label_f, "test", max_bin=max_bin, ref=dtrain
        )
        if n_test_features != n_features:
            raise ValueError(
                f"Train/test feature count mismatch: {n_features} != {n_test_features}"
            )
        y_test = load_labels(test_label_f)
        if y_test.shape[0] != n_test:
            raise ValueError(f"Test label count mismatch: {y_test.shape[0]} != {n_test}")
    elif data_mode == "in_memory":
        log("  Loading complete train/test matrices in memory...")
        X_train = load_dataset(train_feat_f)
        y_train = load_labels(train_label_f)
        X_test = load_dataset(test_feat_f)
        y_test = load_labels(test_label_f)
        if X_train.shape[0] != y_train.shape[0]:
            raise ValueError(
                f"Train feature/label row mismatch: {X_train.shape[0]} != {y_train.shape[0]}"
            )
        if X_test.shape[0] != y_test.shape[0]:
            raise ValueError(
                f"Test feature/label row mismatch: {X_test.shape[0]} != {y_test.shape[0]}"
            )
        if X_train.shape[1] != X_test.shape[1]:
            raise ValueError(
                f"Train/test feature count mismatch: {X_train.shape[1]} != {X_test.shape[1]}"
            )
        n_train = int(X_train.shape[0])
        n_features = int(X_train.shape[1])
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)
    else:
        raise ValueError(f"Unsupported data_mode: {data_mode}")

    model = xgb.train(params, dtrain, num_boost_round=rounds)
    selected = select_topk_from_model(model, n_features, top_k)
    if len(selected) == 0:
        raise RuntimeError(f"No features were selected for {output_dir}")

    pred = model.predict(dtest)
    metrics = evaluate_predictions(y_test, pred)
    score = float(metrics["MAPE"])
    log(
        f"  Single {effective_data_mode} model: train_rows={n_train}, selected={len(selected)}, "
        f"XGB_MAPE={score:.3f}%, RMSE={metrics['RMSE']:.6g}, "
        f"NRMSE={metrics['NRMSE']:.6g}, R2={metrics['R2']:.6g}, "
        f"t={time.time()-started:.1f}s"
    )

    model_path = os.path.join(output_dir, "xgb_proxy_model.json")
    model.save_model(model_path)
    np.savetxt(os.path.join(output_dir, "selected_indices.txt"), selected, fmt="%d")
    save_selected_names(namefile_path, selected, output_dir)
    save_eval_outputs(output_dir, metrics, pred)

    return OrderedDict([
        ("selector", "xgboost"),
        ("version", "2.1-data-mode"),
        ("training_mode", effective_data_mode),
        ("importance_type", "gain"),
        ("top_k", int(top_k)),
        ("rounds", int(rounds)),
        ("max_bin", int(max_bin)),
        ("n_train", int(n_train)),
        ("n_features_total", int(n_features)),
        ("n_selected", int(len(selected))),
        ("best_mape", score),
        ("test_rmse", float(metrics.get("RMSE", float("nan")))),
        ("test_nrmse", float(metrics.get("NRMSE", float("nan")))),
        ("test_r2", float(metrics.get("R2", float("nan")))),
        ("eval_metrics_file", os.path.join(output_dir, "eval", "metrics.json")),
        ("selected_indices_file", os.path.join(output_dir, "selected_indices.txt")),
        ("model_file", model_path),
    ])


def main() -> None:
    ap = argparse.ArgumentParser(description="Per-offset XGBoost proxy selection")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--namefile", default=None)
    ap.add_argument("--batch", type=int, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--top-k", type=int, default=200)
    ap.add_argument("--rounds", type=int, default=50)
    ap.add_argument("--max-bin", type=int, default=256)
    ap.add_argument(
        "--data-mode", choices=("iter", "in_memory"), default="iter",
        help="iter streams shards with DataIter; in_memory vstacks all shards before training",
    )
    ap.add_argument("--eta", type=float, default=0.08)
    ap.add_argument("--max-depth", type=int, default=4)
    ap.add_argument("--subsample", type=float, default=0.9)
    ap.add_argument("--colsample-bytree", type=float, default=0.8)
    ap.add_argument("--min-child-weight", type=float, default=1.0)
    ap.add_argument("--reg-lambda", type=float, default=1.0)
    ap.add_argument("--nthread", type=int, default=1)
    ap.add_argument("--seed", type=int, default=108)
    ap.add_argument("--jobs", type=int, default=1,
                    help="Parallel workers across (offset, data_type) combinations")
    ap.add_argument("--include-combos", default="",
                    help="Comma-separated combo keys to process, e.g. tog_offset12")
    args = ap.parse_args()

    if args.top_k <= 0:
        raise SystemExit("ERROR: --top-k must be positive")
    if args.max_bin <= 1:
        raise SystemExit("ERROR: --max-bin must be greater than 1")
    if args.batch is not None:
        log("WARNING: --batch is deprecated and ignored; one DataIter model uses all files")

    with open(args.manifest) as f:
        manifest = json.load(f, object_pairs_hook=OrderedDict)
    split_indices = indexed_split_paths(manifest)

    params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "tree_method": "hist",
        "eta": args.eta,
        "max_depth": args.max_depth,
        "subsample": args.subsample,
        "colsample_bytree": args.colsample_bytree,
        "min_child_weight": args.min_child_weight,
        "lambda": args.reg_lambda,
        "nthread": args.nthread,
        "seed": args.seed,
        "max_bin": args.max_bin,
    }

    combos = manifest.get("combinations", {})
    log(f"Processing {len(combos)} combinations (selector=xgboost, top_k={args.top_k})")

    out_manifest = OrderedDict([
        ("selector", "xgboost"),
        ("top_k", args.top_k),
        ("rounds", args.rounds),
        ("version", "2.1-data-mode"),
        ("training_mode", args.data_mode),
        ("params", OrderedDict(sorted(params.items()))),
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
            namefile_path=args.namefile,
            top_k=args.top_k,
            rounds=args.rounds,
            params=params,
            max_bin=args.max_bin,
            data_mode=args.data_mode,
            indexed_combo=combo_info,
            split_indices=split_indices,
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
        out_manifest["combinations"][combo_key] = combo_results[combo_key]

    mp = os.path.join(args.output_dir, "mcp_per_offset_manifest.json")
    with open(mp, "w", encoding="utf-8") as f:
        json.dump(out_manifest, f, indent=2, ensure_ascii=False)
    log(f"\nDone. Manifest: {mp}")


if __name__ == "__main__":
    main()
