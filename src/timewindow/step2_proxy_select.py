#!/usr/bin/env python3
"""Step 2: pluggable proxy feature selection with the MCP manifest contract.

The selector never uses the test labels to choose features.  It selects on a
bounded, stratified-by-file training sample, fits a Ridge probe on that sample,
and touches the test set only once for reporting.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed

import joblib
import numpy as np
from scipy.sparse import csr_matrix, issparse, load_npz, vstack
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.feature_selection import f_regression, mutual_info_regression
from sklearn.linear_model import Ridge, SGDRegressor


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def mkdirp(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def read_filelist(path: str) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip() and not line.lstrip().startswith("#")]


def load_labels(path: str) -> np.ndarray:
    files = read_filelist(path)
    return np.concatenate([np.load(p).reshape(-1).astype(np.float64) for p in files])


def load_dataset(path: str):
    mats = [load_npz(p).tocsr() for p in read_filelist(path)]
    return vstack(mats, format="csr") if mats else csr_matrix((0, 0), dtype=np.float64)


def load_training_sample(feature_list: str, label_list: str, max_rows: int, seed: int):
    """Load a per-file balanced sample so one long case cannot dominate."""
    feat_files = read_filelist(feature_list)
    label_files = read_filelist(label_list)
    if len(feat_files) != len(label_files) or not feat_files:
        raise ValueError("train feature/label filelists must be non-empty and have equal length")
    rng = np.random.default_rng(seed)
    quota = 0 if max_rows <= 0 else max(1, int(np.ceil(max_rows / len(feat_files))))
    xs, ys = [], []
    for feat_path, label_path in zip(feat_files, label_files):
        x = load_npz(feat_path).tocsr()
        y = np.load(label_path).reshape(-1).astype(np.float64)
        if x.shape[0] != y.shape[0]:
            raise ValueError(f"row mismatch: {feat_path}={x.shape[0]}, {label_path}={y.shape[0]}")
        if quota and x.shape[0] > quota:
            idx = np.sort(rng.choice(x.shape[0], size=quota, replace=False))
            x, y = x[idx], y[idx]
        xs.append(x)
        ys.append(y)
    x_all = vstack(xs, format="csr")
    y_all = np.concatenate(ys)
    if max_rows > 0 and x_all.shape[0] > max_rows:
        idx = np.sort(rng.choice(x_all.shape[0], size=max_rows, replace=False))
        x_all, y_all = x_all[idx], y_all[idx]
    return x_all, y_all


def sparse_variance(x) -> np.ndarray:
    mean = np.asarray(x.mean(axis=0)).ravel()
    second = np.asarray(x.multiply(x).mean(axis=0)).ravel()
    return np.maximum(0.0, second - mean * mean)


def top_indices(scores: np.ndarray, top_k: int) -> np.ndarray:
    scores = np.nan_to_num(np.asarray(scores, dtype=np.float64), nan=-np.inf)
    k = min(max(1, top_k), scores.size)
    order = np.lexsort((np.arange(scores.size), -scores))[:k]
    return np.sort(order.astype(np.int64))


def dense_candidates(x, candidate_idx: np.ndarray, max_dense_elements: int) -> np.ndarray:
    elements = int(x.shape[0]) * int(candidate_idx.size)
    if elements > max_dense_elements:
        raise MemoryError(
            f"dense candidate matrix needs {elements:,} elements; "
            f"raise --max-dense-elements or reduce --prefilter-k/--max-train-rows"
        )
    block = x[:, candidate_idx]
    return block.toarray().astype(np.float32, copy=False) if issparse(block) else np.asarray(block, np.float32)


def prefilter_by_variance(x, prefilter_k: int) -> np.ndarray:
    n_features = x.shape[1]
    if prefilter_k <= 0 or prefilter_k >= n_features:
        return np.arange(n_features, dtype=np.int64)
    return top_indices(sparse_variance(x), prefilter_k)


def select_features(x, y: np.ndarray, args) -> tuple[np.ndarray, np.ndarray, dict]:
    """Return selected indices, full-length scores, and method metadata."""
    n_features = x.shape[1]
    scores = np.zeros(n_features, dtype=np.float64)
    meta: dict = {}
    method = args.selector

    if method == "f_regression":
        local_scores, _ = f_regression(x, y, center=False)
        scores[:] = np.nan_to_num(np.abs(local_scores))

    elif method == "elastic_net":
        model = SGDRegressor(
            loss="squared_error", penalty="elasticnet", alpha=args.alpha,
            l1_ratio=args.l1_ratio, max_iter=args.max_iter, tol=args.tol,
            random_state=args.seed, average=True,
        )
        model.fit(x, y)
        scores[:] = np.abs(model.coef_)
        meta["nonzero_coefficients"] = int(np.count_nonzero(model.coef_))

    elif method == "stability":
        rng = np.random.default_rng(args.seed)
        counts = np.zeros(n_features, dtype=np.float64)
        magnitudes = np.zeros(n_features, dtype=np.float64)
        sample_n = max(2, int(round(x.shape[0] * args.stability_fraction)))
        per_run_k = min(n_features, max(args.top_k, args.stability_candidates))
        for run in range(args.stability_runs):
            rows = rng.choice(x.shape[0], size=sample_n, replace=False)
            model = SGDRegressor(
                loss="squared_error", penalty="elasticnet", alpha=args.alpha,
                l1_ratio=args.l1_ratio, max_iter=args.max_iter, tol=args.tol,
                random_state=args.seed + run, average=True,
            )
            model.fit(x[rows], y[rows])
            coef = np.abs(model.coef_)
            chosen = top_indices(coef, per_run_k)
            counts[chosen] += 1.0
            magnitudes += coef
        frequency = counts / args.stability_runs
        scores[:] = frequency + 1e-9 * magnitudes / max(1, args.stability_runs)
        meta["selection_frequency_threshold"] = args.stability_threshold
        meta["stable_above_threshold"] = int(np.sum(frequency >= args.stability_threshold))

    else:
        candidates = prefilter_by_variance(x, args.prefilter_k)
        xd = dense_candidates(x, candidates, args.max_dense_elements)
        if method == "mutual_info":
            local = mutual_info_regression(xd, y, random_state=args.seed, n_neighbors=args.mi_neighbors)
        elif method == "extra_trees":
            model = ExtraTreesRegressor(
                n_estimators=args.n_estimators, max_depth=args.max_depth,
                min_samples_leaf=args.min_samples_leaf, max_features=args.max_features,
                n_jobs=args.n_jobs, random_state=args.seed,
            )
            model.fit(xd, y)
            local = model.feature_importances_
        elif method == "mrmr":
            relevance = mutual_info_regression(
                xd, y, random_state=args.seed, n_neighbors=args.mi_neighbors,
            )
            pool_n = min(args.mrmr_pool_k, candidates.size)
            pool_local = top_indices(relevance, pool_n)
            xp = xd[:, pool_local].astype(np.float64, copy=False)
            xp -= xp.mean(axis=0, keepdims=True)
            norm = np.linalg.norm(xp, axis=0)
            norm[norm < 1e-12] = 1.0
            xp /= norm
            chosen_local: list[int] = []
            remaining = set(range(pool_n))
            pool_rel = relevance[pool_local]
            while remaining and len(chosen_local) < min(args.top_k, pool_n):
                if not chosen_local:
                    pick = int(np.argmax(pool_rel))
                else:
                    rem = np.fromiter(sorted(remaining), dtype=np.int64)
                    redundancy = np.mean(np.abs(xp[:, rem].T @ xp[:, chosen_local]), axis=1)
                    criterion = pool_rel[rem] - args.mrmr_redundancy * redundancy
                    pick = int(rem[int(np.argmax(criterion))])
                chosen_local.append(pick)
                remaining.remove(pick)
            selected = candidates[pool_local[np.asarray(chosen_local, dtype=np.int64)]]
            scores[candidates] = relevance
            meta.update({"prefiltered_features": int(candidates.size), "mrmr_pool": int(pool_n)})
            return np.sort(selected), scores, meta
        else:
            raise ValueError(f"unknown selector: {method}")
        scores[candidates] = local
        meta["prefiltered_features"] = int(candidates.size)

    return top_indices(scores, args.top_k), scores, meta


def evaluate_predictions(y: np.ndarray, pred: np.ndarray) -> OrderedDict:
    denom = np.maximum(np.abs(y), 1e-12)
    err = y - pred
    mse = float(np.mean(err * err))
    var = float(np.var(y))
    mean = float(np.mean(y))
    return OrderedDict([
        ("n", int(y.size)), ("MAE", float(np.mean(np.abs(err)))),
        ("MAPE", float(np.mean(np.abs(err) / denom) * 100.0)),
        ("RMSE", float(np.sqrt(mse))),
        ("NRMSE", float(np.sqrt(mse) / mean) if abs(mean) > 1e-12 else float("nan")),
        ("R2", float(1.0 - mse / var) if var > 1e-12 else float("nan")),
    ])


def save_names(namefile: str | None, selected: np.ndarray, output_dir: str) -> None:
    if not namefile or not os.path.exists(namefile):
        return
    with open(namefile, "r", encoding="utf-8") as f:
        names = [line.strip() for line in f]
    with open(os.path.join(output_dir, "selected_feature_names.txt"), "w", encoding="utf-8") as f:
        for idx in selected:
            if idx < len(names):
                f.write(names[idx] + "\n")


def run_combo(combo_key: str, combo_info: dict, args) -> OrderedDict:
    output_dir = os.path.join(args.output_dir, combo_key)
    mkdirp(output_dir)
    fd = combo_info["filelist_dir"]
    x_train, y_train = load_training_sample(
        os.path.join(fd, "train_feature.f"), os.path.join(fd, "train_label.f"),
        args.max_train_rows, args.seed,
    )
    log(f"{combo_key}: training sample={x_train.shape}, nnz={x_train.nnz}")
    selected, scores, method_meta = select_features(x_train, y_train, args)

    probe = Ridge(alpha=args.probe_alpha, solver="lsqr")
    probe.fit(x_train[:, selected], y_train)
    x_test = load_dataset(os.path.join(fd, "test_feature.f"))
    y_test = load_labels(os.path.join(fd, "test_label.f"))
    pred = probe.predict(x_test[:, selected])
    metrics = evaluate_predictions(y_test, pred)

    np.savetxt(os.path.join(output_dir, "selected_indices.txt"), selected, fmt="%d")
    np.savetxt(os.path.join(output_dir, "feature_scores.txt"), scores, fmt="%.12g")
    joblib.dump(probe, os.path.join(output_dir, "probe_model.joblib"))
    save_names(args.namefile, selected, output_dir)
    eval_dir = os.path.join(output_dir, "eval")
    mkdirp(eval_dir)
    with open(os.path.join(eval_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    np.savetxt(os.path.join(eval_dir, "predictions.txt"), pred)
    log(f"{combo_key}: selected={selected.size}, test MAPE={metrics['MAPE']:.3f}%")
    return OrderedDict([
        ("selector", args.selector), ("top_k", int(args.top_k)),
        ("n_features_total", int(x_train.shape[1])), ("n_selected", int(selected.size)),
        ("selection_uses_test_labels", False), ("train_sample_rows", int(x_train.shape[0])),
        ("test_mape", float(metrics["MAPE"])), ("test_rmse", float(metrics["RMSE"])),
        ("selected_indices_file", os.path.join(output_dir, "selected_indices.txt")),
        ("score_file", os.path.join(output_dir, "feature_scores.txt")),
        ("method_meta", method_meta),
    ])


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Pluggable per-offset proxy feature selection")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--namefile", default=None)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--selector", choices=["f_regression", "mutual_info", "elastic_net", "stability", "extra_trees", "mrmr"], default="stability")
    ap.add_argument("--top-k", type=int, default=200)
    ap.add_argument("--max-train-rows", type=int, default=10000)
    ap.add_argument("--prefilter-k", type=int, default=5000)
    ap.add_argument("--max-dense-elements", type=int, default=50_000_000)
    ap.add_argument("--alpha", type=float, default=1e-4)
    ap.add_argument("--l1-ratio", type=float, default=0.8)
    ap.add_argument("--max-iter", type=int, default=1000)
    ap.add_argument("--tol", type=float, default=1e-3)
    ap.add_argument("--stability-runs", type=int, default=20)
    ap.add_argument("--stability-fraction", type=float, default=0.5)
    ap.add_argument("--stability-threshold", type=float, default=0.6)
    ap.add_argument("--stability-candidates", type=int, default=400)
    ap.add_argument("--mi-neighbors", type=int, default=3)
    ap.add_argument("--mrmr-pool-k", type=int, default=1000)
    ap.add_argument("--mrmr-redundancy", type=float, default=1.0)
    ap.add_argument("--n-estimators", type=int, default=200)
    ap.add_argument("--max-depth", type=int, default=None)
    ap.add_argument("--min-samples-leaf", type=int, default=2)
    ap.add_argument("--max-features", type=float, default=0.5)
    ap.add_argument("--probe-alpha", type=float, default=8.0)
    ap.add_argument("--n-jobs", type=int, default=1)
    ap.add_argument("--jobs", type=int, default=1, help="parallel workers across combinations")
    ap.add_argument("--seed", type=int, default=108)
    ap.add_argument("--include-combos", default="")
    return ap


def main() -> None:
    args = build_parser().parse_args()
    if args.top_k <= 0 or args.stability_runs <= 0:
        raise SystemExit("ERROR: --top-k and --stability-runs must be positive")
    with open(args.manifest, "r", encoding="utf-8") as f:
        manifest = json.load(f, object_pairs_hook=OrderedDict)
    items = list(manifest.get("combinations", {}).items())
    if args.include_combos.strip():
        include = {s.strip() for s in args.include_combos.split(",") if s.strip()}
        missing = include.difference(dict(items))
        if missing:
            raise SystemExit(f"ERROR: --include-combos not in manifest: {sorted(missing)}")
        items = [(key, value) for key, value in items if key in include]

    results: dict[str, OrderedDict] = {}
    if args.jobs == 1 or len(items) <= 1:
        for key, info in items:
            results[key] = run_combo(key, info, args)
    else:
        with ThreadPoolExecutor(max_workers=min(args.jobs, len(items))) as executor:
            futures = {executor.submit(run_combo, key, info, args): key for key, info in items}
            for future in as_completed(futures):
                results[futures[future]] = future.result()

    output = OrderedDict([
        ("selector", args.selector), ("top_k", args.top_k),
        ("selection_uses_test_labels", False), ("combinations", OrderedDict()),
    ])
    for key, _ in items:
        output["combinations"][key] = results[key]
    mkdirp(args.output_dir)
    manifest_path = os.path.join(args.output_dir, "mcp_per_offset_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    log(f"Done. Compatible manifest: {manifest_path}")


if __name__ == "__main__":
    main()
