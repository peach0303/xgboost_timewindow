# -*- coding: UTF-8 -*-
"""
Step2 后的流式 Ridge 功耗预测验证。

Version: 1.0-ridge-streaming (2026-08-25)
Reference: xgb_power_v2_3.py

保持参考实现不变的部分：
1. Step1/Step2 manifest 输入与 build_pipeline_chunks 数据构建。
2. cache CSR chunk 格式、固定 pipeline test files、train/val shuffle split。
3. 逐 chunk 预测 CSV 的字段与 split_type 语义。
4. evaluate_prediction_csv 的参数和全部评价输出。

唯一的模型替换：
- XGBoost Booster -> L2 Ridge。
- Ridge 使用矩阵自由的共轭梯度求解正规方程；每次 matvec 按相同 chunk
  和 split 索引加载数据，不 vstack 全量训练矩阵。
"""
from __future__ import print_function

import argparse
import json
import os
import time

import joblib
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.sparse import load_npz
from scipy.sparse.linalg import LinearOperator, cg

from pipeline_combo_dataset import build_pipeline_chunks
from xgb_power_eval_from_prediction_csv import evaluate_prediction_csv
from xgb_power_v2_3 import (
    count_chunk_rows,
    ensure_dir,
    list_chunk_meta_files,
    parse_top_ns,
    setup_directories,
)


VERSION = "1.0-ridge-streaming"


class RidgeLinearModel(object):
    """Small serializable predictor with the sklearn-compatible predict API."""

    def __init__(self, coef, intercept, alpha, solver_info):
        self.coef_ = np.asarray(coef, dtype=np.float64).reshape(-1)
        self.intercept_ = float(intercept)
        self.alpha = float(alpha)
        self.solver_info = int(solver_info)

    def predict(self, X):
        return np.asarray(X.dot(self.coef_)).reshape(-1) + self.intercept_


class ChunkSplitView(object):
    """Iterate the rows belonging to one split without materializing all chunks."""

    def __init__(self, chunk_x_files, chunk_y_files, row_indices, name):
        self.chunk_x_files = list(chunk_x_files)
        self.chunk_y_files = list(chunk_y_files)
        self.row_indices = np.sort(np.asarray(row_indices, dtype=np.int64).reshape(-1))
        self.name = str(name)

        if len(self.chunk_x_files) != len(self.chunk_y_files):
            raise ValueError("chunk_x_files count != chunk_y_files count")
        if self.row_indices.size == 0:
            raise ValueError("row_indices is empty for %s" % self.name)

        self.chunk_rows, self.total_rows = count_chunk_rows(self.chunk_y_files)
        self.chunk_offsets = np.zeros(len(self.chunk_rows) + 1, dtype=np.int64)
        if self.chunk_rows:
            self.chunk_offsets[1:] = np.cumsum(np.asarray(self.chunk_rows, dtype=np.int64))
        if int(self.row_indices[0]) < 0 or int(self.row_indices[-1]) >= self.total_rows:
            raise ValueError("row_indices out of range for %s" % self.name)

        first_shape = load_npz(self.chunk_x_files[0]).shape
        self.n_features = int(first_shape[1])
        self.n_rows = int(self.row_indices.size)

    def iter_chunks(self, verbose=False):
        yielded = 0
        for i, (xf, yf) in enumerate(zip(self.chunk_x_files, self.chunk_y_files)):
            c0 = int(self.chunk_offsets[i])
            c1 = int(self.chunk_offsets[i + 1])
            left = np.searchsorted(self.row_indices, c0, side="left")
            right = np.searchsorted(self.row_indices, c1, side="left")
            if right <= left:
                continue

            t0 = time.time()
            local_rows = self.row_indices[left:right] - c0
            X_full = load_npz(xf).tocsr()
            y_full = np.load(yf).reshape(-1)
            if X_full.shape[0] != y_full.shape[0]:
                raise ValueError(
                    "X/y rows mismatch in chunk %d: %d != %d"
                    % (i, X_full.shape[0], y_full.shape[0])
                )
            if X_full.shape[1] != self.n_features:
                raise ValueError(
                    "feature count mismatch in chunk %d: %d != %d"
                    % (i, X_full.shape[1], self.n_features)
                )

            X = X_full[local_rows, :].tocsr()
            y = np.asarray(y_full[local_rows], dtype=np.float64).reshape(-1)
            yielded += int(X.shape[0])
            if verbose:
                print("[RIDGE-%s] chunk %d/%d rows=%d X=%s nnz=%d time=%.2fs" % (
                    self.name, i + 1, len(self.chunk_x_files), X.shape[0],
                    str(X.shape), X.nnz, time.time() - t0
                ))
            yield X, y
            del X_full, y_full, X, y

        if yielded != self.n_rows:
            raise RuntimeError(
                "split %s yielded rows %d != expected %d"
                % (self.name, yielded, self.n_rows)
            )


class StreamingRidgeSolver(object):
    """Solve (X'X + alpha I)w = X'y with chunk-streamed CG."""

    def __init__(self, alpha=8.0, max_iter=200, tol=1e-6, log_every=1,
                 fit_intercept=True):
        if float(alpha) < 0:
            raise ValueError("ridge alpha must be non-negative")
        if int(max_iter) <= 0:
            raise ValueError("ridge max_iter must be positive")
        if float(tol) <= 0:
            raise ValueError("ridge tol must be positive")
        self.alpha = float(alpha)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.log_every = max(1, int(log_every))
        self.fit_intercept = bool(fit_intercept)
        self.model = None
        self.iterations = 0

    def fit(self, train_view):
        p = int(train_view.n_features)
        q = p + (1 if self.fit_intercept else 0)
        rhs = np.zeros(q, dtype=np.float64)
        diag = np.zeros(q, dtype=np.float64)

        print("[RIDGE] build rhs/preconditioner: rows=%d features=%d alpha=%g" % (
            train_view.n_rows, p, self.alpha
        ))
        for X, y in train_view.iter_chunks(verbose=True):
            rhs[:p] += np.asarray(X.T.dot(y)).reshape(-1)
            diag[:p] += np.asarray(X.multiply(X).sum(axis=0)).reshape(-1)
            if self.fit_intercept:
                rhs[p] += float(np.sum(y))
                diag[p] += float(X.shape[0])

        alpha = self.alpha
        fit_intercept = self.fit_intercept
        matvec_calls = [0]

        def normal_matvec(v):
            matvec_calls[0] += 1
            v = np.asarray(v, dtype=np.float64).reshape(-1)
            out = np.zeros(q, dtype=np.float64)
            out[:p] = alpha * v[:p]
            intercept = float(v[p]) if fit_intercept else 0.0
            for X, _y in train_view.iter_chunks(verbose=False):
                projected = np.asarray(X.dot(v[:p])).reshape(-1) + intercept
                out[:p] += np.asarray(X.T.dot(projected)).reshape(-1)
                if fit_intercept:
                    out[p] += float(np.sum(projected))
            return out

        operator = LinearOperator((q, q), matvec=normal_matvec, dtype=np.float64)
        precond_diag = diag.copy()
        precond_diag[:p] += alpha
        safe_diag = np.where(precond_diag > np.finfo(np.float64).eps, precond_diag, 1.0)
        preconditioner = LinearOperator(
            (q, q), matvec=lambda v: np.asarray(v) / safe_diag, dtype=np.float64
        )

        started = time.time()

        def callback(_xk):
            self.iterations += 1
            if self.iterations == 1 or self.iterations % self.log_every == 0:
                print("[RIDGE-CG] iteration=%d matvec_calls=%d elapsed=%.2fs" % (
                    self.iterations, matvec_calls[0], time.time() - started
                ))

        kwargs = dict(A=operator, b=rhs, M=preconditioner,
                      maxiter=self.max_iter, callback=callback)
        try:
            solution, info = cg(rtol=self.tol, atol=0.0, **kwargs)
        except TypeError:
            solution, info = cg(tol=self.tol, **kwargs)

        if int(info) < 0:
            raise RuntimeError("Ridge CG failed with illegal input/breakdown: info=%d" % int(info))
        if int(info) > 0:
            print("[WARN] Ridge CG did not reach tolerance; iterations/info=%d" % int(info))
        else:
            print("[RIDGE-CG] converged iterations=%d elapsed=%.2fs" % (
                self.iterations, time.time() - started
            ))

        coef = solution[:p]
        intercept = solution[p] if fit_intercept else 0.0
        self.model = RidgeLinearModel(coef, intercept, alpha, info)
        return self.model


def split_metrics(model, split_view):
    count = 0
    sum_sq = 0.0
    sum_abs = 0.0
    for X, y in split_view.iter_chunks(verbose=False):
        pred = model.predict(X)
        residual = y - pred
        count += int(y.shape[0])
        sum_sq += float(np.dot(residual, residual))
        sum_abs += float(np.sum(np.abs(residual)))
    mse = sum_sq / count
    return {
        "n": int(count),
        "mae": float(sum_abs / count),
        "mse": float(mse),
        "rmse": float(np.sqrt(mse)),
    }


def build_split_indices(chunk_y_files, fold, shuffle_seed, chunk_meta_files,
                        fixed_test_file_indices):
    fold = int(fold)
    _chunk_rows, total_rows = count_chunk_rows(chunk_y_files)
    if fold < 2:
        raise ValueError("fold must be at least 2")

    rng = np.random.RandomState(int(shuffle_seed))
    if fixed_test_file_indices is None:
        start = int(total_rows * (fold - 2) / fold)
        end = int(total_rows * (fold - 1) / fold)
        shuffled = rng.permutation(total_rows).astype(np.int64, copy=False)
        train_indices = shuffled[:start]
        val_indices = shuffled[start:end]
        test_indices = shuffled[end:]
        split_mode = "random-all-samples"
        fixed_ids = None
    else:
        if chunk_meta_files is None:
            raise ValueError("fixed file split requires chunk metadata")
        file_idx_parts = []
        for meta_file in chunk_meta_files:
            meta = np.load(meta_file, mmap_mode="r")
            file_idx_parts.append(np.asarray(meta[:, 0], dtype=np.int64))
        sample_file_idx = np.concatenate(file_idx_parts)
        if sample_file_idx.shape[0] != total_rows:
            raise RuntimeError("chunk metadata row count != total rows")

        fixed_ids = np.asarray(fixed_test_file_indices, dtype=np.int64).reshape(-1)
        test_mask = np.isin(sample_file_idx, fixed_ids)
        test_indices = np.flatnonzero(test_mask).astype(np.int64, copy=False)
        train_pool = np.flatnonzero(~test_mask).astype(np.int64, copy=False)
        if test_indices.size == 0:
            raise ValueError("fixed pipeline test file indices have no samples")
        if train_pool.size < fold:
            raise ValueError("training pool is too small for fold=%d" % fold)

        shuffled_pool = rng.permutation(train_pool)
        val_rows = int(train_pool.size / fold)
        val_indices = shuffled_pool[-val_rows:]
        train_indices = shuffled_pool[:-val_rows]
        split_mode = "fixed-pipeline-test-files"

    assignment = np.empty(total_rows, dtype=object)
    assignment[train_indices] = "train"
    assignment[val_indices] = "val"
    assignment[test_indices] = "test"

    print("[TRAIN] total rows=%d" % total_rows)
    print("[TRAIN] split mode: %s" % split_mode)
    if fixed_ids is not None:
        print("[TRAIN] fixed test file indices=%s" % fixed_ids.tolist())
    print("[TRAIN] shuffle training pool before val split: enabled seed=%d" % int(shuffle_seed))
    print("[TRAIN] train n=%d" % train_indices.size)
    print("[TRAIN] val   n=%d" % val_indices.size)
    print("[TRAIN] test  n=%d" % test_indices.size)
    return train_indices, val_indices, test_indices, assignment, total_rows


def export_sample_prediction_csv(model, chunk_x_files, chunk_y_files, chunk_meta_files,
                                 out_csv, split_assignment):
    """Same prediction CSV schema and chunk loading as xgb_power_v2_3.py."""
    ensure_dir(os.path.dirname(out_csv))
    rows_written = 0
    global_offset = 0
    first = True

    for chunk_id, (xf, yf) in enumerate(zip(chunk_x_files, chunk_y_files)):
        t0 = time.time()
        X = load_npz(xf).tocsr()
        y = np.load(yf).reshape(-1)
        pred = model.predict(X).reshape(-1)
        n = int(y.shape[0])
        if X.shape[0] != n or pred.shape[0] != n:
            raise RuntimeError("prediction rows mismatch at chunk %d" % chunk_id)

        if chunk_meta_files is not None:
            meta = np.load(chunk_meta_files[chunk_id])
            if meta.shape[0] != n:
                raise RuntimeError("meta rows mismatch at chunk %d" % chunk_id)
            file_idx = meta[:, 0].astype(np.int64)
            label_idx = meta[:, 1].astype(np.int64)
            cycle_index = meta[:, 2].astype(np.int64)
        else:
            file_idx = np.zeros(n, dtype=np.int64)
            label_idx = np.arange(global_offset, global_offset + n, dtype=np.int64)
            cycle_index = label_idx.copy()

        global_index = np.arange(global_offset, global_offset + n, dtype=np.int64)
        split_type = np.asarray(split_assignment[global_index], dtype=object)
        frame = pd.DataFrame({
            "true_label": y.astype(np.float64),
            "pred_label": pred.astype(np.float64),
            "file_idx": file_idx,
            "cycle_index": cycle_index,
            "split_type": split_type,
            "global_index": global_index,
            "label_index": label_idx,
        })
        frame.to_csv(out_csv, mode="w" if first else "a", header=first, index=False)
        first = False
        rows_written += n
        global_offset += n
        print("[PRED-CSV] chunk %d/%d rows=%d saved_rows=%d time=%.2fs" % (
            chunk_id + 1, len(chunk_x_files), n, rows_written, time.time() - t0
        ))

    print("[SAVE] prediction csv:", out_csv, "rows=", rows_written)
    return out_csv


def save_ridge_importance(model, importance_csv, indice_out_dir, top_ns, feature_names):
    """Preserve the reference importance CSV/top-N output contract using |coef|."""
    ensure_dir(os.path.dirname(importance_csv))
    ensure_dir(indice_out_dir)
    coef = np.asarray(model.coef_).reshape(-1)
    order = np.argsort(-np.abs(coef), kind="stable")
    rows = []
    for fid in order:
        name = str(feature_names[fid]) if fid < len(feature_names) else "f%d" % int(fid)
        rows.append({
            "xgb_feature": "f%d" % int(fid),
            "fid": int(fid),
            "raw_col": int(fid),
            "bc": "",
            "type": "step2_proxy",
            "signal_name": name,
            "gain": float(abs(coef[fid])),
            "coefficient": float(coef[fid]),
        })
    columns = ["xgb_feature", "fid", "raw_col", "bc", "type",
               "signal_name", "gain", "coefficient"]
    frame = pd.DataFrame(rows, columns=columns)
    frame.to_csv(importance_csv, index=False)
    print("[SAVE] Ridge coefficient importance:", importance_csv)

    for top_n in top_ns:
        top_n = int(top_n)
        top_frame = frame.head(top_n)
        raw_file = os.path.join(indice_out_dir, "%d_feature_raw_cols.txt" % top_n)
        name_file = os.path.join(indice_out_dir, "%d_feature_names.txt" % top_n)
        detail_file = os.path.join(indice_out_dir, "%d_feature_detail.csv" % top_n)
        np.savetxt(raw_file, top_frame["raw_col"].values.astype(np.int64), fmt="%d")
        with open(name_file, "w") as f:
            for name in top_frame["signal_name"].values:
                f.write("%s\n" % str(name))
        top_frame.to_csv(detail_file, index=False)
        print("[SAVE] top%d Ridge features: %s" % (top_n, detail_file))


def plot_importance(importance_csv, fig_file, max_num_features):
    frame = pd.read_csv(importance_csv).head(int(max_num_features))
    if frame.empty:
        print("[WARN] empty importance csv, skip plot")
        return
    plt.figure(figsize=(14, max(6, int(max_num_features) * 0.16)))
    plt.barh(np.arange(len(frame)), frame["gain"].values)
    plt.yticks(np.arange(len(frame)), frame["signal_name"].values, fontsize=6)
    plt.gca().invert_yaxis()
    plt.xlabel("absolute Ridge coefficient")
    plt.tight_layout()
    ensure_dir(os.path.dirname(fig_file))
    plt.savefig(fig_file, dpi=300, bbox_inches="tight")
    plt.close()
    print("[SAVE] importance figure:", fig_file)


def main():
    parser = argparse.ArgumentParser(
        description="Post-Step2 streaming Ridge validation with XGBoost-equivalent data/evaluation"
    )
    parser.add_argument("--tw-manifest", required=True)
    parser.add_argument("--proxy-manifest", required=True)
    parser.add_argument("--namefile", default="")
    parser.add_argument("--combos", default="")

    parser.add_argument("--rounds", "-r", type=int, default=200,
                        help="Ridge CG maximum iterations (legacy-compatible option name)")
    parser.add_argument("--ridge-alpha", type=float, default=8.0)
    parser.add_argument("--ridge-tol", type=float, default=1e-6)
    parser.add_argument("--ridge-log-every", type=int, default=1)
    parser.add_argument("--no-fit-intercept", action="store_true")

    parser.add_argument("--length", "-l", type=int, default=8)
    parser.add_argument("--before-cycles", "-bc", type=int, default=6)
    parser.add_argument("--label-start-cycle", type=int, default=20)
    parser.add_argument("--label-scale", type=float, default=1000000.0)
    parser.add_argument("--cache-dir", default="./cache_ultrawide_no_topk")
    parser.add_argument("--build-block-size", type=int, default=200)
    parser.add_argument("--data-dtype", choices=["float32", "float64"], default="float32")
    parser.add_argument("--force-rebuild-dataset", action="store_true")
    parser.add_argument("--build-dataset-only", action="store_true")

    parser.add_argument("--target", "-t", default="tile_reset_domain")
    parser.add_argument("--hpotag", "-hpo", default="allbm_iso_ridge")
    parser.add_argument("--fold", type=int, default=8)
    parser.add_argument("--shuffle-seed", type=int, default=108)
    parser.add_argument("--fixed-test-last-file", action="store_true")
    parser.add_argument("--top-ns", default="4096")
    parser.add_argument("--no-dump-model", action="store_true")
    parser.add_argument("--plot-importance", action="store_true")

    # Accept the reference XGBoost CLI flags so an existing command can switch
    # scripts without argparse failures. They are intentionally unused by Ridge.
    parser.add_argument("--tree-method", choices=["hist", "approx"], default="hist", help=argparse.SUPPRESS)
    parser.add_argument("--max-bin", type=int, default=32, help=argparse.SUPPRESS)
    parser.add_argument("--max-depth", type=int, default=6, help=argparse.SUPPRESS)
    parser.add_argument("--max-leaves", type=int, default=64, help=argparse.SUPPRESS)
    parser.add_argument("--learning-rate", type=float, default=0.05, help=argparse.SUPPRESS)
    parser.add_argument("--subsample", type=float, default=0.8, help=argparse.SUPPRESS)
    parser.add_argument("--colsample-bytree", type=float, default=0.01, help=argparse.SUPPRESS)
    parser.add_argument("--colsample-bynode", type=float, default=0.01, help=argparse.SUPPRESS)
    parser.add_argument("--nthread", type=int, default=32, help=argparse.SUPPRESS)
    parser.add_argument("--seed", type=int, default=108, help=argparse.SUPPRESS)
    parser.add_argument("--device", default="", help=argparse.SUPPRESS)
    args = parser.parse_args()

    selected_combos = [key.strip() for key in args.combos.split(",") if key.strip()]
    dirs = setup_directories(args.hpotag, args.target, args.before_cycles)
    model_file = os.path.join(dirs["result"], "%s-ridge-model.joblib" % args.target)
    importance_csv = os.path.join(
        dirs["result"], "%s-feature_importance_%d.csv" % (args.target, args.rounds)
    )
    importance_fig = os.path.join(
        dirs["result"], "%s-feature_importance_%d.png" % (args.target, args.rounds)
    )

    print("========== PIPELINE RIDGE VALIDATION CONFIG ==========")
    print("version          :", VERSION)
    print("target           :", args.target)
    print("step1 manifest   :", args.tw_manifest)
    print("step2 manifest   :", args.proxy_manifest)
    print("selected combos  :", selected_combos if selected_combos else "all Step1 combos")
    print("cache_dir        :", args.cache_dir)
    print("build block rows :", args.build_block_size)
    print("ridge alpha      :", args.ridge_alpha)
    print("ridge max iter   :", args.rounds)
    print("ridge tol        :", args.ridge_tol)
    print("======================================================")

    chunk_x_files, chunk_y_files, pipeline_meta = build_pipeline_chunks(
        tw_manifest=args.tw_manifest,
        proxy_manifest=args.proxy_manifest,
        cache_dir=args.cache_dir,
        label_scale=args.label_scale,
        data_dtype=args.data_dtype,
        namefile=args.namefile or None,
        block_size=args.build_block_size,
        combo_keys=selected_combos,
    )
    feature_names = list(pipeline_meta["feature_names"])
    print("[PIPELINE-DATA] combos=%d columns=%d rows=%d train_files=%d test_files=%d" % (
        len(pipeline_meta["combo_order"]), int(pipeline_meta["total_columns"]),
        int(pipeline_meta["total_rows"]), len(pipeline_meta["train_file_indices"]),
        len(pipeline_meta["test_file_indices"])
    ))
    if args.build_dataset_only:
        print("[DONE] dataset-only")
        return

    chunk_meta_files = list_chunk_meta_files(chunk_x_files)
    if chunk_meta_files is None:
        raise ValueError("Ridge fixed pipeline test split requires chunk metadata")

    train_idx, val_idx, test_idx, assignment, total_rows = build_split_indices(
        chunk_y_files=chunk_y_files,
        fold=args.fold,
        shuffle_seed=args.shuffle_seed,
        chunk_meta_files=chunk_meta_files,
        fixed_test_file_indices=pipeline_meta["test_file_indices"],
    )
    train_view = ChunkSplitView(chunk_x_files, chunk_y_files, train_idx, "train")
    val_view = ChunkSplitView(chunk_x_files, chunk_y_files, val_idx, "val")
    test_view = ChunkSplitView(chunk_x_files, chunk_y_files, test_idx, "test")

    solver = StreamingRidgeSolver(
        alpha=args.ridge_alpha,
        max_iter=args.rounds,
        tol=args.ridge_tol,
        log_every=args.ridge_log_every,
        fit_intercept=not args.no_fit_intercept,
    )
    model = solver.fit(train_view)
    train_metrics = split_metrics(model, train_view)
    val_metrics = split_metrics(model, val_view)
    test_metrics = split_metrics(model, test_view)
    summary = {
        "version": VERSION,
        "model": "ridge",
        "alpha": float(args.ridge_alpha),
        "max_iter": int(args.rounds),
        "tol": float(args.ridge_tol),
        "fit_intercept": bool(not args.no_fit_intercept),
        "cg_iterations": int(solver.iterations),
        "cg_info": int(model.solver_info),
        "total_rows": int(total_rows),
        "train": train_metrics,
        "val": val_metrics,
        "test": test_metrics,
    }
    summary_file = os.path.join(dirs["eval"], "%s_ridge_split_metrics.json" % args.target)
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    print("[RIDGE METRICS]", json.dumps(summary, sort_keys=True))
    print("[SAVE] split metrics:", summary_file)

    prediction_csv = os.path.join(
        dirs["eval"], "%s_prediction_samples_%d.csv" % (args.target, args.rounds)
    )
    export_sample_prediction_csv(
        model=model,
        chunk_x_files=chunk_x_files,
        chunk_y_files=chunk_y_files,
        chunk_meta_files=chunk_meta_files,
        out_csv=prediction_csv,
        split_assignment=assignment,
    )
    evaluate_prediction_csv(
        prediction_csv=prediction_csv,
        out_dir=dirs["eval"],
        require_consecutive=True,
        threshold_ratios=[x / 10.0 for x in range(1, 10)],
        plot_jump_hist=True,
        jump_hist_bin_step=10.0,
        plot_per_file=False,
    )

    top_ns = parse_top_ns(args.top_ns)
    save_ridge_importance(
        model=model,
        importance_csv=importance_csv,
        indice_out_dir=dirs["indices"],
        top_ns=top_ns,
        feature_names=feature_names,
    )
    if args.plot_importance:
        plot_importance(
            importance_csv, importance_fig, min(100, max(top_ns))
        )

    ensure_dir(os.path.dirname(model_file))
    joblib.dump({
        "version": VERSION,
        "model": "ridge",
        "coef": model.coef_,
        "intercept": model.intercept_,
        "alpha": model.alpha,
        "cg_info": model.solver_info,
        "feature_names": feature_names,
    }, model_file)
    print("[SAVE] Ridge model:", model_file)
    print("[DONE]")


if __name__ == "__main__":
    main()
