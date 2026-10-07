# -*- coding: UTF-8 -*-
"""
XGBoost power prediction evaluation from sample-level prediction CSV.

Input CSV columns required:
    true_label,pred_label,file_idx,cycle_index,split_type

This module computes:
    1) point-wise metrics for each file, each split, and overall
    2) jump/change metrics within continuous samples of the same file
    3) thresholded jump relative-error summaries and histograms

The evaluator is intentionally independent from the training script: it only
needs the exported prediction CSV, so metrics/plots can be regenerated without
re-running XGBoost prediction.
"""

from __future__ import print_function

import argparse
import json
import os

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


REQUIRED_COLUMNS = ["true_label", "pred_label", "file_idx", "cycle_index", "split_type"]


def ensure_dir(path):
    if path:
        os.makedirs(path, exist_ok=True)


def save_json(path, obj):
    ensure_dir(os.path.dirname(path))
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def _safe_float(x):
    try:
        if x is None:
            return None
        if isinstance(x, (float, np.floating)) and (np.isnan(x) or np.isinf(x)):
            return None
        return float(x)
    except Exception:
        return None


def _metrics_dict(y_true, y_pred, name, group_type):
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    n = int(y_true.size)

    out = {
        "group_type": group_type,
        "group_name": str(name),
        "count": n,
        "true_mean": None,
        "pred_mean": None,
        "mean_error": None,
        "mean_abs_error": None,
        "median_abs_error": None,
        "max_error": None,
        "max_abs_error": None,
        "rmse": None,
        "nrmse": None,
        "mape": None,
        "r2": None,
        "p90_abs_error": None,
        "p95_abs_error": None,
        "p99_abs_error": None,
    }

    if n == 0:
        return out

    err = y_pred - y_true
    abs_err = np.abs(err)
    sq_err = err ** 2
    rmse = float(np.sqrt(np.mean(sq_err)))

    true_min = float(np.min(y_true))
    true_max = float(np.max(y_true))
    true_range = true_max - true_min
    true_mean_abs = float(np.mean(np.abs(y_true)))

    # NRMSE prefers range normalization. If range is zero, fall back to mean abs true.
    if true_range > 0:
        nrmse = rmse / true_range
    elif true_mean_abs > 0:
        nrmse = rmse / true_mean_abs
    else:
        nrmse = np.nan

    denom_mask = np.abs(y_true) > 1e-12
    if np.any(denom_mask):
        mape = float(np.mean(np.abs(err[denom_mask] / y_true[denom_mask])) * 100.0)
    else:
        mape = np.nan

    sst = float(np.sum((y_true - np.mean(y_true)) ** 2))
    sse = float(np.sum(sq_err))
    r2 = 1.0 - sse / sst if sst > 0 else np.nan

    out.update({
        "true_mean": float(np.mean(y_true)),
        "pred_mean": float(np.mean(y_pred)),
        "mean_error": float(np.mean(err)),
        "mean_abs_error": float(np.mean(abs_err)),
        "median_abs_error": float(np.median(abs_err)),
        "max_error": float(np.max(err)),
        "max_abs_error": float(np.max(abs_err)),
        "rmse": rmse,
        "nrmse": float(nrmse) if not np.isnan(nrmse) else None,
        "mape": float(mape) if not np.isnan(mape) else None,
        "r2": float(r2) if not np.isnan(r2) else None,
        "p90_abs_error": float(np.percentile(abs_err, 90)),
        "p95_abs_error": float(np.percentile(abs_err, 95)),
        "p99_abs_error": float(np.percentile(abs_err, 99)),
    })
    return out


def compute_point_metrics(df):
    rows = []

    rows.append(_metrics_dict(
        df["true_label"].values,
        df["pred_label"].values,
        name="overall",
        group_type="overall",
    ))

    for split, g in df.groupby("split_type", sort=True):
        rows.append(_metrics_dict(
            g["true_label"].values,
            g["pred_label"].values,
            name=split,
            group_type="split",
        ))

    for file_idx, g in df.groupby("file_idx", sort=True):
        rows.append(_metrics_dict(
            g["true_label"].values,
            g["pred_label"].values,
            name=file_idx,
            group_type="file",
        ))

    # file x split is often useful for checking if one trace behaves differently
    # in train/val/test regions.
    for (file_idx, split), g in df.groupby(["file_idx", "split_type"], sort=True):
        rows.append(_metrics_dict(
            g["true_label"].values,
            g["pred_label"].values,
            name="file_%s__%s" % (str(file_idx), str(split)),
            group_type="file_split",
        ))

    return pd.DataFrame(rows)


def build_jump_detail(df, require_consecutive=True):
    """
    Build one row per valid jump:
      current sample t and previous sample t-1 must be from the same file.

    If require_consecutive=True, cycle_index must satisfy curr = prev + 1.
    The first cycle of each file is skipped automatically.
    """
    rows = []
    sort_cols = ["file_idx", "cycle_index"]
    if "global_index" in df.columns:
        sort_cols = ["file_idx", "cycle_index", "global_index"]

    for file_idx, g in df.sort_values(sort_cols).groupby("file_idx", sort=True):
        g = g.reset_index(drop=True)
        if len(g) <= 1:
            continue

        true_arr = g["true_label"].values.astype(np.float64)
        pred_arr = g["pred_label"].values.astype(np.float64)
        cycles = g["cycle_index"].values.astype(np.int64)
        splits = g["split_type"].astype(str).values

        for i in range(1, len(g)):
            prev_cycle = int(cycles[i - 1])
            curr_cycle = int(cycles[i])
            if require_consecutive and curr_cycle != prev_cycle + 1:
                continue

            real_jump = float(true_arr[i] - true_arr[i - 1])
            pred_jump = float(pred_arr[i] - pred_arr[i - 1])
            abs_jump_error = abs(pred_jump - real_jump)
            if abs(real_jump) > 1e-12:
                rel_jump_error_pct = abs_jump_error / abs(real_jump) * 100.0
            else:
                rel_jump_error_pct = np.nan

            rows.append({
                "file_idx": int(file_idx),
                "prev_cycle_index": prev_cycle,
                "curr_cycle_index": curr_cycle,
                "prev_split_type": str(splits[i - 1]),
                "curr_split_type": str(splits[i]),
                "same_split": bool(splits[i - 1] == splits[i]),
                "split_type": str(splits[i]) if splits[i - 1] == splits[i] else "cross_split",
                "real_prev": float(true_arr[i - 1]),
                "real_curr": float(true_arr[i]),
                "pred_prev": float(pred_arr[i - 1]),
                "pred_curr": float(pred_arr[i]),
                "real_jump": real_jump,
                "pred_jump": pred_jump,
                "abs_real_jump": abs(real_jump),
                "abs_pred_jump": abs(pred_jump),
                "abs_jump_error": abs_jump_error,
                "relative_jump_error_pct": rel_jump_error_pct,
            })

    return pd.DataFrame(rows)


def _jump_threshold_summary(jump_df, name, group_type, threshold_ratios):
    columns = [
        "group_type", "group_name", "threshold_pct", "threshold_value", "sample_count",
        "max_abs_real_jump", "max_relative_jump_error_pct", "mean_relative_jump_error_pct",
        "median_relative_jump_error_pct", "p90_relative_jump_error_pct",
        "p95_relative_jump_error_pct", "p99_relative_jump_error_pct",
        "mean_abs_jump_error", "max_abs_jump_error",
    ]

    if jump_df is None or jump_df.empty:
        return pd.DataFrame(columns=columns)

    valid = jump_df[np.isfinite(jump_df["relative_jump_error_pct"].values)].copy()
    if valid.empty:
        return pd.DataFrame(columns=columns)

    max_abs_jump = float(np.max(valid["abs_real_jump"].values))
    rows = []
    for ratio in threshold_ratios:
        threshold_value = max_abs_jump * float(ratio)
        sub = valid[valid["abs_real_jump"].values > threshold_value]
        if sub.empty:
            rows.append({
                "group_type": group_type,
                "group_name": str(name),
                "threshold_pct": float(ratio) * 100.0,
                "threshold_value": threshold_value,
                "sample_count": 0,
                "max_abs_real_jump": max_abs_jump,
                "max_relative_jump_error_pct": None,
                "mean_relative_jump_error_pct": None,
                "median_relative_jump_error_pct": None,
                "p90_relative_jump_error_pct": None,
                "p95_relative_jump_error_pct": None,
                "p99_relative_jump_error_pct": None,
                "mean_abs_jump_error": None,
                "max_abs_jump_error": None,
            })
            continue

        rel = sub["relative_jump_error_pct"].values.astype(np.float64)
        abs_e = sub["abs_jump_error"].values.astype(np.float64)
        rows.append({
            "group_type": group_type,
            "group_name": str(name),
            "threshold_pct": float(ratio) * 100.0,
            "threshold_value": threshold_value,
            "sample_count": int(len(sub)),
            "max_abs_real_jump": max_abs_jump,
            "max_relative_jump_error_pct": float(np.max(rel)),
            "mean_relative_jump_error_pct": float(np.mean(rel)),
            "median_relative_jump_error_pct": float(np.median(rel)),
            "p90_relative_jump_error_pct": float(np.percentile(rel, 90)),
            "p95_relative_jump_error_pct": float(np.percentile(rel, 95)),
            "p99_relative_jump_error_pct": float(np.percentile(rel, 99)),
            "mean_abs_jump_error": float(np.mean(abs_e)),
            "max_abs_jump_error": float(np.max(abs_e)),
        })

    return pd.DataFrame(rows, columns=columns)


def compute_jump_summaries(jump_df, threshold_ratios=None):
    if threshold_ratios is None:
        threshold_ratios = [x / 10.0 for x in range(1, 10)]

    frames = []
    # Overall includes all continuous adjacent samples within each file, including
    # train/val/test boundaries if the samples are adjacent. Split-specific groups
    # below only include jumps whose two endpoints belong to the same split.
    frames.append(_jump_threshold_summary(jump_df, "overall", "overall", threshold_ratios))

    same_split = jump_df[jump_df["same_split"].values == True] if not jump_df.empty else jump_df
    if same_split is not None and not same_split.empty:
        for split, g in same_split.groupby("split_type", sort=True):
            frames.append(_jump_threshold_summary(g, split, "split", threshold_ratios))

    if jump_df is not None and not jump_df.empty:
        for file_idx, g in jump_df.groupby("file_idx", sort=True):
            frames.append(_jump_threshold_summary(g, file_idx, "file", threshold_ratios))

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _histogram_counts(errors, bin_step):
    errors = np.asarray(errors, dtype=np.float64)
    errors = errors[np.isfinite(errors)]
    if errors.size == 0:
        return pd.DataFrame(columns=["bin_left_pct", "bin_right_pct", "count"])
    max_err = max(100.0, float(np.max(errors)))
    bins = np.arange(0.0, max_err + float(bin_step), float(bin_step))
    if bins.size < 2:
        bins = np.asarray([0.0, float(bin_step)])
    counts, edges = np.histogram(errors, bins=bins)
    return pd.DataFrame({
        "bin_left_pct": edges[:-1],
        "bin_right_pct": edges[1:],
        "count": counts.astype(np.int64),
    })


def plot_jump_histograms(jump_df, out_dir, prefix, threshold_ratios=None, bin_step=10.0):
    if threshold_ratios is None:
        threshold_ratios = [x / 10.0 for x in range(1, 10)]
    ensure_dir(out_dir)

    if jump_df is None or jump_df.empty:
        print("[PLOT] skip empty jump histogram:", prefix)
        return []

    valid = jump_df[np.isfinite(jump_df["relative_jump_error_pct"].values)].copy()
    if valid.empty:
        print("[PLOT] skip jump histogram without finite relative errors:", prefix)
        return []

    max_abs_jump = float(np.max(valid["abs_real_jump"].values))
    saved = []
    for ratio in threshold_ratios:
        threshold_value = max_abs_jump * float(ratio)
        sub = valid[valid["abs_real_jump"].values > threshold_value]
        if sub.empty:
            continue

        errors = sub["relative_jump_error_pct"].values.astype(np.float64)
        hist_df = _histogram_counts(errors, bin_step=bin_step)

        tag = "%s_jump_error_hist_%02d" % (prefix, int(round(float(ratio) * 100.0)))
        csv_path = os.path.join(out_dir, tag + ".csv")
        png_path = os.path.join(out_dir, tag + ".png")
        hist_df.to_csv(csv_path, index=False)

        plt.figure(figsize=(10, 6))
        plt.hist(errors, bins=np.r_[hist_df["bin_left_pct"].values, hist_df["bin_right_pct"].values[-1]])
        plt.xlabel("Relative Jump Error (%)")
        plt.ylabel("Frequency")
        plt.title("%s jump error distribution: |real jump| > %.1f" % (prefix, threshold_value))
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(png_path, dpi=300)
        plt.close()

        saved.append((csv_path, png_path))
        print("[SAVE]", csv_path)
        print("[SAVE]", png_path)

    return saved


def plot_jump_histograms_by_scope(jump_df, out_dir, threshold_ratios=None,
                                  bin_step=10.0, plot_per_file=False):
    ensure_dir(out_dir)
    if jump_df is None or jump_df.empty:
        return

    plot_jump_histograms(jump_df, out_dir, "overall", threshold_ratios, bin_step)

    same_split = jump_df[jump_df["same_split"].values == True]
    for split, g in same_split.groupby("split_type", sort=True):
        plot_jump_histograms(g, out_dir, "split_%s" % str(split), threshold_ratios, bin_step)

    if plot_per_file:
        file_plot_dir = os.path.join(out_dir, "file_jump_histograms")
        ensure_dir(file_plot_dir)
        for file_idx, g in jump_df.groupby("file_idx", sort=True):
            plot_jump_histograms(g, file_plot_dir, "file_%s" % str(file_idx), threshold_ratios, bin_step)


def evaluate_prediction_csv(prediction_csv, out_dir, require_consecutive=True,
                            threshold_ratios=None, plot_jump_hist=True,
                            jump_hist_bin_step=10.0, plot_per_file=False,
                            require_test_split=True):
    ensure_dir(out_dir)
    df = pd.read_csv(prediction_csv)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError("prediction csv missing required columns: %s" % ",".join(missing))

    # Normalize dtypes and ordering.
    df = df.copy()
    df["true_label"] = df["true_label"].astype(np.float64)
    df["pred_label"] = df["pred_label"].astype(np.float64)
    df["file_idx"] = df["file_idx"].astype(np.int64)
    df["cycle_index"] = df["cycle_index"].astype(np.int64)
    df["split_type"] = df["split_type"].astype(str)
    if require_test_split and "test" not in set(df["split_type"].values):
        raise ValueError("prediction csv has no pipeline test samples")

    metrics_df = compute_point_metrics(df)
    metrics_path = os.path.join(out_dir, "point_metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)
    print("[SAVE]", metrics_path)

    # Convenience split files.
    metrics_df[metrics_df["group_type"] == "overall"].to_csv(
        os.path.join(out_dir, "point_metrics_overall.csv"), index=False)
    metrics_df[metrics_df["group_type"] == "split"].to_csv(
        os.path.join(out_dir, "point_metrics_by_split.csv"), index=False)
    metrics_df[metrics_df["group_type"] == "file"].to_csv(
        os.path.join(out_dir, "point_metrics_by_file.csv"), index=False)
    metrics_df[metrics_df["group_type"] == "file_split"].to_csv(
        os.path.join(out_dir, "point_metrics_by_file_split.csv"), index=False)

    jump_df = build_jump_detail(df, require_consecutive=require_consecutive)
    jump_detail_path = os.path.join(out_dir, "jump_detail.csv")
    jump_df.to_csv(jump_detail_path, index=False)
    print("[SAVE]", jump_detail_path)

    jump_summary_df = compute_jump_summaries(jump_df, threshold_ratios=threshold_ratios)
    jump_summary_path = os.path.join(out_dir, "jump_threshold_summary.csv")
    jump_summary_df.to_csv(jump_summary_path, index=False)
    print("[SAVE]", jump_summary_path)

    if not jump_summary_df.empty:
        jump_summary_df[jump_summary_df["group_type"] == "overall"].to_csv(
            os.path.join(out_dir, "jump_threshold_summary_overall.csv"), index=False)
        jump_summary_df[jump_summary_df["group_type"] == "split"].to_csv(
            os.path.join(out_dir, "jump_threshold_summary_by_split.csv"), index=False)
        jump_summary_df[jump_summary_df["group_type"] == "file"].to_csv(
            os.path.join(out_dir, "jump_threshold_summary_by_file.csv"), index=False)

    if plot_jump_hist:
        hist_dir = os.path.join(out_dir, "jump_histograms")
        plot_jump_histograms_by_scope(
            jump_df=jump_df,
            out_dir=hist_dir,
            threshold_ratios=threshold_ratios,
            bin_step=jump_hist_bin_step,
            plot_per_file=plot_per_file,
        )

    summary = {
        "prediction_csv": os.path.abspath(prediction_csv),
        "num_samples": int(len(df)),
        "num_files": int(df["file_idx"].nunique()),
        "splits": sorted([str(x) for x in df["split_type"].unique()]),
        "num_jumps": int(len(jump_df)),
        "num_test_samples": int(np.sum(df["split_type"].values == "test")),
        "point_metrics_csv": os.path.abspath(metrics_path),
        "jump_detail_csv": os.path.abspath(jump_detail_path),
        "jump_threshold_summary_csv": os.path.abspath(jump_summary_path),
    }
    save_json(os.path.join(out_dir, "evaluation_summary.json"), summary)
    return {
        "point_metrics": metrics_df,
        "jump_detail": jump_df,
        "jump_threshold_summary": jump_summary_df,
        "summary": summary,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate power prediction CSV exported by training script")
    parser.add_argument("--prediction-csv", required=True, help="CSV with true_label,pred_label,file_idx,cycle_index,split_type")
    parser.add_argument("--out-dir", required=True, help="evaluation output directory")
    parser.add_argument("--no-plot-jump-hist", action="store_true", help="disable jump-error histograms")
    parser.add_argument("--plot-per-file", action="store_true", help="also plot jump histograms for every file_idx; can generate many PNGs")
    parser.add_argument("--jump-hist-bin-step", type=float, default=10.0, help="histogram bin size in percent")
    parser.add_argument("--thresholds", type=str, default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9",
                        help="comma-separated threshold ratios relative to max abs real jump")
    parser.add_argument("--allow-nonconsecutive", action="store_true",
                        help="allow adjacent rows within file even if cycle_index is not consecutive")
    args = parser.parse_args()

    ratios = []
    for x in args.thresholds.split(","):
        x = x.strip()
        if x:
            ratios.append(float(x))

    evaluate_prediction_csv(
        prediction_csv=args.prediction_csv,
        out_dir=args.out_dir,
        require_consecutive=(not args.allow_nonconsecutive),
        threshold_ratios=ratios,
        plot_jump_hist=(not args.no_plot_jump_hist),
        jump_hist_bin_step=args.jump_hist_bin_step,
        plot_per_file=args.plot_per_file,
    )


if __name__ == "__main__":
    main()
