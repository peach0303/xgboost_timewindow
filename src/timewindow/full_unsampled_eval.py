#!/usr/bin/env python3
"""Prepare, execute, and reduce per-case full-unsampled evaluation tasks."""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.sparse import hstack

from utils.evaluation_plots import save_true_vs_pred_scatter
from utils.posttrain_power_eval import (
    _discover_case_sources,
    _feature_blocks,
    _label_path,
    _load_json,
    _load_row_window,
    _resolve_path,
)
from utils.two_phase_trainer import get_offset_column_groups


THRESHOLD_RATIOS = tuple(value / 10.0 for value in range(1, 10))


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [full-eval] {message}", flush=True)


def elapsed(stage: str, started: float, *, case: str | None = None) -> None:
    prefix = f"case={case} " if case else ""
    log(f"{prefix}stage={stage} elapsed_sec={time.monotonic() - started:.3f}")


def _write_json(path: str | Path, value: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
    os.replace(temporary, path)


def prepare_tasks(tw_manifest_path: str, out_dir: str) -> dict:
    started = time.monotonic()
    tw = _load_json(tw_manifest_path)
    tasks = []
    for split in ("train", "test"):
        for stem in tw.get(f"{split}_stems", []):
            tasks.append(OrderedDict([
                ("task_index", len(tasks)),
                ("file_idx", len(tasks)),
                ("split", split),
                ("stem", stem),
                ("case_dir", f"cases/case_{len(tasks):05d}"),
            ]))
    manifest = OrderedDict([
        ("tw_manifest", os.path.abspath(tw_manifest_path)),
        ("num_tasks", len(tasks)),
        ("tasks", tasks),
    ])
    out_path = os.path.join(out_dir, "case_tasks.json")
    _write_json(out_path, manifest)
    with open(os.path.join(out_dir, "case_tasks.tsv"), "w", encoding="utf-8") as handle:
        handle.write("task_index\tfile_idx\tsplit\tstem\tcase_dir\n")
        for task in tasks:
            handle.write(
                f"{task['task_index']}\t{task['file_idx']}\t{task['split']}\t"
                f"{task['stem']}\t{task['case_dir']}\n"
            )
    elapsed("prepare_tasks", started)
    log(f"prepared tasks={len(tasks)} manifest={out_path}")
    return manifest


def _empty_stats() -> dict:
    return {
        "n": 0, "sum_y": 0.0, "sum_y2": 0.0, "min_y": None, "max_y": None,
        "sum_pred": 0.0, "sum_error": 0.0, "min_error": None, "max_error": None,
        "sum_sq_error": 0.0, "sum_abs_error": 0.0, "sum_rel_error": 0.0,
        "sum_rel_error2": 0.0, "max_abs_error": None, "max_rel_error": None,
    }


def _update_stats(stats: dict, y: np.ndarray, pred: np.ndarray) -> None:
    y = np.asarray(y, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    if y.size == 0:
        return
    error = pred - y
    abs_error = np.abs(error)
    denominator = np.where(np.abs(y) > 1e-12, np.abs(y), 1e-12)
    relative = abs_error / denominator
    stats["n"] += int(y.size)
    stats["sum_y"] += float(np.sum(y))
    stats["sum_y2"] += float(np.sum(y * y))
    stats["sum_pred"] += float(np.sum(pred))
    stats["sum_error"] += float(np.sum(error))
    stats["sum_sq_error"] += float(np.sum(error * error))
    stats["sum_abs_error"] += float(np.sum(abs_error))
    stats["sum_rel_error"] += float(np.sum(relative))
    stats["sum_rel_error2"] += float(np.sum(relative * relative))
    local_min, local_max = float(np.min(y)), float(np.max(y))
    local_error_min, local_error_max = float(np.min(error)), float(np.max(error))
    local_abs_error_max = float(np.max(abs_error))
    local_rel_max = float(np.max(relative))
    stats["min_error"] = (
        local_error_min if stats["min_error"] is None else min(stats["min_error"], local_error_min)
    )
    stats["max_error"] = (
        local_error_max if stats["max_error"] is None else max(stats["max_error"], local_error_max)
    )
    stats["max_abs_error"] = (
        local_abs_error_max if stats["max_abs_error"] is None
        else max(stats["max_abs_error"], local_abs_error_max)
    )
    stats["min_y"] = local_min if stats["min_y"] is None else min(stats["min_y"], local_min)
    stats["max_y"] = local_max if stats["max_y"] is None else max(stats["max_y"], local_max)
    stats["max_rel_error"] = (
        local_rel_max if stats["max_rel_error"] is None
        else max(stats["max_rel_error"], local_rel_max)
    )


def _combine_stats(items: list[dict]) -> dict:
    combined = _empty_stats()
    for item in items:
        combined["n"] += int(item["n"])
        for key in (
            "sum_y", "sum_y2", "sum_pred", "sum_error", "sum_sq_error", "sum_abs_error",
            "sum_rel_error", "sum_rel_error2",
        ):
            combined[key] += float(item[key])
        for key, reducer in (
            ("min_y", min), ("max_y", max), ("min_error", min), ("max_error", max),
            ("max_abs_error", max), ("max_rel_error", max),
        ):
            value = item.get(key)
            if value is not None:
                combined[key] = value if combined[key] is None else reducer(combined[key], value)
    return combined


def _power_metrics(stats: dict) -> dict:
    n = int(stats["n"])
    if n <= 0:
        raise ValueError("Cannot finalize empty power statistics")
    mean = stats["sum_y"] / n
    variance_y = max(0.0, stats["sum_y2"] / n - mean * mean)
    mse = stats["sum_sq_error"] / n
    rmse = math.sqrt(mse)
    mean_relative = stats["sum_rel_error"] / n
    variance_relative = max(
        0.0, stats["sum_rel_error2"] / n - mean_relative * mean_relative,
    )
    return {
        "n": n,
        "label_mean": mean,
        "label_std": math.sqrt(variance_y),
        "label_min": stats["min_y"],
        "label_max": stats["max_y"],
        "Max Error": stats["max_rel_error"],
        "MAPE": mean_relative * 100.0,
        "MSE": mse,
        "var": variance_relative,
        "R2": 1.0 - mse / variance_y if variance_y > 1e-12 else float("nan"),
        "RMSE": rmse,
        "NRMSE": rmse / mean if abs(mean) > 1e-12 else float("nan"),
        "NMAE": stats["sum_abs_error"] / stats["sum_y"]
        if abs(stats["sum_y"]) > 1e-12 else float("nan"),
    }


def _point_row(stats: dict, name: str, group_type: str) -> dict:
    metrics = _power_metrics(stats)
    return {
        "group_type": group_type, "group_name": str(name), "count": metrics["n"],
        "true_mean": metrics["label_mean"], "pred_mean": stats["sum_pred"] / stats["n"],
        "mean_error": stats["sum_error"] / stats["n"],
        "mean_abs_error": stats["sum_abs_error"] / stats["n"],
        "median_abs_error": None, "max_error": stats["max_error"],
        "max_abs_error": stats["max_abs_error"], "rmse": metrics["RMSE"],
        "nrmse": metrics["NRMSE"], "mape": metrics["MAPE"], "r2": metrics["R2"],
        "p90_abs_error": None, "p95_abs_error": None, "p99_abs_error": None,
    }


def _load_sample_info(tw_manifest_path: str, tw: dict) -> dict | None:
    sample_manifest = tw.get("sample_index_manifest", "")
    if not sample_manifest:
        return None
    resolved = _resolve_path(sample_manifest, os.path.dirname(os.path.abspath(tw_manifest_path)))
    return _load_json(resolved)


def _load_predictor(
    result_dir: str, feature_names: list[str], model_type: str,
    two_phase_method: str, cutoff: int,
):
    if model_type == "mlp":
        import joblib

        checkpoint_path = os.path.join(result_dir, "mlp_model.joblib")
        checkpoint = joblib.load(checkpoint_path)
        model = checkpoint["model"]
        feature_scaler = checkpoint["feature_scaler"]
        label_scaler = checkpoint["label_scaler"]
        expected_features = getattr(model, "n_features_in_", len(feature_names))
        if expected_features != len(feature_names):
            raise ValueError(
                f"MLP feature count mismatch: model={expected_features}, "
                f"feature_names={len(feature_names)}"
            )

        def predict(matrix):
            scaled = feature_scaler.transform(matrix)
            scaled_prediction = model.predict(scaled)
            return label_scaler.inverse_transform(
                np.asarray(scaled_prediction).reshape(-1, 1)
            ).reshape(-1)

        return predict

    if model_type != "xgboost":
        raise ValueError(f"Unsupported model type: {model_type}")
    if two_phase_method == "B":
        far = xgb.Booster()
        residual = xgb.Booster()
        far.load_model(os.path.join(result_dir, "xgbmodel_far.json"))
        residual.load_model(os.path.join(result_dir, "xgbmodel_residual.json"))
        far_columns, _ = get_offset_column_groups(feature_names, cutoff)

        def predict(matrix):
            far_prediction = far.predict(xgb.DMatrix(matrix[:, far_columns]))
            return residual.predict(xgb.DMatrix(matrix, base_margin=far_prediction))

        return predict

    filename = "xgbmodel_phase2.json" if two_phase_method == "A" else "xgbmodel.json"
    booster = xgb.Booster()
    booster.load_model(os.path.join(result_dir, filename))
    return lambda matrix: booster.predict(xgb.DMatrix(matrix, feature_names=feature_names))


def _write_csv_chunk(path: str, frame: pd.DataFrame, first: bool) -> bool:
    frame.to_csv(path, mode="w" if first else "a", header=first, index=False)
    return False


def _jump_threshold_rows(case_dir: str, name: str, group_type: str) -> list[dict]:
    abs_real = np.load(os.path.join(case_dir, "abs_real_jump.npy"), mmap_mode="r")
    abs_error = np.load(os.path.join(case_dir, "abs_jump_error.npy"), mmap_mode="r")
    relative = np.load(os.path.join(case_dir, "relative_jump_error_pct.npy"), mmap_mode="r")
    finite = np.isfinite(relative)
    max_jump = float(np.max(abs_real[finite])) if np.any(finite) else 0.0
    rows = []
    for ratio in THRESHOLD_RATIOS:
        threshold = max_jump * ratio
        mask = finite & (abs_real > threshold)
        rel = np.asarray(relative[mask], dtype=np.float64)
        err = np.asarray(abs_error[mask], dtype=np.float64)
        rows.append({
            "group_type": group_type, "group_name": str(name),
            "threshold_pct": ratio * 100.0, "threshold_value": threshold,
            "sample_count": int(rel.size), "max_abs_real_jump": max_jump,
            "max_relative_jump_error_pct": float(np.max(rel)) if rel.size else None,
            "mean_relative_jump_error_pct": float(np.mean(rel)) if rel.size else None,
            "median_relative_jump_error_pct": float(np.median(rel)) if rel.size else None,
            "p90_relative_jump_error_pct": float(np.percentile(rel, 90)) if rel.size else None,
            "p95_relative_jump_error_pct": float(np.percentile(rel, 95)) if rel.size else None,
            "p99_relative_jump_error_pct": float(np.percentile(rel, 99)) if rel.size else None,
            "mean_abs_jump_error": float(np.mean(err)) if err.size else None,
            "max_abs_jump_error": float(np.max(err)) if err.size else None,
        })
    return rows


def evaluate_case(
    *, task_manifest: str, task_index: int, level_dir: str, tog_dir: str,
    label_dir: str, raw_namefile: str, feature_names_path: str, result_dir: str,
    out_dir: str, model_type: str = "xgboost", two_phase_method: str = "",
    two_phase_cutoff: int = 9, chunk_rows: int = 250_000,
) -> dict:
    total_started = time.monotonic()
    task_data = _load_json(task_manifest)
    task = task_data["tasks"][task_index]
    stem, split, file_idx = task["stem"], task["split"], int(task["file_idx"])
    case_dir = os.path.join(out_dir, task["case_dir"])
    complete_path = os.path.join(case_dir, "complete")
    if os.path.exists(complete_path):
        existing = _load_json(os.path.join(case_dir, "case_summary.json"))
        existing_model_type = existing.get("model_type", "xgboost")
        if (
            existing.get("stem") == stem and existing.get("split") == split
            and existing_model_type == model_type
        ):
            log(f"case={stem} task={task_index} model={model_type} already complete; skip")
            return existing
        log(f"case={stem} task={task_index} stale complete marker; rebuild")
    os.makedirs(case_dir, exist_ok=True)
    for stale in ("fail", "complete"):
        try:
            os.remove(os.path.join(case_dir, stale))
        except FileNotFoundError:
            pass
    log(f"case={stem} task={task_index} split={split} start")

    stage_started = time.monotonic()
    tw_path = task_data["tw_manifest"]
    tw = _load_json(tw_path)
    sample_info = _load_sample_info(tw_path, tw)
    boundary = int(tw.get("boundary_discard", 0))
    raw_y = np.load(_label_path(stem, label_dir, sample_info), mmap_mode="r").reshape(-1)
    y = raw_y[boundary:]
    with open(feature_names_path, "r", encoding="utf-8") as handle:
        feature_names = [line.strip() for line in handle if line.strip()]
    with open(raw_namefile, "r", encoding="utf-8") as handle:
        raw_names = [line.strip() for line in handle]
    blocks = _feature_blocks(feature_names, raw_names)
    elapsed("load_metadata_and_labels", stage_started, case=stem)

    stage = str(tw.get("stage", "after_one"))
    source_dirs = {"level": level_dir, "tog": tog_dir}
    sources = {}
    source_origins = {}
    source_columns = {}
    for dtype in ("level", "tog"):
        dtype_blocks = [(offset, indices) for (kind, offset), indices in blocks if kind == dtype]
        if not dtype_blocks:
            continue
        stage_started = time.monotonic()
        files = _discover_case_sources(source_dirs[dtype], stage, [stem], dtype)[stem]
        begins = [boundary - offset if dtype == "tog" else boundary + 1 - offset
                  for offset, _ in dtype_blocks]
        source_begin = min(begins)
        source_end = max(value + y.size for value in begins)
        raw_source = _load_row_window(files, source_begin, source_end).tocsr()
        raw_column_count = raw_source.shape[1]
        selected_columns = np.unique(np.concatenate([indices for _, indices in dtype_blocks]))
        sources[dtype] = raw_source[:, selected_columns].tocsr()
        del raw_source
        source_columns[dtype] = selected_columns
        source_origins[dtype] = source_begin
        elapsed(
            f"load_{dtype}_source rows={sources[dtype].shape[0]} "
            f"raw_cols={raw_column_count} selected_cols={sources[dtype].shape[1]}",
            stage_started, case=stem,
        )

    stage_started = time.monotonic()
    predictor = _load_predictor(
        result_dir, feature_names, model_type, two_phase_method, two_phase_cutoff,
    )
    elapsed("load_model", stage_started, case=stem)
    stage_started = time.monotonic()
    prediction_path = os.path.join(case_dir, "predictions.npy")
    prediction = np.lib.format.open_memmap(prediction_path, mode="w+", dtype=np.float64, shape=(y.size,))
    prediction_csv = os.path.join(case_dir, "prediction_samples.csv")
    stats = _empty_stats()
    first_write = True
    chunk_count = max(1, math.ceil(y.size / chunk_rows))
    for chunk_number, row_begin in enumerate(range(0, y.size, chunk_rows), start=1):
        chunk_started = time.monotonic()
        row_end = min(y.size, row_begin + chunk_rows)
        parts = []
        for (dtype, offset), indices in blocks:
            absolute_begin = boundary - offset if dtype == "tog" else boundary + 1 - offset
            relative_begin = absolute_begin - source_origins[dtype] + row_begin
            relative_end = relative_begin + (row_end - row_begin)
            local_indices = np.searchsorted(source_columns[dtype], indices)
            parts.append(sources[dtype][relative_begin:relative_end, local_indices].tocsr())
        matrix = parts[0] if len(parts) == 1 else hstack(parts, format="csr")
        predicted = np.asarray(predictor(matrix), dtype=np.float64).reshape(-1)
        if predicted.size != row_end - row_begin:
            raise ValueError(f"Prediction rows mismatch for case '{stem}' chunk {chunk_number}")
        prediction[row_begin:row_end] = predicted
        current_y = np.asarray(y[row_begin:row_end], dtype=np.float64)
        _update_stats(stats, current_y, predicted)
        frame = pd.DataFrame({
            "true_label": current_y, "pred_label": predicted,
            "file_idx": np.full(predicted.size, file_idx, dtype=np.int64),
            "case_name": np.full(predicted.size, stem, dtype=object),
            "cycle_index": np.arange(boundary + row_begin, boundary + row_end, dtype=np.int64),
            "split_type": np.full(predicted.size, split, dtype=object),
            "label_index": np.arange(row_begin, row_end, dtype=np.int64),
        })
        first_write = _write_csv_chunk(prediction_csv, frame, first_write)
        log(
            f"case={stem} stage=predict_chunk chunk={chunk_number}/{chunk_count} "
            f"rows={row_end - row_begin} elapsed_sec={time.monotonic() - chunk_started:.3f}"
        )
    prediction.flush()
    sources.clear()
    elapsed("predict_and_write", stage_started, case=stem)

    stage_started = time.monotonic()
    jump_count = max(0, y.size - 1)
    jump_paths = {
        "abs_real": os.path.join(case_dir, "abs_real_jump.npy"),
        "abs_error": os.path.join(case_dir, "abs_jump_error.npy"),
        "relative": os.path.join(case_dir, "relative_jump_error_pct.npy"),
    }
    jump_arrays = {
        key: np.lib.format.open_memmap(path, mode="w+", dtype=np.float64, shape=(jump_count,))
        for key, path in jump_paths.items()
    }
    jump_csv = os.path.join(case_dir, "jump_detail.csv")
    first_write = True
    for jump_begin in range(0, jump_count, chunk_rows):
        jump_end = min(jump_count, jump_begin + chunk_rows)
        current = np.arange(jump_begin + 1, jump_end + 1, dtype=np.int64)
        previous = current - 1
        real_prev = np.asarray(y[previous], dtype=np.float64)
        real_curr = np.asarray(y[current], dtype=np.float64)
        pred_prev = np.asarray(prediction[previous], dtype=np.float64)
        pred_curr = np.asarray(prediction[current], dtype=np.float64)
        real_jump = real_curr - real_prev
        pred_jump = pred_curr - pred_prev
        abs_error = np.abs(pred_jump - real_jump)
        relative = np.full(real_jump.shape, np.nan, dtype=np.float64)
        nonzero = np.abs(real_jump) > 1e-12
        relative[nonzero] = abs_error[nonzero] / np.abs(real_jump[nonzero]) * 100.0
        target = slice(jump_begin, jump_end)
        jump_arrays["abs_real"][target] = np.abs(real_jump)
        jump_arrays["abs_error"][target] = abs_error
        jump_arrays["relative"][target] = relative
        frame = pd.DataFrame({
            "file_idx": np.full(current.size, file_idx, dtype=np.int64),
            "prev_cycle_index": boundary + previous,
            "curr_cycle_index": boundary + current,
            "prev_split_type": np.full(current.size, split, dtype=object),
            "curr_split_type": np.full(current.size, split, dtype=object),
            "same_split": np.ones(current.size, dtype=bool),
            "split_type": np.full(current.size, split, dtype=object),
            "real_prev": real_prev, "real_curr": real_curr,
            "pred_prev": pred_prev, "pred_curr": pred_curr,
            "real_jump": real_jump, "pred_jump": pred_jump,
            "abs_real_jump": np.abs(real_jump), "abs_pred_jump": np.abs(pred_jump),
            "abs_jump_error": abs_error, "relative_jump_error_pct": relative,
        })
        first_write = _write_csv_chunk(jump_csv, frame, first_write)
    for array in jump_arrays.values():
        array.flush()
    elapsed("build_and_write_jump_detail", stage_started, case=stem)

    stage_started = time.monotonic()
    threshold_rows = _jump_threshold_rows(case_dir, file_idx, "file") if jump_count else []
    pd.DataFrame(threshold_rows).to_csv(
        os.path.join(case_dir, "jump_threshold_summary_file.csv"), index=False,
    )
    summary = OrderedDict([
        ("task_index", task_index), ("file_idx", file_idx), ("stem", stem), ("split", split),
        ("model_type", model_type),
        ("num_samples", int(y.size)), ("num_jumps", int(jump_count)),
        ("stats", stats), ("prediction_csv", os.path.abspath(prediction_csv)),
        ("jump_detail_csv", os.path.abspath(jump_csv)),
        ("elapsed_sec", time.monotonic() - total_started),
    ])
    _write_json(os.path.join(case_dir, "case_summary.json"), summary)
    Path(complete_path).touch()
    elapsed("case_finalize", stage_started, case=stem)
    elapsed("case_total", total_started, case=stem)
    return summary


def _merge_csv(inputs: list[str], output: str) -> None:
    first = True
    with open(output, "wb") as destination:
        for path in inputs:
            with open(path, "rb") as source:
                header = source.readline()
                if first:
                    destination.write(header)
                    first = False
                shutil.copyfileobj(source, destination, length=16 * 1024 * 1024)


def _read_scatter_sample(
    prediction_path: str, total_rows: int, max_points: int = 50_000,
) -> tuple[np.ndarray, np.ndarray]:
    sample_count = min(total_rows, max_points)
    selected = np.linspace(0, total_rows - 1, sample_count, dtype=np.int64)
    true_parts, predicted_parts = [], []
    row_offset = 0
    selected_offset = 0
    for frame in pd.read_csv(
        prediction_path, usecols=["true_label", "pred_label"], chunksize=250_000,
    ):
        row_end = row_offset + len(frame)
        selection_end = int(np.searchsorted(selected, row_end, side="left"))
        local = selected[selected_offset:selection_end] - row_offset
        if local.size:
            true_parts.append(frame["true_label"].to_numpy(dtype=np.float64)[local])
            predicted_parts.append(frame["pred_label"].to_numpy(dtype=np.float64)[local])
        selected_offset = selection_end
        row_offset = row_end
    if row_offset != total_rows or selected_offset != sample_count:
        raise ValueError(
            f"prediction CSV row mismatch: expected={total_rows}, read={row_offset}, "
            f"selected={selected_offset}/{sample_count}"
        )
    return np.concatenate(true_parts), np.concatenate(predicted_parts)


def _aggregate_jump_rows(case_dirs: list[str], name: str, group_type: str) -> list[dict]:
    if not case_dirs:
        return []
    max_jump = 0.0
    for directory in case_dirs:
        values = np.load(os.path.join(directory, "abs_real_jump.npy"), mmap_mode="r")
        if values.size:
            max_jump = max(max_jump, float(np.max(values)))
    rows = []
    for ratio in THRESHOLD_RATIOS:
        threshold = max_jump * ratio
        relative_parts, error_parts = [], []
        for directory in case_dirs:
            abs_real = np.load(os.path.join(directory, "abs_real_jump.npy"), mmap_mode="r")
            relative = np.load(os.path.join(directory, "relative_jump_error_pct.npy"), mmap_mode="r")
            abs_error = np.load(os.path.join(directory, "abs_jump_error.npy"), mmap_mode="r")
            mask = np.isfinite(relative) & (abs_real > threshold)
            if np.any(mask):
                relative_parts.append(np.asarray(relative[mask], dtype=np.float64))
                error_parts.append(np.asarray(abs_error[mask], dtype=np.float64))
        rel = np.concatenate(relative_parts) if relative_parts else np.empty(0)
        err = np.concatenate(error_parts) if error_parts else np.empty(0)
        rows.append({
            "group_type": group_type, "group_name": str(name),
            "threshold_pct": ratio * 100.0, "threshold_value": threshold,
            "sample_count": int(rel.size), "max_abs_real_jump": max_jump,
            "max_relative_jump_error_pct": float(np.max(rel)) if rel.size else None,
            "mean_relative_jump_error_pct": float(np.mean(rel)) if rel.size else None,
            "median_relative_jump_error_pct": float(np.median(rel)) if rel.size else None,
            "p90_relative_jump_error_pct": float(np.percentile(rel, 90)) if rel.size else None,
            "p95_relative_jump_error_pct": float(np.percentile(rel, 95)) if rel.size else None,
            "p99_relative_jump_error_pct": float(np.percentile(rel, 99)) if rel.size else None,
            "mean_abs_jump_error": float(np.mean(err)) if err.size else None,
            "max_abs_jump_error": float(np.max(err)) if err.size else None,
        })
    return rows


def reduce_cases(task_manifest: str, out_dir: str) -> dict:
    total_started = time.monotonic()
    task_data = _load_json(task_manifest)
    summaries, case_dirs = [], []
    for task in task_data["tasks"]:
        case_dir = os.path.join(out_dir, task["case_dir"])
        if not os.path.exists(os.path.join(case_dir, "complete")):
            raise RuntimeError(f"Full-eval case is incomplete: {task['stem']} ({case_dir})")
        summaries.append(_load_json(os.path.join(case_dir, "case_summary.json")))
        case_dirs.append(case_dir)

    stage_started = time.monotonic()
    prediction_path = os.path.join(out_dir, "prediction_samples.csv")
    _merge_csv([item["prediction_csv"] for item in summaries], prediction_path)
    elapsed("reduce_merge_predictions", stage_started)

    stage_started = time.monotonic()
    power_metrics = {}
    point_rows = []
    for split in ("train", "test"):
        selected = [item["stats"] for item in summaries if item["split"] == split]
        combined = _combine_stats(selected)
        power_metrics[split] = _power_metrics(combined)
        point_rows.append(_point_row(combined, split, "split"))
    overall_stats = _combine_stats([item["stats"] for item in summaries])
    power_metrics["overall"] = _power_metrics(overall_stats)
    point_rows.insert(0, _point_row(overall_stats, "overall", "overall"))
    for item in summaries:
        point_rows.append(_point_row(item["stats"], item["file_idx"], "file"))
        point_rows.append(_point_row(
            item["stats"], f"file_{item['file_idx']}__{item['split']}", "file_split",
        ))
    for scope, metrics in power_metrics.items():
        pd.DataFrame.from_dict(metrics, orient="index", columns=["value"]).to_csv(
            os.path.join(out_dir, f"power_metrics_{scope}.csv")
        )
    _write_json(os.path.join(out_dir, "power_metrics.json"), power_metrics)
    detail_dir = os.path.join(out_dir, "power_and_delta_detail")
    os.makedirs(detail_dir, exist_ok=True)
    point_df = pd.DataFrame(point_rows)
    point_df.to_csv(os.path.join(detail_dir, "point_metrics.csv"), index=False)
    for group_type, filename in (
        ("overall", "point_metrics_overall.csv"), ("split", "point_metrics_by_split.csv"),
        ("file", "point_metrics_by_file.csv"), ("file_split", "point_metrics_by_file_split.csv"),
    ):
        point_df[point_df["group_type"] == group_type].to_csv(
            os.path.join(detail_dir, filename), index=False,
        )
    elapsed("reduce_power_metrics", stage_started)

    stage_started = time.monotonic()
    scatter_path = os.path.join(out_dir, "true_vs_pred_scatter.png")
    scatter_true, scatter_predicted = _read_scatter_sample(
        prediction_path, int(overall_stats["n"]),
    )
    save_true_vs_pred_scatter(
        scatter_true, scatter_predicted, scatter_path,
        r2=power_metrics["overall"]["R2"],
    )
    elapsed("reduce_true_vs_pred_scatter", stage_started)

    stage_started = time.monotonic()
    jump_detail_path = os.path.join(detail_dir, "jump_detail.csv")
    _merge_csv([item["jump_detail_csv"] for item in summaries], jump_detail_path)
    elapsed("reduce_merge_jump_detail", stage_started)

    stage_started = time.monotonic()
    jump_rows = _aggregate_jump_rows(case_dirs, "overall", "overall")
    for split in ("train", "test"):
        selected_dirs = [directory for directory, item in zip(case_dirs, summaries)
                         if item["split"] == split]
        jump_rows.extend(_aggregate_jump_rows(selected_dirs, split, "split"))
    for case_dir in case_dirs:
        file_summary = os.path.join(case_dir, "jump_threshold_summary_file.csv")
        if os.path.getsize(file_summary):
            jump_rows.extend(pd.read_csv(file_summary).to_dict("records"))
    jump_df = pd.DataFrame(jump_rows)
    jump_summary_path = os.path.join(detail_dir, "jump_threshold_summary.csv")
    jump_df.to_csv(jump_summary_path, index=False)
    for group_type, filename in (
        ("overall", "jump_threshold_summary_overall.csv"),
        ("split", "jump_threshold_summary_by_split.csv"),
        ("file", "jump_threshold_summary_by_file.csv"),
    ):
        jump_df[jump_df["group_type"] == group_type].to_csv(
            os.path.join(detail_dir, filename), index=False,
        )
    elapsed("reduce_jump_summaries", stage_started)

    summary = OrderedDict([
        ("prediction_csv", os.path.abspath(prediction_path)),
        ("power_metrics", power_metrics),
        ("num_samples", int(sum(item["num_samples"] for item in summaries))),
        ("num_jumps", int(sum(item["num_jumps"] for item in summaries))),
        ("num_files", len(summaries)),
        ("case_task_manifest", os.path.abspath(task_manifest)),
        ("case_results_dir", os.path.abspath(os.path.join(out_dir, "cases"))),
        ("jump_detail_csv", os.path.abspath(jump_detail_path)),
        ("jump_threshold_summary_csv", os.path.abspath(jump_summary_path)),
        ("true_vs_pred_scatter", os.path.abspath(scatter_path)),
        ("elapsed_sec", time.monotonic() - total_started),
    ])
    _write_json(os.path.join(detail_dir, "evaluation_summary.json"), summary)
    _write_json(os.path.join(out_dir, "evaluation_summary.json"), summary)
    elapsed("reduce_total", total_started)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--tw-manifest", required=True)
    prepare.add_argument("--output-dir", required=True)

    case = subparsers.add_parser("case")
    case.add_argument("--task-manifest", required=True)
    case.add_argument("--task-index", required=True, type=int)
    case.add_argument("--level-dir", required=True)
    case.add_argument("--tog-dir", required=True)
    case.add_argument("--label-dir", required=True)
    case.add_argument("--raw-namefile", required=True)
    case.add_argument("--feature-names", required=True)
    case.add_argument("--result-dir", required=True)
    case.add_argument("--output-dir", required=True)
    case.add_argument("--model-type", choices=["xgboost", "mlp"], default="xgboost")
    case.add_argument("--two-phase-method", default="", choices=["", "A", "B"])
    case.add_argument("--two-phase-cutoff", type=int, default=9)
    case.add_argument("--chunk-rows", type=int, default=250_000)

    reduce_parser = subparsers.add_parser("reduce")
    reduce_parser.add_argument("--task-manifest", required=True)
    reduce_parser.add_argument("--output-dir", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "prepare":
        prepare_tasks(args.tw_manifest, args.output_dir)
    elif args.command == "case":
        evaluate_case(
            task_manifest=args.task_manifest, task_index=args.task_index,
            level_dir=args.level_dir, tog_dir=args.tog_dir, label_dir=args.label_dir,
            raw_namefile=args.raw_namefile, feature_names_path=args.feature_names,
            result_dir=args.result_dir, out_dir=args.output_dir, model_type=args.model_type,
            two_phase_method=args.two_phase_method, two_phase_cutoff=args.two_phase_cutoff,
            chunk_rows=args.chunk_rows,
        )
    else:
        reduce_cases(args.task_manifest, args.output_dir)


if __name__ == "__main__":
    main()
