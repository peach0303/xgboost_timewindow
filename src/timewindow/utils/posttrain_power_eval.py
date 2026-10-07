"""Post-training power and delta-power evaluation helpers.

The sampled-data path anchors every sampled point to the *true* power from
the immediately preceding raw cycle.  The full-data path reconstructs the
screened time-window features directly from the unsampled reduction caches.
"""
from __future__ import annotations

import json
import os
import re
from collections import OrderedDict, defaultdict
from typing import Callable

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack, load_npz, vstack

from step1_prepare_timewindow import choose_power_sample_indices
from utils.training_policy import shuffled_indices
from step2_validation.xgb_power_eval_from_prediction_csv import (
    compute_jump_summaries,
    evaluate_prediction_csv,
    plot_jump_histograms_by_scope,
)


FEATURE_RE = re.compile(r"^(tog|level)_o(\d+)_(.+)$")
MERGE_PREFIX_RE = re.compile(r"^m\d{3}_")


def restore_train_prediction_order(
    train_true: np.ndarray,
    train_pred: np.ndarray,
    *,
    feature_input_mode: str,
    model_type: str,
    shuffle_seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return training labels and predictions in Step1 manifest order."""
    counts = {
        "xgboost": {"proxy_columns": 3, "signal_all_windows": 1},
        "mlp": {"proxy_columns": 2, "signal_all_windows": 0},
    }
    if model_type not in counts or feature_input_mode not in counts[model_type]:
        raise ValueError(f"Unsupported train row order: {model_type}/{feature_input_mode}")
    true = np.asarray(train_true).reshape(-1)
    pred = np.asarray(train_pred).reshape(-1)
    if true.shape != pred.shape:
        raise ValueError(f"Training true/pred shape mismatch: {true.shape} != {pred.shape}")
    order = np.arange(true.size)
    permutation = shuffled_indices(true.size, shuffle_seed)
    for _ in range(counts[model_type][feature_input_mode]):
        order = order[permutation]
    inverse = np.argsort(order)
    return true[inverse], pred[inverse]


def _resolve_path(path: str, base_dir: str) -> str:
    if os.path.isabs(path):
        return path
    beside = os.path.abspath(os.path.join(base_dir, path))
    if os.path.exists(beside):
        return beside
    return os.path.abspath(path)


def _load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f, object_pairs_hook=OrderedDict)


def _label_path(stem: str, label_dir: str, sample_info: dict | None) -> str:
    if sample_info:
        entry = sample_info.get("stems", {}).get(stem, {})
        candidate = entry.get("label_file", "")
        if candidate and os.path.exists(candidate):
            return candidate
    direct = os.path.join(label_dir, f"{stem}.npy")
    if os.path.exists(direct):
        return direct
    stripped = MERGE_PREFIX_RE.sub("", stem)
    fallback = os.path.join(label_dir, f"{stripped}.npy")
    if stripped != stem and os.path.exists(fallback):
        return fallback
    raise FileNotFoundError(f"No power label for case '{stem}' under {label_dir}")


def build_sampled_prediction_frame(
    tw_manifest_path: str,
    label_dir: str,
    train_true: np.ndarray,
    train_pred: np.ndarray,
    test_true: np.ndarray,
    test_pred: np.ndarray,
) -> pd.DataFrame:
    """Restore case/cycle metadata for flattened sampled Step6 predictions."""
    tw = _load_json(tw_manifest_path)
    tw_dir = os.path.dirname(os.path.abspath(tw_manifest_path))
    sample_info = None
    sample_manifest = tw.get("sample_index_manifest", "")
    if sample_manifest:
        sample_manifest = _resolve_path(sample_manifest, tw_dir)
        sample_info = _load_json(sample_manifest)

    boundary = int(tw.get("boundary_discard", 0))
    rows = []
    file_idx = 0
    split_inputs = (
        ("train", list(tw.get("train_stems", [])), train_true, train_pred),
        ("test", list(tw.get("test_stems", [])), test_true, test_pred),
    )
    for split, stems, flat_true, flat_pred in split_inputs:
        flat_true = np.asarray(flat_true, dtype=np.float64).reshape(-1)
        flat_pred = np.asarray(flat_pred, dtype=np.float64).reshape(-1)
        if flat_true.shape != flat_pred.shape:
            raise ValueError(f"{split} true/pred shape mismatch: {flat_true.shape} != {flat_pred.shape}")
        cursor = 0
        for stem in stems:
            label_path = _label_path(stem, label_dir, sample_info)
            raw_y = np.load(label_path).reshape(-1).astype(np.float64)
            available = max(0, int(raw_y.size - boundary))
            entry = sample_info.get("stems", {}).get(stem, {}) if sample_info else {}
            index_file = entry.get("index_file", "")
            if sample_info and sample_info.get("sampled", False):
                if not index_file:
                    raise ValueError(f"Sample manifest lacks index_file for case '{stem}'")
                index_file = _resolve_path(index_file, os.path.dirname(sample_manifest))
                local_indices = np.load(index_file).reshape(-1).astype(np.int64)
            elif int(tw.get("sample_rows_per_file", 0)) > 0:
                local_indices = choose_power_sample_indices(
                    raw_y[boundary:],
                    max_rows=int(tw["sample_rows_per_file"]),
                    bins=int(tw.get("sample_bins", 20)),
                    min_per_bin=int(tw.get("sample_min_per_bin", 1)),
                    seed=int(tw.get("sample_seed", 108)) + file_idx,
                )
            else:
                local_indices = np.arange(available, dtype=np.int64)
            if local_indices.size and (
                int(local_indices.min()) < 0 or int(local_indices.max()) >= available
            ):
                raise ValueError(f"Sample index outside label range for case '{stem}'")

            count = int(local_indices.size)
            end = cursor + count
            if end > flat_true.size:
                raise ValueError(f"{split} predictions end inside case '{stem}'")
            current_indices = boundary + local_indices
            expected_true = raw_y[current_indices]
            actual_true = flat_true[cursor:end]
            if not np.allclose(expected_true, actual_true, rtol=1e-5, atol=1e-6):
                max_diff = float(np.max(np.abs(expected_true - actual_true)))
                raise ValueError(
                    f"Flattened labels do not match sample indices for case '{stem}'; "
                    f"max_abs_diff={max_diff}"
                )
            for pos, cycle in enumerate(current_indices):
                previous = float(raw_y[cycle - 1]) if cycle > 0 else np.nan
                rows.append({
                    "true_label": float(actual_true[pos]),
                    "pred_label": float(flat_pred[cursor + pos]),
                    "true_prev_label": previous,
                    "file_idx": file_idx,
                    "case_name": stem,
                    "cycle_index": int(cycle),
                    "label_index": int(local_indices[pos]),
                    "split_type": split,
                })
            cursor = end
            file_idx += 1
        if cursor != flat_true.size:
            raise ValueError(
                f"{split} metadata covers {cursor} rows but buffer contains {flat_true.size}"
            )
    return pd.DataFrame(rows)


def build_anchored_jump_detail(prediction_df: pd.DataFrame) -> pd.DataFrame:
    """Build jump rows assuming the preceding-cycle prediction is exact."""
    rows = []
    for row in prediction_df.itertuples(index=False):
        real_prev = float(row.true_prev_label)
        if not np.isfinite(real_prev):
            continue
        real_curr = float(row.true_label)
        pred_curr = float(row.pred_label)
        real_jump = real_curr - real_prev
        pred_jump = pred_curr - real_prev
        abs_error = abs(pred_jump - real_jump)
        rel_error = abs_error / abs(real_jump) * 100.0 if abs(real_jump) > 1e-12 else np.nan
        rows.append({
            "file_idx": int(row.file_idx),
            "case_name": str(row.case_name),
            "prev_cycle_index": int(row.cycle_index) - 1,
            "curr_cycle_index": int(row.cycle_index),
            "prev_split_type": str(row.split_type),
            "curr_split_type": str(row.split_type),
            "same_split": True,
            "split_type": str(row.split_type),
            "real_prev": real_prev,
            "real_curr": real_curr,
            "pred_prev": real_prev,
            "pred_curr": pred_curr,
            "real_jump": real_jump,
            "pred_jump": pred_jump,
            "abs_real_jump": abs(real_jump),
            "abs_pred_jump": abs(pred_jump),
            "abs_jump_error": abs_error,
            "relative_jump_error_pct": rel_error,
        })
    return pd.DataFrame(rows)


def save_anchored_delta_evaluation(
    prediction_df: pd.DataFrame,
    out_dir: str,
    threshold_ratios: list[float] | None = None,
) -> dict:
    """Save sampled delta-power outputs using the existing jump contract."""
    os.makedirs(out_dir, exist_ok=True)
    if threshold_ratios is None:
        threshold_ratios = [x / 10.0 for x in range(1, 10)]
    prediction_path = os.path.join(out_dir, "prediction_samples.csv")
    prediction_df.to_csv(prediction_path, index=False)
    jump_df = build_anchored_jump_detail(prediction_df)
    jump_path = os.path.join(out_dir, "jump_detail.csv")
    jump_df.to_csv(jump_path, index=False)
    summary_df = compute_jump_summaries(jump_df, threshold_ratios)
    summary_path = os.path.join(out_dir, "jump_threshold_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    for group_type, filename in (
        ("overall", "jump_threshold_summary_overall.csv"),
        ("split", "jump_threshold_summary_by_split.csv"),
        ("file", "jump_threshold_summary_by_file.csv"),
    ):
        summary_df[summary_df["group_type"] == group_type].to_csv(
            os.path.join(out_dir, filename), index=False,
        )
    plot_jump_histograms_by_scope(
        jump_df, os.path.join(out_dir, "jump_histograms"),
        threshold_ratios=threshold_ratios, bin_step=10.0, plot_per_file=False,
    )
    summary = {
        "mode": "sampled_true_previous_anchor",
        "num_samples": int(len(prediction_df)),
        "num_jumps": int(len(jump_df)),
        "num_files": int(prediction_df["file_idx"].nunique()),
        "prediction_csv": os.path.abspath(prediction_path),
        "jump_detail_csv": os.path.abspath(jump_path),
        "jump_threshold_summary_csv": os.path.abspath(summary_path),
    }
    with open(os.path.join(out_dir, "evaluation_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    return summary


def _discover_case_sources(source_dir: str, stage: str, stems: list[str], dtype: str) -> dict[str, list[str]]:
    # The directory determines the logical cache type. Existing LEVEL caches
    # commonly retain the historical .tog.csr.npz filename suffix used by
    # Step1, while newer datasets may use an explicit .level.csr.npz suffix.
    suffixes = ("tog", "level") if dtype == "level" else ("tog",)
    suffix_pattern = "(?:" + "|".join(re.escape(value) for value in suffixes) + ")"
    single = re.compile(
        rf"^(.+)\.{re.escape(stage)}\.{suffix_pattern}\.csr\.npz$"
    )
    sharded = re.compile(
        rf"^(.+)\.{re.escape(stage)}\.row(\d+)\.{suffix_pattern}\.csr\.npz$"
    )
    found: dict[str, list[str]] = defaultdict(list)
    for name in os.listdir(source_dir):
        match = single.match(name) or sharded.match(name)
        if match and match.group(1) in stems:
            found[match.group(1)].append(os.path.join(source_dir, name))
    for stem in stems:
        found[stem].sort()
        if not found[stem]:
            raise FileNotFoundError(f"No {dtype} reduction cache for case '{stem}'")
    return found


def _load_row_window(files: list[str], begin: int, end: int):
    if begin < 0:
        raise ValueError(f"Negative source row requested: {begin}")
    if len(files) == 1:
        return load_npz(files[0]).tocsr()[begin:end].tocsr()
    pieces = []
    cursor = 0
    template = None
    for path in files:
        shard = load_npz(path).tocsr()
        if template is None:
            template = shard
        shard_end = cursor + shard.shape[0]
        take0, take1 = max(begin, cursor), min(end, shard_end)
        if take0 < take1:
            pieces.append(shard[take0 - cursor:take1 - cursor].tocsr())
        cursor = shard_end
        if cursor >= end:
            break
    if not pieces:
        if template is None:
            raise ValueError("Empty source shard list")
        return csr_matrix((0, template.shape[1]), dtype=template.dtype)
    return pieces[0] if len(pieces) == 1 else vstack(pieces, format="csr")


def _feature_blocks(feature_names: list[str], raw_names: list[str]):
    lookup: dict[str, int] = {}
    for index, raw_name in enumerate(raw_names):
        sanitized = raw_name.replace("[", "_").replace("]", "_").replace("<", "_")
        if sanitized in lookup:
            raise ValueError(f"Sanitized raw signal name is not unique: {sanitized}")
        lookup[sanitized] = index
    blocks = []
    for name in feature_names:
        match = FEATURE_RE.match(name)
        if not match:
            raise ValueError(f"Invalid screened feature name: {name}")
        dtype, offset, signal = match.group(1), int(match.group(2)), match.group(3)
        if signal not in lookup:
            raise ValueError(f"Screened signal '{signal}' is absent from raw namefile")
        key = (dtype, offset)
        if not blocks or blocks[-1][0] != key:
            blocks.append([key, []])
        blocks[-1][1].append(lookup[signal])
    return [(key, np.asarray(indices, dtype=np.int64)) for key, indices in blocks]


def evaluate_full_unsampled(
    tw_manifest_path: str,
    level_dir: str,
    tog_dir: str,
    label_dir: str,
    raw_namefile: str,
    feature_names: list[str],
    predict_sparse: Callable,
    out_dir: str,
    power_evaluator: Callable[[np.ndarray, np.ndarray], dict],
) -> dict:
    """Predict every unsampled valid row, then run point and jump evaluation."""
    tw = _load_json(tw_manifest_path)
    tw_dir = os.path.dirname(os.path.abspath(tw_manifest_path))
    sample_info = None
    sample_manifest = tw.get("sample_index_manifest", "")
    if sample_manifest:
        sample_manifest = _resolve_path(sample_manifest, tw_dir)
        sample_info = _load_json(sample_manifest)
    stems_by_split = OrderedDict([
        ("train", list(tw.get("train_stems", []))),
        ("test", list(tw.get("test_stems", []))),
    ])
    all_stems = stems_by_split["train"] + stems_by_split["test"]
    stage = str(tw.get("stage", "after_one"))
    boundary = int(tw.get("boundary_discard", 0))
    with open(raw_namefile, "r", encoding="utf-8") as f:
        raw_names = [line.strip() for line in f]
    blocks = _feature_blocks(feature_names, raw_names)
    source_maps = {
        "level": _discover_case_sources(level_dir, stage, all_stems, "level"),
        "tog": _discover_case_sources(tog_dir, stage, all_stems, "tog"),
    }

    os.makedirs(out_dir, exist_ok=True)
    prediction_path = os.path.join(out_dir, "prediction_samples.csv")
    first_write = True
    file_idx = 0
    split_values: dict[str, list[tuple[np.ndarray, np.ndarray]]] = defaultdict(list)
    for split, stems in stems_by_split.items():
        for stem in stems:
            raw_y = np.load(_label_path(stem, label_dir, sample_info)).reshape(-1).astype(np.float64)
            y = raw_y[boundary:]
            matrix_parts = []
            for (dtype, offset), indices in blocks:
                begin = boundary - offset if dtype == "tog" else boundary + 1 - offset
                X = _load_row_window(source_maps[dtype][stem], begin, begin + y.size)
                if X.shape[0] == y.size + 1:
                    X = X[:-1].tocsr()
                if X.shape[0] != y.size:
                    raise ValueError(
                        f"Full row mismatch for {stem}/{dtype}_offset{offset}: "
                        f"X={X.shape[0]} y={y.size}"
                    )
                matrix_parts.append(X[:, indices].tocsr())
            X_case = matrix_parts[0] if len(matrix_parts) == 1 else hstack(matrix_parts, format="csr")
            if X_case.shape[1] != len(feature_names):
                raise ValueError(
                    f"Full feature count mismatch for {stem}: {X_case.shape[1]} != {len(feature_names)}"
                )
            pred = np.asarray(predict_sparse(X_case), dtype=np.float64).reshape(-1)
            if pred.size != y.size:
                raise ValueError(f"Prediction rows mismatch for case '{stem}'")
            split_values[split].append((y, pred))
            frame = pd.DataFrame({
                "true_label": y,
                "pred_label": pred,
                "file_idx": np.full(y.size, file_idx, dtype=np.int64),
                "case_name": np.full(y.size, stem, dtype=object),
                "cycle_index": np.arange(boundary, boundary + y.size, dtype=np.int64),
                "split_type": np.full(y.size, split, dtype=object),
                "label_index": np.arange(y.size, dtype=np.int64),
            })
            frame.to_csv(prediction_path, mode="w" if first_write else "a", header=first_write, index=False)
            first_write = False
            file_idx += 1

    power_metrics = {}
    all_y, all_pred = [], []
    for split in ("train", "test"):
        values = split_values[split]
        y = np.concatenate([item[0] for item in values])
        pred = np.concatenate([item[1] for item in values])
        all_y.append(y)
        all_pred.append(pred)
        metrics = power_evaluator(y, pred)
        power_metrics[split] = metrics
        pd.DataFrame.from_dict(metrics, orient="index", columns=["value"]).to_csv(
            os.path.join(out_dir, f"power_metrics_{split}.csv")
        )
    metrics = power_evaluator(np.concatenate(all_y), np.concatenate(all_pred))
    power_metrics["overall"] = metrics
    pd.DataFrame.from_dict(metrics, orient="index", columns=["value"]).to_csv(
        os.path.join(out_dir, "power_metrics_overall.csv")
    )
    with open(os.path.join(out_dir, "power_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(power_metrics, f, indent=2, ensure_ascii=False)

    evaluator_result = evaluate_prediction_csv(
        prediction_csv=prediction_path,
        out_dir=os.path.join(out_dir, "power_and_delta_detail"),
        require_consecutive=True,
        plot_jump_hist=True,
        jump_hist_bin_step=10.0,
        plot_per_file=False,
    )
    return {
        "prediction_csv": os.path.abspath(prediction_path),
        "power_metrics": power_metrics,
        "num_samples": int(sum(len(item[0]) for values in split_values.values() for item in values)),
        "num_jumps": int(evaluator_result["summary"]["num_jumps"]),
        "num_files": int(file_idx),
    }
