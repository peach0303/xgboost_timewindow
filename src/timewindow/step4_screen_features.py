#!/usr/bin/env python3
"""
Step 4: Signal-Level Feature Importance Screening.

1. Trains a default XGBoost on the Step3 concatenated feature matrix.
2. Aggregates column-level importance to signal-level (max across all
   time-window columns belonging to the same signal).
3. Selects top-K signals.
4. By default, expands those signals from Step1 full time-window features when
   --tw-manifest and --namefile are provided. This keeps all available
   offset/type columns for each selected signal, instead of only the Step2
   proxy-selected columns present in Step3.

Feature naming: "{type}_o{offset}_{signal_name}"
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from collections import OrderedDict, defaultdict

import numpy as np
from scipy.sparse import load_npz, hstack, vstack
import xgboost as xgb
from utils.training_policy import XGB_LOSSES, configure_xgb_loss, shuffle_xyw


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def mkdirp(path: str) -> None:
    os.makedirs(path, exist_ok=True)


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
    mkdirp(output_dir)
    with open(os.path.join(output_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    with open(os.path.join(output_dir, "metrics.csv"), "w", encoding="utf-8") as f:
        f.write("metric,value\n")
        for key, value in metrics.items():
            f.write(f"{key},{value}\n")
    if pred is not None:
        np.savetxt(os.path.join(output_dir, "predictions.txt"), pred)


def log_test_metrics(tag: str, metrics: OrderedDict) -> None:
    """Print one consistent post-training test-set verification line."""
    log(
        f"[TEST VERIFY] {tag}: n={int(metrics['n'])}, "
        f"MAPE={metrics['MAPE']:.3f}%, MAE={metrics['MAE']:.6g}, "
        f"RMSE={metrics['RMSE']:.6g}, NRMSE={metrics['NRMSE']:.6g}, "
        f"R2={metrics['R2']:.6g}"
    )


def save_test_verification_summary(output_dir: str, rows: list[OrderedDict]) -> None:
    """Save all Step4 post-training test metrics in compact JSON/CSV files."""
    mkdirp(output_dir)
    with open(os.path.join(output_dir, "test_verification_summary.json"), "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)

    columns = [
        "model", "K", "columns", "rounds", "n", "MAPE", "MAE",
        "RMSE", "NRMSE", "R2",
    ]
    with open(os.path.join(output_dir, "test_verification_summary.csv"), "w", encoding="utf-8") as f:
        f.write(",".join(columns) + "\n")
        for row in rows:
            f.write(",".join(str(row.get(column, "")) for column in columns) + "\n")


def sanitize_feature_name(name: str) -> str:
    return name.replace("[", "_").replace("]", "_").replace("<", "_")


def extract_signal_name(full_name: str) -> str:
    """Extract signal name from 'TYPE_oOFFSET_original_signal_name'."""
    m = re.match(r"^(?:tog|level)_o\d+_(.+)$", full_name)
    if m:
        return m.group(1)
    return full_name


def read_filelist(path: str) -> list[str]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s and not s.startswith("#"):
                out.append(s)
    return out


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


def load_namefile(path: str) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f]


def selected_indices_for_combo(
    dtype: str,
    offset_k: int,
    signal_names: list[str],
    top_sig_set: set[str],
    n_cols: int,
) -> tuple[list[int], list[str]]:
    indices: list[int] = []
    names: list[str] = []
    limit = min(len(signal_names), n_cols)
    for idx in range(limit):
        full_name = sanitize_feature_name(f"{dtype}_o{offset_k}_{signal_names[idx]}")
        if extract_signal_name(full_name) in top_sig_set:
            indices.append(idx)
            names.append(full_name)
    return indices, names


def build_top_matrices_from_step1(
    tw_manifest: dict,
    namefile_path: str,
    top_sig_set: set[str],
) -> tuple[object, np.ndarray, object, np.ndarray, list[str], list[OrderedDict]]:
    """Build screened train/test matrices from Step1 full features."""
    signal_names = load_namefile(namefile_path)
    combos = tw_manifest["combinations"]
    train_stems = tw_manifest["train_stems"]
    test_stems = tw_manifest["test_stems"]
    if not train_stems and not test_stems:
        raise ValueError("Step1 manifest contains no train/test stems")

    split_file_index = {
        split: {
            combo_key: build_split_index(combo_info, split)
            for combo_key, combo_info in combos.items()
        }
        for split in ("train", "test")
    }

    combo_selected: OrderedDict[str, tuple[list[int], list[str]]] = OrderedDict()
    source_blocks: list[OrderedDict] = []
    col_offset = 0
    probe_split = "train" if train_stems else "test"
    probe_stem = train_stems[0] if train_stems else test_stems[0]
    for combo_key, combo_info in combos.items():
        feat_path, _label_path = split_file_index[probe_split][combo_key][probe_stem]
        n_cols = load_npz(feat_path).shape[1]
        dtype = combo_info["data_type"]
        offset_k = combo_info["offset"]
        indices, names = selected_indices_for_combo(dtype, offset_k, signal_names, top_sig_set, n_cols)
        combo_selected[combo_key] = (indices, names)
        if indices:
            source_blocks.append(OrderedDict([
                ("combo_key", combo_key),
                ("offset", offset_k),
                ("data_type", dtype),
                ("n_selected", len(indices)),
                ("start_col", col_offset),
                ("end_col", col_offset + len(indices)),
            ]))
            col_offset += len(indices)

    feature_names = [name for _combo_key, (_indices, names) in combo_selected.items() for name in names]
    if not feature_names:
        raise ValueError("No Step1 columns matched selected top signals")

    def build_split(split: str, stem_list: list[str]):
        X_parts = []
        y_parts = []
        w_parts = []
        first_combo = next(iter(combos.keys()))
        for stem in stem_list:
            row_parts = []
            try:
                _first_feat, label_path = split_file_index[split][first_combo][stem]
            except KeyError as exc:
                raise FileNotFoundError(
                    f"Missing {split} stem '{stem}' in Step1 filelist for {first_combo}"
                ) from exc
            labels = np.load(label_path).reshape(-1).astype(np.float64)
            y_parts.append(labels)
            weight_path = label_path.replace(".label.npy", ".weight.npy")
            w_parts.append(np.load(weight_path).reshape(-1) if os.path.exists(weight_path) else np.ones(len(labels)))

            for combo_key in combos.keys():
                indices, _names = combo_selected[combo_key]
                if not indices:
                    continue
                try:
                    feat_path, _label_path = split_file_index[split][combo_key][stem]
                except KeyError as exc:
                    raise FileNotFoundError(
                        f"Missing {split} stem '{stem}' in Step1 filelist for {combo_key}"
                    ) from exc
                X = load_npz(feat_path).tocsr()
                row_parts.append(X[:, indices].tocsr())
            if not row_parts:
                raise ValueError(f"No selected columns for {split} stem '{stem}'")
            X_parts.append(hstack(row_parts, format="csr"))

        X_all = vstack(X_parts, format="csr") if len(X_parts) > 1 else X_parts[0]
        y_all = np.concatenate(y_parts)
        return X_all, y_all, np.concatenate(w_parts).astype(np.float64)

    X_train, y_train, w_train = build_split("train", train_stems)
    X_test, y_test, w_test = build_split("test", test_stems)
    return X_train, y_train, w_train, X_test, y_test, w_test, feature_names, source_blocks


def train_default_eval(
    X_train,
    y_train: np.ndarray,
    X_test,
    y_test: np.ndarray,
    params: dict,
    train_weight: np.ndarray,
    test_weight: np.ndarray,
    objective,
    rounds: int,
    out_dir: str,
    feature_names: list[str] | None = None,
) -> OrderedDict:
    dtrain_eval = xgb.DMatrix(X_train, label=y_train, weight=train_weight, feature_names=feature_names)
    dtest_eval = xgb.DMatrix(X_test, label=y_test, weight=test_weight, feature_names=feature_names)
    model = xgb.train(
        params, dtrain_eval, num_boost_round=rounds,
        evals=[(dtest_eval, "eval")], verbose_eval=False, obj=objective,
    )
    pred = model.predict(dtest_eval)
    metrics = evaluate_predictions(y_test, pred)
    save_eval_outputs(out_dir, metrics, pred)
    model.save_model(os.path.join(out_dir, "xgb_default_model.json"))
    return metrics


def save_screen_outputs(
    out_dir: str,
    K: int,
    X_train,
    y_train: np.ndarray,
    X_test,
    y_test: np.ndarray,
    train_weight: np.ndarray,
    test_weight: np.ndarray,
    feature_names: list[str],
    sorted_signals: list[tuple[str, float]],
    top_cols: list[int] | None,
    source_blocks: list[OrderedDict] | None,
) -> None:
    mkdirp(out_dir)
    dtrain_top = xgb.DMatrix(X_train, label=y_train, weight=train_weight, feature_names=feature_names)
    dtest_top = xgb.DMatrix(X_test, label=y_test, weight=test_weight, feature_names=feature_names)
    dtrain_top.save_binary(os.path.join(out_dir, f"train_top{K}.buffer"))
    dtest_top.save_binary(os.path.join(out_dir, f"test_top{K}.buffer"))

    with open(os.path.join(out_dir, "feature_names.txt"), "w", encoding="utf-8") as f:
        for name in feature_names:
            f.write(name + "\n")
    with open(os.path.join(out_dir, f"top{K}_signals.txt"), "w", encoding="utf-8") as f:
        for name, score in sorted_signals[:K]:
            f.write(f"{name}\t{score}\n")
    if top_cols is not None:
        with open(os.path.join(out_dir, f"top{K}_col_indices.txt"), "w", encoding="utf-8") as f:
            for idx in top_cols:
                f.write(f"{idx}\n")
        np.savetxt(os.path.join(out_dir, f"top{K}_col_indices.npy"), np.array(top_cols, dtype=np.int64))
    with open(os.path.join(out_dir, "signal_importance.csv"), "w", encoding="utf-8") as f:
        f.write("signal,max_importance_gain\n")
        for name, score in sorted_signals:
            f.write(f"{name},{score}\n")
    with open(os.path.join(out_dir, "screen_meta.json"), "w", encoding="utf-8") as f:
        json.dump(OrderedDict([
            ("K", K),
            ("columns", len(feature_names)),
            ("train_shape", list(X_train.shape)),
            ("test_shape", list(X_test.shape)),
            ("source_blocks", source_blocks or []),
        ]), f, indent=2, ensure_ascii=False)


def main():
    ap = argparse.ArgumentParser(description="Signal-level feature importance screening")
    ap.add_argument("--train-buffer", required=True, help="Full Step3 train DMatrix buffer")
    ap.add_argument("--test-buffer", required=True, help="Full Step3 test DMatrix buffer")
    ap.add_argument("--concat-meta", required=True, help="concat_meta.json from step3")
    ap.add_argument("--top-ks", default="256,512,1024,2048,4096",
                    help="Comma-separated K values")
    ap.add_argument("--default-rounds", type=int, default=50, help="Default XGBoost rounds")
    ap.add_argument("--screen-eval-rounds", type=int, default=None,
                    help="Rounds for per-K screened-buffer test evaluation; default uses --default-rounds, 0 disables")
    ap.add_argument("--seed", type=int, default=108)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--tw-manifest", default=None,
                    help="timewindow_manifest.json from Step1; enables all-available expansion")
    ap.add_argument("--namefile", default=None,
                    help="Signal name file used to map Step1 columns to signal names")
    ap.add_argument("--screen-source", choices=["auto", "step3", "step1"], default="auto",
                    help="Where to build screened buffers from. auto uses Step1 when --tw-manifest and --namefile are provided.")
    ap.add_argument("--loss", choices=XGB_LOSSES, default="relative_error_focus")
    ap.add_argument("--relative-floor", type=float, default=1e-3)
    ap.add_argument("--focus-power", type=float, default=3.0)
    ap.add_argument("--shuffle-seed", type=int, default=108)
    args = ap.parse_args()

    top_ks = [int(x.strip()) for x in args.top_ks.split(",") if x.strip()]
    screen_eval_rounds = args.default_rounds if args.screen_eval_rounds is None else args.screen_eval_rounds
    verification_rows: list[OrderedDict] = []

    with open(args.concat_meta) as f:
        concat_meta = json.load(f, object_pairs_hook=OrderedDict)
    feature_names = concat_meta.get("feature_names", [])
    if not feature_names:
        with open(os.path.join(os.path.dirname(args.concat_meta), "feature_names.txt")) as f:
            feature_names = [line.strip() for line in f]
    log(f"Total Step3 feature columns: {len(feature_names)}")

    dtrain = xgb.DMatrix(args.train_buffer)
    dtest = xgb.DMatrix(args.test_buffer)
    X_train = dtrain.get_data().tocsr()
    y_train = dtrain.get_label()
    w_train = dtrain.get_weight()
    if not w_train.size:
        w_train = np.ones(len(y_train))
    X_test = dtest.get_data().tocsr()
    y_test = dtest.get_label()
    w_test = dtest.get_weight()
    if not w_test.size:
        w_test = np.ones(len(y_test))
    X_train, y_train, w_train, _ = shuffle_xyw(X_train, y_train, w_train, args.shuffle_seed)
    log(f"Step3 Train: {X_train.shape}, Test: {X_test.shape}")

    default_params = {
        "n_jobs": -1,
        "objective": "reg:squarederror",
        "booster": "gbtree",
        "tree_method": "approx",
        "grow_policy": "lossguide",
        "seed": args.seed,
        "eval_metric": "rmse",
        "learning_rate": 0.1,
        "max_depth": 6,
    }
    default_params, objective = configure_xgb_loss(
        default_params, args.loss, args.relative_floor, args.focus_power
    )
    dtrain = xgb.DMatrix(X_train, label=y_train, weight=w_train, feature_names=feature_names)
    dtest = xgb.DMatrix(X_test, label=y_test, weight=w_test, feature_names=feature_names)

    log("Training default XGBoost for feature importance...")
    t0 = time.time()
    model = xgb.train(
        default_params, dtrain,
        num_boost_round=args.default_rounds,
        evals=[(dtest, "eval")],
        verbose_eval=10,
        obj=objective,
    )
    log(f"Training done in {time.time() - t0:.1f}s")

    pred_step3 = model.predict(dtest)
    step3_metrics = evaluate_predictions(y_test, pred_step3)
    eval_dir = os.path.join(args.output_dir, "default_model_eval")
    save_eval_outputs(eval_dir, step3_metrics, pred_step3)
    model.save_model(os.path.join(eval_dir, "xgb_default_model.json"))
    log_test_metrics("Step3 concatenated default model", step3_metrics)
    verification_rows.append(OrderedDict([
        ("model", "step3_default"), ("K", ""), ("columns", int(X_train.shape[1])),
        ("rounds", int(args.default_rounds)), *step3_metrics.items(),
    ]))

    col_importance = model.get_score(importance_type="gain")
    log(f"Got importance for {len(col_importance)} columns")

    signal_scores: dict[str, float] = {}
    signal_col_map: dict[str, list[int]] = defaultdict(list)
    for col_idx, col_name in enumerate(feature_names):
        sig = extract_signal_name(col_name)
        score = col_importance.get(col_name, col_importance.get(f"f{col_idx}", 0.0))
        signal_scores[sig] = max(signal_scores.get(sig, 0.0), score)
        signal_col_map[sig].append(col_idx)

    sorted_signals = sorted(signal_scores.items(), key=lambda x: x[1], reverse=True)
    log(f"Unique signals in Step3: {len(signal_scores)}, columns per signal: ~{len(feature_names)/max(1,len(signal_scores)):.1f}")

    use_step1 = args.screen_source == "step1" or (
        args.screen_source == "auto" and args.tw_manifest and args.namefile
    )
    if use_step1:
        if not args.tw_manifest or not args.namefile:
            raise SystemExit("ERROR: --screen-source step1 requires --tw-manifest and --namefile")
        with open(args.tw_manifest) as f:
            tw_manifest = json.load(f, object_pairs_hook=OrderedDict)
        log("Screen source: Step1 full features (all available columns for selected signals)")
    else:
        tw_manifest = None
        log("Screen source: Step3 concatenated proxy-selected columns")

    for K in top_ks:
        if K > len(sorted_signals):
            log(f"  K={K} > available signals ({len(sorted_signals)}), skipping")
            continue

        top_sig_set = set(name for name, _ in sorted_signals[:K])
        out_dir = os.path.join(args.output_dir, f"screen_K{K}")

        if use_step1:
            X_train_top, y_train_top, w_train_top, X_test_top, y_test_top, w_test_top, top_names, source_blocks = build_top_matrices_from_step1(
                tw_manifest=tw_manifest,
                namefile_path=args.namefile,
                top_sig_set=top_sig_set,
            )
            X_train_top, y_train_top, w_train_top, _ = shuffle_xyw(
                X_train_top, y_train_top, w_train_top, args.shuffle_seed + K
            )
            log(f"\n  K={K}: {len(top_sig_set)} signals -> {len(top_names)} Step1 columns")
            save_screen_outputs(
                out_dir, K, X_train_top, y_train_top, X_test_top, y_test_top, w_train_top, w_test_top,
                top_names, sorted_signals, top_cols=None, source_blocks=source_blocks,
            )
            if screen_eval_rounds > 0:
                metrics_k = train_default_eval(
                    X_train_top, y_train_top, X_test_top, y_test_top, default_params,
                    w_train_top, w_test_top, objective,
                    screen_eval_rounds, os.path.join(out_dir, "default_model_eval"), top_names,
                )
                log_test_metrics(f"screen_K{K}", metrics_k)
                verification_rows.append(OrderedDict([
                    ("model", f"screen_K{K}"), ("K", int(K)), ("columns", len(top_names)),
                    ("rounds", int(screen_eval_rounds)), *metrics_k.items(),
                ]))
        else:
            top_cols = []
            for sig_name in sorted_signals[:K]:
                top_cols.extend(signal_col_map[sig_name[0]])
            top_cols = sorted(set(top_cols))
            top_names = [feature_names[i] for i in top_cols]
            log(f"\n  K={K}: {len(top_sig_set)} signals -> {len(top_cols)} Step3 columns")
            X_train_top = X_train[:, top_cols].tocsr()
            X_test_top = X_test[:, top_cols].tocsr()
            save_screen_outputs(
                out_dir, K,
                X_train_top, y_train,
                X_test_top, y_test, w_train, w_test,
                top_names, sorted_signals, top_cols=top_cols, source_blocks=None,
            )
            if screen_eval_rounds > 0:
                metrics_k = train_default_eval(
                    X_train_top, y_train, X_test_top, y_test, default_params,
                    w_train, w_test, objective,
                    screen_eval_rounds, os.path.join(out_dir, "default_model_eval"), top_names,
                )
                log_test_metrics(f"screen_K{K}", metrics_k)
                verification_rows.append(OrderedDict([
                    ("model", f"screen_K{K}"), ("K", int(K)), ("columns", len(top_names)),
                    ("rounds", int(screen_eval_rounds)), *metrics_k.items(),
                ]))

    save_test_verification_summary(args.output_dir, verification_rows)
    log(f"Test verification summary: {os.path.join(args.output_dir, 'test_verification_summary.csv')}")
    log(f"\nAll K values processed. Output: {args.output_dir}")


if __name__ == "__main__":
    main()
