#!/usr/bin/env python3
"""Step 6: retrain the selected MLP and emit compatible evaluation artifacts."""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from math import sqrt
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning

from utils.evaluation_plots import save_true_vs_pred_scatter
from mlp_core import (
    SEED, connection_importance, count_parameters, load_buffer, load_hpo_config,
    predict, refit_full, save_checkpoint,
)


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def evaluate(y: np.ndarray, pred: np.ndarray) -> dict[str, float | int]:
    denominator = np.maximum(np.abs(y), 1e-12)
    absolute_error = np.abs(y - pred)
    relative_error = absolute_error / denominator
    mse = float(np.mean((y - pred) ** 2))
    rmse = sqrt(mse)
    y_mean, y_var, y_sum = float(np.mean(y)), float(np.var(y)), float(np.sum(y))
    return {
        "n": int(y.size), "label_mean": y_mean, "label_std": float(np.std(y)),
        "label_min": float(np.min(y)), "label_max": float(np.max(y)),
        "Max Error": float(np.max(relative_error)), "MAPE": float(np.mean(relative_error) * 100),
        "MSE": mse, "var": float(np.var(relative_error)),
        "R2": float(1 - mse / y_var) if y_var > 1e-12 else float("nan"),
        "RMSE": rmse, "NRMSE": float(rmse / y_mean) if abs(y_mean) > 1e-12 else float("nan"),
        "NMAE": float(np.sum(absolute_error) / y_sum) if abs(y_sum) > 1e-12 else float("nan"),
    }


def read_feature_names(path: str | None, count: int) -> list[str]:
    if not path:
        return [f"f{index}" for index in range(count)]
    with open(path, encoding="utf-8") as handle:
        names = [line.strip() for line in handle if line.strip()]
    if len(names) != count:
        raise SystemExit(f"ERROR: feature name count {len(names)} != buffer columns {count}")
    return names


def main() -> None:
    parser = argparse.ArgumentParser(description="Time-window MLP final retrain/evaluation")
    parser.add_argument("--train-buffer", required=True)
    parser.add_argument("--test-buffer", required=True)
    parser.add_argument("--hpo-result", required=True, help="best_trial.json or compatible CSV")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--feature-names", default=None)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--feature-input-mode", choices=["proxy_columns", "signal_all_windows"],
                        default="proxy_columns", help="Step4 buffer row-order branch for sampled evaluation")
    parser.add_argument("--shuffle-seed", type=int, default=108, help="Step3/4 row shuffle seed")
    parser.add_argument("--tw-manifest", default=None)
    parser.add_argument("--level-dir", default=None)
    parser.add_argument("--tog-dir", default=None)
    parser.add_argument("--label-dir", default=None)
    parser.add_argument("--raw-namefile", default=None)
    parser.add_argument("--skip-full-eval", action="store_true")
    args = parser.parse_args()

    output = Path(args.output_dir)
    result_dir, pred_dir, eval_dir = output / "result", output / "pred", output / "eval"
    for directory in (result_dir, pred_dir, eval_dir):
        directory.mkdir(parents=True, exist_ok=True)
    x_train, y_train = load_buffer(args.train_buffer)
    x_test, y_test = load_buffer(args.test_buffer)
    if x_train.shape[1] != x_test.shape[1]:
        raise SystemExit("ERROR: train/test feature counts differ")
    feature_names = read_feature_names(args.feature_names, x_train.shape[1])
    config = load_hpo_config(args.hpo_result)
    config["random_state"] = args.seed
    log(f"config={json.dumps(config, ensure_ascii=False)}")

    started = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        model, x_scaler, y_scaler, selected_epochs = refit_full(x_train, y_train, config, log)
    pred_train = predict(model, x_scaler, y_scaler, x_train)
    pred_test = predict(model, x_scaler, y_scaler, x_test)
    metadata = {
        "model_type": "sklearn_mlp", "selected_epochs": selected_epochs,
        "trained_epochs": int(model.n_iter_), "parameter_count": count_parameters(model),
        "input_features": x_train.shape[1], "seed": args.seed, "config": config,
    }
    save_checkpoint(str(result_dir / "mlp_model.joblib"), model, x_scaler, y_scaler, metadata)
    with open(result_dir / "model_meta.json", "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)
    np.savetxt(pred_dir / "predictions.txt", pred_test)
    np.savetxt(pred_dir / "train_predictions.txt", pred_train)

    test_metrics, train_metrics = evaluate(y_test, pred_test), evaluate(y_train, pred_train)
    for name, metrics in (("metrics", test_metrics), ("metrics_test", test_metrics), ("metrics_train", train_metrics)):
        pd.DataFrame.from_dict(metrics, orient="index", columns=["value"]).to_csv(eval_dir / f"{name}.csv")
    log(f"test MAPE={test_metrics['MAPE']:.6g}%, R2={test_metrics['R2']:.6g}, RMSE={test_metrics['RMSE']:.6g}")

    importance = connection_importance(model, feature_names)
    pd.DataFrame(importance, columns=["feature", "path_weight_importance"]).to_csv(
        result_dir / "feature_importance.csv", index=False
    )
    top = importance[:50]
    if top:
        names, scores = zip(*reversed(top))
        plt.figure(figsize=(10, 8), facecolor="white")
        plt.barh(range(len(names)), scores)
        plt.yticks(range(len(names)), names, fontsize=6)
        plt.title("Top 50 MLP feature importance (absolute path weight)")
        plt.tight_layout()
        plt.savefig(result_dir / "feature_importance.png", dpi=150)
        plt.close()

    shown = min(500, y_test.size)
    plt.figure(figsize=(12, 6), facecolor="white")
    plt.plot(np.arange(shown), y_test[:shown], label="Real", alpha=0.7)
    plt.plot(np.arange(shown), pred_test[:shown], label="Pred", alpha=0.7)
    plt.legend()
    plt.title(f"MAPE={test_metrics['MAPE']:.3f}% R2={test_metrics['R2']:.4f}")
    plt.tight_layout()
    plt.savefig(pred_dir / "pred_vs_real.png", dpi=150)
    plt.close()
    save_true_vs_pred_scatter(
        y_test, pred_test, pred_dir / "true_vs_pred_scatter.png", r2=test_metrics["R2"],
    )

    # Reuse the proven reconstruction/evaluation implementation from the source pipeline.
    shared_utils = Path(__file__).resolve().parent / "utils"
    if (args.tw_manifest or args.label_dir) and not (args.tw_manifest and args.label_dir):
        raise SystemExit("ERROR: sampled delta evaluation requires both --tw-manifest and --label-dir")
    if args.tw_manifest and args.label_dir:
        sys.path.insert(0, str(shared_utils))
        from posttrain_power_eval import (
            build_sampled_prediction_frame, restore_train_prediction_order,
            save_anchored_delta_evaluation,
        )
        train_true_ordered, train_pred_ordered = restore_train_prediction_order(
            y_train, pred_train, feature_input_mode=args.feature_input_mode,
            model_type="mlp", shuffle_seed=args.shuffle_seed,
        )
        frame = build_sampled_prediction_frame(
            args.tw_manifest, args.label_dir, train_true_ordered, train_pred_ordered, y_test, pred_test
        )
        save_anchored_delta_evaluation(frame, str(eval_dir / "sampled_delta_power"))

    full_inputs = [args.tw_manifest, args.level_dir, args.tog_dir, args.label_dir, args.raw_namefile]
    if not args.skip_full_eval and all(full_inputs):
        sys.path.insert(0, str(shared_utils))
        from posttrain_power_eval import evaluate_full_unsampled
        summary = evaluate_full_unsampled(
            tw_manifest_path=args.tw_manifest, level_dir=args.level_dir, tog_dir=args.tog_dir,
            label_dir=args.label_dir, raw_namefile=args.raw_namefile, feature_names=feature_names,
            predict_sparse=lambda features: predict(model, x_scaler, y_scaler, features),
            out_dir=str(eval_dir / "full_unsampled"), power_evaluator=evaluate,
        )
        target = eval_dir / "full_unsampled" / "evaluation_summary.json"
        with open(target, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, ensure_ascii=False)
    elif not args.skip_full_eval and any([args.level_dir, args.tog_dir, args.raw_namefile]):
        required = ["--tw-manifest", "--level-dir", "--tog-dir", "--label-dir", "--raw-namefile"]
        missing = [name for name, value in zip(required, full_inputs) if not value]
        raise SystemExit("ERROR: incomplete full-evaluation inputs; missing " + ", ".join(missing))
    log(f"all done in {time.time() - started:.1f}s: {output}")


if __name__ == "__main__":
    main()
