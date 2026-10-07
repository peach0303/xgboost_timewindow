#!/usr/bin/env python3
"""
Step 6: Final Retrain & Evaluation.

Trains XGBoost, BNN, LSTM, or liquid time-constant (LTC) models and evaluates
them through one output contract. XGBoost retains optional two-phase training.

Output: model files, predictions, evaluation metrics, feature importance.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from math import sqrt
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from utils.evaluation_plots import save_true_vs_pred_scatter
from utils.training_policy import XGB_LOSSES, configure_xgb_loss, shuffled_indices

SEED = 108


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def mkdirp(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def evaluate(y, pred):
    denom = np.where(np.abs(y) > 1e-12, np.abs(y), 1e-12)
    abs_err = np.abs(y - pred)
    rel_err = abs_err / denom
    mse = float(np.mean((y - pred) ** 2))
    rmse = sqrt(mse)
    y_mean = float(np.average(y))
    y_var = float(np.var(y))
    y_sum = float(np.sum(y))
    return {
        "n": int(y.shape[0]),
        "label_mean": y_mean,
        "label_std": float(np.std(y)),
        "label_min": float(np.min(y)),
        "label_max": float(np.max(y)),
        "Max Error": float(np.max(rel_err)),
        "MAPE": float(np.average(rel_err) * 100),
        "MSE": mse,
        "var": float(np.var(rel_err)),
        "R2": float(1 - mse / y_var) if y_var > 1e-12 else float("nan"),
        "RMSE": rmse,
        "NRMSE": float(rmse / y_mean) if abs(y_mean) > 1e-12 else float("nan"),
        "NMAE": float(np.sum(abs_err) / y_sum) if abs(y_sum) > 1e-12 else float("nan"),
    }


def load_best_params(hpo_csv: str) -> dict:
    """Load best params from HPO trials CSV."""
    df = pd.read_csv(hpo_csv)
    if "mape" in df.columns:
        df = df.sort_values("mape").iloc[:1]
    row = df.iloc[0]
    return {
        "lambda": float(row["lambda"]),
        "alpha": float(row["alpha"]),
        "max_leaves": int(row["max_leaves"]),
        "max_depth": int(row["max_depth"]),
        "eta": float(row["eta"]),
        "gamma": float(row["gamma"]),
        "subsample": float(row["subsample"]),
        "colsample_bytree": float(row["colsample_bytree"]),
        "min_child_weight": float(row["min_child_weight"]),
        "grow_policy": "lossguide",
        "seed": SEED,
    }


def main():
    ap = argparse.ArgumentParser(description="Final retrain & evaluation")
    ap.add_argument("--train-buffer", required=True)
    ap.add_argument("--test-buffer", required=True)
    ap.add_argument("--hpo-csv", default=None, help="Best trials CSV; required only for XGBoost")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--rounds", type=int, default=50)
    ap.add_argument(
        "--model-type", choices=["xgboost", "bnn", "bnn_fp32", "lstm", "ltc"],
        default="xgboost",
    )
    ap.add_argument("--hpo-result", default=None,
                    help="Neural best_trial.json; valid for bnn/bnn_fp32/lstm/ltc")
    ap.add_argument("--feature-names", default=None)
    ap.add_argument("--two-phase-method", default="", choices=["", "A", "B"])
    ap.add_argument("--two-phase-cutoff", type=int, default=9)
    ap.add_argument("--two-phase-ratio", type=float, default=0.8)
    ap.add_argument("--loss", choices=XGB_LOSSES, default="relative_error_focus")
    ap.add_argument("--relative-floor", type=float, default=1e-3)
    ap.add_argument("--focus-power", type=float, default=3.0)
    ap.add_argument("--shuffle-seed", type=int, default=108)
    ap.add_argument("--feature-input-mode", choices=["proxy_columns", "signal_all_windows"],
                    default="proxy_columns", help="Step4 buffer row-order branch for sampled evaluation")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--learning-rate", type=float, default=1e-3)
    ap.add_argument("--hidden-size", type=int, default=128)
    ap.add_argument("--num-layers", type=int, default=1)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--weight-decay", type=float, default=1e-5)
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--validation-ratio", type=float, default=0.15)
    ap.add_argument("--gradient-clip", type=float, default=1.0)
    ap.add_argument("--sequence-cache-mb", type=int, default=512)
    ap.add_argument(
        "--mixed-precision", choices=("auto", "on", "off"), default="auto"
    )
    ap.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--tw-manifest", default=None,
                    help="Step1 manifest for sampled delta metadata")
    ap.add_argument("--level-dir", default=None,
                    help="Unsampled reduced LEVEL cache directory")
    ap.add_argument("--tog-dir", default=None,
                    help="Unsampled reduced TOG cache directory")
    ap.add_argument("--label-dir", default=None,
                    help="Raw power-label directory")
    ap.add_argument("--raw-namefile", default=None,
                    help="Original reduced-cache signal namefile")
    ap.add_argument("--skip-full-eval", action="store_true",
                    help="Skip unsampled full-data evaluation")
    args = ap.parse_args()

    result_dir = os.path.join(args.output_dir, "result")
    pred_dir = os.path.join(args.output_dir, "pred")
    eval_dir = os.path.join(args.output_dir, "eval")
    for d in [result_dir, pred_dir, eval_dir]:
        mkdirp(d)

    dtrain = xgb.DMatrix(args.train_buffer)
    dtest = xgb.DMatrix(args.test_buffer)
    dtrain = dtrain.slice(shuffled_indices(dtrain.num_row(), args.shuffle_seed).astype(np.int32))
    y_train = dtrain.get_label()
    y_test = dtest.get_label()

    feature_names = None
    if args.feature_names and os.path.exists(args.feature_names):
        with open(args.feature_names) as f:
            feature_names = [line.strip() for line in f]

    if args.model_type == "xgboost":
        if not args.hpo_csv:
            raise SystemExit("ERROR: --hpo-csv is required for --model-type xgboost")
        params = load_best_params(args.hpo_csv)
        params.update({
            "n_jobs": -1,
            "objective": "reg:squarederror",
            "booster": "gbtree",
            "tree_method": "hist",
            "eval_metric": "rmse",
        })
        params, custom_objective = configure_xgb_loss(
            params, args.loss, args.relative_floor, args.focus_power
        )
        log(f"Params: {json.dumps(params, indent=2)}")
    else:
        params = None
        custom_objective = None

    two_phase_enabled = (
        args.model_type == "xgboost"
        and args.two_phase_method in ("A", "B")
        and feature_names is not None
    )
    if args.model_type == "xgboost" and args.two_phase_method in ("A", "B") and feature_names is None:
        log(f"Two-phase disabled: missing or unreadable feature names file: {args.feature_names}")

    if two_phase_enabled:
        from utils.two_phase_trainer import get_offset_column_groups
        far_cols, near_cols = get_offset_column_groups(feature_names, args.two_phase_cutoff)
        log(
            f"Two-phase {args.two_phase_method}: cutoff={args.two_phase_cutoff}, "
            f"far={len(far_cols)}, near={len(near_cols)}"
        )
        if args.two_phase_method == "B" and not far_cols:
            log(f"Two-phase B disabled: no columns have offset > {args.two_phase_cutoff}")
            two_phase_enabled = False

    t0 = time.time()

    if args.model_type != "xgboost":
        from utils.neural_models import (
            NeuralConfig, load_neural_hpo_config, train_neural_regressor,
        )
        config_values = {
            "epochs": args.epochs, "batch_size": args.batch_size,
            "learning_rate": args.learning_rate, "hidden_size": args.hidden_size,
            "num_layers": args.num_layers, "dropout": args.dropout,
            "weight_decay": args.weight_decay, "patience": args.patience,
            "validation_ratio": args.validation_ratio,
            "gradient_clip": args.gradient_clip,
            "sequence_cache_mb": args.sequence_cache_mb,
            "mixed_precision": args.mixed_precision,
        }
        if args.hpo_result:
            config_values.update(
                load_neural_hpo_config(args.hpo_result, args.model_type)
            )
            log(f"Loaded neural HPO result: {args.hpo_result}")
        config_values.update({
            "model_type": args.model_type, "seed": args.seed, "device": args.device,
        })
        config = NeuralConfig(**config_values)
        log(f"Neural config: {json.dumps(config_values, ensure_ascii=False)}")
        x_train = dtrain.get_data().tocsr()
        x_test = dtest.get_data().tocsr()
        pred_train, pred, model_meta, neural_predict = train_neural_regressor(
            x_train, y_train, x_test, feature_names, config,
            os.path.join(result_dir, f"{args.model_type}_model.pt"), log,
            return_predictor=True,
        )
        with open(os.path.join(result_dir, "model_meta.json"), "w", encoding="utf-8") as f:
            json.dump(model_meta, f, indent=2, ensure_ascii=False)
        model = None
        predict_sparse = neural_predict
        log(f"{args.model_type.upper()} done in {time.time() - t0:.1f}s")
    elif not two_phase_enabled:
        # Single-phase training
        model = xgb.train(
            params, dtrain, num_boost_round=args.rounds,
            evals=[(dtest, "eval"), (dtrain, "train")],
            verbose_eval=10,
            obj=custom_objective,
        )
        pred = model.predict(dtest)
        pred_train = model.predict(dtrain)
        model.save_model(os.path.join(result_dir, "xgbmodel.json"))
        log(f"Single-phase done in {time.time() - t0:.1f}s")
    elif args.two_phase_method == "A":
        from utils.two_phase_trainer import TwoPhaseTrainerA
        trainer = TwoPhaseTrainerA(
            feature_names, args.two_phase_cutoff, args.two_phase_ratio, args.rounds,
        )
        model = trainer.train(dtrain, dtest, params, custom_objective)
        pred = trainer.predict(dtest)
        pred_train = trainer.predict(dtrain)
        trainer.model_final.save_model(os.path.join(result_dir, "xgbmodel_phase2.json"))
        log(f"Method A done in {time.time() - t0:.1f}s")
    else:
        from utils.two_phase_trainer import TwoPhaseTrainerB
        trainer = TwoPhaseTrainerB(
            feature_names, args.two_phase_cutoff, args.two_phase_ratio, args.rounds,
        )
        models = trainer.train(dtrain, dtest, params, custom_objective)
        pred = trainer.predict(dtest)
        pred_train = trainer.predict(dtrain)
        models["far"].save_model(os.path.join(result_dir, "xgbmodel_far.json"))
        models["residual"].save_model(os.path.join(result_dir, "xgbmodel_residual.json"))
        model = models  # for importance
        log(f"Method B done in {time.time() - t0:.1f}s")

    if args.model_type == "xgboost":
        if two_phase_enabled:
            predict_sparse = lambda X: trainer.predict(
                xgb.DMatrix(X, feature_names=feature_names))
        else:
            predict_sparse = lambda X: model.predict(
                xgb.DMatrix(X, feature_names=feature_names))

    # Save predictions
    np.savetxt(os.path.join(pred_dir, "predictions.txt"), pred)
    np.savetxt(os.path.join(pred_dir, "train_predictions.txt"), pred_train)

    # Evaluate
    metrics = evaluate(y_test, pred)
    train_metrics = evaluate(y_train, pred_train)
    pd.DataFrame.from_dict(metrics, orient="index", columns=["value"]).to_csv(
        os.path.join(eval_dir, "metrics.csv")
    )
    pd.DataFrame.from_dict(metrics, orient="index", columns=["value"]).to_csv(
        os.path.join(eval_dir, "metrics_test.csv")
    )
    pd.DataFrame.from_dict(train_metrics, orient="index", columns=["value"]).to_csv(
        os.path.join(eval_dir, "metrics_train.csv")
    )
    log(
        f"Test: MAPE={metrics['MAPE']:.3f}%, R2={metrics['R2']:.6f}, "
        f"RMSE={metrics['RMSE']:.6g}, NRMSE={metrics['NRMSE']:.6f}"
    )
    log(
        f"Train: MAPE={train_metrics['MAPE']:.3f}%, R2={train_metrics['R2']:.6f}, "
        f"RMSE={train_metrics['RMSE']:.6g}, NRMSE={train_metrics['NRMSE']:.6f}"
    )

    # Sampled delta power uses the true preceding-cycle power as its anchor.
    if args.tw_manifest and args.label_dir:
        from utils.posttrain_power_eval import (
            build_sampled_prediction_frame,
            save_anchored_delta_evaluation,
            restore_train_prediction_order,
        )
        train_true_ordered, train_pred_ordered = restore_train_prediction_order(
            y_train, pred_train, feature_input_mode=args.feature_input_mode,
            model_type="xgboost", shuffle_seed=args.shuffle_seed,
        )
        sampled_frame = build_sampled_prediction_frame(
            args.tw_manifest, args.label_dir,
            train_true_ordered, train_pred_ordered, y_test, pred,
        )
        sampled_summary = save_anchored_delta_evaluation(
            sampled_frame, os.path.join(eval_dir, "sampled_delta_power")
        )
        log(
            f"Sampled delta power: samples={sampled_summary['num_samples']}, "
            f"jumps={sampled_summary['num_jumps']}"
        )
    elif args.tw_manifest or args.label_dir:
        raise SystemExit(
            "ERROR: sampled delta evaluation requires both --tw-manifest and --label-dir"
        )

    # Rebuild only the final screened columns and predict every unsampled row.
    full_inputs = [
        args.tw_manifest, args.level_dir, args.tog_dir,
        args.label_dir, args.raw_namefile,
    ]
    if not args.skip_full_eval and all(full_inputs):
        if not feature_names:
            raise SystemExit("ERROR: full evaluation requires --feature-names")
        from utils.posttrain_power_eval import evaluate_full_unsampled
        full_summary = evaluate_full_unsampled(
            tw_manifest_path=args.tw_manifest,
            level_dir=args.level_dir,
            tog_dir=args.tog_dir,
            label_dir=args.label_dir,
            raw_namefile=args.raw_namefile,
            feature_names=feature_names,
            predict_sparse=predict_sparse,
            out_dir=os.path.join(eval_dir, "full_unsampled"),
            power_evaluator=evaluate,
        )
        with open(
            os.path.join(eval_dir, "full_unsampled", "evaluation_summary.json"),
            "w", encoding="utf-8",
        ) as f:
            json.dump(full_summary, f, indent=2, ensure_ascii=False)
        log(
            f"Full unsampled: samples={full_summary['num_samples']}, "
            f"jumps={full_summary['num_jumps']}, files={full_summary['num_files']}"
        )
    elif not args.skip_full_eval and any([args.level_dir, args.tog_dir, args.raw_namefile]):
        missing = [
            name for name, value in zip(
                ("--tw-manifest", "--level-dir", "--tog-dir", "--label-dir", "--raw-namefile"),
                full_inputs,
            ) if not value
        ]
        raise SystemExit(
            "ERROR: incomplete full-evaluation inputs; missing " + ", ".join(missing)
        )

    # Plot pred vs real
    xs = np.arange(min(500, len(y_test)))
    plt.figure(figsize=(12, 6), facecolor="white")
    plt.plot(xs, y_test[:500], label="Real", alpha=0.7)
    plt.plot(xs, pred[:500], label="Pred", alpha=0.7)
    plt.legend()
    plt.title(f"MAPE={metrics['MAPE']:.3f}% R2={metrics['R2']:.4f}")
    plt.tight_layout()
    plt.savefig(os.path.join(pred_dir, "pred_vs_real.png"), dpi=150)
    plt.close()
    save_true_vs_pred_scatter(
        y_test, pred, os.path.join(pred_dir, "true_vs_pred_scatter.png"),
        r2=metrics["R2"],
    )

    # Feature importance (for single model or Method A)
    if args.model_type == "xgboost" and (not two_phase_enabled or args.two_phase_method == "A"):
        importance = model.get_score(importance_type="gain")
        imp_sorted = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:50]
        if imp_sorted:
            names, scores = zip(*imp_sorted)
            plt.figure(figsize=(10, 8), facecolor="white")
            plt.barh(range(len(names)), scores)
            plt.yticks(range(len(names)), names, fontsize=6)
            plt.gca().invert_yaxis()
            plt.title("Top 50 Feature Importance (gain)")
            plt.tight_layout()
            plt.savefig(os.path.join(result_dir, "feature_importance.png"), dpi=150)
            plt.close()
            pd.DataFrame(imp_sorted, columns=["feature", "gain"]).to_csv(
                os.path.join(result_dir, "feature_importance.csv"), index=False,
            )

    log(f"All done. Results: {args.output_dir}")


if __name__ == "__main__":
    main()
