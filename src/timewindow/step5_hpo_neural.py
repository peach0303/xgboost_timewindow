#!/usr/bin/env python3
"""Step 5: multi-objective HPO for optional PyTorch time-window models."""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import KFold

from mlp_core import load_buffer, safe_mape
from utils.neural_models import NeuralConfig, require_torch, train_neural_regressor


SEED = 108
MODEL_TYPES = ("bnn", "bnn_fp32", "lstm", "ltc")


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def read_feature_names(path: str | None, feature_count: int) -> list[str] | None:
    if not path:
        return None
    with open(path, encoding="utf-8") as handle:
        names = [line.strip() for line in handle if line.strip()]
    if len(names) != feature_count:
        raise SystemExit(
            f"ERROR: feature name count {len(names)} != buffer columns {feature_count}"
        )
    return names


def make_sampler(name: str, population_size: int, seed: int):
    if name in ("NSGAII", "NSGAIII") and population_size < 2:
        raise SystemExit("ERROR: --pop-size must be >= 2 for NSGAII/NSGAIII")
    if name == "NSGAIII":
        return optuna.samplers.NSGAIIISampler(
            population_size=population_size, seed=seed
        )
    if name == "NSGAII":
        return optuna.samplers.NSGAIISampler(
            population_size=population_size, seed=seed
        )
    if name == "TPE":
        return optuna.samplers.TPESampler(
            seed=seed, n_startup_trials=population_size
        )
    return optuna.samplers.RandomSampler(seed=seed)


def suggest_config(
    trial: optuna.Trial, model_type: str, args: argparse.Namespace, fold: int
) -> NeuralConfig:
    if model_type in ("bnn", "bnn_fp32"):
        hidden_choices = [32, 64, 128, 256, 512]
        batch_choices = [64, 128, 256, 512]
        learning_rate_high = 1e-2
        num_layers = 2
    else:
        hidden_choices = [32, 64, 128, 256]
        batch_choices = [32, 64, 128, 256]
        learning_rate_high = 3e-3
        num_layers = (
            trial.suggest_int("num_layers", 1, 3)
            if model_type == "lstm" else 1
        )
    return NeuralConfig(
        model_type=model_type,
        epochs=args.epochs,
        batch_size=trial.suggest_categorical("batch_size", batch_choices),
        learning_rate=trial.suggest_float(
            "learning_rate", 1e-5, learning_rate_high, log=True
        ),
        hidden_size=trial.suggest_categorical("hidden_size", hidden_choices),
        num_layers=num_layers,
        dropout=trial.suggest_float("dropout", 0.0, 0.5, step=0.1),
        weight_decay=trial.suggest_float("weight_decay", 1e-8, 1e-2, log=True),
        patience=args.patience,
        validation_ratio=args.validation_ratio,
        gradient_clip=trial.suggest_categorical(
            "gradient_clip", [0.5, 1.0, 2.0, 5.0]
        ),
        sequence_cache_mb=args.sequence_cache_mb,
        mixed_precision=args.mixed_precision,
        seed=args.seed + trial.number * args.nfold + fold,
        device=args.device,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="BNN/LSTM/LTC multi-objective HPO"
    )
    parser.add_argument("--model-type", choices=MODEL_TYPES, required=True)
    parser.add_argument("--train-buffer", required=True)
    parser.add_argument(
        "--test-buffer", default=None,
        help="Compatibility/input audit only; never used for HPO",
    )
    parser.add_argument("--feature-names", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--trial-num", type=int, default=40)
    parser.add_argument("--pop-size", type=int, default=8)
    parser.add_argument("--job-num", type=int, default=1)
    parser.add_argument("--nfold", type=int, default=3)
    parser.add_argument(
        "--sampler", choices=["NSGAII", "NSGAIII", "TPE", "Random"],
        default="NSGAII",
    )
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--validation-ratio", type=float, default=0.15)
    parser.add_argument("--parameter-threshold", type=int, default=5_000_000)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--sequence-cache-mb", type=int, default=512)
    parser.add_argument(
        "--mixed-precision", choices=("auto", "on", "off"), default="auto"
    )
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    require_torch()
    if args.trial_num < 1:
        raise SystemExit("ERROR: --trial-num must be >= 1")
    if not 0.0 < args.validation_ratio < 1.0:
        raise SystemExit("ERROR: --validation-ratio must be in (0, 1)")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    x_train, y_train = load_buffer(args.train_buffer)
    if args.nfold < 2 or args.nfold > y_train.size:
        raise SystemExit(f"ERROR: --nfold must be in [2, {y_train.size}]")
    feature_names = read_feature_names(args.feature_names, x_train.shape[1])
    if args.model_type in ("lstm", "ltc") and not feature_names:
        raise SystemExit(
            f"ERROR: --feature-names is required for {args.model_type}"
        )
    if args.test_buffer:
        x_test, y_test = load_buffer(args.test_buffer)
        if x_test.shape[1] != x_train.shape[1]:
            raise SystemExit("ERROR: train/test feature counts differ")
        log(f"test buffer audited: rows={y_test.size}; it is not used for HPO")
    if args.job_num > 1 and args.device != "cpu":
        log("WARNING: parallel GPU/auto trials may compete for accelerator memory")

    folds = list(
        KFold(args.nfold, shuffle=True, random_state=args.seed).split(
            np.arange(y_train.size)
        )
    )
    log(
        f"model={args.model_type}, train rows={x_train.shape[0]}, "
        f"features={x_train.shape[1]}, folds={args.nfold}"
    )

    def objective(trial: optuna.Trial):
        scores, epochs = [], []
        parameter_count = None
        for fold, (fit_indices, val_indices) in enumerate(folds):
            config = suggest_config(trial, args.model_type, args, fold)
            _, prediction, metadata = train_neural_regressor(
                x_train[fit_indices],
                y_train[fit_indices],
                x_train[val_indices],
                feature_names,
                config,
                checkpoint_path=None,
                log=lambda _message: None,
                predict_training=False,
            )
            scores.append(safe_mape(y_train[val_indices], prediction))
            epochs.append(int(metadata["trained_epochs"]))
            parameter_count = int(metadata["parameter_count"])
        trial.set_user_attr("mean_epochs", float(np.mean(epochs)))
        trial.set_user_attr("fold_mape", scores)
        return float(np.mean(scores)), float(parameter_count)

    sampler = make_sampler(args.sampler, args.pop_size, args.seed)
    study = optuna.create_study(
        directions=["minimize", "minimize"],
        sampler=sampler,
        study_name=f"tw-{args.model_type}-hpo",
    )
    study.optimize(
        objective,
        n_trials=args.trial_num,
        n_jobs=args.job_num,
        show_progress_bar=True,
    )

    trials = study.trials_dataframe(
        attrs=("number", "values", "params", "user_attrs", "state")
    )
    trials.to_csv(output_dir / "neural_trials.csv", index=False)
    candidates = [
        trial for trial in study.best_trials
        if trial.values and trial.values[1] <= args.parameter_threshold
    ]
    if not candidates:
        candidates = [min(study.best_trials, key=lambda item: item.values[0])]
        log("no Pareto trial met parameter threshold; selecting lowest-MAPE trial")
    best = min(candidates, key=lambda item: item.values[0])

    best_config = suggest_config(best, args.model_type, args, fold=0)
    best_config.seed = args.seed
    payload = {
        "model_type": args.model_type,
        "trial_number": best.number,
        "val_mape": float(best.values[0]),
        "parameter_count": int(best.values[1]),
        "mean_epochs": best.user_attrs.get("mean_epochs"),
        "nfold": args.nfold,
        "params": asdict(best_config),
    }
    with open(output_dir / "best_trial.json", "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    csv_row = {key: value for key, value in payload.items() if key != "params"}
    csv_row.update(payload["params"])
    pd.DataFrame([csv_row]).to_csv(output_dir / "best_trial.csv", index=False)
    log(
        f"best trial #{best.number}: MAPE={best.values[0]:.6g}%, "
        f"parameters={int(best.values[1])}"
    )


if __name__ == "__main__":
    main()
