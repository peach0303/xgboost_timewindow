#!/usr/bin/env python3
"""Step 5: resumable multi-objective HPO for a time-window MLP regressor."""
from __future__ import annotations

import argparse
import json
import math
import os
import threading
import time
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import KFold

from mlp_core import SEED, count_parameters, fit_model, load_buffer, predict, safe_mape


STUDY_NAME = "tw-mlp-hpo"


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def _atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    os.replace(temporary, path)


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def _finished_trials(study: optuna.Study) -> list[optuna.trial.FrozenTrial]:
    return [trial for trial in study.get_trials(deepcopy=False) if trial.state.is_finished()]


def _select_best(study: optuna.Study, parameter_threshold: int):
    pareto = [trial for trial in study.best_trials if trial.values]
    if not pareto:
        return None, False
    candidates = [trial for trial in pareto if trial.values[1] <= parameter_threshold]
    used_fallback = not candidates
    if used_fallback:
        candidates = pareto
    return min(candidates, key=lambda item: item.values[0]), used_fallback


def _best_payload(best, args: argparse.Namespace, finished: int, target: int) -> dict:
    best_config = {
        "hidden_layer_sizes": [int(best.params["hidden_size"])] * int(best.params["num_layers"]),
        "activation": best.params["activation"],
        "alpha": float(best.params["alpha"]),
        "batch_size": int(best.params["batch_size"]),
        "learning_rate_init": float(best.params["learning_rate_init"]),
        "max_iter": args.epochs,
        "n_iter_no_change": args.patience,
        "validation_fraction": args.validation_ratio,
        "random_state": args.seed,
    }
    return {
        "trial_number": best.number,
        "val_mape": float(best.values[0]),
        "parameter_count": int(best.values[1]),
        "mean_epochs": best.user_attrs.get("mean_epochs"),
        "finished_trials": finished,
        "target_trials": target,
        "params": best_config,
    }


def _write_snapshot(
    study: optuna.Study, args: argparse.Namespace, output_dir: Path, *, reason: str,
) -> None:
    finished_trials = _finished_trials(study)
    finished = len(finished_trials)
    trials = study.trials_dataframe(
        attrs=("number", "values", "params", "user_attrs", "state")
    )
    _atomic_csv(output_dir / "mlp_trials.csv", trials)
    pareto = pd.DataFrame([
        {
            "trial_number": trial.number,
            "val_mape": trial.values[0],
            "parameter_count": trial.values[1],
            **trial.params,
        }
        for trial in study.best_trials if trial.values
    ])
    _atomic_csv(output_dir / "pareto_trials.csv", pareto)

    best, used_fallback = _select_best(study, args.parameter_threshold)
    if best is not None:
        payload = _best_payload(best, args, finished, args.trial_num)
        payload["parameter_threshold_fallback"] = used_fallback
        _atomic_json(output_dir / "best_trial.json", payload)
        csv_row = {key: value for key, value in payload.items() if key != "params"}
        csv_row.update(best.params)
        _atomic_csv(output_dir / "best_trial.csv", pd.DataFrame([csv_row]))

    states = Counter(trial.state.name for trial in study.get_trials(deepcopy=False))
    progress = {
        "study_name": study.study_name,
        "storage": str((output_dir / "optuna_study.db").resolve()),
        "target_trials": args.trial_num,
        "finished_trials": finished,
        "completed_trials": states.get("COMPLETE", 0),
        "failed_trials": states.get("FAIL", 0),
        "pruned_trials": states.get("PRUNED", 0),
        "running_trials": states.get("RUNNING", 0),
        "progress_fraction": min(1.0, finished / args.trial_num),
        "checkpoint_reason": reason,
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    _atomic_json(output_dir / "hpo_progress.json", progress)
    if best is None:
        log(f"checkpoint reason={reason} finished={finished}/{args.trial_num}; no complete trial")
    else:
        log(
            f"checkpoint reason={reason} finished={finished}/{args.trial_num} "
            f"best=#{best.number} MAPE={best.values[0]:.6g}% params={int(best.values[1])}"
        )


def _sampler(args: argparse.Namespace, seed: int):
    if args.sampler == "NSGAIII":
        return optuna.samplers.NSGAIIISampler(population_size=args.pop_size, seed=seed)
    if args.sampler == "NSGAII":
        return optuna.samplers.NSGAIISampler(population_size=args.pop_size, seed=seed)
    if args.sampler == "TPE":
        return optuna.samplers.TPESampler(seed=seed, n_startup_trials=args.pop_size)
    return optuna.samplers.RandomSampler(seed=seed)


def _study_signature(args: argparse.Namespace, x_train, y_train: np.ndarray) -> dict:
    return {
        "train_buffer": str(Path(args.train_buffer).resolve()),
        "train_rows": int(x_train.shape[0]),
        "feature_count": int(x_train.shape[1]),
        "label_sum": float(np.sum(y_train)),
        "label_sum_squares": float(np.sum(np.square(y_train))),
        "nfold": args.nfold,
        "epochs": args.epochs,
        "patience": args.patience,
        "validation_ratio": args.validation_ratio,
        "sampler": args.sampler,
        "pop_size": args.pop_size,
        "seed": args.seed,
    }


def _validate_signature(study: optuna.Study, signature: dict) -> None:
    existing = study.user_attrs.get("hpo_signature")
    if existing is None:
        study.set_user_attr("hpo_signature", signature)
        return
    if existing != signature:
        differences = [
            f"{key}: stored={existing.get(key)!r}, current={signature.get(key)!r}"
            for key in sorted(set(existing) | set(signature))
            if existing.get(key) != signature.get(key)
        ]
        raise SystemExit(
            "ERROR: existing Optuna study is incompatible with current HPO inputs:\n  "
            + "\n  ".join(differences)
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Time-window MLP HPO")
    parser.add_argument("--train-buffer", required=True)
    parser.add_argument("--test-buffer", default=None, help="Compatibility/input audit only; never used for HPO")
    parser.add_argument("--feature-names", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--trial-num", type=int, default=40)
    parser.add_argument("--pop-size", type=int, default=8)
    parser.add_argument("--job-num", type=int, default=1)
    parser.add_argument("--nfold", type=int, default=3)
    parser.add_argument("--sampler", choices=["NSGAII", "NSGAIII", "TPE", "Random"], default="NSGAII")
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--validation-ratio", type=float, default=0.15)
    parser.add_argument("--parameter-threshold", type=int, default=5_000_000)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    if args.trial_num <= 0:
        parser.error("--trial-num must be positive")
    if args.job_num <= 0:
        parser.error("--job-num must be positive")
    if args.sampler in ("NSGAII", "NSGAIII") and args.pop_size < 2:
        parser.error("--pop-size must be >= 2 for NSGAII/NSGAIII")
    return args


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    x_train, y_train = load_buffer(args.train_buffer)
    if args.nfold < 2 or args.nfold > y_train.size:
        raise SystemExit(f"ERROR: --nfold must be in [2, {y_train.size}]")
    if args.test_buffer:
        x_test, y_test = load_buffer(args.test_buffer)
        if x_test.shape[1] != x_train.shape[1]:
            raise SystemExit("ERROR: train/test feature counts differ")
        log(f"test buffer audited: rows={y_test.size}; it is not used for HPO")
    if args.feature_names:
        with open(args.feature_names, encoding="utf-8") as handle:
            feature_name_count = sum(1 for line in handle if line.strip())
        if feature_name_count != x_train.shape[1]:
            raise SystemExit(
                f"ERROR: feature name count {feature_name_count} != buffer columns {x_train.shape[1]}"
            )
    log(f"train rows={x_train.shape[0]}, features={x_train.shape[1]}, folds={args.nfold}")

    split = list(KFold(args.nfold, shuffle=True, random_state=args.seed).split(np.arange(y_train.size)))

    def objective(trial: optuna.Trial):
        layers = trial.suggest_int("num_layers", 1, 4)
        width = trial.suggest_categorical("hidden_size", [32, 64, 128, 256, 512])
        config = {
            "hidden_layer_sizes": [width] * layers,
            "activation": trial.suggest_categorical("activation", ["relu", "tanh"]),
            "alpha": trial.suggest_float("alpha", 1e-8, 1e-2, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [64, 128, 256, 512]),
            "learning_rate_init": trial.suggest_float("learning_rate_init", 1e-5, 1e-2, log=True),
            "max_iter": args.epochs,
            "n_iter_no_change": args.patience,
            "validation_fraction": args.validation_ratio,
            "random_state": args.seed + trial.number,
        }
        scores, sizes, epochs = [], [], []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            for fit_indices, val_indices in split:
                model, x_scaler, y_scaler = fit_model(
                    x_train[fit_indices], y_train[fit_indices], config, early_stopping=True
                )
                fold_pred = predict(model, x_scaler, y_scaler, x_train[val_indices])
                scores.append(safe_mape(y_train[val_indices], fold_pred))
                sizes.append(count_parameters(model))
                epochs.append(model.n_iter_)
        trial.set_user_attr("mean_epochs", float(np.mean(epochs)))
        return float(np.mean(scores)), float(np.mean(sizes))

    storage_path = (output_dir / "optuna_study.db").resolve()
    storage = optuna.storages.RDBStorage(
        url=f"sqlite:///{storage_path}", engine_kwargs={"connect_args": {"timeout": 60}}
    )
    existing_names = {summary.study_name for summary in optuna.study.get_all_study_summaries(storage)}
    if STUDY_NAME in existing_names:
        preliminary = optuna.load_study(study_name=STUDY_NAME, storage=storage)
        resume_seed = args.seed + len(_finished_trials(preliminary))
        study = optuna.load_study(
            study_name=STUDY_NAME, storage=storage, sampler=_sampler(args, resume_seed)
        )
        log(f"resume study from {storage_path}")
    else:
        study = optuna.create_study(
            directions=["minimize", "minimize"], sampler=_sampler(args, args.seed),
            study_name=STUDY_NAME, storage=storage,
        )
        log(f"created persistent study at {storage_path}")
    _validate_signature(study, _study_signature(args, x_train, y_train))

    finished_before = len(_finished_trials(study))
    remaining = max(0, args.trial_num - finished_before)
    checkpoint_interval = max(1, math.ceil(args.trial_num / 10))
    checkpoint_lock = threading.Lock()
    checkpoint_bucket = finished_before // checkpoint_interval
    _write_snapshot(study, args, output_dir, reason="resume" if finished_before else "initial")
    log(
        f"HPO progress={finished_before}/{args.trial_num}, remaining={remaining}, "
        f"checkpoint_interval={checkpoint_interval}"
    )

    def checkpoint_callback(current_study: optuna.Study, _trial) -> None:
        nonlocal checkpoint_bucket
        with checkpoint_lock:
            finished = len(_finished_trials(current_study))
            bucket = finished // checkpoint_interval
            if bucket > checkpoint_bucket or finished >= args.trial_num:
                checkpoint_bucket = bucket
                _write_snapshot(current_study, args, output_dir, reason="10_percent")

    if remaining:
        try:
            study.optimize(
                objective, n_trials=remaining, n_jobs=args.job_num,
                callbacks=[checkpoint_callback], show_progress_bar=True,
            )
        finally:
            _write_snapshot(study, args, output_dir, reason="interrupt_or_finish")
    else:
        log("target trial count already reached; skip optimization")

    finished_after = len(_finished_trials(study))
    if finished_after < args.trial_num:
        raise RuntimeError(
            f"HPO stopped before target: finished={finished_after}, target={args.trial_num}"
        )
    _write_snapshot(study, args, output_dir, reason="complete")
    best, used_fallback = _select_best(study, args.parameter_threshold)
    if best is None:
        raise RuntimeError("HPO completed without a successful trial")
    if used_fallback:
        log("no Pareto trial met parameter threshold; selected lowest-MAPE Pareto trial")
    log(f"best trial #{best.number}: MAPE={best.values[0]:.6g}%, parameters={int(best.values[1])}")


if __name__ == "__main__":
    main()
