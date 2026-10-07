#!/usr/bin/env python3
"""
Step 5: HPO with Two-Phase Training.

Runs Optuna multi-objective HPO on a top-K feature subset.
Supports Method A (Column Mask) and Method B (Residual) via --two-phase-method.

Reuses sum_leaf_node from existing HPO scripts.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import optuna
import xgboost as xgb
from utils.training_policy import XGB_LOSSES, configure_xgb_loss, shuffled_indices

try:
    from utils.sum_leaf_node import sum_leave_nodes
except ImportError:
    # Fallback for running this file from the original powermodeling-demo tree.
    _hpo_dir = os.path.join(
        os.path.dirname(__file__), "..", "src", "step6-power-model-retraining", "hpo"
    )
    if os.path.isdir(_hpo_dir):
        sys.path.insert(0, _hpo_dir)
    from sum_leaf_node import sum_leave_nodes


SEED = 108


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def mape_func(predt, eval_data):
    y = eval_data.get_label()
    return "mape", float(abs((y - predt) / y).mean())


def closeness_score(y, prediction, metric: str, floor: float) -> float:
    error = np.abs(np.asarray(y) - np.asarray(prediction))
    relative = error / np.maximum(np.abs(y), floor)
    if metric == "rmse":
        return float(np.sqrt(np.mean(error * error)))
    if metric == "max_ape":
        return float(np.max(relative))
    if metric == "p95_ape":
        return float(np.quantile(relative, 0.95))
    if metric == "tail_mean_ape":
        threshold = np.quantile(relative, 0.95)
        return float(np.mean(relative[relative >= threshold]))
    return float(np.mean(relative))


def run_hpo(
    train_buffer: str,
    test_buffer: str,
    output_dir: str,
    round_num: int,
    trial_num: int,
    pop_size: int,
    job_num: int,
    nfold: int,
    sampler_name: str,
    pruner_name: str,
    leaf_num_threshold: int,
    two_phase_method: str,
    two_phase_cutoff: int,
    two_phase_ratio: float,
    feature_names_file: str | None,
    loss_name: str,
    relative_floor: float,
    focus_power: float,
    closeness_metric: str,
    shuffle_seed: int,
):
    os.makedirs(output_dir, exist_ok=True)
    dtrain = xgb.DMatrix(train_buffer)
    dtest = xgb.DMatrix(test_buffer)
    dtrain = dtrain.slice(shuffled_indices(dtrain.num_row(), shuffle_seed).astype(np.int32))

    # Load feature names for two-phase column grouping
    feature_names = None
    if feature_names_file and os.path.exists(feature_names_file):
        with open(feature_names_file) as f:
            feature_names = [line.strip() for line in f]
    elif dtrain.feature_names:
        feature_names = dtrain.feature_names

    two_phase_enabled = two_phase_method in ("A", "B") and feature_names is not None
    early_stopping = max(1, int(round_num * 0.9))

    if two_phase_enabled:
        from utils.two_phase_trainer import (
            TwoPhaseTrainerA, TwoPhaseTrainerB,
            get_offset_column_groups,
        )
        if two_phase_method == "A":
            far_cols, near_cols = get_offset_column_groups(feature_names, two_phase_cutoff)
            phase1_rounds = int(round_num * two_phase_ratio)
            phase2_rounds = round_num - phase1_rounds
            log(f"Two-Phase A: cutoff={two_phase_cutoff}, far={len(far_cols)}, "
                f"near={len(near_cols)}, r1={phase1_rounds}, r2={phase2_rounds}")
        else:
            far_cols, _ = get_offset_column_groups(feature_names, two_phase_cutoff)
            if not far_cols:
                two_phase_enabled = False
                log(f"Two-Phase B: disabled because no columns have offset > {two_phase_cutoff}")
            else:
                phase1_rounds = int(round_num * two_phase_ratio)
                phase2_rounds = round_num - phase1_rounds
                log(f"Two-Phase B: cutoff={two_phase_cutoff}, far={len(far_cols)}, "
                    f"r1={phase1_rounds}, r2={phase2_rounds}")
    if not two_phase_enabled:
        log("Two-Phase: disabled (single-phase training)")
        # XGBoost 2.0 calls np.array(copy=False, dtype=int32) for its internally
        # generated int64 folds, which is rejected by NumPy 2.x.  Supplying
        # deterministic contiguous int32 folds is compatible with both NumPy 1.x
        # and 2.x and preserves the previous shuffle/seed semantics.
        from sklearn.model_selection import KFold
        if nfold > dtrain.num_row():
            raise ValueError(f"nfold={nfold} exceeds train rows={dtrain.num_row()}")
        cv_folds = [
            (
                np.asarray(train_idx, dtype=np.int32),
                np.asarray(val_idx, dtype=np.int32),
            )
            for train_idx, val_idx in KFold(
                n_splits=nfold, shuffle=True, random_state=SEED
            ).split(np.arange(dtrain.num_row()))
        ]

    def objective(trial):
        params = {
            "verbosity": 0,
            "objective": "reg:squarederror",
            "n_jobs": 64,
            "booster": "gbtree",
            "lambda": trial.suggest_float("lambda", 1e-8, 1.0, log=True),
            "alpha": trial.suggest_float("alpha", 1e-8, 1.0, log=True),
            "max_leaves": trial.suggest_int("max_leaves", 16, 64, log=True),
            "max_depth": trial.suggest_int("max_depth", 4, 32),
            "eta": trial.suggest_float("eta", 1e-8, 1.0, log=True),
            "gamma": trial.suggest_float("gamma", 1e-8, 1.0, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_float("min_child_weight", 1e-8, 5.0, log=True),
            "grow_policy": "lossguide",
            "seed": SEED,
        }
        params, custom_objective = configure_xgb_loss(
            params, loss_name, relative_floor, focus_power
        )
        def custom_metric(prediction, data):
            return closeness_metric, closeness_score(
                data.get_label(), prediction, closeness_metric, relative_floor
            )

        if not two_phase_enabled:
            result = xgb.cv(
                params, dtrain, num_boost_round=round_num,
                early_stopping_rounds=early_stopping, folds=cv_folds,
                seed=SEED, custom_metric=custom_metric, obj=custom_objective,
            )
            # xgboost 2.x returns tuple (df, cvpack)
            cv_df = result[0] if isinstance(result, tuple) else result
            # Find mape column
            mape_cols = [c for c in cv_df.columns if closeness_metric in str(c) and "test" in str(c) and "mean" in str(c)]
            mape_col = mape_cols[0] if mape_cols else cv_df.columns[-1]
            leaf_num = round_num
            return float(cv_df[mape_col].values[-1]), float(leaf_num)

        # Two-phase CV (manual k-fold)
        from sklearn.model_selection import KFold
        kf = KFold(n_splits=nfold, shuffle=True, random_state=SEED)
        n = dtrain.num_row()
        if nfold > n:
            raise ValueError(f"nfold={nfold} exceeds train rows={n}")
        mape_scores, leaf_counts = [], []

        for train_idx, val_idx in kf.split(range(n)):
            train_idx = np.asarray(train_idx, dtype=np.int32)
            val_idx = np.asarray(val_idx, dtype=np.int32)
            dt_fold = dtrain.slice(train_idx)
            # Validation fold must be sliced from dtrain. Using train-row indices
            # on dtest can hit XGBoost native out-of-range paths and segfault.
            dv_fold = dtrain.slice(val_idx)
            y_val = dtrain.get_label()[val_idx]

            if two_phase_method == "A":
                trainer = TwoPhaseTrainerA(
                    feature_names, two_phase_cutoff, two_phase_ratio, round_num,
                )
                model = trainer.train(dt_fold, dv_fold, params, custom_objective)
                pred = model.predict(dv_fold)
                leaf_num = sum_leave_nodes(
                    model.get_dump(dump_format="json", with_stats=True)
                )
            else:
                trainer = TwoPhaseTrainerB(
                    feature_names, two_phase_cutoff, two_phase_ratio, round_num,
                )
                models = trainer.train(dt_fold, dv_fold, params, custom_objective)
                pred = trainer.predict(dv_fold)
                dump_far = models["far"].get_dump(dump_format="json", with_stats=True)
                dump_res = models["residual"].get_dump(dump_format="json", with_stats=True)
                leaf_num = sum_leave_nodes(dump_far) + sum_leave_nodes(dump_res)

            mape = closeness_score(y_val, pred, closeness_metric, relative_floor)
            mape_scores.append(mape)
            leaf_counts.append(float(leaf_num))

        return float(np.mean(mape_scores)), float(np.mean(leaf_counts))

    # Sampler
    if sampler_name == "NSGAIII":
        sampler = optuna.samplers.NSGAIIISampler(population_size=pop_size, seed=SEED)
    elif sampler_name == "NSGAII":
        sampler = optuna.samplers.NSGAIISampler(population_size=pop_size, seed=SEED)
    elif sampler_name == "TPE":
        sampler = optuna.samplers.TPESampler(seed=SEED, n_startup_trials=pop_size)
    else:
        sampler = optuna.samplers.RandomSampler(seed=SEED)

    if pruner_name == "Median":
        pruner = optuna.pruners.MedianPruner(n_warmup_steps=5)
    else:
        pruner = optuna.pruners.HyperbandPruner()

    study = optuna.create_study(
        sampler=sampler, pruner=pruner,
        directions=["minimize", "minimize"],
        study_name="tw-hpo",
    )
    log(f"Sampler: {study.sampler.__class__.__name__}, Pruner: {study.pruner.__class__.__name__}")
    study.optimize(objective, n_trials=trial_num, n_jobs=job_num, show_progress_bar=True)

    log(f"Finished trials: {len(study.trials)}")
    for i, trial in enumerate(study.best_trials):
        log(f"Best trial {i} (#{trial.number}): values={trial.values}")

    # Save results
    df = study.trials_dataframe(attrs=("number", "values", "params", "state"))
    tag = f"tw-{two_phase_method}-c{two_phase_cutoff}-r{round_num}-{trial_num}"
    df.to_csv(os.path.join(output_dir, f"{tag}-trials_data.csv"), index=False)

    # Save best params
    best_params_list = []
    for trial in study.best_trials:
        leaf_count = sum_leave_nodes_from_trial(trial)
        if leaf_count <= leaf_num_threshold:
            best_params_list.append({
                "number": trial.number,
                "mape": float(trial.values[0]),
                "leaf_num": float(leaf_count),
                **trial.params,
            })

    import pandas as pd
    if best_params_list:
        pd.DataFrame(best_params_list).to_csv(
            os.path.join(output_dir, f"best_trials_leaf_{leaf_num_threshold}.csv"), index=False,
        )
        log(f"Best params (leaf<={leaf_num_threshold}): {len(best_params_list)} candidates")

    # Plots
    try:
        optuna.visualization.plot_pareto_front(
            study, target_names=["MAPE", "Leaf Num"]
        ).write_image(os.path.join(output_dir, f"{tag}-pf.png"))
    except Exception as e:
        log(f"[warn] Pareto plot: {e}")

    return study


def sum_leave_nodes_from_trial(trial) -> int:
    """Estimate leaf count from trial attributes (may not be exact for 2-phase)."""
    n_estimators = trial.user_attrs.get("n_estimators", 0)
    if n_estimators > 0:
        return n_estimators
    # Fallback: guess based on max_leaves
    return trial.params.get("max_leaves", 32)


def main():
    ap = argparse.ArgumentParser(description="Time-window HPO")
    ap.add_argument("--train-buffer", required=True)
    ap.add_argument("--test-buffer", required=True)
    ap.add_argument("--feature-names", default=None, help="feature_names.txt from step3")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--round-num", type=int, default=50)
    ap.add_argument("--trial-num", type=int, default=64)
    ap.add_argument("--pop-size", type=int, default=8)
    ap.add_argument("--job-num", type=int, default=1)
    ap.add_argument("--nfold", type=int, default=3)
    ap.add_argument("--sampler", default="NSGAII")
    ap.add_argument("--pruner", default="Hyperband")
    ap.add_argument("--leaf-threshold", type=int, default=1000)
    ap.add_argument("--two-phase-method", default="", choices=["", "A", "B"])
    ap.add_argument("--two-phase-cutoff", type=int, default=9)
    ap.add_argument("--two-phase-ratio", type=float, default=0.8)
    ap.add_argument("--loss", choices=XGB_LOSSES, default="relative_error_focus")
    ap.add_argument("--relative-floor", type=float, default=1e-3)
    ap.add_argument("--focus-power", type=float, default=3.0)
    ap.add_argument("--closeness-metric", choices=["mape", "rmse", "p95_ape", "max_ape", "tail_mean_ape"], default="tail_mean_ape")
    ap.add_argument("--shuffle-seed", type=int, default=108)
    args = ap.parse_args()

    run_hpo(
        train_buffer=args.train_buffer,
        test_buffer=args.test_buffer,
        output_dir=args.output_dir,
        round_num=args.round_num,
        trial_num=args.trial_num,
        pop_size=args.pop_size,
        job_num=args.job_num,
        nfold=args.nfold,
        sampler_name=args.sampler,
        pruner_name=args.pruner,
        leaf_num_threshold=args.leaf_threshold,
        two_phase_method=args.two_phase_method,
        two_phase_cutoff=args.two_phase_cutoff,
        two_phase_ratio=args.two_phase_ratio,
        feature_names_file=args.feature_names,
        loss_name=args.loss,
        relative_floor=args.relative_floor,
        focus_power=args.focus_power,
        closeness_metric=args.closeness_metric,
        shuffle_seed=args.shuffle_seed,
    )


if __name__ == "__main__":
    main()
