#!/usr/bin/env python3
"""Step 5 (hybrid): Ridge baseline plus XGBoost residual compensation HPO."""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from collections import OrderedDict

import joblib
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from utils.training_policy import XGB_LOSSES, configure_xgb_loss, shuffled_indices



SEED = 108
RATIOS = tuple(index / 8.0 for index in range(1, 9))


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def sum_leave_nodes(json_trees: list[str]) -> int:
    """Count XGBoost leaf nodes without repository-local imports."""
    def count_tree(node: dict) -> int:
        if "leaf" in node:
            return 1
        return sum(count_tree(child) for child in node.get("children", []))

    return sum(count_tree(json.loads(tree)) for tree in json_trees)


def safe_mape(y_true: np.ndarray, prediction: np.ndarray) -> float:
    denominator = np.maximum(np.abs(y_true), 1e-12)
    return float(np.mean(np.abs(y_true - prediction) / denominator))


def evaluate(y_true: np.ndarray, prediction: np.ndarray) -> OrderedDict:
    error = y_true - prediction
    mse = float(np.mean(error * error))
    variance = float(np.var(y_true))
    return OrderedDict([
        ("n", int(y_true.size)), ("MAE", float(np.mean(np.abs(error)))),
        ("MAPE", safe_mape(y_true, prediction) * 100.0),
        ("RMSE", float(math.sqrt(mse))),
        ("R2", float(1.0 - mse / variance) if variance > 1e-12 else float("nan")),
    ])


def residual_params(trial: optuna.Trial, nthread: int) -> dict:
    return {
        "verbosity": 0, "objective": "reg:squarederror", "nthread": nthread,
        "booster": "gbtree", "tree_method": "hist", "grow_policy": "lossguide",
        "lambda": trial.suggest_float("lambda", 1e-8, 1.0, log=True),
        "alpha": trial.suggest_float("alpha", 1e-8, 1.0, log=True),
        "max_leaves": trial.suggest_int("max_leaves", 16, 64, log=True),
        "max_depth": trial.suggest_int("max_depth", 4, 32),
        "eta": trial.suggest_float("eta", 1e-8, 1.0, log=True),
        "gamma": trial.suggest_float("gamma", 1e-8, 1.0, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_float("min_child_weight", 1e-8, 5.0, log=True),
        "seed": SEED, "eval_metric": "rmse",
    }


def params_from_best(best_params: dict, nthread: int) -> dict:
    keys = (
        "lambda", "alpha", "max_leaves", "max_depth", "eta", "gamma",
        "subsample", "colsample_bytree", "min_child_weight",
    )
    result = {key: best_params[key] for key in keys}
    result.update({
        "verbosity": 0, "objective": "reg:squarederror", "nthread": nthread,
        "booster": "gbtree", "tree_method": "hist", "grow_policy": "lossguide",
        "seed": SEED, "eval_metric": "rmse",
    })
    return result


def fit_fold(
    train_x,
    train_y: np.ndarray,
    val_x,
    val_y: np.ndarray,
    train_weight: np.ndarray,
    val_weight: np.ndarray,
    linear_columns: list[int],
    ridge_alpha: float,
    ridge_max_iter: int,
    params: dict,
    rounds: int,
    early_stopping_rounds: int,
    feature_names: list[str],
    custom_objective,
):
    ridge = Ridge(
        alpha=ridge_alpha, fit_intercept=True, solver="lsqr",
        max_iter=ridge_max_iter, tol=1e-6,
    )
    ridge.fit(train_x[:, linear_columns], train_y, sample_weight=train_weight)
    train_linear = ridge.predict(train_x[:, linear_columns])
    val_linear = ridge.predict(val_x[:, linear_columns])
    residual = train_y - train_linear
    dtrain = xgb.DMatrix(train_x, label=residual, weight=train_weight, feature_names=feature_names)
    # Validation labels are residual targets only for early stopping.  The HPO
    # objective below is always computed on the combined final prediction.
    dval = xgb.DMatrix(
        val_x, label=val_y - val_linear, weight=val_weight, feature_names=feature_names
    )
    booster = xgb.train(
        params, dtrain, num_boost_round=rounds, evals=[(dval, "residual_val")],
        early_stopping_rounds=early_stopping_rounds, verbose_eval=False, obj=custom_objective,
    )
    prediction = val_linear + booster.predict(dval)
    leaves = sum_leave_nodes(booster.get_dump(dump_format="json", with_stats=True))
    return ridge, booster, prediction, int(leaves)


def fit_full_and_evaluate(
    train_x,
    train_y: np.ndarray,
    test_x,
    train_weight: np.ndarray,
    test_weight: np.ndarray,
    linear_columns: list[int],
    ridge_alpha: float,
    ridge_max_iter: int,
    params: dict,
    rounds: int,
    feature_names: list[str],
    custom_objective,
):
    """Retrain on all training rows; never use test labels for early stopping."""
    ridge = Ridge(
        alpha=ridge_alpha, fit_intercept=True, solver="lsqr",
        max_iter=ridge_max_iter, tol=1e-6,
    )
    ridge.fit(train_x[:, linear_columns], train_y, sample_weight=train_weight)
    train_linear = ridge.predict(train_x[:, linear_columns])
    test_linear = ridge.predict(test_x[:, linear_columns])
    dtrain = xgb.DMatrix(
        train_x, label=train_y - train_linear, weight=train_weight, feature_names=feature_names
    )
    dtest = xgb.DMatrix(test_x, weight=test_weight, feature_names=feature_names)
    booster = xgb.train(params, dtrain, num_boost_round=rounds, verbose_eval=False, obj=custom_objective)
    prediction = test_linear + booster.predict(dtest)
    leaves = sum_leave_nodes(booster.get_dump(dump_format="json", with_stats=True))
    return ridge, booster, prediction, int(leaves)


def run_hpo(args):
    with open(args.union_meta, "r", encoding="utf-8") as handle:
        meta = json.load(handle, object_pairs_hook=OrderedDict)
    final_top_k = int(meta["final_top_k"])
    k_mcp = int(meta["k_mcp"])
    k_xgb = int(meta["k_xgb"])
    if not (final_top_k == k_mcp == k_xgb):
        raise ValueError("union metadata violates K_MCP = K_XGB = final_top_k")
    if k_mcp < 8:
        raise ValueError("final_top_k must be >= 8 so ratio=1/8 selects at least one proxy")

    train_matrix = xgb.DMatrix(meta["train_buffer"])
    test_matrix = xgb.DMatrix(meta["test_buffer"])
    train_x = train_matrix.get_data().tocsr()
    train_y = train_matrix.get_label()
    train_weight = train_matrix.get_weight()
    if not train_weight.size:
        train_weight = np.ones(len(train_y))
    test_x = test_matrix.get_data().tocsr()
    test_y = test_matrix.get_label()
    test_weight = test_matrix.get_weight()
    if not test_weight.size:
        test_weight = np.ones(len(test_y))
    order = shuffled_indices(len(train_y), args.shuffle_seed)
    train_x, train_y, train_weight = train_x[order], train_y[order], train_weight[order]
    with open(meta["feature_names_file"], "r", encoding="utf-8") as handle:
        feature_names = [line.strip() for line in handle if line.strip()]
    if train_x.shape[1] != len(feature_names) or test_x.shape[1] != len(feature_names):
        raise ValueError("union feature-name count does not match buffers")
    ranked_linear_columns = [
        int(item["union_column"]) for item in meta["mcp_ranked_proxies"]
    ]
    if len(ranked_linear_columns) != k_mcp:
        raise ValueError("union metadata does not contain exactly K_MCP ranked MCP proxies")
    if len(set(ranked_linear_columns)) != len(ranked_linear_columns):
        raise ValueError("duplicate MCP linear columns in union metadata")

    os.makedirs(args.output_dir, exist_ok=True)
    early_stopping = max(1, int(round(args.round_num * args.early_stopping_ratio)))
    folds = list(KFold(
        n_splits=args.nfold, shuffle=True, random_state=SEED
    ).split(np.arange(train_x.shape[0])))

    def objective(trial: optuna.Trial):
        ratio = trial.suggest_categorical("ratio", RATIOS)
        linear_count = int(math.floor(k_mcp * float(ratio) + 1e-12))
        if linear_count < 1 or linear_count > k_mcp * float(ratio) + 1e-9:
            raise RuntimeError("invalid linear proxy count for ratio")
        linear_columns = ranked_linear_columns[:linear_count]
        params, custom_objective = configure_xgb_loss(
            residual_params(trial, args.nthread), args.loss,
            args.relative_floor, args.focus_power,
        )
        scores = []
        leaf_counts = []
        for train_indices, val_indices in folds:
            _, _, prediction, leaves = fit_fold(
                train_x[train_indices], train_y[train_indices],
                train_x[val_indices], train_y[val_indices],
                train_weight[train_indices], train_weight[val_indices],
                linear_columns, args.ridge_alpha, args.ridge_max_iter,
                params, args.round_num, early_stopping, feature_names, custom_objective,
            )
            relative = np.abs(train_y[val_indices] - prediction) / np.maximum(
                np.abs(train_y[val_indices]), args.relative_floor
            )
            threshold = np.quantile(relative, args.closeness_quantile)
            scores.append(float(np.mean(relative[relative >= threshold])))
            leaf_counts.append(leaves)
        mean_leaves = float(np.mean(leaf_counts))
        trial.set_user_attr("linear_proxy_count", linear_count)
        trial.set_user_attr("mean_leaf_count", mean_leaves)
        trial.set_user_attr("fold_mape", scores)
        trial.set_user_attr("fold_leaf_count", leaf_counts)
        return float(np.mean(scores)), mean_leaves

    if args.sampler == "NSGAIII":
        sampler = optuna.samplers.NSGAIIISampler(
            population_size=args.pop_size, seed=SEED
        )
    elif args.sampler == "NSGAII":
        sampler = optuna.samplers.NSGAIISampler(
            population_size=args.pop_size, seed=SEED
        )
    elif args.sampler == "TPE":
        sampler = optuna.samplers.TPESampler(seed=SEED, n_startup_trials=args.pop_size)
    else:
        sampler = optuna.samplers.RandomSampler(seed=SEED)
    study = optuna.create_study(
        directions=["minimize", "minimize"], sampler=sampler,
        study_name=f"hybrid-K{final_top_k}",
    )
    log(
        f"HPO K={final_top_k}, union={train_x.shape[1]}, ratios={RATIOS}, "
        f"trials={args.trial_num}, folds={args.nfold}"
    )
    study.optimize(
        objective, n_trials=args.trial_num, n_jobs=args.job_num,
        show_progress_bar=True,
    )

    trials_path = os.path.join(args.output_dir, "hybrid_trials.csv")
    study.trials_dataframe(
        attrs=("number", "values", "params", "user_attrs", "state")
    ).to_csv(trials_path, index=False)
    pareto_rows = []
    for trial in study.best_trials:
        pareto_rows.append({
            "number": trial.number, "mape": float(trial.values[0]),
            "leaf_num": float(trial.values[1]),
            "linear_proxy_count": trial.user_attrs["linear_proxy_count"],
            **trial.params,
        })
    pd.DataFrame(pareto_rows).to_csv(
        os.path.join(args.output_dir, "best_trials_pareto.csv"), index=False
    )
    eligible = [
        trial for trial in study.best_trials
        if float(trial.values[1]) <= args.leaf_threshold
    ]
    candidates = eligible or list(study.best_trials)
    best = min(candidates, key=lambda trial: (trial.values[0], trial.values[1]))
    if not eligible:
        log(f"warning: no Pareto trial satisfies leaf threshold {args.leaf_threshold}")

    ratio = float(best.params["ratio"])
    linear_count = int(math.floor(k_mcp * ratio + 1e-12))
    linear_columns = ranked_linear_columns[:linear_count]
    best_params, custom_objective = configure_xgb_loss(
        params_from_best(best.params, args.nthread), args.loss,
        args.relative_floor, args.focus_power,
    )
    ridge, booster, prediction, leaves = fit_full_and_evaluate(
        train_x, train_y, test_x, train_weight, test_weight, linear_columns,
        args.ridge_alpha, args.ridge_max_iter, best_params,
        args.round_num, feature_names, custom_objective,
    )
    joblib.dump(ridge, os.path.join(args.output_dir, "best_ridge_model.joblib"))
    booster.save_model(os.path.join(args.output_dir, "best_residual_xgb_model.json"))
    np.savetxt(os.path.join(args.output_dir, "best_test_predictions.txt"), prediction)
    best_meta = OrderedDict([
        ("trial_number", best.number), ("final_top_k", final_top_k),
        ("ratio", ratio), ("linear_proxy_count", linear_count),
        ("linear_proxy_limit", k_mcp * ratio),
        ("linear_columns", linear_columns),
        ("linear_proxies", [feature_names[index] for index in linear_columns]),
        ("ridge_alpha", args.ridge_alpha), ("xgb_params", best_params),
        ("xgb_rounds_limit", args.round_num), ("leaf_count", leaves),
        ("cv_mape", float(best.values[0])), ("cv_mean_leaf_count", float(best.values[1])),
        ("test_metrics", evaluate(test_y, prediction)),
        ("loss", args.loss),
        ("closeness_quantile", args.closeness_quantile),
    ])
    with open(os.path.join(args.output_dir, "best_hybrid_model_meta.json"), "w", encoding="utf-8") as handle:
        json.dump(best_meta, handle, indent=2, ensure_ascii=False)
    log(
        f"best trial={best.number}, ratio={ratio:g}, linear={linear_count}/{k_mcp}, "
        f"CV MAPE={best.values[0] * 100.0:.4f}%, test MAPE={best_meta['test_metrics']['MAPE']:.4f}%"
    )
    return study


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Ridge + residual XGBoost HPO")
    parser.add_argument("--union-meta", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--round-num", type=int, default=50)
    parser.add_argument("--trial-num", type=int, default=64)
    parser.add_argument("--pop-size", type=int, default=8)
    parser.add_argument("--job-num", type=int, default=1)
    parser.add_argument("--nthread", type=int, default=32)
    parser.add_argument("--nfold", type=int, default=3)
    parser.add_argument("--sampler", choices=["NSGAIII", "NSGAII", "TPE", "Random"], default="NSGAII")
    parser.add_argument("--leaf-threshold", type=int, default=1000)
    parser.add_argument("--ridge-alpha", type=float, default=8.0)
    parser.add_argument("--ridge-max-iter", type=int, default=2000)
    parser.add_argument("--loss", choices=XGB_LOSSES, default="error_focus")
    parser.add_argument("--relative-floor", type=float, default=1e-3)
    parser.add_argument("--focus-power", type=float, default=3.0)
    parser.add_argument("--closeness-quantile", type=float, default=0.95)
    parser.add_argument("--shuffle-seed", type=int, default=108)
    parser.add_argument("--early-stopping-ratio", type=float, default=0.1)
    args = parser.parse_args()
    if args.nfold < 2:
        parser.error("--nfold must be >= 2")
    if args.ridge_alpha < 0:
        parser.error("--ridge-alpha must be non-negative")
    if not 0 < args.early_stopping_ratio < 1:
        parser.error("--early-stopping-ratio must be in (0, 1)")
    run_hpo(args)


if __name__ == "__main__":
    main()
