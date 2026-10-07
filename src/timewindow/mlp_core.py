#!/usr/bin/env python3
"""Shared utilities for sparse-buffer MLP HPO and retraining."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import joblib
import numpy as np
import xgboost as xgb
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler


SEED = 108


def load_buffer(path: str):
    """Load an XGBoost binary/LibSVM buffer as CSR features and float labels."""
    matrix = xgb.DMatrix(path)
    features = matrix.get_data().tocsr().astype(np.float32)
    labels = matrix.get_label().astype(np.float64)
    if features.shape[0] != labels.size:
        raise ValueError(f"row/label mismatch in {path}: {features.shape[0]} != {labels.size}")
    return features, labels


def safe_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denominator = np.maximum(np.abs(y_true), 1e-12)
    return float(np.mean(np.abs(y_true - y_pred) / denominator) * 100.0)


def count_parameters(model: MLPRegressor) -> int:
    return int(sum(value.size for value in model.coefs_) + sum(value.size for value in model.intercepts_))


def normalize_config(config: dict[str, Any]) -> dict[str, Any]:
    result = dict(config)
    hidden = result.get("hidden_layer_sizes", (128, 128))
    if isinstance(hidden, str):
        hidden = tuple(int(item) for item in hidden.split(",") if item.strip())
    elif isinstance(hidden, (int, np.integer)):
        hidden = (int(hidden),)
    else:
        hidden = tuple(int(item) for item in hidden)
    result["hidden_layer_sizes"] = hidden
    for key in ("batch_size", "max_iter", "n_iter_no_change", "random_state"):
        if key in result:
            result[key] = int(result[key])
    for key in ("alpha", "learning_rate_init", "validation_fraction", "tol"):
        if key in result:
            result[key] = float(result[key])
    return result


def build_model(config: dict[str, Any], early_stopping: bool) -> MLPRegressor:
    config = normalize_config(config)
    return MLPRegressor(
        hidden_layer_sizes=config["hidden_layer_sizes"],
        activation=config.get("activation", "relu"),
        solver="adam",
        alpha=config.get("alpha", 1e-5),
        batch_size=config.get("batch_size", 128),
        learning_rate="adaptive",
        learning_rate_init=config.get("learning_rate_init", 1e-3),
        max_iter=config.get("max_iter", 100),
        shuffle=True,
        random_state=config.get("random_state", SEED),
        tol=config.get("tol", 1e-4),
        early_stopping=early_stopping,
        validation_fraction=config.get("validation_fraction", 0.15),
        n_iter_no_change=config.get("n_iter_no_change", 12),
    )


def fit_model(
    x_train,
    y_train: np.ndarray,
    config: dict[str, Any],
    *,
    early_stopping: bool = True,
    progress: Callable[[str], None] | None = None,
):
    """Fit feature/label scalers and one MLP without densifying the full matrix."""
    feature_scaler = StandardScaler(with_mean=False, copy=True)
    x_scaled = feature_scaler.fit_transform(x_train)
    label_scaler = StandardScaler()
    y_scaled = label_scaler.fit_transform(y_train.reshape(-1, 1)).ravel()
    fitted_config = dict(config)
    usable_rows = y_train.size
    if early_stopping:
        usable_rows = max(1, int(y_train.size * (1.0 - float(config.get("validation_fraction", 0.15)))))
    fitted_config["batch_size"] = min(int(config.get("batch_size", 128)), usable_rows)
    model = build_model(fitted_config, early_stopping=early_stopping)
    model.fit(x_scaled, y_scaled)
    if progress:
        progress(
            f"trained epochs={model.n_iter_}, loss={model.loss_:.6g}, "
            f"parameters={count_parameters(model)}"
        )
    return model, feature_scaler, label_scaler


def predict(model, feature_scaler, label_scaler, features) -> np.ndarray:
    scaled = feature_scaler.transform(features)
    pred_scaled = model.predict(scaled)
    return label_scaler.inverse_transform(np.asarray(pred_scaled).reshape(-1, 1)).ravel()


def refit_full(
    x_train,
    y_train: np.ndarray,
    config: dict[str, Any],
    progress: Callable[[str], None] | None = None,
):
    """Use early stopping to select epoch count, then refit on every train row."""
    probe, _, _ = fit_model(x_train, y_train, config, early_stopping=True, progress=progress)
    selected_epochs = max(1, int(probe.n_iter_))
    final_config = dict(config)
    final_config["max_iter"] = selected_epochs
    model, x_scaler, y_scaler = fit_model(
        x_train, y_train, final_config, early_stopping=False, progress=progress
    )
    return model, x_scaler, y_scaler, selected_epochs


def save_checkpoint(path: str, model, feature_scaler, label_scaler, metadata: dict[str, Any]):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "feature_scaler": feature_scaler,
            "label_scaler": label_scaler,
            "metadata": metadata,
        },
        path,
        compress=3,
    )


def load_hpo_config(path: str) -> dict[str, Any]:
    """Read the preferred JSON artifact or a compatible Optuna CSV."""
    if path.lower().endswith(".json"):
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
        return normalize_config(payload.get("params", payload))

    import pandas as pd

    frame = pd.read_csv(path)
    score_col = "val_mape" if "val_mape" in frame.columns else "values_0"
    row = frame.sort_values(score_col).iloc[0]
    params = {}
    for name in ("num_layers", "hidden_size", "activation", "alpha", "batch_size", "learning_rate_init"):
        for candidate in (name, f"params_{name}"):
            if candidate in row.index:
                params[name] = row[candidate]
                break
    layers = int(params.pop("num_layers"))
    width = int(params.pop("hidden_size"))
    params["hidden_layer_sizes"] = [width] * layers
    return normalize_config(params)


def connection_importance(model: MLPRegressor, feature_names: list[str] | None = None):
    """Return global absolute path-weight importance (descriptive, not causal)."""
    score = np.abs(model.coefs_[0]).astype(np.float64)
    for weights in model.coefs_[1:]:
        score = score @ np.abs(weights)
    score = score.reshape(-1)
    total = float(score.sum())
    if total > 0:
        score /= total
    names = feature_names or [f"f{index}" for index in range(score.size)]
    if len(names) != score.size:
        raise ValueError(f"feature name count {len(names)} != model inputs {score.size}")
    order = np.argsort(-score)
    return [(names[index], float(score[index])) for index in order]
