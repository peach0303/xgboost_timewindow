"""Shared deterministic shuffle, sample-weight and XGBoost loss helpers."""
from __future__ import annotations

import numpy as np


XGB_LOSSES = (
    "squared_error",
    "pseudo_huber",
    "absolute_error",
    "relative_squared",
    "error_focus",
    "relative_error_focus",
)


def shuffled_indices(size: int, seed: int = 108) -> np.ndarray:
    return np.random.default_rng(seed).permutation(int(size)).astype(np.int64)


def shuffle_xyw(x, y: np.ndarray, weight: np.ndarray | None = None, seed: int = 108):
    order = shuffled_indices(len(y), seed)
    shuffled_weight = None if weight is None or len(weight) == 0 else np.asarray(weight)[order]
    return x[order], np.asarray(y)[order], shuffled_weight, order


def load_weight_filelist(filelist_dir: str, split: str, expected_rows: int) -> np.ndarray:
    path = f"{filelist_dir}/{split}_weight.f"
    try:
        with open(path, "r", encoding="utf-8") as handle:
            files = [line.strip() for line in handle if line.strip()]
    except FileNotFoundError:
        return np.ones(expected_rows, dtype=np.float64)
    parts = [np.load(item).reshape(-1).astype(np.float64) for item in files]
    result = np.concatenate(parts) if parts else np.ones(expected_rows, dtype=np.float64)
    if len(result) != expected_rows:
        raise ValueError(f"Weight rows mismatch: {len(result)} != {expected_rows} ({path})")
    return result


def _weighted_power_objective(loss: str, relative_floor: float, focus_power: float):
    if loss not in {"relative_squared", "error_focus", "relative_error_focus"}:
        raise ValueError(f"Unsupported custom loss: {loss}")
    p = max(2.0, float(focus_power))
    eps = max(float(relative_floor), 1e-12)

    def objective(prediction: np.ndarray, dtrain):
        label = dtrain.get_label().astype(np.float64)
        error = prediction.astype(np.float64) - label
        scale = np.maximum(np.abs(label), eps) if loss.startswith("relative_") else np.ones_like(label)
        normalized = error / scale
        if loss in {"error_focus", "relative_error_focus"}:
            smooth = normalized * normalized + 1e-12
            grad = p * normalized * np.power(smooth, p / 2.0 - 1.0) / scale
            hess = p * np.power(smooth, p / 2.0 - 2.0) * (1e-12 + (p - 1.0) * normalized * normalized) / (scale * scale)
        else:
            grad = 2.0 * error / (scale * scale)
            hess = 2.0 / (scale * scale)
        weight = dtrain.get_weight()
        if weight.size:
            grad *= weight
            hess *= weight
        return grad.astype(np.float32), np.maximum(hess, 1e-12).astype(np.float32)

    return objective


def configure_xgb_loss(
    params: dict,
    loss: str,
    relative_floor: float = 1e-3,
    focus_power: float = 3.0,
) -> tuple[dict, object | None]:
    """Return copied parameters plus optional custom objective callable."""
    if loss not in XGB_LOSSES:
        raise ValueError(f"Unknown loss {loss!r}; choices={XGB_LOSSES}")
    result = dict(params)
    builtin = {
        "squared_error": "reg:squarederror",
        "pseudo_huber": "reg:pseudohubererror",
        "absolute_error": "reg:absoluteerror",
    }
    if loss in builtin:
        result["objective"] = builtin[loss]
        return result, None
    # A valid built-in objective is retained for serialization/config output;
    # xgb.train receives the callable through obj= below.
    result["objective"] = "reg:squarederror"
    return result, _weighted_power_objective(loss, relative_floor, focus_power)
