"""统一回归指标，便于不同模型/损失横向比较。"""
from __future__ import annotations

from typing import Dict

import numpy as np


def regression_metrics(
    target: np.ndarray,
    prediction: np.ndarray,
    weights: np.ndarray,
    relative_floor: float,
) -> Dict[str, float]:
    target = np.asarray(target, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    error = prediction - target
    absolute = np.abs(error)
    denominator = np.maximum(np.abs(target), relative_floor)
    normalized_weights = weights / weights.sum()
    mse = float(np.mean(error ** 2))
    weighted_mse = float(np.sum(normalized_weights * error ** 2))
    variance = float(np.var(target))
    return {
        "n": int(target.size),
        "MAE": float(np.mean(absolute)),
        "MAPE": float(np.mean(absolute / denominator) * 100.0),
        "MSE": mse,
        "RMSE": float(np.sqrt(mse)),
        "R2": float(1.0 - mse / variance) if variance > 1e-12 else float("nan"),
        "weighted_MAE": float(np.sum(normalized_weights * absolute)),
        "weighted_MAPE": float(np.sum(normalized_weights * absolute / denominator) * 100.0),
        "weighted_RMSE": float(np.sqrt(weighted_mse)),
    }
