#!/usr/bin/env python3
"""Memory-bounded plots shared by post-training evaluators."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def save_true_vs_pred_scatter(
    y_true, y_pred, output_path: str | Path, *, r2: float, max_points: int = 50_000,
) -> None:
    """Plot predicted versus true values with an equal-axis y=x reference line."""
    true_values = np.asarray(y_true, dtype=np.float64).reshape(-1)
    predicted_values = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    if true_values.size != predicted_values.size:
        raise ValueError(
            f"scatter row mismatch: true={true_values.size}, pred={predicted_values.size}"
        )
    finite = np.isfinite(true_values) & np.isfinite(predicted_values)
    true_values = true_values[finite]
    predicted_values = predicted_values[finite]
    if true_values.size == 0:
        raise ValueError("cannot draw true-vs-predicted scatter without finite points")
    total_points = int(true_values.size)
    if total_points > max_points:
        selected = np.linspace(0, total_points - 1, max_points, dtype=np.int64)
        true_values = true_values[selected]
        predicted_values = predicted_values[selected]

    lower = float(min(np.min(true_values), np.min(predicted_values)))
    upper = float(max(np.max(true_values), np.max(predicted_values)))
    span = upper - lower
    padding = span * 0.03 if span > 0 else max(abs(lower) * 0.03, 1e-9)
    bounds = (lower - padding, upper + padding)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 7), facecolor="white")
    plt.scatter(
        true_values, predicted_values, s=8, alpha=0.25,
        edgecolors="none", rasterized=True,
    )
    plt.plot(bounds, bounds, linestyle="--", color="black", linewidth=1.2, label="Ideal: y = x")
    plt.xlim(bounds)
    plt.ylim(bounds)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.xlabel("True power")
    plt.ylabel("Predicted power")
    plt.title(f"True vs Predicted Power (R²={r2:.6f}, n={total_points})")
    plt.grid(alpha=0.2)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()
