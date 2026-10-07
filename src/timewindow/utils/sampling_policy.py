"""Two-dimensional power/delta-power sampling and weighting utilities."""
from __future__ import annotations

from collections import OrderedDict
from typing import Iterable

import numpy as np


def delta_power(values: np.ndarray) -> np.ndarray:
    """Return per-row power change without crossing case/file boundaries."""
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    result = np.zeros_like(values)
    if values.size > 1:
        result[1:] = np.diff(values)
    return result


def quantile_edges(values: np.ndarray, bins: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size == 0:
        return np.array([-np.inf, np.inf], dtype=np.float64)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.array([-np.inf, np.inf], dtype=np.float64)
    edges = np.unique(np.quantile(finite, np.linspace(0.0, 1.0, max(1, bins) + 1)))
    if edges.size == 1:
        pad = max(abs(float(edges[0])) * 1e-9, 1e-12)
        edges = np.array([edges[0] - pad, edges[0] + pad])
    return edges.astype(np.float64)


def equal_width_edges(values: np.ndarray, bins: int) -> np.ndarray:
    """Return fixed-width numeric bin edges spanning the finite value range."""
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.array([-np.inf, np.inf], dtype=np.float64)
    lower, upper = float(np.min(finite)), float(np.max(finite))
    if lower == upper:
        pad = max(abs(lower) * 1e-9, 1e-12)
        return np.array([lower - pad, upper + pad], dtype=np.float64)
    return np.linspace(lower, upper, max(1, int(bins)) + 1, dtype=np.float64)


def assign_joint_bins(
    power: np.ndarray,
    delta: np.ndarray,
    power_edges: np.ndarray,
    delta_edges: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    power_bin = np.searchsorted(power_edges[1:-1], power, side="right").astype(np.int32)
    delta_bin = np.searchsorted(delta_edges[1:-1], delta, side="right").astype(np.int32)
    joint = power_bin * (len(delta_edges) - 1) + delta_bin
    return power_bin, delta_bin, joint.astype(np.int32)


def _balanced_dense_sample(
    joint: np.ndarray,
    dense_bins: list[int],
    budget: int,
    rng: np.random.Generator,
) -> list[int]:
    """Water-fill the remaining budget across dense non-empty bins."""
    if budget <= 0 or not dense_bins:
        return []
    pools = {bid: np.flatnonzero(joint == bid) for bid in dense_bins}
    chosen: list[int] = []
    active = list(dense_bins)
    remaining = int(budget)
    while remaining > 0 and active:
        quota = max(1, remaining // len(active))
        next_active = []
        for bid in active:
            pool = pools[bid]
            take = min(quota, len(pool), remaining)
            if take:
                picked = rng.choice(pool, size=take, replace=False)
                chosen.extend(int(i) for i in picked)
                pools[bid] = np.setdiff1d(pool, picked, assume_unique=False)
                remaining -= take
            if pools[bid].size and remaining:
                next_active.append(bid)
            if remaining <= 0:
                break
        active = next_active
    return chosen


def choose_2d_sample_indices(
    power: np.ndarray,
    delta: np.ndarray,
    max_rows: int,
    power_edges: np.ndarray,
    delta_edges: np.ndarray,
    rare_threshold: int,
    seed: int,
) -> tuple[np.ndarray, dict]:
    """Balance joint bins while retaining every row in genuinely sparse bins."""
    power = np.asarray(power, dtype=np.float64).reshape(-1)
    delta = np.asarray(delta, dtype=np.float64).reshape(-1)
    if power.shape != delta.shape:
        raise ValueError("power and delta_power must have identical shapes")
    _, _, joint = assign_joint_bins(power, delta, power_edges, delta_edges)
    unique, counts = np.unique(joint, return_counts=True)
    count_map = {int(b): int(c) for b, c in zip(unique, counts)}
    rare_bins = [b for b, c in count_map.items() if c <= max(0, rare_threshold)]
    dense_bins = [b for b, c in count_map.items() if c > max(0, rare_threshold)]

    n = int(power.size)
    if max_rows <= 0 or max_rows >= n:
        selected = np.arange(n, dtype=np.int64)
    else:
        # The requested cap is soft: all rare-bin rows are retained even when
        # their total exceeds it, because silently dropping them violates the
        # sampling contract and destroys tail coverage.
        rare_rows = np.flatnonzero(np.isin(joint, rare_bins)).astype(np.int64)
        remaining = max(0, int(max_rows) - int(rare_rows.size))
        dense_rows = _balanced_dense_sample(
            joint, dense_bins, remaining, np.random.default_rng(seed)
        )
        selected = np.unique(np.concatenate([rare_rows, np.asarray(dense_rows, dtype=np.int64)]))

    selected_counts = np.bincount(joint[selected], minlength=(len(power_edges) - 1) * (len(delta_edges) - 1))
    return np.sort(selected), {
        "rare_joint_bins": sorted(rare_bins),
        "rare_rows_retained": int(sum(count_map[b] for b in rare_bins)),
        "requested_rows": int(max_rows),
        "selected_rows": int(selected.size),
        "selected_joint_counts": selected_counts,
    }


def recommend_joint_weights(
    selected_joint: np.ndarray,
    joint_bin_count: int,
    exponent: float = 0.5,
    min_weight: float = 0.25,
    max_weight: float = 8.0,
) -> np.ndarray:
    """Recommend clipped inverse-frequency weights, normalized to mean one."""
    counts = np.bincount(selected_joint, minlength=joint_bin_count).astype(np.float64)
    nonzero = counts > 0
    weights = np.zeros(joint_bin_count, dtype=np.float64)
    if not np.any(nonzero):
        return weights
    reference = float(np.median(counts[nonzero]))
    weights[nonzero] = np.power(reference / counts[nonzero], float(exponent))
    weights[nonzero] = np.clip(weights[nonzero], min_weight, max_weight)
    row_mean = float(np.average(weights[nonzero], weights=counts[nonzero]))
    if row_mean > 0:
        weights[nonzero] /= row_mean
    return weights


def distribution_report(
    power: np.ndarray,
    delta: np.ndarray,
    selected: np.ndarray,
    power_edges: np.ndarray,
    delta_edges: np.ndarray,
    rare_bins: Iterable[int],
    joint_weights: np.ndarray,
) -> OrderedDict:
    power_bin, delta_bin, joint = assign_joint_bins(power, delta, power_edges, delta_edges)
    p_bins = len(power_edges) - 1
    d_bins = len(delta_edges) - 1
    before = np.bincount(joint, minlength=p_bins * d_bins).reshape(p_bins, d_bins)
    after = np.bincount(joint[selected], minlength=p_bins * d_bins).reshape(p_bins, d_bins)
    return OrderedDict([
        ("power", {"edges": power_edges.tolist(), "counts_before": np.bincount(power_bin, minlength=p_bins).tolist(),
                   "counts_after": np.bincount(power_bin[selected], minlength=p_bins).tolist()}),
        ("delta_power", {"edges": delta_edges.tolist(), "counts_before": np.bincount(delta_bin, minlength=d_bins).tolist(),
                         "counts_after": np.bincount(delta_bin[selected], minlength=d_bins).tolist()}),
        ("joint", {"shape": [p_bins, d_bins], "counts_before": before.tolist(), "counts_after": after.tolist(),
                   "sample_rates": np.divide(after, before, out=np.zeros_like(after, dtype=float), where=before > 0).tolist(),
                   "rare_bin_ids": sorted(int(x) for x in rare_bins), "recommended_weights": joint_weights.reshape(p_bins, d_bins).tolist()}),
    ])


def row_weights_from_policy(power: np.ndarray, delta: np.ndarray, policy: dict) -> np.ndarray:
    power_edges = np.asarray(policy["power_edges"], dtype=np.float64)
    delta_edges = np.asarray(policy["delta_power_edges"], dtype=np.float64)
    _, _, joint = assign_joint_bins(power, delta, power_edges, delta_edges)
    table = np.asarray(policy["joint_bin_weights"], dtype=np.float64)
    return table[joint].astype(np.float64)
