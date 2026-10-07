#!/usr/bin/env python3
"""Collect two compatible evaluation directories into one delta report."""
from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from pathlib import Path

import pandas as pd


LOWER_IS_BETTER = {"MAPE", "MSE", "RMSE", "NRMSE", "NMAE", "Max Error", "MAE"}
HIGHER_IS_BETTER = {"R2"}


def locate_metrics(path: str) -> Path:
    root = Path(path)
    candidates = [root, root / "metrics.csv", root / "eval" / "metrics.csv"]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    matches = sorted(root.glob("**/eval/metrics.csv")) if root.is_dir() else []
    if len(matches) == 1:
        return matches[0].resolve()
    raise FileNotFoundError(f"cannot uniquely locate eval/metrics.csv below {path}")


def read_metrics(path: Path) -> OrderedDict[str, float]:
    frame = pd.read_csv(path)
    if "value" not in frame.columns:
        raise ValueError(f"{path}: expected a value column")
    key_column = next((name for name in frame.columns if name != "value"), None)
    if key_column is None:
        raise ValueError(f"{path}: expected a metric-name column")
    return OrderedDict(
        (str(row[key_column]), float(row["value"])) for _, row in frame.iterrows()
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare baseline and augmented model metrics")
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--baseline-name", default="baseline")
    parser.add_argument("--candidate-name", default="candidate")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    baseline_path, candidate_path = locate_metrics(args.baseline), locate_metrics(args.candidate)
    baseline, candidate = read_metrics(baseline_path), read_metrics(candidate_path)
    common = [name for name in baseline if name in candidate]
    rows = []
    for metric in common:
        old, new = baseline[metric], candidate[metric]
        delta = new - old
        if metric in LOWER_IS_BETTER:
            improved = delta < 0
        elif metric in HIGHER_IS_BETTER:
            improved = delta > 0
        else:
            improved = None
        rows.append({
            "metric": metric, args.baseline_name: old, args.candidate_name: new,
            "candidate_minus_baseline": delta, "improved": improved,
        })
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "metrics_comparison.csv", index=False)
    summary = {
        "baseline_metrics": str(baseline_path),
        "candidate_metrics": str(candidate_path),
        "rows": rows,
    }
    (output / "metrics_comparison.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
