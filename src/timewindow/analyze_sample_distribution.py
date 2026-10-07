#!/usr/bin/env python3
"""Audit time-window label imbalance and recommend Step3-compatible weights."""
from __future__ import annotations

import argparse
import csv
import json
import os
from collections import OrderedDict
from pathlib import Path

import numpy as np

from utils.sampling_policy import (
    assign_joint_bins,
    delta_power,
    distribution_report,
    quantile_edges,
    recommend_joint_weights,
)


def read_filelist(path: Path) -> list[str]:
    with path.open(encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip() and not line.startswith("#")]


def resolve_path(value: str, base: Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute() or path.exists():
        return path.resolve()
    return (base / path).resolve()


def feature_stem(path: str) -> str:
    name = Path(path).name
    suffix = ".feature.npz"
    if not name.endswith(suffix):
        raise ValueError(f"unexpected feature path in filelist: {path}")
    return name[:-len(suffix)]


def sibling_weight_path(label_path: Path) -> Path:
    suffix = ".label.npy"
    if not label_path.name.endswith(suffix):
        raise ValueError(
            f"Step3-compatible labels must end with {suffix!r}: {label_path}"
        )
    return label_path.with_name(label_path.name[:-len(suffix)] + ".weight.npy")


def load_manifest_labels(manifest_path: Path):
    with manifest_path.open(encoding="utf-8") as handle:
        manifest = json.load(handle, object_pairs_hook=OrderedDict)
    combinations = manifest.get("combinations", {})
    if not combinations:
        raise ValueError("manifest has no combinations")
    combo_key, combo = next(iter(combinations.items()))
    filelist_dir = resolve_path(combo["filelist_dir"], manifest_path.parent)
    result = OrderedDict()
    for split in ("train", "test"):
        features = read_filelist(filelist_dir / f"{split}_feature.f")
        labels = read_filelist(filelist_dir / f"{split}_label.f")
        if len(features) != len(labels):
            raise ValueError(
                f"{split} feature/label filelist count mismatch: "
                f"{len(features)} != {len(labels)}"
            )
        indexed = OrderedDict()
        for feature, label in zip(features, labels):
            stem = feature_stem(feature)
            if stem in indexed:
                raise ValueError(f"duplicate {split} stem: {stem}")
            label_path = resolve_path(label, filelist_dir)
            if not label_path.is_file():
                raise FileNotFoundError(label_path)
            indexed[stem] = label_path
        expected = manifest.get(f"{split}_stems", [])
        if expected:
            missing = [stem for stem in expected if stem not in indexed]
            if missing:
                raise ValueError(f"{split} stems missing from filelist: {missing}")
            indexed = OrderedDict((stem, indexed[stem]) for stem in expected)
        result[split] = indexed
    return combo_key, result


def load_split(label_paths: OrderedDict[str, Path]):
    powers, deltas, spans = [], [], OrderedDict()
    cursor = 0
    for stem, path in label_paths.items():
        values = np.load(path).reshape(-1).astype(np.float64)
        if not values.size:
            raise ValueError(f"empty label file: {path}")
        if not np.all(np.isfinite(values)):
            raise ValueError(f"non-finite label found: {path}")
        change = delta_power(values)
        spans[stem] = (cursor, cursor + values.size)
        cursor += values.size
        powers.append(values)
        deltas.append(change)
    if not powers:
        raise ValueError("split has no label rows")
    return np.concatenate(powers), np.concatenate(deltas), spans


def histogram(values: np.ndarray, bins: int) -> dict:
    counts, edges = np.histogram(values, bins=max(1, bins))
    nonzero = counts[counts > 0]
    ratio = float(nonzero.max() / nonzero.min()) if nonzero.size else 0.0
    return {
        "edges": edges.tolist(),
        "counts": counts.tolist(),
        "nonempty_max_min_ratio": ratio,
        "empty_bin_fraction": float(np.mean(counts == 0)),
    }


def weight_summary(weights: np.ndarray) -> dict:
    total = float(weights.sum())
    ess = total * total / float(np.dot(weights, weights)) if weights.size else 0.0
    return {
        "min": float(weights.min()),
        "p50": float(np.quantile(weights, 0.5)),
        "p95": float(np.quantile(weights, 0.95)),
        "max": float(weights.max()),
        "mean": float(weights.mean()),
        "effective_sample_size": ess,
        "effective_sample_ratio": float(ess / weights.size),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scan label distribution and recommend Step3-compatible row weights"
    )
    parser.add_argument("--tw-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--power-bins", type=int, default=12)
    parser.add_argument("--delta-bins", type=int, default=12)
    parser.add_argument("--histogram-bins", type=int, default=40)
    parser.add_argument("--weight-exponent", type=float, default=0.5)
    parser.add_argument("--weight-min", type=float, default=0.25)
    parser.add_argument("--weight-max", type=float, default=8.0)
    parser.add_argument(
        "--write-weights", action="store_true",
        help="Write sibling *.weight.npy files consumed by Step3",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Allow --write-weights to replace existing sibling weight files",
    )
    args = parser.parse_args()
    if args.power_bins < 1 or args.delta_bins < 1 or args.histogram_bins < 1:
        parser.error("bin counts must be positive")
    if not 0.0 <= args.weight_exponent <= 1.0:
        parser.error("--weight-exponent must be in [0, 1]")
    if args.weight_min <= 0 or args.weight_max < args.weight_min:
        parser.error("require 0 < --weight-min <= --weight-max")
    if args.overwrite and not args.write_weights:
        parser.error("--overwrite requires --write-weights")

    manifest_path = Path(args.tw_manifest).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    source_combo, labels = load_manifest_labels(manifest_path)
    train_power, train_delta, train_spans = load_split(labels["train"])
    test_power, test_delta, test_spans = load_split(labels["test"])

    power_edges = quantile_edges(train_power, args.power_bins)
    delta_edges = quantile_edges(train_delta, args.delta_bins)
    _, _, train_joint = assign_joint_bins(
        train_power, train_delta, power_edges, delta_edges
    )
    joint_count = (len(power_edges) - 1) * (len(delta_edges) - 1)
    joint_weights = recommend_joint_weights(
        train_joint, joint_count, args.weight_exponent,
        args.weight_min, args.weight_max,
    )
    joint_weights[joint_weights == 0] = 1.0
    _, _, test_joint = assign_joint_bins(
        test_power, test_delta, power_edges, delta_edges
    )
    train_row_weights = joint_weights[train_joint]
    test_row_weights = joint_weights[test_joint]
    all_train = np.arange(train_power.size, dtype=np.int64)
    all_test = np.arange(test_power.size, dtype=np.int64)

    policy = OrderedDict([
        ("method", "joint_inverse_frequency"),
        ("power_edges", power_edges.tolist()),
        ("delta_power_edges", delta_edges.tolist()),
        ("joint_bin_weights", joint_weights.tolist()),
        ("exponent", args.weight_exponent),
        ("min_weight", args.weight_min),
        ("max_weight", args.weight_max),
        ("normalization", "training-row mean equals one"),
        ("recommended_default", "exponent=0.5, clip=[0.25, 8.0]"),
        ("tuning_note", "Use 0.25 for milder weighting; avoid 1.0 until weighted validation confirms tail improvement."),
    ])
    report = OrderedDict([
        ("schema_version", 1),
        ("tw_manifest", str(manifest_path)),
        ("label_source_combo", source_combo),
        ("input_format", {
            "label": "one-dimensional NumPy *.label.npy, one value per sample row",
            "weight": "one-dimensional NumPy *.weight.npy beside the label; same row count/order; finite and > 0",
        }),
        ("train", {
            "rows": int(train_power.size), "files": len(labels["train"]),
            "power_histogram_equal_width": histogram(train_power, args.histogram_bins),
            "delta_histogram_equal_width": histogram(train_delta, args.histogram_bins),
            "joint_quantile_distribution": distribution_report(
                train_power, train_delta, all_train, power_edges, delta_edges, [], joint_weights
            ),
            "recommended_row_weight": weight_summary(train_row_weights),
        }),
        ("test", {
            "rows": int(test_power.size), "files": len(labels["test"]),
            "power_histogram_equal_width": histogram(test_power, args.histogram_bins),
            "delta_histogram_equal_width": histogram(test_delta, args.histogram_bins),
            "joint_quantile_distribution": distribution_report(
                test_power, test_delta, all_test, power_edges, delta_edges, [], joint_weights
            ),
            "mapped_row_weight": weight_summary(test_row_weights),
        }),
        ("weight_recommendation", policy),
        ("weights_written", bool(args.write_weights)),
    ])

    report_path = output_dir / "sample_distribution_and_weight_recommendation.json"
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False, allow_nan=False)

    table_path = output_dir / "joint_bin_weight_recommendation.csv"
    train_counts = np.bincount(train_joint, minlength=joint_count)
    test_counts = np.bincount(test_joint, minlength=joint_count)
    d_bins = len(delta_edges) - 1
    with table_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "power_bin", "delta_bin", "power_left", "power_right",
            "delta_left", "delta_right", "train_count", "test_count", "recommended_weight",
        ])
        for joint_id in range(joint_count):
            power_bin, delta_bin = divmod(joint_id, d_bins)
            writer.writerow([
                power_bin, delta_bin,
                power_edges[power_bin], power_edges[power_bin + 1],
                delta_edges[delta_bin], delta_edges[delta_bin + 1],
                int(train_counts[joint_id]), int(test_counts[joint_id]),
                float(joint_weights[joint_id]),
            ])

    inventory_path = output_dir / "label_file_inventory.csv"
    with inventory_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["split", "stem", "label_file", "rows", "weight_file", "weight_exists"])
        for split, paths, spans in (
            ("train", labels["train"], train_spans),
            ("test", labels["test"], test_spans),
        ):
            for stem, label_path in paths.items():
                start, end = spans[stem]
                weight_path = sibling_weight_path(label_path)
                writer.writerow([
                    split, stem, label_path, end - start, weight_path, weight_path.exists()
                ])

    if args.write_weights:
        pending = []
        for split, paths, spans, flat_weights in (
            ("train", labels["train"], train_spans, train_row_weights),
            ("test", labels["test"], test_spans, test_row_weights),
        ):
            for stem, label_path in paths.items():
                start, end = spans[stem]
                target = sibling_weight_path(label_path)
                if target.exists() and not args.overwrite:
                    raise FileExistsError(
                        f"refusing to overwrite {target}; rerun with --overwrite"
                    )
                pending.append((split, stem, target, flat_weights[start:end]))
        for split, stem, target, weights in pending:
            np.save(target, weights.astype(np.float64))
            print(f"[{split}] wrote {stem}: {target} ({weights.size} rows)")

    print(f"report: {report_path}")
    print(f"joint weights: {table_path}")
    print(f"label inventory: {inventory_path}")


if __name__ == "__main__":
    main()
