#!/usr/bin/env python3
"""Backfill a weight-recommendation report from already sampled Step 1 labels."""
from __future__ import annotations

import argparse
import json
import os
from collections import OrderedDict
from pathlib import Path

import numpy as np
from utils.indexed_split import indexed_split_paths, load_indexed_combo_labels

from utils.sampling_policy import (
    assign_joint_bins,
    delta_power,
    distribution_report,
    quantile_edges,
    recommend_joint_weights,
)


def read_filelist(path: Path) -> list[str]:
    with path.open(encoding="utf-8") as handle:
        return [
            line.strip()
            for line in handle
            if line.strip() and not line.lstrip().startswith("#")
        ]


def resolve_path(value: str, base: Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (base / path).resolve()


def feature_stem(path: str) -> str:
    name = Path(path).name
    suffix = ".feature.npz"
    if not name.endswith(suffix):
        raise ValueError(f"unexpected feature path in filelist: {path}")
    return name[: -len(suffix)]


def load_sampled_labels(manifest_path: Path):
    with manifest_path.open(encoding="utf-8") as handle:
        manifest = json.load(handle, object_pairs_hook=OrderedDict)
    combinations = manifest.get("combinations", {})
    if not combinations:
        raise ValueError("time-window manifest has no combinations")

    combo_key, combo = next(iter(combinations.items()))
    split_indices = indexed_split_paths(manifest)
    if split_indices is not None:
        labels_by_split = OrderedDict()
        for split in ("train", "test"):
            labels, _weights = load_indexed_combo_labels(combo, split_indices[split])
            stems = manifest.get(f"{split}_stems", [])
            stem = stems[0] if stems else f"split_after_shuffle_{split}"
            labels_by_split[split] = OrderedDict([(stem, labels)])
        return manifest, combo_key, labels_by_split

    filelist_dir = resolve_path(combo["filelist_dir"], manifest_path.parent)
    labels_by_split = OrderedDict()
    for split in ("train", "test"):
        features = read_filelist(filelist_dir / f"{split}_feature.f")
        labels = read_filelist(filelist_dir / f"{split}_label.f")
        if len(features) != len(labels):
            raise ValueError(
                f"{split} feature/label count mismatch: "
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
        if not indexed:
            raise ValueError(f"time-window manifest has no {split} labels")
        labels_by_split[split] = indexed
    return manifest, combo_key, labels_by_split


def load_split(labels: OrderedDict[str, Path | np.ndarray]):
    power_parts, delta_parts = [], []
    inventory = []
    for stem, path in labels.items():
        if isinstance(path, np.ndarray):
            values = path.reshape(-1).astype(np.float64, copy=False)
            label_file = "<indexed-split>"
        else:
            values = np.load(path).reshape(-1).astype(np.float64)
            label_file = str(path)
        if values.size == 0:
            raise ValueError(f"empty sampled label file: {path}")
        if not np.all(np.isfinite(values)):
            raise ValueError(f"non-finite sampled label value: {path}")
        power_parts.append(values)
        delta_parts.append(delta_power(values))
        inventory.append(
            OrderedDict(
                stem=stem,
                label_file=label_file,
                sampled_rows=int(values.size),
            )
        )
    return np.concatenate(power_parts), np.concatenate(delta_parts), inventory


def rare_joint_bins(
    power: np.ndarray,
    delta: np.ndarray,
    power_edges: np.ndarray,
    delta_edges: np.ndarray,
    threshold: int,
) -> list[int]:
    _, _, joint = assign_joint_bins(power, delta, power_edges, delta_edges)
    unique, counts = np.unique(joint, return_counts=True)
    return [
        int(bin_id)
        for bin_id, count in zip(unique, counts)
        if int(count) <= max(0, threshold)
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate the current power/delta-power weight recommendation "
            "from already sampled Step 1 labels"
        )
    )
    parser.add_argument("--tw-manifest", required=True)
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--power-bins", type=int, default=12)
    parser.add_argument("--delta-bins", type=int, default=12)
    parser.add_argument("--rare-threshold", type=int, default=32)
    parser.add_argument("--weight-exponent", type=float, default=0.5)
    parser.add_argument("--weight-min", type=float, default=0.25)
    parser.add_argument("--weight-max", type=float, default=8.0)
    args = parser.parse_args()

    if args.power_bins < 1 or args.delta_bins < 1:
        parser.error("bin counts must be positive")
    if not 0.0 <= args.weight_exponent <= 1.0:
        parser.error("--weight-exponent must be in [0, 1]")
    if args.weight_min <= 0 or args.weight_max < args.weight_min:
        parser.error("require 0 < --weight-min <= --weight-max")

    manifest_path = Path(args.tw_manifest).resolve()
    output_file = Path(args.output_file).expanduser().resolve()
    manifest, combo_key, labels = load_sampled_labels(manifest_path)
    train_power, train_delta, train_inventory = load_split(labels["train"])
    test_power, test_delta, test_inventory = load_split(labels["test"])

    power_edges = quantile_edges(train_power, args.power_bins)
    delta_edges = quantile_edges(train_delta, args.delta_bins)
    _, _, train_joint = assign_joint_bins(
        train_power, train_delta, power_edges, delta_edges
    )
    joint_count = (len(power_edges) - 1) * (len(delta_edges) - 1)
    joint_weights = recommend_joint_weights(
        train_joint,
        joint_count,
        args.weight_exponent,
        args.weight_min,
        args.weight_max,
    )
    joint_weights[joint_weights == 0] = 1.0

    policy = OrderedDict(
        method="sampled_joint_inverse_frequency",
        power_edges=power_edges.tolist(),
        delta_power_edges=delta_edges.tolist(),
        joint_bin_weights=joint_weights.tolist(),
        exponent=args.weight_exponent,
        min_weight=args.weight_min,
        max_weight=args.weight_max,
        normalization="sample-row mean equals one",
    )
    train_selected = np.arange(train_power.size, dtype=np.int64)
    test_selected = np.arange(test_power.size, dtype=np.int64)
    report = OrderedDict(
        schema_version=1,
        generated_from="already_sampled_timewindow_labels",
        timewindow_manifest=str(manifest_path),
        sampled=bool(manifest.get("sampled", True)),
        label_source_combo=combo_key,
        train=distribution_report(
            train_power,
            train_delta,
            train_selected,
            power_edges,
            delta_edges,
            rare_joint_bins(
                train_power, train_delta, power_edges, delta_edges,
                args.rare_threshold,
            ),
            joint_weights,
        ),
        test=distribution_report(
            test_power,
            test_delta,
            test_selected,
            power_edges,
            delta_edges,
            rare_joint_bins(
                test_power, test_delta, power_edges, delta_edges,
                args.rare_threshold,
            ),
            joint_weights,
        ),
        weight_recommendation=policy,
        sampled_label_inventory=OrderedDict(
            train=train_inventory, test=test_inventory
        ),
    )

    output_file.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_file.with_name(
        f".{output_file.name}.{os.getpid()}.tmp"
    )
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(
            report,
            handle,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        handle.write("\n")
    os.replace(temporary, output_file)
    print(f"weight recommendation: {output_file}")


if __name__ == "__main__":
    main()
