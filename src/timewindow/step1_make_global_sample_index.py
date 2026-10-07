#!/usr/bin/env python3
"""Build global 2-D (power, delta-power) sampling indices and row weights."""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from collections import OrderedDict, defaultdict

import numpy as np

from utils.manifest_stems import write_stem_lists
from utils.sampling_policy import (
    assign_joint_bins,
    choose_2d_sample_indices,
    delta_power,
    distribution_report,
    equal_width_edges,
    recommend_joint_weights,
    row_weights_from_policy,
)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def discover_stems(level_dir: str, tog_dir: str, label_dir: str, stage: str):
    single_pat = re.compile(rf"^(.+)\.{re.escape(stage)}\.tog\.csr\.npz$")
    shard_pat = re.compile(rf"^(.+)\.{re.escape(stage)}\.row(\d+)\.tog\.csr\.npz$")
    merge_prefix_pat = re.compile(r"^m\d{3}_")
    stem_files = defaultdict(lambda: {"level": [], "tog": []})
    for dir_key, scan_dir in (("level", level_dir), ("tog", tog_dir)):
        if not os.path.isdir(scan_dir):
            continue
        for name in os.listdir(scan_dir):
            match = single_pat.match(name) or shard_pat.match(name)
            if match:
                stem_files[match.group(1)][dir_key].append(os.path.join(scan_dir, name))

    def find_label(stem: str):
        candidates = [stem, merge_prefix_pat.sub("", stem)]
        for candidate in dict.fromkeys(candidates):
            path = os.path.join(label_dir, f"{candidate}.npy")
            if os.path.exists(path):
                return os.path.abspath(path)
        return None

    labels = OrderedDict()
    for stem in sorted(stem_files):
        label = find_label(stem)
        if stem_files[stem]["tog"] and label:
            labels[stem] = label
    return list(labels), labels


def split_flatten(stems, powers, deltas):
    spans, power_parts, delta_parts, cursor = OrderedDict(), [], [], 0
    for stem in stems:
        spans[stem] = (cursor, cursor + len(powers[stem]))
        power_parts.append(powers[stem])
        delta_parts.append(deltas[stem])
        cursor += len(powers[stem])
    empty = np.array([], dtype=np.float64)
    flat_power = np.concatenate(power_parts) if power_parts else empty
    flat_delta = np.concatenate(delta_parts) if delta_parts else empty
    return spans, flat_power, flat_delta


def ensure_minimum_per_case(selected, spans, minimum: int, seed: int):
    if minimum <= 0:
        return np.sort(selected)
    rng = np.random.default_rng(seed)
    chosen = set(int(i) for i in selected)
    for start, end in spans.values():
        present = sum(start <= i < end for i in chosen)
        need = min(minimum, end - start) - present
        if need > 0:
            pool = np.asarray([i for i in range(start, end) if i not in chosen], dtype=np.int64)
            chosen.update(int(i) for i in rng.choice(pool, size=min(need, len(pool)), replace=False))
    return np.asarray(sorted(chosen), dtype=np.int64)


def main() -> None:
    parser = argparse.ArgumentParser(description="Global two-dimensional power sampling index builder")
    parser.add_argument("--level-dir", required=True)
    parser.add_argument("--tog-dir", required=True)
    parser.add_argument("--label-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--offsets", default="12,11,10,9,8,7,6")
    parser.add_argument("--boundary-discard", type=int, default=None)
    parser.add_argument("--test-count", type=int, default=1)
    parser.add_argument("--stage", default="after_one")
    parser.add_argument("--sample-rows-total", type=int, default=0)
    parser.add_argument("--sample-rows-per-file", type=int, default=0)
    parser.add_argument("--sample-power-bins", type=int, default=12,
                        help="Number of equal-width power-value bins")
    parser.add_argument("--sample-delta-bins", type=int, default=12,
                        help="Number of equal-width delta-power-value bins")
    parser.add_argument("--sample-bins", type=int, default=None, help="Deprecated fallback applied to both axes")
    parser.add_argument("--sample-all-threshold", type=int, default=32,
                        help="Keep every row when a joint bucket has at most this many rows")
    parser.add_argument("--sample-min-per-bin", type=int, default=1,
                        help="Compatibility option; sparse buckets are retained in full")
    parser.add_argument("--sample-min-per-case", type=int, default=1)
    parser.add_argument("--sample-test-ratio", type=float, default=-1.0)
    parser.add_argument("--weight-exponent", type=float, default=0.5)
    parser.add_argument("--weight-min", type=float, default=0.25)
    parser.add_argument("--weight-max", type=float, default=8.0)
    parser.add_argument("--sample-seed", type=int, default=108)
    args = parser.parse_args()

    offsets = [int(value) for value in args.offsets.split(",") if value.strip()]
    discard = args.boundary_discard if args.boundary_discard is not None else max(offsets)
    os.makedirs(args.output_dir, exist_ok=True)
    index_dir = os.path.join(args.output_dir, "indices")
    os.makedirs(index_dir, exist_ok=True)
    stems, label_map = discover_stems(args.level_dir, args.tog_dir, args.label_dir, args.stage)
    if len(stems) <= args.test_count:
        raise SystemExit(f"ERROR: Need > {args.test_count} valid stems, got {len(stems)}")
    train_stems, test_stems = stems[:-args.test_count], stems[-args.test_count:]

    powers, deltas = OrderedDict(), OrderedDict()
    for stem in stems:
        powers[stem] = np.load(label_map[stem]).reshape(-1).astype(np.float64)[discard:]
        deltas[stem] = delta_power(powers[stem])
    train_spans, train_power, train_delta = split_flatten(train_stems, powers, deltas)
    test_spans, test_power, test_delta = split_flatten(test_stems, powers, deltas)
    power_bins = args.sample_bins or args.sample_power_bins
    delta_bins = args.sample_bins or args.sample_delta_bins
    # Use fixed numeric ranges so a bin has the same physical meaning
    # regardless of how many training rows happen to occupy that range.
    power_edges = equal_width_edges(train_power, power_bins)
    delta_edges = equal_width_edges(train_delta, delta_bins)

    total_available = len(train_power) + len(test_power)
    requested_total = int(args.sample_rows_total)
    if requested_total <= 0 and args.sample_rows_per_file > 0:
        requested_total = args.sample_rows_per_file * len(stems)
    sampled = 0 < requested_total < total_available
    ratio = args.sample_test_ratio if args.sample_test_ratio >= 0 else len(test_stems) / len(stems)
    test_requested = int(round(requested_total * ratio)) if sampled else 0
    train_requested = requested_total - test_requested if sampled else 0

    train_selected, train_meta = choose_2d_sample_indices(
        train_power, train_delta, train_requested, power_edges, delta_edges,
        args.sample_all_threshold, args.sample_seed,
    )
    test_selected, test_meta = choose_2d_sample_indices(
        test_power, test_delta, test_requested, power_edges, delta_edges,
        args.sample_all_threshold, args.sample_seed + 7919,
    )
    train_selected = ensure_minimum_per_case(
        train_selected, train_spans, args.sample_min_per_case, args.sample_seed + 17
    )
    test_selected = ensure_minimum_per_case(
        test_selected, test_spans, args.sample_min_per_case, args.sample_seed + 7936
    )

    _, _, selected_train_joint = assign_joint_bins(
        train_power[train_selected], train_delta[train_selected], power_edges, delta_edges
    )
    joint_count = (len(power_edges) - 1) * (len(delta_edges) - 1)
    joint_weights = recommend_joint_weights(
        selected_train_joint, joint_count, args.weight_exponent, args.weight_min, args.weight_max
    )
    joint_weights[joint_weights == 0] = 1.0
    policy = OrderedDict([
        ("method", "sampled_joint_inverse_frequency"),
        ("binning", "equal_width_value_range"),
        ("power_edges", power_edges.tolist()),
        ("delta_power_edges", delta_edges.tolist()),
        ("joint_bin_weights", joint_weights.tolist()),
        ("exponent", args.weight_exponent),
        ("min_weight", args.weight_min),
        ("max_weight", args.weight_max),
        ("normalization", "sample-row mean equals one"),
    ])

    selected_by_stem = OrderedDict()
    for split, split_stems, spans, selected in (
        ("train", train_stems, train_spans, train_selected),
        ("test", test_stems, test_spans, test_selected),
    ):
        for stem in split_stems:
            start, end = spans[stem]
            local = selected[(selected >= start) & (selected < end)] - start
            index_path = os.path.abspath(os.path.join(index_dir, f"{stem}.sample_idx.npy"))
            weight_path = os.path.abspath(os.path.join(index_dir, f"{stem}.sample_weight.npy"))
            np.save(index_path, local)
            np.save(weight_path, row_weights_from_policy(powers[stem][local], deltas[stem][local], policy))
            selected_by_stem[stem] = OrderedDict([
                ("split", split),
                ("label_file", label_map[stem]),
                ("index_file", index_path),
                ("weight_file", weight_path),
                ("available_rows", len(powers[stem])),
                ("sampled_rows", len(local)),
            ])
            log(f"{stem} [{split}]: {len(local)}/{len(powers[stem])} rows")

    report = OrderedDict([
        ("train", distribution_report(
            train_power, train_delta, train_selected, power_edges, delta_edges,
            train_meta["rare_joint_bins"], joint_weights,
        )),
        ("test", distribution_report(
            test_power, test_delta, test_selected, power_edges, delta_edges,
            test_meta["rare_joint_bins"], joint_weights,
        )),
        ("weight_recommendation", policy),
    ])
    report_path = os.path.abspath(os.path.join(
        args.output_dir, "sampling_distribution_and_weight_recommendation.json"
    ))
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False, allow_nan=False)

    manifest = OrderedDict([
        ("stage", args.stage),
        ("boundary_discard", discard),
        ("test_count", args.test_count),
        ("sampled", sampled),
        ("sample_scope", "split_global_power_delta_2d"),
        ("sample_binning", "equal_width_value_range"),
        ("sample_rows_total_requested", requested_total),
        ("sample_rows_total_actual", int(len(train_selected) + len(test_selected))),
        ("sample_rows_train_actual", int(len(train_selected))),
        ("sample_rows_test_actual", int(len(test_selected))),
        ("available_rows_total", int(total_available)),
        ("sample_power_bins", len(power_edges) - 1),
        ("sample_delta_bins", len(delta_edges) - 1),
        ("sample_all_threshold", args.sample_all_threshold),
        ("sample_seed", args.sample_seed),
        ("train_stems", train_stems),
        ("test_stems", test_stems),
        ("weight_policy", policy),
        ("distribution_report", report_path),
        ("stems", selected_by_stem),
    ])
    manifest_path = os.path.join(args.output_dir, "sample_index_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False, allow_nan=False)
    write_stem_lists(args.output_dir, train_stems, test_stems)
    log(f"Done: {manifest_path}; distribution/weight report: {report_path}")


if __name__ == "__main__":
    main()
