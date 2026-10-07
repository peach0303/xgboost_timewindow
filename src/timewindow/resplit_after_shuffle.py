#!/usr/bin/env python3
"""Re-split existing Step1 rows after one deterministic global shuffle.

This is an experimental alternative to testcase-level splitting.  It consumes
Step1 artifacts without changing them and emits a new Step1-compatible
manifest whose tag is ``split_after_shuffle``.  Step2 and later stages can use
the emitted manifest unchanged.
"""
from __future__ import annotations

import argparse
import json
import os
from collections import OrderedDict
from pathlib import Path

import numpy as np
from scipy.sparse import load_npz, save_npz, vstack


TAG = "split_after_shuffle"


def read_filelist(path: Path) -> list[str]:
    with path.open(encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip() and not line.startswith("#")]


def write_filelist(path: Path, values: list[Path]) -> None:
    path.write_text("".join(f"{value.resolve()}\n" for value in values), encoding="utf-8")


def load_split(combo: dict, split: str, materialize: bool = True):
    directory = Path(combo["filelist_dir"])
    feature_files = read_filelist(directory / f"{split}_feature.f")
    label_files = read_filelist(directory / f"{split}_label.f")
    weight_list = directory / f"{split}_weight.f"
    weight_files = read_filelist(weight_list) if weight_list.exists() else []
    if len(feature_files) != len(label_files):
        raise ValueError(f"{directory}: {split} feature/label file count mismatch")
    if weight_files and len(weight_files) != len(label_files):
        raise ValueError(f"{directory}: {split} weight file count mismatch")
    labels = [np.load(path).reshape(-1) for path in label_files]
    matrices = [load_npz(path).tocsr() for path in feature_files] if materialize else []
    weights = []
    if materialize:
        weights = (
            [np.load(path).reshape(-1) for path in weight_files]
            if weight_files else [np.ones(value.size, dtype=np.float64) for value in labels]
        )
        for feature, label, weight in zip(matrices, labels, weights):
            if feature.shape[0] != label.size or label.size != weight.size:
                raise ValueError(f"{directory}: {split} row mismatch")
    return matrices, labels, weights


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Globally shuffle all Step1 rows, then create a row-level train/test split"
    )
    parser.add_argument("--input-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--test-ratio", type=float, default=None,
        help="New test fraction; default preserves the original test-row fraction",
    )
    parser.add_argument("--seed", type=int, default=108)
    parser.add_argument(
        "--storage-mode", choices=("index", "materialize"), default="materialize",
        help="index stores only row indices; materialize writes shuffled feature shards",
    )
    parser.add_argument("--index-row-batch-size", type=int, default=200_000)
    parser.add_argument("--index-row-batch-size-tog", type=int, default=50_000)
    parser.add_argument("--index-row-batch-size-level", type=int, default=5_000)
    parser.add_argument("--compress", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    for option, value in (
        ("--index-row-batch-size", args.index_row_batch_size),
        ("--index-row-batch-size-tog", args.index_row_batch_size_tog),
        ("--index-row-batch-size-level", args.index_row_batch_size_level),
    ):
        if value <= 0:
            parser.error(f"{option} must be positive")

    output = Path(args.output_dir).resolve()
    output_manifest = output / "timewindow_manifest.json"
    if output_manifest.exists() and not args.force:
        print(f"reuse existing {TAG} manifest: {output_manifest}")
        return
    with open(args.input_manifest, encoding="utf-8") as handle:
        source = json.load(handle, object_pairs_hook=OrderedDict)
    combos = source.get("combinations", {})
    if not combos:
        raise SystemExit("ERROR: input manifest contains no combinations")

    output.mkdir(parents=True, exist_ok=True)
    permutation = None
    train_count = test_count = total_rows = None
    reference_labels = None
    result_combos = OrderedDict()
    train_stem = f"{TAG}_train"
    test_stem = f"{TAG}_test"

    for combo_key, combo in combos.items():
        split_data = [
            load_split(combo, split, materialize=args.storage_mode == "materialize")
            for split in ("train", "test")
        ]
        labels = np.concatenate(split_data[0][1] + split_data[1][1])

        if permutation is None:
            total_rows = int(labels.size)
            original_test_rows = int(sum(value.size for value in split_data[1][1]))
            ratio = args.test_ratio
            if ratio is None:
                ratio = original_test_rows / total_rows
            if not 0.0 < ratio < 1.0:
                raise ValueError(f"test ratio must be in (0, 1), got {ratio}")
            test_count = max(1, min(total_rows - 1, int(round(total_rows * ratio))))
            train_count = total_rows - test_count
            permutation = np.random.default_rng(args.seed).permutation(total_rows)
            reference_labels = labels.copy()
        elif labels.size != total_rows or not np.array_equal(labels, reference_labels):
            raise ValueError(f"{combo_key}: labels/order differ from the first combination")

        item = OrderedDict(combo)
        if args.storage_mode == "materialize":
            matrices = split_data[0][0] + split_data[1][0]
            weights = np.concatenate(split_data[0][2] + split_data[1][2])
            matrix = vstack(matrices, format="csr")
            if matrix.shape[0] != labels.size:
                raise ValueError(f"{combo_key}: combined row mismatch")
            shuffled_matrix = matrix[permutation]
            shuffled_labels = labels[permutation]
            shuffled_weights = weights[permutation]
            combo_output = output / combo_key
            combo_output.mkdir(parents=True, exist_ok=True)
            train_feature = combo_output / f"{train_stem}.feature.npz"
            test_feature = combo_output / f"{test_stem}.feature.npz"
            train_label = combo_output / f"{train_stem}.label.npy"
            test_label = combo_output / f"{test_stem}.label.npy"
            train_weight = combo_output / f"{train_stem}.weight.npy"
            test_weight = combo_output / f"{test_stem}.weight.npy"
            save_npz(train_feature, shuffled_matrix[:train_count], compressed=args.compress)
            save_npz(test_feature, shuffled_matrix[train_count:], compressed=args.compress)
            np.save(train_label, shuffled_labels[:train_count])
            np.save(test_label, shuffled_labels[train_count:])
            np.save(train_weight, shuffled_weights[:train_count])
            np.save(test_weight, shuffled_weights[train_count:])
            for split, feature, label, weight in (
                ("train", train_feature, train_label, train_weight),
                ("test", test_feature, test_label, test_weight),
            ):
                write_filelist(combo_output / f"{split}_feature.f", [feature])
                write_filelist(combo_output / f"{split}_label.f", [label])
                write_filelist(combo_output / f"{split}_weight.f", [weight])
            item["filelist_dir"] = str(combo_output)
            item["train_files"] = 1
            item["test_files"] = 1
            item["valid_rows_per_file"] = None
        result_combos[combo_key] = item

    split_indices = None
    if args.storage_mode == "index":
        index_dtype = np.uint32 if total_rows <= np.iinfo(np.uint32).max else np.uint64
        train_indices_file = output / "train_indices.npy"
        test_indices_file = output / "test_indices.npy"
        np.save(train_indices_file, permutation[:train_count].astype(index_dtype))
        np.save(test_indices_file, permutation[train_count:].astype(index_dtype))
        split_indices = OrderedDict([
            ("train", str(train_indices_file)), ("test", str(test_indices_file)),
        ])

    result = OrderedDict((key, value) for key, value in source.items() if key != "combinations")
    result["train_stems"] = [train_stem]
    result["test_stems"] = [test_stem]
    result["test_count"] = 1
    result["split_strategy"] = TAG
    result["experiment_tag"] = TAG
    result["split_seed"] = args.seed
    result["split_storage_mode"] = args.storage_mode
    if split_indices is not None:
        result["split_indices"] = split_indices
        result["index_row_batch_size"] = args.index_row_batch_size
        result["index_row_batch_sizes"] = OrderedDict([
            ("tog", args.index_row_batch_size_tog),
            ("level", args.index_row_batch_size_level),
        ])
    result["split_train_rows"] = train_count
    result["split_test_rows"] = test_count
    result["split_source_manifest"] = os.path.abspath(args.input_manifest)
    result["supports_case_level_full_evaluation"] = False
    result["combinations"] = result_combos
    with output_manifest.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
    (output / "complete").touch()
    print(
        f"wrote {output_manifest}: train={train_count}, test={test_count}, "
        f"seed={args.seed}, storage={args.storage_mode}, tag={TAG}"
    )


if __name__ == "__main__":
    main()
