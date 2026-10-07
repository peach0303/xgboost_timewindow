#!/usr/bin/env python3
"""Step 3 (hybrid): independently concatenate MCP and XGBoost proxies.

Both Step2 selectors write the same manifest schema.  This script keeps their
feature spaces separate so Step4 can rank MCP proxies with Ridge coefficients
and XGBoost proxies with XGBoost gain before taking their union.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from collections import OrderedDict

import numpy as np
import xgboost as xgb
from utils.training_policy import shuffle_xyw
from utils.indexed_split import (
    index_row_batch_size,
    indexed_split_paths,
    load_indexed_combo_split,
)
from scipy.sparse import hstack, load_npz, vstack


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def read_filelist(path: str) -> list[str]:
    with open(path, "r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip() and not line.startswith("#")]


def load_indices(path: str) -> np.ndarray:
    with open(path, "r", encoding="utf-8") as handle:
        raw = handle.read().replace(",", " ")
    return np.asarray([int(item) for item in raw.split()], dtype=np.int64)


def resolve_path(path: str, manifest_dir: str) -> str:
    if os.path.isabs(path):
        return path
    cwd_path = os.path.abspath(path)
    if os.path.exists(cwd_path):
        return cwd_path
    return os.path.abspath(os.path.join(manifest_dir, path))


def feature_stem(path: str) -> str:
    suffix = ".feature.npz"
    name = os.path.basename(path)
    if not name.endswith(suffix):
        raise ValueError(f"unexpected feature path: {path}")
    return name[:-len(suffix)]


def split_index(combo: dict, split: str) -> OrderedDict:
    directory = combo["filelist_dir"]
    features = read_filelist(os.path.join(directory, f"{split}_feature.f"))
    labels = read_filelist(os.path.join(directory, f"{split}_label.f"))
    if len(features) != len(labels):
        raise ValueError(f"{split} feature/label count mismatch in {directory}")
    result = OrderedDict()
    for feature_path, label_path in zip(features, labels):
        stem = feature_stem(feature_path)
        if stem in result:
            raise ValueError(f"duplicate stem {stem!r} in {directory}")
        result[stem] = (feature_path, label_path)
    return result


def safe_name(name: str) -> str:
    return name.replace("[", "_").replace("]", "_").replace("<", "_")


def build_selector_dataset(
    selector: str,
    tw_manifest: OrderedDict,
    proxy_manifest_path: str,
    signal_names: list[str],
) -> tuple[object, np.ndarray, object, np.ndarray, list[str], list[OrderedDict]]:
    with open(proxy_manifest_path, "r", encoding="utf-8") as handle:
        proxy_manifest = json.load(handle, object_pairs_hook=OrderedDict)
    proxy_dir = os.path.dirname(os.path.abspath(proxy_manifest_path))
    combos = tw_manifest["combinations"]
    proxy_combos = proxy_manifest["combinations"]

    missing = [key for key in combos if key not in proxy_combos]
    if missing:
        raise KeyError(f"{selector} manifest misses combos: {missing}")

    selections = OrderedDict()
    feature_names: list[str] = []
    blocks: list[OrderedDict] = []
    start_col = 0
    for combo_key, combo in combos.items():
        selected_file = resolve_path(
            proxy_combos[combo_key]["selected_indices_file"], proxy_dir
        )
        indices = load_indices(selected_file)
        if indices.size != np.unique(indices).size:
            raise ValueError(f"duplicate selected indices in {selector}/{combo_key}")
        dtype = combo["data_type"]
        offset = combo["offset"]
        names = [
            safe_name(
                f"{dtype}_o{offset}_{signal_names[index]}"
                if index < len(signal_names)
                else f"{dtype}_o{offset}_idx{index}"
            )
            for index in indices
        ]
        selections[combo_key] = indices
        feature_names.extend(names)
        blocks.append(OrderedDict([
            ("combo_key", combo_key), ("data_type", dtype), ("offset", offset),
            ("n_selected", int(indices.size)), ("start_col", start_col),
            ("end_col", start_col + int(indices.size)),
        ]))
        start_col += int(indices.size)
        log(f"{selector}/{combo_key}: {indices.size} proxies")

    if len(feature_names) != len(set(feature_names)):
        raise ValueError(
            f"{selector} contains duplicate proxy names; combo-qualified names must be unique"
        )

    split_indices = indexed_split_paths(tw_manifest)
    if split_indices is not None:
        def build_indexed_split(split: str):
            row_parts = []
            reference_labels = None
            reference_weights = None
            for combo_key, combo in combos.items():
                matrix, labels, weights = load_indexed_combo_split(
                    combo, split_indices[split], columns=selections[combo_key],
                    row_batch_size=index_row_batch_size(tw_manifest, combo),
                )
                if reference_labels is None:
                    reference_labels = labels
                    reference_weights = weights
                elif not np.array_equal(reference_labels, labels):
                    raise ValueError(f"indexed labels differ for {split}/{combo_key}")
                elif not np.array_equal(reference_weights, weights):
                    raise ValueError(f"indexed weights differ for {split}/{combo_key}")
                row_parts.append(matrix)
            if not row_parts or reference_labels is None or reference_weights is None:
                raise ValueError(f"no indexed {split} data")
            return (
                hstack(row_parts, format="csr"), reference_labels, reference_weights,
            )

        train_x, train_y, train_w = build_indexed_split("train")
        test_x, test_y, test_w = build_indexed_split("test")
        return (
            train_x, train_y, train_w, test_x, test_y, test_w,
            feature_names, blocks,
        )

    indices_by_split = {
        split: {key: split_index(combo, split) for key, combo in combos.items()}
        for split in ("train", "test")
    }

    def build_split(split: str, stems: list[str]):
        matrix_parts = []
        label_parts = []
        weight_parts = []
        first_combo = next(iter(combos))
        for stem in stems:
            if stem not in indices_by_split[split][first_combo]:
                raise FileNotFoundError(f"missing {split} stem {stem!r} in {first_combo}")
            _, label_path = indices_by_split[split][first_combo][stem]
            labels = np.load(label_path).reshape(-1).astype(np.float32)
            row_parts = []
            for combo_key in combos:
                if stem not in indices_by_split[split][combo_key]:
                    raise FileNotFoundError(f"missing {split} stem {stem!r} in {combo_key}")
                feature_path, other_label_path = indices_by_split[split][combo_key][stem]
                other_rows = np.load(other_label_path, mmap_mode="r").size
                if other_rows != labels.size:
                    raise ValueError(f"label row mismatch for {split}/{stem}/{combo_key}")
                matrix = load_npz(feature_path).tocsr()
                selected = selections[combo_key]
                if selected.size and int(selected.max()) >= matrix.shape[1]:
                    raise IndexError(
                        f"{selector}/{combo_key} selected index {selected.max()} "
                        f">= feature columns {matrix.shape[1]}"
                    )
                row_parts.append(matrix[:, selected].tocsr())
            joined = hstack(row_parts, format="csr")
            if joined.shape[0] != labels.size:
                raise ValueError(f"feature/label row mismatch for {split}/{stem}")
            matrix_parts.append(joined)
            label_parts.append(labels)
            weight_path = label_path.replace(".label.npy", ".weight.npy")
            weight_parts.append(np.load(weight_path).reshape(-1) if os.path.exists(weight_path) else np.ones(len(labels)))
        if not matrix_parts:
            raise ValueError(f"no {split} data")
        return vstack(matrix_parts, format="csr"), np.concatenate(label_parts), np.concatenate(weight_parts)

    train_x, train_y, train_w = build_split("train", tw_manifest["train_stems"])
    test_x, test_y, test_w = build_split("test", tw_manifest["test_stems"])
    return train_x, train_y, train_w, test_x, test_y, test_w, feature_names, blocks


def save_selector(
    output_dir: str,
    selector: str,
    train_x,
    train_y: np.ndarray,
    test_x,
    test_y: np.ndarray,
    train_weight: np.ndarray,
    test_weight: np.ndarray,
    feature_names: list[str],
    blocks: list[OrderedDict],
) -> OrderedDict:
    selector_dir = os.path.join(output_dir, selector)
    os.makedirs(selector_dir, exist_ok=True)
    train_buffer = os.path.join(selector_dir, "train.buffer")
    test_buffer = os.path.join(selector_dir, "test.buffer")
    xgb.DMatrix(train_x, label=train_y, weight=train_weight, feature_names=feature_names).save_binary(train_buffer)
    xgb.DMatrix(test_x, label=test_y, weight=test_weight, feature_names=feature_names).save_binary(test_buffer)
    names_file = os.path.join(selector_dir, "feature_names.txt")
    with open(names_file, "w", encoding="utf-8") as handle:
        handle.write("\n".join(feature_names) + "\n")
    meta = OrderedDict([
        ("selector", selector), ("train_buffer", os.path.abspath(train_buffer)),
        ("test_buffer", os.path.abspath(test_buffer)),
        ("feature_names_file", os.path.abspath(names_file)),
        ("train_shape", list(train_x.shape)), ("test_shape", list(test_x.shape)),
        ("blocks", blocks),
    ])
    with open(os.path.join(selector_dir, "concat_meta.json"), "w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2, ensure_ascii=False)
    log(f"saved {selector}: train={train_x.shape}, test={test_x.shape}")
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Step3 MCP/XGB proxy concatenation")
    parser.add_argument("--tw-manifest", required=True)
    parser.add_argument("--mcp-manifest")
    parser.add_argument("--xgb-manifest")
    parser.add_argument(
        "--selectors", default="mcp,xgb",
        help="Comma-separated selectors to concatenate: mcp,xgb or one selector",
    )
    parser.add_argument("--namefile", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--shuffle-seed", type=int, default=108)
    args = parser.parse_args()

    with open(args.tw_manifest, "r", encoding="utf-8") as handle:
        tw_manifest = json.load(handle, object_pairs_hook=OrderedDict)
    with open(args.namefile, "r", encoding="utf-8") as handle:
        signal_names = [line.strip() for line in handle]
    os.makedirs(args.output_dir, exist_ok=True)

    selectors = [value.strip() for value in args.selectors.split(",") if value.strip()]
    if not selectors or any(value not in {"mcp", "xgb"} for value in selectors):
        parser.error("--selectors must contain mcp and/or xgb")
    if len(selectors) != len(set(selectors)):
        parser.error("--selectors contains duplicates")
    manifests = {"mcp": args.mcp_manifest, "xgb": args.xgb_manifest}
    missing = [selector for selector in selectors if not manifests[selector]]
    if missing:
        parser.error(f"missing manifest for selectors: {missing}")

    output = OrderedDict()
    labels_by_selector = {}
    for selector in selectors:
        manifest = manifests[selector]
        data = build_selector_dataset(selector, tw_manifest, manifest, signal_names)
        train_x, train_y, train_w, test_x, test_y, test_w, feature_names, blocks = data
        train_x, train_y, train_w, _ = shuffle_xyw(
            train_x, train_y, train_w, args.shuffle_seed
        )
        labels_by_selector[selector] = (train_y, test_y)
        output[selector] = save_selector(
            args.output_dir, selector, train_x, train_y, test_x, test_y, train_w, test_w,
            feature_names, blocks,
        )
    reference = labels_by_selector[selectors[0]]
    for selector in selectors[1:]:
        if not np.array_equal(reference[0], labels_by_selector[selector][0]):
            raise ValueError(f"{selectors[0]} and {selector} train labels differ")
        if not np.array_equal(reference[1], labels_by_selector[selector][1]):
            raise ValueError(f"{selectors[0]} and {selector} test labels differ")

    hybrid_meta = OrderedDict([
        ("schema_version", 1), ("tw_manifest", os.path.abspath(args.tw_manifest)),
        (
            "mcp_manifest",
            os.path.abspath(args.mcp_manifest) if args.mcp_manifest else None,
        ),
        (
            "xgb_manifest",
            os.path.abspath(args.xgb_manifest) if args.xgb_manifest else None,
        ),
        ("selected_selectors", selectors), ("selectors", output),
    ])
    meta_path = os.path.join(args.output_dir, "hybrid_concat_meta.json")
    with open(meta_path, "w", encoding="utf-8") as handle:
        json.dump(hybrid_meta, handle, indent=2, ensure_ascii=False)
    log(f"done: {meta_path}")


if __name__ == "__main__":
    main()
