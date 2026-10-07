"""Load a virtual train/test split backed by existing Step1 shards."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from scipy.sparse import load_npz, vstack


def read_filelist(path: str | Path) -> list[str]:
    path = Path(path)
    with path.open(encoding="utf-8") as handle:
        values = [line.strip() for line in handle if line.strip() and not line.startswith("#")]
    return [value if os.path.isabs(value) else str((path.parent / value).resolve()) for value in values]


def _source_files(combo: dict) -> tuple[list[str], list[str], list[str]]:
    directory = Path(combo["filelist_dir"])
    features: list[str] = []
    labels: list[str] = []
    weights: list[str] = []
    has_all_weights = True
    for source_split in ("train", "test"):
        split_features = read_filelist(directory / f"{source_split}_feature.f")
        split_labels = read_filelist(directory / f"{source_split}_label.f")
        weight_list = directory / f"{source_split}_weight.f"
        split_weights = read_filelist(weight_list) if weight_list.exists() else []
        if len(split_features) != len(split_labels):
            raise ValueError(f"{directory}: {source_split} feature/label count mismatch")
        if split_weights and len(split_weights) != len(split_labels):
            raise ValueError(f"{directory}: {source_split} weight/label count mismatch")
        if not split_weights:
            has_all_weights = False
        features.extend(split_features)
        labels.extend(split_labels)
        weights.extend(split_weights)
    return features, labels, weights if has_all_weights else []


def load_indexed_combo_split(
    combo: dict,
    indices_file: str | Path,
    columns: np.ndarray | None = None,
    row_batch_size: int = 200_000,
):
    """Materialize an indexed split from bounded row batches."""
    batches = list(iter_indexed_combo_batches(
        combo, indices_file, row_batch_size=row_batch_size, columns=columns
    ))
    if not batches:
        raise ValueError(f"indexed split has no rows: {indices_file}")
    return (
        vstack([item[0] for item in batches], format="csr"),
        np.concatenate([item[1] for item in batches]),
        np.concatenate([item[2] for item in batches]),
    )


def iter_indexed_combo_batches(
    combo: dict,
    indices_file: str | Path,
    row_batch_size: int,
    columns: np.ndarray | None = None,
):
    """Yield indexed rows in split order without stacking all source features."""
    if row_batch_size <= 0:
        raise ValueError(f"row_batch_size must be positive, got {row_batch_size}")
    feature_files, label_files, weight_files = _source_files(combo)
    if not feature_files:
        raise ValueError(f"{combo['filelist_dir']}: no source feature shards")
    label_arrays = [np.load(path, mmap_mode="r").reshape(-1) for path in label_files]
    if weight_files:
        weight_arrays = [np.load(path, mmap_mode="r").reshape(-1) for path in weight_files]
    else:
        weight_arrays = []
    row_counts = np.asarray([values.size for values in label_arrays], dtype=np.int64)
    offsets = np.concatenate(([0], np.cumsum(row_counts)))
    indices = np.load(indices_file).reshape(-1).astype(np.int64, copy=False)
    if indices.size and (indices.min() < 0 or indices.max() >= offsets[-1]):
        raise IndexError(f"split indices exceed source rows: {indices_file}")

    for batch_start in range(0, indices.size, row_batch_size):
        batch_indices = indices[batch_start:batch_start + row_batch_size]
        matrix_parts = []
        source_positions = []
        labels = np.empty(batch_indices.size, dtype=np.float64)
        weights = np.ones(batch_indices.size, dtype=np.float64)
        for shard_index, (feature_file, label_values) in enumerate(
            zip(feature_files, label_arrays)
        ):
            positions = np.flatnonzero(
                (batch_indices >= offsets[shard_index])
                & (batch_indices < offsets[shard_index + 1])
            )
            if positions.size == 0:
                continue
            local_rows = batch_indices[positions] - offsets[shard_index]
            matrix = load_npz(feature_file).tocsr()
            if matrix.shape[0] != label_values.size:
                raise ValueError(f"feature/label row mismatch in {feature_file}")
            selected = matrix[local_rows]
            if columns is not None:
                selected = selected[:, columns]
            matrix_parts.append(selected.tocsr())
            source_positions.append(positions)
            labels[positions] = label_values[local_rows]
            if weight_arrays:
                weight_values = weight_arrays[shard_index]
                if weight_values.size != label_values.size:
                    raise ValueError(f"weight/label row mismatch in {feature_file}")
                weights[positions] = weight_values[local_rows]
        if not matrix_parts:
            raise ValueError(f"indexed batch resolves to no source rows: {indices_file}")
        grouped = vstack(matrix_parts, format="csr")
        grouped_positions = np.concatenate(source_positions)
        restore_order = np.argsort(grouped_positions)
        yield grouped[restore_order].tocsr(), labels, weights


def load_indexed_combo_labels(
    combo: dict,
    indices_file: str | Path,
) -> tuple[np.ndarray, np.ndarray]:
    """Load only labels/weights for an indexed split without reading features."""
    _feature_files, label_files, weight_files = _source_files(combo)
    labels = np.concatenate([np.load(path).reshape(-1) for path in label_files])
    if weight_files:
        weights = np.concatenate([np.load(path).reshape(-1) for path in weight_files])
        if weights.size != labels.size:
            raise ValueError(f"{combo['filelist_dir']}: weight/label row mismatch")
    else:
        weights = np.ones(labels.size, dtype=np.float64)
    indices = np.load(indices_file).reshape(-1).astype(np.int64, copy=False)
    if indices.size and (indices.min() < 0 or indices.max() >= labels.size):
        raise IndexError(f"split indices exceed source labels: {indices_file}")
    return labels[indices], weights[indices]


def indexed_split_paths(manifest: dict) -> dict[str, str] | None:
    if manifest.get("split_storage_mode") != "index":
        return None
    paths = manifest.get("split_indices", {})
    if not paths.get("train") or not paths.get("test"):
        raise ValueError("index split manifest misses train/test split_indices")
    return {split: os.path.abspath(path) for split, path in paths.items()}


def index_row_batch_size(
    manifest: dict,
    combo: dict | None = None,
    default: int = 200_000,
) -> int:
    value = manifest.get("index_row_batch_size", default)
    if combo is not None:
        data_type = str(combo.get("data_type", ""))
        value = manifest.get("index_row_batch_sizes", {}).get(data_type, value)
    value = int(value)
    if value <= 0:
        raise ValueError(f"index_row_batch_size must be positive, got {value}")
    return value
