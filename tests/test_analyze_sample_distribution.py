from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "src" / "timewindow" / "analyze_sample_distribution.py"


def make_dataset(tmp_path: Path) -> Path:
    combo = tmp_path / "tog_offset2"
    combo.mkdir()
    train_features, train_labels, test_features, test_labels = [], [], [], []
    values_by_split = {
        "train": [np.r_[np.ones(90), np.linspace(5, 20, 10)]],
        "test": [np.r_[np.ones(18), [8.0, 16.0]]],
    }
    for split, groups in values_by_split.items():
        for index, values in enumerate(groups):
            stem = f"{split}{index}"
            feature = combo / f"{stem}.feature.npz"
            label = combo / f"{stem}.label.npy"
            feature.touch()
            np.save(label, values)
            (train_features if split == "train" else test_features).append(str(feature))
            (train_labels if split == "train" else test_labels).append(str(label))
    for split, features, labels in (
        ("train", train_features, train_labels), ("test", test_features, test_labels)
    ):
        (combo / f"{split}_feature.f").write_text("\n".join(features) + "\n")
        (combo / f"{split}_label.f").write_text("\n".join(labels) + "\n")
    manifest = tmp_path / "timewindow_manifest.json"
    manifest.write_text(json.dumps({
        "train_stems": ["train0"], "test_stems": ["test0"],
        "combinations": {"tog_offset2": {
            "data_type": "tog", "offset": 2, "filelist_dir": str(combo),
        }},
    }))
    return manifest


def test_distribution_scan_and_optional_weight_output(tmp_path: Path) -> None:
    manifest = make_dataset(tmp_path)
    output = tmp_path / "audit"
    command = [
        sys.executable, str(SCRIPT), "--tw-manifest", str(manifest),
        "--output-dir", str(output), "--power-bins", "4", "--delta-bins", "4",
    ]
    subprocess.run(command, check=True)
    report = json.loads(
        (output / "sample_distribution_and_weight_recommendation.json").read_text()
    )
    assert report["train"]["rows"] == 100
    assert report["weight_recommendation"]["normalization"] == "training-row mean equals one"
    assert (output / "joint_bin_weight_recommendation.csv").exists()
    assert not (tmp_path / "tog_offset2" / "train0.weight.npy").exists()

    subprocess.run(command + ["--write-weights"], check=True)
    train_weight = np.load(tmp_path / "tog_offset2" / "train0.weight.npy")
    test_weight = np.load(tmp_path / "tog_offset2" / "test0.weight.npy")
    assert train_weight.shape == (100,)
    assert test_weight.shape == (20,)
    assert np.all(np.isfinite(train_weight)) and np.all(train_weight > 0)
    assert np.isclose(train_weight.mean(), 1.0)
