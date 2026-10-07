from __future__ import annotations

import sys
import json
import subprocess
from pathlib import Path

import numpy as np


TIMEWINDOW = Path(__file__).resolve().parents[1] / "src" / "timewindow"
sys.path.insert(0, str(TIMEWINDOW))

from utils.sampling_policy import (  # noqa: E402
    assign_joint_bins,
    choose_2d_sample_indices,
    equal_width_edges,
    recommend_joint_weights,
)
from utils.training_policy import configure_xgb_loss, shuffle_xyw  # noqa: E402


def test_sparse_joint_bins_are_retained_and_weights_are_normalized():
    power = np.r_[np.zeros(80), np.ones(18), 10.0, 20.0]
    delta = np.r_[np.zeros(98), 7.0, -9.0]
    power_edges = np.array([-1.0, 0.5, 2.0, 30.0])
    delta_edges = np.array([-10.0, -1.0, 1.0, 10.0])
    selected, metadata = choose_2d_sample_indices(
        power, delta, 20, power_edges, delta_edges, rare_threshold=2, seed=108
    )
    _, _, joint = assign_joint_bins(power, delta, power_edges, delta_edges)
    counts = np.bincount(joint, minlength=9)
    sparse_rows = np.flatnonzero(counts[joint] <= 2)
    assert set(sparse_rows).issubset(set(selected))
    assert metadata["rare_rows_retained"] >= 2

    selected_joint = joint[selected]
    table = recommend_joint_weights(selected_joint, 9, exponent=0.5)
    nonzero = table[selected_joint]
    assert np.isclose(nonzero.mean(), 1.0)
    assert table[selected_joint[-1]] >= table[selected_joint[0]]


def test_equal_width_edges_follow_numeric_values_not_quantiles():
    values = np.array([0.0] * 90 + [10.0])
    assert np.array_equal(equal_width_edges(values, 5), np.arange(0.0, 12.0, 2.0))

    constant_edges = equal_width_edges(np.ones(20) * 3.0, 5)
    assert len(constant_edges) == 2
    assert constant_edges[0] < 3.0 < constant_edges[1]


def test_shuffle_keeps_rows_labels_and_weights_aligned():
    x = np.arange(30).reshape(10, 3)
    y = np.arange(10)
    weight = y + 100
    sx, sy, sw, order = shuffle_xyw(x, y, weight, seed=7)
    assert np.array_equal(sx, x[order])
    assert np.array_equal(sy, y[order])
    assert np.array_equal(sw, weight[order])


def test_error_focus_objective_emphasizes_larger_individual_errors():
    class Matrix:
        def get_label(self):
            return np.array([1.0, 1.0], dtype=np.float32)

        def get_weight(self):
            return np.array([1.0, 2.0], dtype=np.float32)

    _, objective = configure_xgb_loss(
        {"objective": "reg:squarederror"}, "error_focus", focus_power=3.0
    )
    gradient, hessian = objective(np.array([1.5, 3.0]), Matrix())
    assert abs(gradient[1]) > abs(gradient[0])
    assert hessian[1] > hessian[0]


def test_global_sampler_writes_distribution_policy_indices_and_weights(tmp_path):
    level_dir, tog_dir, label_dir, output_dir = (
        tmp_path / "level", tmp_path / "tog", tmp_path / "label", tmp_path / "output"
    )
    for directory in (level_dir, tog_dir, label_dir):
        directory.mkdir()
    for case_index in range(3):
        stem = f"case{case_index}"
        (tog_dir / f"{stem}.after_one.tog.csr.npz").touch()
        baseline = np.linspace(1.0, 3.0, 60)
        baseline[-2:] = [10.0 + case_index, 20.0 + case_index]
        np.save(label_dir / f"{stem}.npy", baseline)

    subprocess.run([
        sys.executable, str(TIMEWINDOW / "step1_make_global_sample_index.py"),
        "--level-dir", str(level_dir), "--tog-dir", str(tog_dir),
        "--label-dir", str(label_dir), "--output-dir", str(output_dir),
        "--offsets", "2,1", "--test-count", "1", "--sample-rows-total", "45",
        "--sample-power-bins", "4", "--sample-delta-bins", "4",
        "--sample-all-threshold", "2",
    ], check=True)
    manifest = json.loads((output_dir / "sample_index_manifest.json").read_text())
    assert (output_dir / "train_stems.txt").read_text().splitlines() == manifest["train_stems"]
    assert (output_dir / "test_stems.txt").read_text().splitlines() == manifest["test_stems"]
    report = json.loads((output_dir / "sampling_distribution_and_weight_recommendation.json").read_text())
    assert manifest["sample_scope"] == "split_global_power_delta_2d"
    assert manifest["sample_binning"] == "equal_width_value_range"
    assert manifest["weight_policy"]["binning"] == "equal_width_value_range"
    assert report["train"]["joint"]["counts_before"]
    assert report["train"]["joint"]["counts_after"]
    for item in manifest["stems"].values():
        indices = np.load(item["index_file"])
        weights = np.load(item["weight_file"])
        assert len(indices) == len(weights)
        assert np.all(weights > 0)


def test_existing_sample_manifest_can_backfill_legacy_stem_lists(tmp_path):
    manifest_path = tmp_path / "sample_index_manifest.json"
    manifest_path.write_text(json.dumps({
        "train_stems": ["case0", "case1"],
        "test_stems": ["case2"],
    }))
    subprocess.run([
        sys.executable, str(TIMEWINDOW / "utils" / "manifest_stems.py"),
        "--manifest", str(manifest_path), "--output-dir", str(tmp_path),
    ], check=True)
    assert (tmp_path / "train_stems.txt").read_text() == "case0\ncase1\n"
    assert (tmp_path / "test_stems.txt").read_text() == "case2\n"

    pipeline = (TIMEWINDOW.parents[1] / "cmd" / "run_timewindow_pipeline.csh").read_text()
    assert "utils/manifest_stems.py" in pipeline
    assert "Step 1 stem lists missing; backfill" in pipeline
