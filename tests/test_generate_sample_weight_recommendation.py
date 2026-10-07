from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT / "src" / "timewindow" / "generate_sample_weight_recommendation.py"
)


def make_sampled_manifest(tmp_path: Path) -> tuple[Path, np.ndarray]:
    combo_dir = tmp_path / "tog_offset2"
    combo_dir.mkdir()
    train_values = np.r_[np.ones(80), np.linspace(2.0, 10.0, 20)]
    test_values = np.r_[np.ones(16), np.linspace(2.0, 8.0, 4)]

    filelists = {}
    for split, values in (("train", train_values), ("test", test_values)):
        stem = f"{split}_sampled"
        feature = combo_dir / f"{stem}.feature.npz"
        label = combo_dir / f"{stem}.label.npy"
        feature.touch()
        np.save(label, values)
        filelists[split] = (stem, feature, label)
        (combo_dir / f"{split}_feature.f").write_text(
            f"{feature}\n", encoding="utf-8"
        )
        (combo_dir / f"{split}_label.f").write_text(
            f"{label}\n", encoding="utf-8"
        )

    manifest = tmp_path / "timewindow_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "sampled": True,
                "train_stems": [filelists["train"][0]],
                "test_stems": [filelists["test"][0]],
                "combinations": {
                    "tog_offset2": {
                        "data_type": "tog",
                        "offset": 2,
                        "filelist_dir": str(combo_dir),
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    return manifest, train_values


def test_generate_report_to_explicit_output_file(tmp_path: Path) -> None:
    manifest, train_values = make_sampled_manifest(tmp_path)
    output = tmp_path / "custom" / "weight_advice.json"
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--tw-manifest",
            str(manifest),
            "--output-file",
            str(output),
            "--power-bins",
            "4",
            "--delta-bins",
            "4",
        ],
        check=True,
    )

    report = json.loads(output.read_text(encoding="utf-8"))
    policy = report["weight_recommendation"]
    power_edges = np.asarray(policy["power_edges"])
    delta_edges = np.asarray(policy["delta_power_edges"])
    delta = np.zeros_like(train_values)
    delta[1:] = np.diff(train_values)
    power_bin = np.searchsorted(power_edges[1:-1], train_values, side="right")
    delta_bin = np.searchsorted(delta_edges[1:-1], delta, side="right")
    joint = power_bin * (len(delta_edges) - 1) + delta_bin
    weights = np.asarray(policy["joint_bin_weights"])[joint]

    assert report["generated_from"] == "already_sampled_timewindow_labels"
    assert report["label_source_combo"] == "tog_offset2"
    assert report["train"]["joint"]["counts_after"]
    assert policy["normalization"] == "sample-row mean equals one"
    assert np.all(np.isfinite(weights)) and np.all(weights > 0)
    assert np.isclose(weights.mean(), 1.0)


def test_pipeline_checks_and_backfills_missing_report() -> None:
    pipeline = (ROOT / "cmd" / "run_timewindow_pipeline.csh").read_text()
    assert "WEIGHT_RECOMMENDATION_FILE" in pipeline
    assert "generate_sample_weight_recommendation.py" in pipeline
    assert 'if ( -f "${WEIGHT_RECOMMENDATION_FILE}" ) then' in pipeline
    assert (
        "automatically rerun failed Step 1 weight recommendation only"
        in pipeline
    )
    assert "-s \\!:1" in pipeline
