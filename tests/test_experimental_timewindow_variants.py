from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xgboost as xgb
from scipy.sparse import csr_matrix, load_npz, save_npz


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "timewindow"


def run_script(name: str, *args) -> None:
    subprocess.run(
        [sys.executable, str(SRC / name), *(str(value) for value in args)],
        check=True,
    )


def write_combo(
    root: Path, key: str, offset: int, scale: float = 1.0,
    data_type: str = "tog",
) -> dict:
    directory = root / key
    directory.mkdir(parents=True)
    rows = {
        "train": ("case_a", np.array([1.0, 2.0, 3.0], dtype=np.float32)),
        "test": ("case_b", np.array([4.0, 5.0], dtype=np.float32)),
    }
    for split, (stem, labels) in rows.items():
        feature = directory / f"{stem}.feature.npz"
        label = directory / f"{stem}.label.npy"
        weight = directory / f"{stem}.weight.npy"
        values = np.column_stack((labels * scale, labels * scale + offset, labels * 0 + 7))
        save_npz(feature, csr_matrix(values))
        np.save(label, labels)
        np.save(weight, np.ones(labels.size))
        (directory / f"{split}_feature.f").write_text(f"{feature}\n")
        (directory / f"{split}_label.f").write_text(f"{label}\n")
        (directory / f"{split}_weight.f").write_text(f"{weight}\n")
    return {"offset": offset, "data_type": data_type, "filelist_dir": str(directory)}


def test_resplit_after_shuffle_is_deterministic_and_combo_aligned(tmp_path: Path) -> None:
    combinations = {
        "tog_offset12": write_combo(tmp_path / "source", "tog_offset12", 12),
        "level_offset6": write_combo(
            tmp_path / "source", "level_offset6", 6, 2.0, data_type="level"
        ),
    }
    source = tmp_path / "source" / "timewindow_manifest.json"
    source.write_text(json.dumps({
        "offsets": [12, 6], "data_types": ["tog", "level"],
        "train_stems": ["case_a"], "test_stems": ["case_b"],
        "combinations": combinations,
    }))
    output = tmp_path / "split"
    run_script(
        "resplit_after_shuffle.py", "--input-manifest", source,
        "--output-dir", output, "--test-ratio", "0.4", "--seed", "7",
    )
    manifest = json.loads((output / "timewindow_manifest.json").read_text())
    assert manifest["experiment_tag"] == "split_after_shuffle"
    assert manifest["split_train_rows"] == 3
    assert manifest["split_test_rows"] == 2
    labels_by_combo = []
    for key in combinations:
        directory = output / key
        train_feature = load_npz(directory / "split_after_shuffle_train.feature.npz")
        test_feature = load_npz(directory / "split_after_shuffle_test.feature.npz")
        train = np.load(directory / "split_after_shuffle_train.label.npy")
        test = np.load(directory / "split_after_shuffle_test.label.npy")
        train_weight = np.load(directory / "split_after_shuffle_train.weight.npy")
        test_weight = np.load(directory / "split_after_shuffle_test.weight.npy")
        labels_by_combo.append((train, test))
        assert train_feature.shape[0] == train.size == train_weight.size == 3
        assert test_feature.shape[0] == test.size == test_weight.size == 2
        assert sorted(np.concatenate((train, test)).tolist()) == [1, 2, 3, 4, 5]
    assert np.array_equal(labels_by_combo[0][0], labels_by_combo[1][0])
    assert np.array_equal(labels_by_combo[0][1], labels_by_combo[1][1])


def test_resplit_index_mode_reuses_source_feature_files(tmp_path: Path) -> None:
    combinations = {
        "tog_offset12": write_combo(tmp_path / "source", "tog_offset12", 12),
        "level_offset6": write_combo(
            tmp_path / "source", "level_offset6", 6, 2.0, data_type="level"
        ),
    }
    source = tmp_path / "source" / "timewindow_manifest.json"
    source.write_text(json.dumps({
        "offsets": [12, 6], "data_types": ["tog", "level"],
        "train_stems": ["case_a"], "test_stems": ["case_b"],
        "combinations": combinations,
    }))
    output = tmp_path / "index_split"
    run_script(
        "resplit_after_shuffle.py", "--input-manifest", source,
        "--output-dir", output, "--test-ratio", "0.4", "--seed", "7",
        "--storage-mode", "index", "--index-row-batch-size", "9",
        "--index-row-batch-size-tog", "2",
        "--index-row-batch-size-level", "1",
    )
    manifest = json.loads((output / "timewindow_manifest.json").read_text())
    assert manifest["split_storage_mode"] == "index"
    assert manifest["index_row_batch_size"] == 9
    assert manifest["index_row_batch_sizes"] == {"tog": 2, "level": 1}
    sys.path.insert(0, str(SRC))
    from utils.indexed_split import index_row_batch_size
    assert index_row_batch_size(manifest, manifest["combinations"]["tog_offset12"]) == 2
    assert index_row_batch_size(manifest, manifest["combinations"]["level_offset6"]) == 1
    train_indices = np.load(manifest["split_indices"]["train"])
    test_indices = np.load(manifest["split_indices"]["test"])
    assert train_indices.size == 3
    assert test_indices.size == 2
    assert sorted(np.concatenate((train_indices, test_indices)).tolist()) == list(range(5))
    assert not list(output.glob("**/*.feature.npz"))
    assert all(
        manifest["combinations"][key]["filelist_dir"] == combinations[key]["filelist_dir"]
        for key in combinations
    )

    recommendation = tmp_path / "weight_recommendation.json"
    run_script(
        "generate_sample_weight_recommendation.py",
        "--tw-manifest", output / "timewindow_manifest.json",
        "--output-file", recommendation,
    )
    recommendation_data = json.loads(recommendation.read_text())
    inventory = recommendation_data["sampled_label_inventory"]
    assert inventory["train"][0]["sampled_rows"] == 3
    assert inventory["test"][0]["sampled_rows"] == 2

    namefile = tmp_path / "signals.txt"
    namefile.write_text("s0\ns1\ns2\n")
    xgb_output = tmp_path / "step2_xgb"
    run_script(
        "step2_xgb_per_offset.py", "--manifest", output / "timewindow_manifest.json",
        "--namefile", namefile, "--output-dir", xgb_output,
        "--rounds", "2", "--top-k", "2", "--include-combos", "tog_offset12",
    )
    xgb_manifest = json.loads((xgb_output / "mcp_per_offset_manifest.json").read_text())
    xgb_result = xgb_manifest["combinations"]["tog_offset12"]
    assert xgb_result["training_mode"] == "indexed_iter"
    assert xgb_result["n_train"] == 3

    mcp_output = tmp_path / "step2_mcp"
    run_script(
        "step2_mcp_per_offset.py", "--manifest", output / "timewindow_manifest.json",
        "--namefile", namefile, "--output-dir", mcp_output,
        "--alpha", "0.001", "--top-k", "2",
        "--include-combos", "tog_offset12",
    )
    mcp_manifest = json.loads((mcp_output / "mcp_per_offset_manifest.json").read_text())
    mcp_result = mcp_manifest["combinations"]["tog_offset12"]
    assert mcp_result["n_train"] == 3
    assert mcp_result["batch_unit"] == "rows"
    assert mcp_result["effective_batch_size"] == 2
    assert mcp_result["n_batches"] == 2

    proxy_combinations = {}
    for key in combinations:
        selected = tmp_path / "proxy" / key / "selected_indices.txt"
        selected.parent.mkdir(parents=True)
        selected.write_text("0\n1\n")
        proxy_combinations[key] = {"selected_indices_file": str(selected)}
    proxy_manifest = tmp_path / "proxy" / "manifest.json"
    proxy_manifest.write_text(json.dumps({"combinations": proxy_combinations}))
    step3_output = tmp_path / "step3"
    run_script(
        "step3_hybrid_concat_features.py",
        "--tw-manifest", output / "timewindow_manifest.json",
        "--xgb-manifest", proxy_manifest, "--selectors", "xgb",
        "--namefile", namefile, "--output-dir", step3_output,
    )
    assert xgb.DMatrix(step3_output / "xgb" / "train.buffer").num_row() == 3
    assert xgb.DMatrix(step3_output / "xgb" / "test.buffer").num_row() == 2


def test_cross_window_uses_source_signals_but_target_offsets(tmp_path: Path) -> None:
    combinations = {
        "tog_offset12": write_combo(tmp_path / "target", "tog_offset12", 12),
        "tog_offset6": write_combo(tmp_path / "target", "tog_offset6", 6),
    }
    manifest = tmp_path / "target" / "timewindow_manifest.json"
    manifest.write_text(json.dumps({
        "offsets": [12, 6], "data_types": ["tog"],
        "train_stems": ["case_a"], "test_stems": ["case_b"],
        "combinations": combinations,
    }))
    namefile = tmp_path / "signals.txt"
    namefile.write_text("s0\ns1\ns2\n")
    baseline = tmp_path / "baseline_features.txt"
    baseline.write_text("tog_o12_s0\n")
    source = tmp_path / "near_proxy_features.txt"
    source.write_text("tog_o5_s1\ntog_o0_s1\ntog_o12_s2\n")
    output = tmp_path / "cross"
    run_script(
        "build_cross_window_features.py", "--target-tw-manifest", manifest,
        "--namefile", namefile, "--baseline-feature-names", baseline,
        "--source-proxy-features", source,
        "--source-offset-min", "0", "--source-offset-max", "5",
        "--output-dir", output,
    )
    names = (output / "feature_names.txt").read_text().splitlines()
    assert names == ["tog_o12_s0", "tog_o12_s1", "tog_o6_s1"]
    assert all("_o5_" not in name and "_o0_" not in name for name in names)
    assert xgb.DMatrix(output / "train_cross_window.buffer").num_col() == 3
    metadata = json.loads((output / "cross_window_union_meta.json").read_text())
    assert metadata["source_offset_min"] == 0
    assert metadata["source_offset_max"] == 5
    assert metadata["source_proxy_features_in_range"] == 2
    assert metadata["source_proxy_features_filtered_by_offset"] == 1
    assert metadata["transferred_signals"] == ["s1"]


def test_compare_experiment_metrics_writes_deltas(tmp_path: Path) -> None:
    for name, mape, r2 in (("baseline", 10.0, 0.80), ("candidate", 8.0, 0.85)):
        directory = tmp_path / name / "eval"
        directory.mkdir(parents=True)
        pd.DataFrame({"metric": ["MAPE", "R2"], "value": [mape, r2]}).to_csv(
            directory / "metrics.csv", index=False
        )
    output = tmp_path / "comparison"
    run_script(
        "compare_experiment_metrics.py", "--baseline", tmp_path / "baseline",
        "--candidate", tmp_path / "candidate", "--output-dir", output,
    )
    result = pd.read_csv(output / "metrics_comparison.csv").set_index("metric")
    assert result.loc["MAPE", "candidate_minus_baseline"] == -2.0
    assert result.loc["R2", "candidate_minus_baseline"] == pytest.approx(0.05)


def test_split_after_shuffle_runner_reuses_each_stage() -> None:
    compatibility = (ROOT / "cmd" / "run_timewindow_split_after_shuffle.csh").read_text()
    assert "run_timewindow_split_after_shuffle_dsub.csh" in compatibility
    runner = (ROOT / "cmd" / "run_timewindow_split_after_shuffle_dsub.csh").read_text()
    common = (ROOT / "cmd" / "timewindow_common.csh").read_text()
    assert 'if ( ! $?SPLIT_STORAGE_MODE ) set SPLIT_STORAGE_MODE = index' in common
    assert 'if ( ! $?INDEX_ROW_BATCH_SIZE ) set INDEX_ROW_BATCH_SIZE = 200000' in common
    assert 'if ( ! $?INDEX_ROW_BATCH_SIZE_TOG ) set INDEX_ROW_BATCH_SIZE_TOG = 50000' in common
    assert 'if ( ! $?INDEX_ROW_BATCH_SIZE_LEVEL ) set INDEX_ROW_BATCH_SIZE_LEVEL = 5000' in common
    assert "resplit_after_shuffle.py" in runner
    assert 'source "${SCRIPT_DIR}/timewindow_common.csh"' in runner
    assert 'setenv OUTPUT_ROOT "${SOURCE_OUTPUT_ROOT}/spilt_after_shuffle"' in runner
    assert "\nsetenv EXTRA_TAG " not in runner
    assert 'if ( $?EXTRA_TAG ) unsetenv EXTRA_TAG' in runner
    assert '--storage-mode "${SPLIT_STORAGE_MODE}"' in runner
    assert '--index-row-batch-size "${INDEX_ROW_BATCH_SIZE}"' in runner
    assert '--index-row-batch-size-tog "${INDEX_ROW_BATCH_SIZE_TOG}"' in runner
    assert '--index-row-batch-size-level "${INDEX_ROW_BATCH_SIZE_LEVEL}"' in runner
    assert 'setenv RUN_STEP2_VALIDATION 0' in runner
    assert "build_timewindow_output_tag.py" in runner
    assert 'set TAG_OUTPUT_ROOT = "${OUTPUT_ROOT}/${OUTPUT_TAG}"' in runner
    assert 'run_timewindow_pipeline.csh" "$HW" prep' in runner
    assert "step5_hpo_timewindow.py" in runner
    assert "step6_retrain_eval.py" in runner
    assert '--skip-full-eval' in runner
    assert "alias split_submit '${DSUB_CMD}" in runner
    assert "alias step5_submit '${DSUB_CMD}" in runner
    assert "alias step6_submit '${DSUB_CMD}" in runner
    assert "foreach K ( $K_LIST )" in runner
    assert "while ( 1 )" in runner


def test_cross_window_runner_submits_all_compute_stages_with_dsub() -> None:
    runner = (ROOT / "cmd" / "run_timewindow_cross_window_dsub.csh").read_text()
    common = (ROOT / "cmd" / "timewindow_common.csh").read_text()
    assert 'source "${SCRIPT_DIR}/timewindow_common.csh"' in runner
    assert 'if ( ! $?CROSS_SOURCE_OFFSET_MIN ) set CROSS_SOURCE_OFFSET_MIN = 0' in common
    assert 'if ( ! $?CROSS_SOURCE_OFFSET_MAX ) set CROSS_SOURCE_OFFSET_MAX = 5' in common
    assert '--source-offset-min "${CROSS_SOURCE_OFFSET_MIN}"' in runner
    assert '--source-offset-max "${CROSS_SOURCE_OFFSET_MAX}"' in runner
    for script in (
        "build_cross_window_features.py", "step5_hpo_timewindow.py",
        "step6_retrain_eval.py", "compare_experiment_metrics.py",
    ):
        assert script in runner
    assert "alias cross_submit '${DSUB_CMD}" in runner
    assert "alias cross_hpo_submit '${DSUB_CMD}" in runner
    assert "alias cross_retrain_submit '${DSUB_CMD}" in runner
    assert "foreach STAGE ( build hpo retrain compare )" in runner
