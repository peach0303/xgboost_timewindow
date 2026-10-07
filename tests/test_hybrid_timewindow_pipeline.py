from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix, save_npz


ROOT = Path(__file__).resolve().parents[1]
TIMEWINDOW = ROOT / "src" / "timewindow"


def run_script(name: str, *arguments: object) -> None:
    script_python = os.environ.get("TIMEWINDOW_TEST_PYTHON", sys.executable)
    subprocess.run(
        [script_python, str(TIMEWINDOW / name), *(str(value) for value in arguments)],
        cwd=ROOT, check=True, capture_output=True, text=True,
    )


def test_hybrid_pipeline_smoke(tmp_path: Path) -> None:
    rng = np.random.RandomState(108)
    names = [f"signal_{index}" for index in range(8)]
    namefile = tmp_path / "signals.txt"
    namefile.write_text("\n".join(names) + "\n", encoding="utf-8")

    train_y = rng.uniform(1.0, 3.0, size=40).astype(np.float32)
    test_y = rng.uniform(1.0, 3.0, size=16).astype(np.float32)
    train_label = tmp_path / "train.label.npy"
    test_label = tmp_path / "test.label.npy"
    np.save(train_label, train_y)
    np.save(test_label, test_y)

    combinations = {}
    for offset in (1, 2):
        combo_key = f"tog_offset{offset}"
        combo_dir = tmp_path / combo_key
        combo_dir.mkdir()
        train_feature = combo_dir / "train.feature.npz"
        test_feature = combo_dir / "test.feature.npz"
        train_x = rng.normal(size=(40, 8)).astype(np.float32)
        test_x = rng.normal(size=(16, 8)).astype(np.float32)
        train_y += (0.3 * train_x[:, 0]).astype(np.float32)
        test_y += (0.3 * test_x[:, 0]).astype(np.float32)
        save_npz(train_feature, csr_matrix(train_x))
        save_npz(test_feature, csr_matrix(test_x))
        (combo_dir / "train_feature.f").write_text(str(train_feature) + "\n")
        (combo_dir / "test_feature.f").write_text(str(test_feature) + "\n")
        (combo_dir / "train_label.f").write_text(str(train_label) + "\n")
        (combo_dir / "test_label.f").write_text(str(test_label) + "\n")
        combinations[combo_key] = {
            "data_type": "tog", "offset": offset, "filelist_dir": str(combo_dir),
        }
    # Labels are loaded after all feature contributions have been applied.
    np.save(train_label, train_y)
    np.save(test_label, test_y)

    tw_manifest = tmp_path / "timewindow_manifest.json"
    tw_manifest.write_text(json.dumps({
        "train_stems": ["train"], "test_stems": ["test"],
        "combinations": combinations,
    }))

    def proxy_manifest(selector: str, selections: list[list[int]]) -> Path:
        manifest_combos = {}
        for (combo_key, _), selected in zip(combinations.items(), selections):
            selector_dir = tmp_path / selector / combo_key
            selector_dir.mkdir(parents=True)
            selected_file = selector_dir / "selected_indices.txt"
            selected_file.write_text("\n".join(str(index) for index in selected) + "\n")
            manifest_combos[combo_key] = {"selected_indices_file": str(selected_file)}
        path = tmp_path / selector / "mcp_per_offset_manifest.json"
        path.write_text(json.dumps({"selector": selector, "combinations": manifest_combos}))
        return path

    mcp_manifest = proxy_manifest("mcp", [[0, 1, 2, 3], list(range(8))])
    xgb_manifest = proxy_manifest("xgb", [[0, 1, 4, 5], list(range(8))])
    filtered = tmp_path / "filtered"
    filtered_tw = filtered / "timewindow_manifest.json"
    filtered_mcp = filtered / "mcp_per_offset_manifest.json"
    filtered_xgb = filtered / "xgb_per_offset_manifest.json"
    run_script(
        "filter_timewindow_manifests.py",
        "--tw-manifest-in", tw_manifest, "--mcp-manifest-in", mcp_manifest,
        "--tw-manifest-out", filtered_tw, "--mcp-manifest-out", filtered_mcp,
        "--include-combos", "tog_offset2",
    )
    run_script(
        "filter_timewindow_manifests.py",
        "--tw-manifest-in", tw_manifest, "--mcp-manifest-in", xgb_manifest,
        "--tw-manifest-out", filtered / "timewindow_manifest_xgb.json",
        "--mcp-manifest-out", filtered_xgb,
        "--include-combos", "tog_offset2",
    )

    step3 = tmp_path / "step3"
    step4 = tmp_path / "step4"
    step5 = tmp_path / "step5"

    run_script(
        "step3_hybrid_concat_features.py",
        "--tw-manifest", filtered_tw, "--mcp-manifest", filtered_mcp,
        "--xgb-manifest", filtered_xgb, "--namefile", namefile,
        "--output-dir", step3,
    )
    hybrid_meta = json.loads((step3 / "hybrid_concat_meta.json").read_text())
    assert [block["combo_key"] for block in hybrid_meta["selectors"]["mcp"]["blocks"]] == ["tog_offset2"]
    assert all(
        name.startswith("tog_o2_")
        for name in (step3 / "mcp" / "feature_names.txt").read_text().splitlines()
    )
    run_script(
        "step4_hybrid_screen_features.py",
        "--hybrid-concat-meta", step3 / "hybrid_concat_meta.json",
        "--final-top-ks", 8, "--xgb-rounds", 3, "--nthread", 1,
        "--output-dir", step4,
    )
    union_meta_path = step4 / "union_K8" / "union_meta.json"
    union_meta = json.loads(union_meta_path.read_text())
    assert union_meta["k_mcp"] == union_meta["k_xgb"] == union_meta["final_top_k"] == 8
    assert 8 <= union_meta["union_columns"] <= 16
    assert len(union_meta["mcp_ranked_proxies"]) == 8

    run_script(
        "step5_hybrid_hpo_timewindow.py",
        "--union-meta", union_meta_path, "--output-dir", step5,
        "--round-num", 3, "--trial-num", 1, "--pop-size", 2,
        "--nfold", 2, "--job-num", 1, "--nthread", 1,
    )
    best = json.loads((step5 / "best_hybrid_model_meta.json").read_text())
    assert best["ratio"] in [index / 8 for index in range(1, 9)]
    assert best["linear_proxy_count"] <= 8 * best["ratio"]
    assert (step5 / "best_ridge_model.joblib").exists()
    assert (step5 / "best_residual_xgb_model.json").exists()

    # The main pipeline now reuses the mixed-proxy union but trains one direct
    # XGBoost model, rather than the standalone Ridge + residual-XGB model.
    step5_direct = tmp_path / "step5_direct"
    step6_direct = tmp_path / "step6_direct"
    union_dir = step4 / "union_K8"
    run_script(
        "step5_hpo_timewindow.py",
        "--train-buffer", union_dir / "train_union_top8.buffer",
        "--test-buffer", union_dir / "test_union_top8.buffer",
        "--feature-names", union_dir / "feature_names.txt",
        "--output-dir", step5_direct,
        "--round-num", 3, "--trial-num", 1, "--pop-size", 2,
        "--nfold", 2, "--job-num", 1, "--two-phase-method", "",
        "--loss", "squared_error",
    )
    best_csv = step5_direct / "best_trials_leaf_1000.csv"
    assert best_csv.exists()
    run_script(
        "step6_retrain_eval.py",
        "--train-buffer", union_dir / "train_union_top8.buffer",
        "--test-buffer", union_dir / "test_union_top8.buffer",
        "--feature-names", union_dir / "feature_names.txt",
        "--hpo-csv", best_csv, "--output-dir", step6_direct,
        "--rounds", 3, "--two-phase-method", "",
        "--loss", "squared_error", "--skip-full-eval",
    )
    assert (step6_direct / "result" / "xgbmodel.json").exists()
    assert not (step6_direct / "result" / "xgbmodel_residual.json").exists()


def test_main_pipeline_wires_dual_step2_to_direct_xgb() -> None:
    pipeline = (ROOT / "cmd" / "run_timewindow_pipeline.csh").read_text(
        encoding="utf-8"
    )
    assert "foreach selector ( mcp xgb )" in pipeline
    assert "step2_mcp_per_offset.py" in pipeline
    assert "step2_xgb_per_offset.py" in pipeline
    assert "step3_hybrid_concat_features.py" in pipeline
    assert "step4_hybrid_screen_features.py" in pipeline
    assert "train_union_top${k}.buffer" in pipeline
    assert 'step5_hpo_timewindow.py' in pipeline
    assert 'step5_hybrid_hpo_timewindow.py' not in pipeline
    assert '--two-phase-method ""' in pipeline


def test_standalone_hybrid_pipeline_wires_manifest_filtering() -> None:
    pipeline = (ROOT / "cmd" / "run_timewindow_hybrid_pipeline.csh").read_text(
        encoding="utf-8"
    )
    assert 'if ( ! $?STEP3_OFFSETS ) set STEP3_OFFSETS = ""' in pipeline
    assert 'if ( ! $?STEP3_DATA_TYPES ) set STEP3_DATA_TYPES = ""' in pipeline
    assert 'if ( ! $?STEP3_COMBOS ) set STEP3_COMBOS = ""' in pipeline
    assert 'if ( ! $?EXTRA_TAG ) set EXTRA_TAG = ""' in pipeline
    assert "build_timewindow_output_tag.py" in pipeline
    assert 'set TAG_OUTPUT_ROOT = "${OUTPUT_ROOT}/${OUTPUT_TAG}"' in pipeline
    assert 'set STEP3_DIR = "${TAG_OUTPUT_ROOT}/step3_hybrid_concat"' in pipeline
    assert 'set MLP_TRAIN_DIR = "${TAG_OUTPUT_ROOT}/step6_mlp_retrain/union_K${FINAL_TOP_K}"' in pipeline
    assert pipeline.count("filter_timewindow_manifests.py") == 2
    assert '--tw-manifest "$EFFECTIVE_TW_MANIFEST"' in pipeline
    assert '--mcp-manifest "$EFFECTIVE_MCP_MANIFEST"' in pipeline
    assert '--xgb-manifest "$EFFECTIVE_XGB_MANIFEST"' in pipeline
    arg_headers = [
        line for line in pipeline.splitlines()
        if line.startswith("#=") and "step " in line and " args" in line
    ]
    assert len(arg_headers) == 6
    assert all(line.count("=") == 60 for line in arg_headers)
    assert pipeline.splitlines().count("#" + "=" * 70) == 6

    assert pipeline.count("#" + "=" * 10 + "user check args" + "=" * 10) == 1
    assert pipeline.splitlines().count("#" + "=" * 25) == 1
    assert "set DSUB_CMD = ( dsub -A root.ug_dsp.hisdr_HClass )" in pipeline
    assert "if ( ! $?DSUB_QUEUE ) set DSUB_QUEUE = normal" in pipeline
    assert "if ( ! $?MAX_RERUN ) set MAX_RERUN = 10" in pipeline
    assert pipeline.count("#!/bin/csh -x") == 5
    assert pipeline.count('/running"') >= 10
    assert pipeline.count('/complete"') >= 10
    assert pipeline.count('/fail"') >= 10
    assert "-q ${DSUB_QUEUE} -oo ${dsub_out} -eo ${dsub_err}" in pipeline
    assert "$DSUB_EXTRA_ARGS:q -s ${job_file}" in pipeline
    assert "if ( $rerun_count >= $MAX_RERUN )" in pipeline
    assert "automatically rerun failed stage only" in pipeline
    assert 'if ( ! $?skip_mlp_train ) set skip_mlp_train = 1' in pipeline
    assert "step5_hpo_mlp.py" in pipeline
    assert "step6_retrain_eval_mlp.py" in pipeline
    assert "step3 step4 step5 mlp_hpo mlp_train" in pipeline
    assert '${MLP_HPO_DIR}/best_trial.json' in pipeline
    assert '${MLP_TRAIN_DIR}/result/mlp_model.joblib' in pipeline
    assert pipeline.count('--shuffle-seed "$SHUFFLE_SEED" --output-dir "$STEP3_DIR"') == 1
