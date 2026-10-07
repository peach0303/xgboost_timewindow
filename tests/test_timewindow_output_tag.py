from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "src" / "timewindow" / "build_timewindow_output_tag.py"


def manifest(tmp_path: Path) -> Path:
    path = tmp_path / "timewindow_manifest.json"
    path.write_text(json.dumps({"combinations": {
        "level_offset12": {}, "tog_offset12": {},
        "level_offset11": {}, "tog_offset11": {},
        "level_offset10": {}, "tog_offset10": {},
    }}))
    return path


def run_tag(path: Path, *arguments: str) -> str:
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--tw-manifest", str(path), *arguments],
        check=True, capture_output=True, text=True,
    )
    return result.stdout.strip()


def test_tag_normalizes_type_order_and_preserves_offset_order(tmp_path: Path) -> None:
    path = manifest(tmp_path)
    assert run_tag(
        path, "--offsets", "12,11,10", "--data-types", "level,tog"
    ) == "tog_level_12_11_10"
    assert run_tag(
        path, "--include-combos", "level_offset11,tog_offset12"
    ) == "tog_level_11_12"


def test_tag_uses_all_manifest_combinations_or_safe_override(tmp_path: Path) -> None:
    path = manifest(tmp_path)
    assert run_tag(path) == "tog_level_12_11_10"
    assert run_tag(path, "--override", "experiment_a") == "experiment_a"


def test_main_pipeline_routes_downstream_outputs_under_tag_root() -> None:
    pipeline = (ROOT / "cmd" / "run_timewindow_pipeline.csh").read_text(
        encoding="utf-8"
    )
    assert 'if ( ! $?EXTRA_TAG ) set EXTRA_TAG = ""' in pipeline
    assert "build_timewindow_output_tag.py" in pipeline
    assert 'set TAG_OUTPUT_ROOT = "${OUTPUT_ROOT}/${OUTPUT_TAG}"' in pipeline
    for assignment in (
        'set STEP3_DIR = "${TAG_OUTPUT_ROOT}/step3_hybrid_concat"',
        'set STEP4_DIR = "${TAG_OUTPUT_ROOT}/step4_hybrid_screen"',
        'set STEP5_DIR = "${TAG_OUTPUT_ROOT}/step5_hpo"',
        'set STEP6_DIR = "${TAG_OUTPUT_ROOT}/step6_retrain"',
        'set MLP_STEP5_DIR = "${TAG_OUTPUT_ROOT}/step5_mlp_hpo"',
        'set MLP_STEP6_DIR = "${TAG_OUTPUT_ROOT}/step6_mlp_retrain"',
        'set STEP3_INPUT_DIR = "${TAG_OUTPUT_ROOT}/input_manifests"',
        'set LOG_DIR = "${TAG_OUTPUT_ROOT}/logs_dsub"',
        'set JOB_DIR = "${TAG_OUTPUT_ROOT}/jobs_dsub"',
    ):
        assert assignment in pipeline
    assert "touch ${TAG_OUTPUT_ROOT}/complete" in pipeline
    assert "touch ${OUTPUT_ROOT}/complete" not in pipeline


def test_main_pipeline_exposes_new_optional_modes() -> None:
    pipeline = (ROOT / "cmd" / "run_timewindow_pipeline.csh").read_text(
        encoding="utf-8"
    )
    assert "STEP2_PARALLEL_DSUB" not in pipeline
    assert "ALPHA_MCP_TOG" in pipeline
    assert "ALPHA_MCP_LEVEL" in pipeline
    assert "BATCH_SIZE_TOG" in pipeline
    assert "BATCH_SIZE_LEVEL" in pipeline
    assert 'if ( "${combo_key}" =~ *level* ) then' in pipeline
    assert "set alpha_mcp = ${ALPHA_MCP_LEVEL}" in pipeline
    assert "--alpha ${alpha_mcp}" in pipeline
    assert "--batch ${batch_size}" in pipeline
    step2_dir_create = pipeline.index("mkdir -p ${STEP2_DIR}")
    step2_complete_check = pipeline.index("if ( -f ${STEP2_DIR}/complete ) then")
    assert step2_dir_create < step2_complete_check
    for resource_var in (
        "STEP1_DSUB_EXTRA_ARGS",
        "STEP2_TOG_DSUB_EXTRA_ARGS",
        "STEP2_LEVEL_DSUB_EXTRA_ARGS",
        "STEP2_MERGE_DSUB_EXTRA_ARGS",
        "STEP2_VALIDATION_DSUB_EXTRA_ARGS",
        "STEP3_DSUB_EXTRA_ARGS",
        "STEP4_DSUB_EXTRA_ARGS",
        "STEP5_DSUB_EXTRA_ARGS",
        "STEP6_DSUB_EXTRA_ARGS",
        "MLP_DSUB_EXTRA_ARGS",
    ):
        assert resource_var in pipeline
    assert 'if ( ! $?STEP2_SELECTOR_MODE ) set STEP2_SELECTOR_MODE = hybrid' in pipeline
    assert 'if ( ! $?FEATURE_INPUT_MODE ) set FEATURE_INPUT_MODE = proxy_columns' in pipeline
    assert 'if ( ! $?TWO_PHASE_METHOD ) set TWO_PHASE_METHOD = ""' in pipeline
    assert "--include-combos ${combo_key}" in pipeline
    assert "merge_timewindow_manifests.py" in pipeline
    assert "--feature-input-mode ${FEATURE_INPUT_MODE}" in pipeline
    assert '--two-phase-method "${TWO_PHASE_METHOD}"' in pipeline
    assert "methodA_c${TWO_PHASE_CUTOFF}_r${TWO_PHASE_RATIO}" in pipeline
    for step in range(1, 7):
        assert f"# ▶ Step {step} 控制流：" in pipeline


def test_legacy_pipeline_keeps_single_selector_defaults() -> None:
    wrapper = (
        ROOT / "cmd" / "run_timewindow_pipeline_single_selector.csh"
    ).read_text(encoding="utf-8")
    assert "setenv STEP2_SELECTOR_MODE single" in wrapper
    assert "setenv PROXY_SELECTOR mcp" in wrapper
    assert "setenv FEATURE_INPUT_MODE signal_all_windows" in wrapper
    assert "setenv TWO_PHASE_METHOD A" in wrapper
    assert 'exec csh "${SCRIPT_DIR}/run_timewindow_pipeline.csh" $argv:q' in wrapper
