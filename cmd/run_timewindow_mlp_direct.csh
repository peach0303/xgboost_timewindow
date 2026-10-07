#!/bin/csh -f
# Train one MLP directly from an existing main-pipeline Step4 result without HPO.
# Usage: csh run_timewindow_mlp_direct.csh <hw> <K> [OUTPUT_ROOT]
# Optional overrides: OUTPUT_ROOT, MLP_STEP4_DIR, MLP_CONFIG, MLP_OUTPUT_DIR,
# FEATURE_INPUT_MODE, MLP_RUN_FULL_EVAL=0|1, MLP_FORCE=0|1.

set SCRIPT_DIR = `dirname "$0"`
set SCRIPT_DIR = `cd "$SCRIPT_DIR" && pwd`
if ( $#argv < 2 || $#argv > 3 ) then
    echo "Usage: csh $0 <hw> <K> [OUTPUT_ROOT]"
    exit 2
endif

set HW = "$1"
set TARGET_K = "$2"
set PACKAGE_ROOT = `cd "${SCRIPT_DIR}/.." && pwd`
set PROJECT_ROOT = `cd "${PACKAGE_ROOT}/.." && pwd`
if ( ! $?OUTPUT_BASE ) set OUTPUT_BASE = "${PROJECT_ROOT}/output/xgboost_timewindow_train"
if ( ! $?STAGE ) set STAGE = after_one
if ( $#argv == 3 ) set OUTPUT_ROOT = "$3"

if ( ! $?OUTPUT_ROOT ) then
    if ( ! -d "$OUTPUT_BASE" ) then
        echo "ERROR: output base does not exist: ${OUTPUT_BASE}"
        exit 1
    endif
    set OUTPUT_CANDIDATES = ( `find "$OUTPUT_BASE" -mindepth 1 -maxdepth 1 -type d -name "${HW}_${STAGE}_*" -print | sort -r` )
    if ( $#OUTPUT_CANDIDATES == 0 ) then
        echo "ERROR: no main-pipeline output matches ${OUTPUT_BASE}/${HW}_${STAGE}_*"
        echo "Set OUTPUT_ROOT or pass it as the third argument."
        exit 1
    endif
    foreach candidate_root ( $OUTPUT_CANDIDATES )
        set TARGET_BUFFERS = ( `find "$candidate_root" -mindepth 4 -maxdepth 4 -type f -name "train_union_top${TARGET_K}.buffer" -path "*/step4_*/union_K${TARGET_K}/*" -print` )
        if ( $#TARGET_BUFFERS > 0 ) then
            set OUTPUT_ROOT = "$candidate_root"
            break
        endif
    end
    if ( ! $?OUTPUT_ROOT ) then
        echo "ERROR: no pipeline output contains a Step4 result for K=${TARGET_K}"
        echo "Set OUTPUT_ROOT or pass it as the third argument."
        exit 1
    endif
endif
if ( ! -d "$OUTPUT_ROOT" ) then
    echo "ERROR: OUTPUT_ROOT does not exist: ${OUTPUT_ROOT}"
    exit 1
endif

set FEATURE_MODE_EXPLICIT = 0
if ( $?FEATURE_INPUT_MODE ) set FEATURE_MODE_EXPLICIT = 1
source "${SCRIPT_DIR}/timewindow_common.csh"

if ( $?MLP_STEP4_DIR ) then
    set SCREEN_DIR = "${MLP_STEP4_DIR}/union_K${TARGET_K}"
    set STEP4_CANDIDATES = ( "$SCREEN_DIR" )
else
    set STEP4_CANDIDATES = ( )
    foreach candidate ( `find "$OUTPUT_ROOT" -mindepth 3 -maxdepth 3 -type d -name "union_K${TARGET_K}" -path "*/step4_*/*" -print | sort` )
        if ( -f "${candidate}/train_union_top${TARGET_K}.buffer" && \
             -f "${candidate}/test_union_top${TARGET_K}.buffer" && \
             -f "${candidate}/feature_names.txt" ) then
            set STEP4_CANDIDATES = ( $STEP4_CANDIDATES "$candidate" )
        endif
    end
    if ( $#STEP4_CANDIDATES == 0 ) then
        echo "ERROR: no valid Step4 union_K${TARGET_K} result found under ${OUTPUT_ROOT}"
        exit 1
    endif
    if ( $#STEP4_CANDIDATES > 1 ) then
        echo "ERROR: multiple Step4 results match K=${TARGET_K}:"
        foreach candidate ( $STEP4_CANDIDATES )
            echo "  $candidate"
        end
        echo "Set MLP_STEP4_DIR to the intended step4_* directory."
        exit 1
    endif
    set SCREEN_DIR = "$STEP4_CANDIDATES[1]"
endif

set STEP4_DIR = `dirname "$SCREEN_DIR"`
set TAG_OUTPUT_ROOT = `dirname "$STEP4_DIR"`
set TRAIN_BUFFER = "${SCREEN_DIR}/train_union_top${TARGET_K}.buffer"
set TEST_BUFFER = "${SCREEN_DIR}/test_union_top${TARGET_K}.buffer"
set FEATURE_NAMES = "${SCREEN_DIR}/feature_names.txt"
if ( ! -f "$TRAIN_BUFFER" || ! -f "$TEST_BUFFER" || ! -f "$FEATURE_NAMES" ) then
    echo "ERROR: incomplete Step4 inputs under ${SCREEN_DIR}"
    exit 1
endif

if ( $FEATURE_MODE_EXPLICIT == 0 ) then
    set STEP4_BASENAME = `basename "$STEP4_DIR"`
    if ( "$STEP4_BASENAME" =~ *_signal_all_windows ) then
        set FEATURE_INPUT_MODE = signal_all_windows
    else
        set FEATURE_INPUT_MODE = proxy_columns
    endif
endif

if ( ! $?MLP_CONFIG ) set MLP_CONFIG = "${PACKAGE_ROOT}/configs/mlp_default.json"
if ( ! -f "$MLP_CONFIG" ) then
    echo "ERROR: MLP config does not exist: ${MLP_CONFIG}"
    exit 1
endif
if ( ! $?MLP_OUTPUT_DIR ) then
    set MLP_OUTPUT_DIR = "${TAG_OUTPUT_ROOT}/step6_mlp_retrain/union_K${TARGET_K}_mlp_default"
endif
if ( ! $?MLP_RUN_FULL_EVAL ) set MLP_RUN_FULL_EVAL = 0
if ( "$MLP_RUN_FULL_EVAL" != "0" && "$MLP_RUN_FULL_EVAL" != "1" ) then
    echo "ERROR: MLP_RUN_FULL_EVAL must be 0 or 1"
    exit 2
endif
if ( ! $?MLP_FORCE ) set MLP_FORCE = 0
if ( "$MLP_FORCE" != "0" && "$MLP_FORCE" != "1" ) then
    echo "ERROR: MLP_FORCE must be 0 or 1"
    exit 2
endif

if ( ! $?MLP_TW_MANIFEST ) then
    if ( -f "${TAG_OUTPUT_ROOT}/input_manifests/timewindow_manifest.json" ) then
        set MLP_TW_MANIFEST = "${TAG_OUTPUT_ROOT}/input_manifests/timewindow_manifest.json"
    else
        set MLP_TW_MANIFEST = "${OUTPUT_ROOT}/step1_shifted/timewindow_manifest.json"
    endif
endif
if ( ! -f "$MLP_TW_MANIFEST" ) then
    echo "ERROR: timewindow manifest does not exist: ${MLP_TW_MANIFEST}"
    exit 1
endif

mkdir -p "$MLP_OUTPUT_DIR"
if ( "$MLP_FORCE" == "0" && \
     -f "${MLP_OUTPUT_DIR}/result/mlp_model.joblib" && \
     -f "${MLP_OUTPUT_DIR}/eval/metrics.csv" ) then
    echo "MLP direct train: SKIP; existing result: ${MLP_OUTPUT_DIR}"
    exit 0
endif

set EVAL_ARGS = ( --skip-full-eval )
if ( "$MLP_RUN_FULL_EVAL" == "1" ) set EVAL_ARGS = ( )

echo "============================================================"
echo "Time-Window MLP Direct Training"
echo "  HW           = ${HW}"
echo "  K            = ${TARGET_K}"
echo "  OUTPUT_ROOT  = ${OUTPUT_ROOT}"
echo "  STEP4_DIR    = ${STEP4_DIR}"
echo "  CONFIG       = ${MLP_CONFIG}"
echo "  OUTPUT       = ${MLP_OUTPUT_DIR}"
echo "  FEATURE_MODE = ${FEATURE_INPUT_MODE}"
echo "  FULL_EVAL    = ${MLP_RUN_FULL_EVAL}"
echo "============================================================"

rm -f "${MLP_OUTPUT_DIR}/fail" "${MLP_OUTPUT_DIR}/status" "${MLP_OUTPUT_DIR}/train_complete"
"$PYTHON" "${TIMEWINDOW_SRC}/step6_retrain_eval_mlp.py" \
    --train-buffer "$TRAIN_BUFFER" \
    --test-buffer "$TEST_BUFFER" \
    --hpo-result "$MLP_CONFIG" \
    --output-dir "$MLP_OUTPUT_DIR" \
    --feature-names "$FEATURE_NAMES" \
    --tw-manifest "$MLP_TW_MANIFEST" \
    --level-dir "$LEVEL_DIR" \
    --tog-dir "$TOG_DIR" \
    --label-dir "$LABEL_DIR" \
    --raw-namefile "$NAMEFILE" \
    --feature-input-mode "$FEATURE_INPUT_MODE" \
    --shuffle-seed "$SHUFFLE_SEED" \
    --seed "$SHUFFLE_SEED" \
    $EVAL_ARGS:q
set rc = $status
echo "$rc" >! "${MLP_OUTPUT_DIR}/status"
if ( $rc == 0 && \
     -f "${MLP_OUTPUT_DIR}/result/mlp_model.joblib" && \
     -f "${MLP_OUTPUT_DIR}/eval/metrics.csv" ) then
    touch "${MLP_OUTPUT_DIR}/train_complete"
    if ( "$MLP_RUN_FULL_EVAL" == "1" && \
         -f "${MLP_OUTPUT_DIR}/eval/full_unsampled/evaluation_summary.json" ) then
        touch "${MLP_OUTPUT_DIR}/complete"
    endif
    echo "MLP direct train complete: ${MLP_OUTPUT_DIR}"
    exit 0
endif

touch "${MLP_OUTPUT_DIR}/fail"
echo "ERROR: MLP direct train failed with status ${rc}"
exit $rc
