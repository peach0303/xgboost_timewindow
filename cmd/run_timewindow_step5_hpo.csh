#!/bin/csh -f
# Step 5: run HPO for one K. Usage: csh run_timewindow_step5_hpo.csh <hw> <K>
set SCRIPT_DIR = `dirname "$0"`
set SCRIPT_DIR = `cd "$SCRIPT_DIR" && pwd`
if ( $#argv >= 1 ) then
    set HW = "$1"
else
    set HW = boom
endif
if ( $#argv >= 2 ) then
    set TARGET_K = "$2"
else if ( $?TARGET_K ) then
    set TARGET_K = "$TARGET_K"
else
    echo "ERROR: Step 5 requires K, e.g. csh $0 ${HW} 1024"
    exit 2
endif
source "${SCRIPT_DIR}/timewindow_common.csh"
mkdir -p "$OUTPUT_ROOT" "$LOG_DIR" "$STEP5_DIR"

set SCREEN_DIR = "${STEP4_DIR}/screen_K${TARGET_K}"
set HPO_OUT = "${STEP5_DIR}/screen_K${TARGET_K}_method${TWO_PHASE_METHOD}"

if ( ! -f "${SCREEN_DIR}/train_top${TARGET_K}.buffer" ) then
    echo "ERROR: missing Step 4 train buffer: ${SCREEN_DIR}/train_top${TARGET_K}.buffer"
    exit 1
endif
if ( ! -f "${SCREEN_DIR}/test_top${TARGET_K}.buffer" ) then
    echo "ERROR: missing Step 4 test buffer: ${SCREEN_DIR}/test_top${TARGET_K}.buffer"
    exit 1
endif
if ( ! -f "${SCREEN_DIR}/feature_names.txt" ) then
    echo "ERROR: missing feature names: ${SCREEN_DIR}/feature_names.txt"
    exit 1
endif
if ( -f "${HPO_OUT}/best_trials_leaf_1000.csv" ) then
    echo "Step 5: SKIP ${HPO_OUT}/best_trials_leaf_1000.csv exists"
    exit 0
endif

echo "[`date '+%H:%M:%S'`] ===== Step 5: HPO K=${TARGET_K} ====="
"$PYTHON" "${TIMEWINDOW_SRC}/step5_hpo_timewindow.py" \
    --train-buffer "${SCREEN_DIR}/train_top${TARGET_K}.buffer" \
    --test-buffer "${SCREEN_DIR}/test_top${TARGET_K}.buffer" \
    --feature-names "${SCREEN_DIR}/feature_names.txt" --output-dir "$HPO_OUT" \
    --round-num "$HPO_ROUNDS" --trial-num "$HPO_TRIALS" --pop-size "$HPO_POP" \
    --job-num "$HPO_JOB_NUM" --nfold "$HPO_NFOLD" --two-phase-method "$TWO_PHASE_METHOD" \
    --two-phase-cutoff "$TWO_PHASE_CUTOFF" --two-phase-ratio "$TWO_PHASE_RATIO" \
    --loss "$XGB_LOSS" --relative-floor "$XGB_RELATIVE_FLOOR" \
    --focus-power "$XGB_FOCUS_POWER" --closeness-metric "$HPO_CLOSENESS_METRIC" \
    --shuffle-seed "$SHUFFLE_SEED"
if ( $status != 0 ) exit 1
