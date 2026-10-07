#!/bin/csh -f
# Step 6: retrain/evaluate one K after Step 5. Usage: csh run_timewindow_step6_retrain.csh <hw> <K>
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
    echo "ERROR: Step 6 requires K, e.g. csh $0 ${HW} 1024"
    exit 2
endif
source "${SCRIPT_DIR}/timewindow_common.csh"
mkdir -p "$OUTPUT_ROOT" "$LOG_DIR" "$STEP6_DIR"

set SCREEN_DIR = "${STEP4_DIR}/screen_K${TARGET_K}"
set HPO_OUT = "${STEP5_DIR}/screen_K${TARGET_K}_method${TWO_PHASE_METHOD}"
set BEST_CSV = "${HPO_OUT}/best_trials_leaf_1000.csv"
set RETRAIN_OUT = "${STEP6_DIR}/screen_K${TARGET_K}_method${TWO_PHASE_METHOD}"

if ( ! -f "${SCREEN_DIR}/train_top${TARGET_K}.buffer" ) then
    echo "ERROR: missing Step 4 train buffer: ${SCREEN_DIR}/train_top${TARGET_K}.buffer"
    exit 1
endif
if ( ! -f "${SCREEN_DIR}/test_top${TARGET_K}.buffer" ) then
    echo "ERROR: missing Step 4 test buffer: ${SCREEN_DIR}/test_top${TARGET_K}.buffer"
    exit 1
endif
if ( ! -f "$BEST_CSV" ) then
    echo "ERROR: missing Step 5 best trials CSV: ${BEST_CSV}"
    exit 1
endif
if ( -f "${RETRAIN_OUT}/eval/metrics.csv" && -f "${RETRAIN_OUT}/eval/full_unsampled/evaluation_summary.json" ) then
    echo "Step 6: SKIP ${RETRAIN_OUT}/eval/metrics.csv exists"
    exit 0
endif

echo "[`date '+%H:%M:%S'`] ===== Step 6: Retrain K=${TARGET_K} ====="
"$PYTHON" "${TIMEWINDOW_SRC}/step6_retrain_eval.py" \
    --train-buffer "${SCREEN_DIR}/train_top${TARGET_K}.buffer" \
    --test-buffer "${SCREEN_DIR}/test_top${TARGET_K}.buffer" \
    --hpo-csv "$BEST_CSV" --output-dir "$RETRAIN_OUT" --rounds "$RETRAIN_ROUNDS" \
    --tw-manifest "${STEP3_TW_MANIFEST}" --level-dir "$LEVEL_DIR" --tog-dir "$TOG_DIR" \
    --label-dir "$LABEL_DIR" --raw-namefile "$NAMEFILE" \
    --feature-names "${SCREEN_DIR}/feature_names.txt" --two-phase-method "$TWO_PHASE_METHOD" \
    --two-phase-cutoff "$TWO_PHASE_CUTOFF" --two-phase-ratio "$TWO_PHASE_RATIO" \
    --loss "$XGB_LOSS" --relative-floor "$XGB_RELATIVE_FLOOR" \
    --focus-power "$XGB_FOCUS_POWER" --shuffle-seed "$SHUFFLE_SEED"
if ( $status != 0 ) exit 1
