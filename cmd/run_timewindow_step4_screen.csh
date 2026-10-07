#!/bin/csh -f
# Step 4: train default model once and save top-K screened buffers.
set SCRIPT_DIR = `dirname "$0"`
set SCRIPT_DIR = `cd "$SCRIPT_DIR" && pwd`
if ( $#argv >= 1 ) then
    set HW = "$1"
else
    set HW = boom
endif
source "${SCRIPT_DIR}/timewindow_common.csh"
mkdir -p "$OUTPUT_ROOT" "$LOG_DIR" "$STEP4_DIR"

if ( ! -f "${STEP3_DIR}/train_tw.buffer" ) then
    echo "ERROR: missing Step 3 train buffer: ${STEP3_DIR}/train_tw.buffer"
    exit 1
endif
if ( ! -f "${STEP3_DIR}/test_tw.buffer" ) then
    echo "ERROR: missing Step 3 test buffer: ${STEP3_DIR}/test_tw.buffer"
    exit 1
endif
if ( ! -f "${STEP3_DIR}/concat_meta.json" ) then
    echo "ERROR: missing Step 3 meta: ${STEP3_DIR}/concat_meta.json"
    exit 1
endif

set first_k = `echo "$TOP_KS" | awk -F, '{print $1}'`
if ( -f "${STEP4_DIR}/screen_K${first_k}/train_top${first_k}.buffer" ) then
    echo "Step 4: SKIP ${STEP4_DIR}/screen_K${first_k}/train_top${first_k}.buffer exists"
    exit 0
endif

if ( "${STEP3_TW_MANIFEST}" != "${STEP1_DIR}/timewindow_manifest.json" && ! -f "${STEP3_TW_MANIFEST}" ) then
    mkdir -p "${STEP3_INPUT_DIR}"
    "$PYTHON" "${TIMEWINDOW_SRC}/filter_timewindow_manifests.py" \
        --tw-manifest-in "${STEP1_DIR}/timewindow_manifest.json" \
        --mcp-manifest-in "${STEP2_DIR}/mcp_per_offset_manifest.json" \
        --tw-manifest-out "${STEP3_TW_MANIFEST}" \
        --mcp-manifest-out "${STEP3_MCP_MANIFEST}" \
        --offsets "$STEP3_OFFSETS" --data-types "$STEP3_DATA_TYPES" \
        --include-combos "$STEP3_COMBOS"
    if ( $status != 0 ) exit 1
endif

echo "[`date '+%H:%M:%S'`] ===== Step 4: Feature Screening ====="
"$PYTHON" "${TIMEWINDOW_SRC}/step4_screen_features.py" \
    --train-buffer "${STEP3_DIR}/train_tw.buffer" --test-buffer "${STEP3_DIR}/test_tw.buffer" \
    --concat-meta "${STEP3_DIR}/concat_meta.json" --top-ks "$TOP_KS" \
    --tw-manifest "${STEP3_TW_MANIFEST}" --namefile "$NAMEFILE" \
    --default-rounds "$DEFAULT_ROUNDS" --screen-eval-rounds "$SCREEN_EVAL_ROUNDS" \
    --loss "$XGB_LOSS" --relative-floor "$XGB_RELATIVE_FLOOR" \
    --focus-power "$XGB_FOCUS_POWER" --shuffle-seed "$SHUFFLE_SEED" \
    --output-dir "$STEP4_DIR"
if ( $status != 0 ) exit 1
