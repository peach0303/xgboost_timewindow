#!/bin/csh -f
# Step 3: concatenate selected time-window features into train/test buffers.
set SCRIPT_DIR = `dirname "$0"`
set SCRIPT_DIR = `cd "$SCRIPT_DIR" && pwd`
if ( $#argv >= 1 ) then
    set HW = "$1"
else
    set HW = boom
endif
source "${SCRIPT_DIR}/timewindow_common.csh"
mkdir -p "$OUTPUT_ROOT" "$LOG_DIR"

if ( ! -f "${STEP1_DIR}/timewindow_manifest.json" ) then
    echo "ERROR: missing Step 1 manifest: ${STEP1_DIR}/timewindow_manifest.json"
    exit 1
endif
if ( ! -f "${STEP2_DIR}/mcp_per_offset_manifest.json" ) then
    echo "ERROR: missing Step 2 manifest: ${STEP2_DIR}/mcp_per_offset_manifest.json"
    exit 1
endif
if ( -f "${STEP3_DIR}/train_tw.buffer" ) then
    echo "Step 3: SKIP ${STEP3_DIR}/train_tw.buffer exists"
    exit 0
endif

if ( "${STEP3_TW_MANIFEST}" != "${STEP1_DIR}/timewindow_manifest.json" ) then
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

echo "[`date '+%H:%M:%S'`] ===== Step 3: Feature Concat ====="
"$PYTHON" "${TIMEWINDOW_SRC}/step3_concat_features.py" \
    --tw-manifest "${STEP3_TW_MANIFEST}" \
    --mcp-manifest "${STEP3_MCP_MANIFEST}" \
    --namefile "$NAMEFILE" --shuffle-seed "$SHUFFLE_SEED" --output-dir "$STEP3_DIR"
if ( $status != 0 ) exit 1
