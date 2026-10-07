#!/bin/csh -f
# Step 2: per-offset/type proxy feature selection.
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
if ( ! -f "$NAMEFILE" ) then
    echo "ERROR: NAMEFILE not found: ${NAMEFILE}"
    exit 1
endif
if ( -f "${STEP2_DIR}/mcp_per_offset_manifest.json" ) then
    echo "Step 2: SKIP ${STEP2_DIR}/mcp_per_offset_manifest.json exists"
    exit 0
endif

echo "[`date '+%H:%M:%S'`] ===== Step 2: MCP Per Offset ====="
"$PYTHON" "${TIMEWINDOW_SRC}/step2_mcp_per_offset.py" \
    --manifest "${STEP1_DIR}/timewindow_manifest.json" --namefile "$NAMEFILE" \
    --alpha "$ALPHA_MCP" --gamma "$GAMMA" --alpha-l2 "$ALPHA_L2" \
    --batch "$BATCH_SIZE" --top-k "$MCP_TOP_K" --jobs "$STEP2_JOBS" \
    --output-dir "$STEP2_DIR"
if ( $status != 0 ) exit 1
