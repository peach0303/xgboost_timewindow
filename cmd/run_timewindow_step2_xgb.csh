#!/bin/csh -f
# Step 2: per-offset/type XGBoost proxy feature selection.
# Usage:
#   csh xgboost_timewindow/cmd/run_timewindow_step2_xgb.csh <hardware_name>
set SCRIPT_DIR = `dirname "$0"`
set SCRIPT_DIR = `cd "$SCRIPT_DIR" && pwd`
if ( $#argv >= 1 ) then
    set HW = "$1"
else
    set HW = boom
endif

source "${SCRIPT_DIR}/timewindow_common.csh"

if ( ! -f "${STEP1_DIR}/timewindow_manifest.json" ) then
    echo "ERROR: missing Step 1 manifest: ${STEP1_DIR}/timewindow_manifest.json"
    exit 1
endif

if ( ! $?XGB_PROXY_TOP_K ) set XGB_PROXY_TOP_K = 200
if ( ! $?XGB_PROXY_ROUNDS ) set XGB_PROXY_ROUNDS = 50
if ( ! $?XGB_PROXY_MAX_DEPTH ) set XGB_PROXY_MAX_DEPTH = 6
if ( ! $?XGB_PROXY_ETA ) set XGB_PROXY_ETA = 0.1
if ( ! $?XGB_PROXY_SUBSAMPLE ) set XGB_PROXY_SUBSAMPLE = 0.9
if ( ! $?XGB_PROXY_COLSAMPLE ) set XGB_PROXY_COLSAMPLE = 0.8
if ( ! $?XGB_PROXY_NTHREAD ) set XGB_PROXY_NTHREAD = 0

set STEP2_XGB_DIR = "${OUTPUT_ROOT}/step2_xgb"
mkdir -p "$STEP2_XGB_DIR"

echo "[`date '+%H:%M:%S'`] ===== Step 2: XGBoost Proxy Per Offset ====="
"$PYTHON" "${TIMEWINDOW_SRC}/step2_xgb_per_offset.py" \
    --manifest "${STEP1_DIR}/timewindow_manifest.json" --namefile "$NAMEFILE" \
    --top-k "$XGB_PROXY_TOP_K" \
    --rounds "$XGB_PROXY_ROUNDS" --max-depth "$XGB_PROXY_MAX_DEPTH" \
    --eta "$XGB_PROXY_ETA" --subsample "$XGB_PROXY_SUBSAMPLE" \
    --colsample-bytree "$XGB_PROXY_COLSAMPLE" \
    --nthread "$XGB_PROXY_NTHREAD" \
    --jobs "$STEP2_JOBS" --output-dir "$STEP2_XGB_DIR"
if ( $status != 0 ) exit 1

echo "Done: ${STEP2_XGB_DIR}/mcp_per_offset_manifest.json"
