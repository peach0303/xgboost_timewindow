#!/bin/csh -f
# Post-Step2 validation using copied/adapted legacy scripts. Step2 combo proxies
# form the dataset; legacy model/training/evaluation defaults remain unchanged.
set SCRIPT_DIR = `dirname "$0"`
set SCRIPT_DIR = `cd "$SCRIPT_DIR" && pwd`
if ( $#argv >= 1 ) then
    set HW = "$1"
else
    set HW = boom
endif

source "${SCRIPT_DIR}/timewindow_common.csh"
set PROXY_MANIFEST = "${STEP2_DIR}/mcp_per_offset_manifest.json"
if ( $#argv >= 2 ) set PROXY_MANIFEST = "$2"
set VALIDATION_PY = "${TIMEWINDOW_SRC}/step2_validation/xgb_power_v2_3.py"
if ( ! $?RUN_STEP2_VALIDATION ) set RUN_STEP2_VALIDATION = 1
if ( ${RUN_STEP2_VALIDATION} == 0 ) then
    echo "Step 2 validation: disabled by RUN_STEP2_VALIDATION=0"
    exit 0
endif

set VALIDATION_DIR = "${OUTPUT_ROOT}/step2_validation_legacy_xgb"
set SUMMARY = "${VALIDATION_DIR}/legacy_xgb_validation/tile_reset_domain-bc6/eval/evaluation_summary.json"
if ( -f "$SUMMARY" ) then
    echo "Step 2 validation: SKIP $SUMMARY exists"
    exit 0
endif
if ( ! -f "${STEP1_DIR}/timewindow_manifest.json" ) then
    echo "ERROR: missing Step 1 manifest: ${STEP1_DIR}/timewindow_manifest.json"
    exit 1
endif
if ( ! -f "$PROXY_MANIFEST" ) then
    echo "ERROR: missing Step 2 proxy manifest: $PROXY_MANIFEST"
    exit 1
endif
if ( ! -f "$VALIDATION_PY" ) then
    echo "ERROR: missing copied validation trainer: $VALIDATION_PY"
    exit 1
endif

mkdir -p "$VALIDATION_DIR"
echo "[`date '+%H:%M:%S'`] ===== Post-Step2 Legacy XGBoost Validation ====="
echo "Validation combos: $STEP2_VALIDATION_COMBOS"
cd "$VALIDATION_DIR"
"$PYTHON" "$VALIDATION_PY" \
    --tw-manifest "${STEP1_DIR}/timewindow_manifest.json" \
    --proxy-manifest "$PROXY_MANIFEST" --namefile "$NAMEFILE" \
    --combos "$STEP2_VALIDATION_COMBOS" \
    --cache-dir "${VALIDATION_DIR}/dataset_cache" \
    --build-block-size "${STEP2_VALIDATION_CHUNK_SIZE}" \
    --hpotag legacy_xgb_validation
if ( $status != 0 ) exit 1
if ( ! -f "$SUMMARY" ) then
    echo "ERROR: legacy validation did not produce $SUMMARY"
    exit 1
endif
echo "Done: $SUMMARY"
