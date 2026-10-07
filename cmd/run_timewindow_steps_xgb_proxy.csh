#!/bin/csh -f
#===============================================================================
# Time-window step-runner orchestration using XGBoost proxy selection in Step 2.
# Step 3 and later runners are unchanged; a step2_mcp compatibility symlink is
# created because existing downstream scripts read step2_mcp/mcp_per_offset_manifest.json.
#
# Usage:
#   csh xgboost_timewindow/cmd/run_timewindow_steps_xgb_proxy.csh <hardware_name>
#
# Useful overrides:
#   setenv XGB_PROXY_TOP_K 512
#   setenv XGB_PROXY_ROUNDS 80
#   setenv BATCH_SIZE 999999
#   setenv TOP_KS "256,512,1024"
#===============================================================================

set SCRIPT_DIR = `dirname "$0"`
set SCRIPT_DIR = `cd "$SCRIPT_DIR" && pwd`
if ( $#argv >= 1 ) then
    set HW = "$1"
else
    set HW = boom
endif

if ( ! $?STAGE ) set STAGE = after_one
if ( ! $?PACKAGE_ROOT ) set PACKAGE_ROOT = `cd "${SCRIPT_DIR}/.." && pwd`
if ( ! $?PROJECT_ROOT ) set PROJECT_ROOT = `cd "${PACKAGE_ROOT}/.." && pwd`
if ( ! $?OUTPUT_BASE ) set OUTPUT_BASE = "${PROJECT_ROOT}/output/xgboost_timewindow_train"
if ( ! $?RUN_TAG ) setenv RUN_TAG `date '+xgb_proxy_%Y%m%d_%H%M%S'`
if ( ! $?OUTPUT_ROOT ) setenv OUTPUT_ROOT "${OUTPUT_BASE}/${HW}_${STAGE}_${RUN_TAG}"
if ( ! $?TOP_KS ) setenv TOP_KS "256,512,1024,2048,4096"
if ( ! $?PARALLEL_TOP_KS ) set PARALLEL_TOP_KS = "$TOP_KS"

set LOG_DIR = "${OUTPUT_ROOT}/logs_steps"
mkdir -p "$OUTPUT_ROOT" "$LOG_DIR"
set k_list = ( `echo "$PARALLEL_TOP_KS" | tr ',' ' '` )

echo "============================================================"
echo "Time-Window XGBoost Proxy Step Runner"
echo "  HW          = ${HW}"
echo "  OUTPUT_ROOT = ${OUTPUT_ROOT}"
echo "  TOP_KS      = ${TOP_KS}"
echo "  PARALLEL_KS = ${PARALLEL_TOP_KS}"
echo "  LOG_DIR     = ${LOG_DIR}"
echo "============================================================"

csh "${SCRIPT_DIR}/run_timewindow_step1_prepare.csh" "$HW" >& "${LOG_DIR}/01_step1_prepare.log"
if ( $status != 0 ) exit 1

csh "${SCRIPT_DIR}/run_timewindow_step2_xgb.csh" "$HW" >& "${LOG_DIR}/02_step2_xgb.log"
if ( $status != 0 ) exit 1

csh "${SCRIPT_DIR}/run_timewindow_step2_validation.csh" "$HW" "${OUTPUT_ROOT}/step2_xgb/mcp_per_offset_manifest.json" >& "${LOG_DIR}/02b_step2_validation.log"
if ( $status != 0 ) exit 1

if ( ! -e "${OUTPUT_ROOT}/step2_mcp" ) then
    ln -s "${OUTPUT_ROOT}/step2_xgb" "${OUTPUT_ROOT}/step2_mcp"
endif

csh "${SCRIPT_DIR}/run_timewindow_step3_concat.csh" "$HW" >& "${LOG_DIR}/03_step3_concat.log"
if ( $status != 0 ) exit 1

csh "${SCRIPT_DIR}/run_timewindow_step4_screen.csh" "$HW" >& "${LOG_DIR}/04_step4_screen.log"
if ( $status != 0 ) exit 1

echo "[`date '+%H:%M:%S'`] ===== Submit Step 5 HPO workers ====="
foreach k ( $k_list )
    set log_file = "${LOG_DIR}/05_step5_hpo_K${k}.log"
    set status_file = "${LOG_DIR}/05_step5_hpo_K${k}.status"
    rm -f "$status_file"
    echo "  submit Step 5 K=${k}"
    sh -c 'csh "$1" "$2" "$3" > "$4" 2>&1; echo $? > "$5"' worker "${SCRIPT_DIR}/run_timewindow_step5_hpo.csh" "$HW" "$k" "$log_file" "$status_file" &
end
wait

set failed = 0
foreach k ( $k_list )
    set status_file = "${LOG_DIR}/05_step5_hpo_K${k}.status"
    if ( ! -f "$status_file" ) then
        echo "ERROR: Step 5 K=${k} did not write status"
        set failed = 1
        continue
    endif
    set code = `cat "$status_file"`
    if ( "$code" != "0" ) then
        echo "ERROR: Step 5 K=${k} failed; see ${LOG_DIR}/05_step5_hpo_K${k}.log"
        set failed = 1
    endif
end
if ( $failed != 0 ) exit 1

echo "[`date '+%H:%M:%S'`] ===== Submit Step 6 retrain workers ====="
foreach k ( $k_list )
    set log_file = "${LOG_DIR}/06_step6_retrain_K${k}.log"
    set status_file = "${LOG_DIR}/06_step6_retrain_K${k}.status"
    rm -f "$status_file"
    echo "  submit Step 6 K=${k}"
    sh -c 'csh "$1" "$2" "$3" > "$4" 2>&1; echo $? > "$5"' worker "${SCRIPT_DIR}/run_timewindow_step6_retrain.csh" "$HW" "$k" "$log_file" "$status_file" &
end
wait

set failed = 0
foreach k ( $k_list )
    set status_file = "${LOG_DIR}/06_step6_retrain_K${k}.status"
    if ( ! -f "$status_file" ) then
        echo "ERROR: Step 6 K=${k} did not write status"
        set failed = 1
        continue
    endif
    set code = `cat "$status_file"`
    if ( "$code" != "0" ) then
        echo "ERROR: Step 6 K=${k} failed; see ${LOG_DIR}/06_step6_retrain_K${k}.log"
        set failed = 1
    endif
end
if ( $failed != 0 ) exit 1

echo "[`date '+%H:%M:%S'`] ALL DONE. Results under ${OUTPUT_ROOT}"
