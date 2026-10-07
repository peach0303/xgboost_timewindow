#!/bin/csh -f
#===============================================================================
# Parallel wrapper for Time-Window Power Modeling Pipeline.
#
# Usage:
#   csh xgboost_timewindow/cmd/run_timewindow_pipeline_parallel.csh <hardware_name>
#
# It runs shared prep steps (Step 1-4) once, then launches one background
# Step5+Step6 worker per K. Override TOP_KS/OUTPUT_ROOT/RUN_TAG via setenv.
#===============================================================================

set SCRIPT_DIR = `dirname "$0"`
set SCRIPT_DIR = `cd "$SCRIPT_DIR" && pwd`
if ( ! $?PACKAGE_ROOT ) set PACKAGE_ROOT = `cd "${SCRIPT_DIR}/.." && pwd`
if ( ! $?PROJECT_ROOT ) set PROJECT_ROOT = `cd "${PACKAGE_ROOT}/.." && pwd`

if ( $#argv >= 1 ) then
    set HW = "$1"
else
    set HW = boom
endif

if ( ! $?STAGE ) set STAGE = after_one
if ( ! $?OUTPUT_BASE ) set OUTPUT_BASE = "${PROJECT_ROOT}/output/xgboost_timewindow_train"
if ( ! $?RUN_TAG ) setenv RUN_TAG `date '+%Y%m%d_%H%M%S'`
if ( ! $?OUTPUT_ROOT ) setenv OUTPUT_ROOT "${OUTPUT_BASE}/${HW}_${STAGE}_${RUN_TAG}"
if ( ! $?TOP_KS ) setenv TOP_KS "256,512,1024,2048,4096"
if ( ! $?PARALLEL_TOP_KS ) set PARALLEL_TOP_KS = "$TOP_KS"

set PIPELINE = "${SCRIPT_DIR}/run_timewindow_pipeline.csh"
set LOG_DIR = "${OUTPUT_ROOT}/logs_parallel"
mkdir -p "$OUTPUT_ROOT" "$LOG_DIR"

set k_list = ( `echo "$PARALLEL_TOP_KS" | tr ',' ' '` )

echo "============================================================"
echo "Time-Window Parallel Training Submit"
echo "  HW          = ${HW}"
echo "  OUTPUT_ROOT = ${OUTPUT_ROOT}"
echo "  TOP_KS      = ${TOP_KS}"
echo "  PARALLEL_KS = ${PARALLEL_TOP_KS}"
echo "  LOG_DIR     = ${LOG_DIR}"
echo "============================================================"

echo "[`date '+%H:%M:%S'`] ===== Shared prep: Step 1-4 ====="
csh "$PIPELINE" "$HW" prep >& "${LOG_DIR}/prep.log"
if ( $status != 0 ) then
    echo "ERROR: prep failed, see ${LOG_DIR}/prep.log"
    exit 1
endif

echo "[`date '+%H:%M:%S'`] ===== Submit parallel K workers ====="
foreach k ( $k_list )
    set log_file = "${LOG_DIR}/K${k}.log"
    set status_file = "${LOG_DIR}/K${k}.status"
    rm -f "$status_file"
    echo "  submit K=${k}, log=${log_file}"
    sh -c 'csh "$1" "$2" train_k "$3" > "$4" 2>&1; echo $? > "$5"' worker "$PIPELINE" "$HW" "$k" "$log_file" "$status_file" &
end

wait

set failed = 0
foreach k ( $k_list )
    set status_file = "${LOG_DIR}/K${k}.status"
    if ( ! -f "$status_file" ) then
        echo "ERROR: K=${k} did not write status file"
        set failed = 1
        continue
    endif
    set code = `cat "$status_file"`
    if ( "$code" != "0" ) then
        echo "ERROR: K=${k} failed with exit=${code}; see ${LOG_DIR}/K${k}.log"
        set failed = 1
    else
        echo "OK: K=${k}"
    endif
end

if ( $failed != 0 ) then
    echo "[`date '+%H:%M:%S'`] PARALLEL RUN FAILED. Logs under ${LOG_DIR}"
    exit 1
endif

echo "[`date '+%H:%M:%S'`] ALL DONE. Results under ${OUTPUT_ROOT}"
