#!/bin/csh -f
# Submit neural HPO/retrain (BNN/LSTM/LTC) to dsub, one model per job.
# Step5 and Step6 for the same model stay in one job: models therefore run
# their Step5 HPO in parallel, while Step6 starts only after its own HPO.
#
# Usage:
#   setenv OUTPUT_ROOT /path/to/existing/pipeline_run
#   csh cmd/run_timewindow_neural_hpo_dsub.csh <hw> <K>
#
# Optional overrides:
#   NEURAL_MODEL_TYPES=bnn,bnn_fp32,lstm,ltc  SKIP_NEURAL_HPO=0|1
#   NEURAL_DSUB_EXTRA_ARGS=( ... )  POLL_SEC=30  MAX_RERUN=10

set SCRIPT_DIR = `dirname "$0"`
set SCRIPT_DIR = `cd "$SCRIPT_DIR" && pwd`

if ( $#argv < 2 ) then
    echo "Usage: csh $0 <hw> <K>  (set OUTPUT_ROOT to an existing Step4 run)"
    exit 2
endif
if ( ! $?OUTPUT_ROOT ) then
    echo "ERROR: OUTPUT_ROOT must point to the existing pipeline run containing Step4 output"
    exit 2
endif

set HW = "$1"
set TARGET_K = "$2"
setenv RESOLVE_DOWNSTREAM_PATHS 1
source "${SCRIPT_DIR}/timewindow_common.csh"

if ( "$SKIP_NEURAL_HPO" != "0" && "$SKIP_NEURAL_HPO" != "1" ) then
    echo "ERROR: SKIP_NEURAL_HPO must be 0 (run HPO) or 1 (direct retrain)"
    exit 2
endif
alias neural_dsub_submit '${DSUB_CMD} -q ${DSUB_QUEUE} -oo \!:2 -eo \!:3 $NEURAL_DSUB_EXTRA_ARGS:q -s \!:1'

set SCREEN_DIR = "${STEP4_DIR}/union_K${TARGET_K}"
set TRAIN_BUFFER = "${SCREEN_DIR}/train_union_top${TARGET_K}.buffer"
set TEST_BUFFER = "${SCREEN_DIR}/test_union_top${TARGET_K}.buffer"
if ( ! -d "$SCREEN_DIR" ) then
    set SCREEN_DIR = "${STEP4_DIR}/screen_K${TARGET_K}"
    set TRAIN_BUFFER = "${SCREEN_DIR}/train_top${TARGET_K}.buffer"
    set TEST_BUFFER = "${SCREEN_DIR}/test_top${TARGET_K}.buffer"
endif
set FEATURE_NAMES = "${SCREEN_DIR}/feature_names.txt"
if ( ! -f "$TRAIN_BUFFER" || ! -f "$TEST_BUFFER" || ! -f "$FEATURE_NAMES" ) then
    echo "ERROR: missing Step4 buffers or feature_names under ${SCREEN_DIR}"
    exit 1
endif

set JOB_DIR = "${NEURAL_OUTPUT_ROOT}/jobs_dsub/neural_K${TARGET_K}"
set LOG_DIR = "${NEURAL_OUTPUT_ROOT}/logs_dsub/neural_K${TARGET_K}"
mkdir -p "$JOB_DIR" "$LOG_DIR"
set TYPES = ( `echo "$NEURAL_MODEL_TYPES" | tr ',' ' '` )
foreach MODEL_TYPE ( $TYPES )
    if ( "$MODEL_TYPE" != "bnn" && "$MODEL_TYPE" != "bnn_fp32" && "$MODEL_TYPE" != "lstm" && "$MODEL_TYPE" != "ltc" ) then
        echo "ERROR: unsupported neural model: ${MODEL_TYPE}"
        exit 2
    endif
end

foreach MODEL_TYPE ( $TYPES )
    if ( "$SKIP_NEURAL_HPO" == "1" ) then
        set MODEL_OUT = "${NEURAL_OUTPUT_ROOT}/step6_neural_retrain/union_K${TARGET_K}_${MODEL_TYPE}_direct"
    else
        set MODEL_OUT = "${NEURAL_OUTPUT_ROOT}/step6_neural_retrain/union_K${TARGET_K}_${MODEL_TYPE}"
    endif
    if ( -f "${MODEL_OUT}/complete" ) then
        echo "Neural ${MODEL_TYPE} K=${TARGET_K}: SKIP (already complete)"
        continue
    endif

    mkdir -p "$MODEL_OUT"
    rm -f "${MODEL_OUT}/fail" "${MODEL_OUT}/status"
    set JOB_FILE = "${JOB_DIR}/${MODEL_TYPE}_hpo_retrain.csh"
    set JOB_LOG = "${LOG_DIR}/${MODEL_TYPE}_hpo_retrain.log"
    set DSUB_OUT = "${LOG_DIR}/${MODEL_TYPE}_hpo_retrain.dsub.out"
    set DSUB_ERR = "${LOG_DIR}/${MODEL_TYPE}_hpo_retrain.dsub.err"
    cat > "$JOB_FILE" << EOF
#!/bin/csh -f
rm -f "${JOB_FILE}.running"
touch "${JOB_FILE}.running"
echo "[job] neural ${MODEL_TYPE} HPO/retrain K=${TARGET_K} start \`date '+%Y-%m-%d %H:%M:%S'\` host=\`hostname\`" >& "${JOB_LOG}"
setenv OUTPUT_ROOT "${OUTPUT_ROOT}"
setenv TIMEWINDOW_SRC "${TIMEWINDOW_SRC}"
setenv NEURAL_PYTHON "${NEURAL_PYTHON}"
setenv STAGE "${STAGE}"
setenv LEVEL_DIR "${LEVEL_DIR}"
setenv TOG_DIR "${TOG_DIR}"
setenv LABEL_DIR "${LABEL_DIR}"
setenv NAMEFILE "${NAMEFILE}"
setenv OFFSETS "${OFFSETS}"
setenv DATA_TYPES "${DATA_TYPES}"
setenv STEP3_OFFSETS "${STEP3_OFFSETS}"
setenv STEP3_DATA_TYPES "${STEP3_DATA_TYPES}"
setenv STEP3_COMBOS "${STEP3_COMBOS}"
setenv STEP2_SELECTOR_MODE "${STEP2_SELECTOR_MODE}"
setenv PROXY_SELECTOR "${PROXY_SELECTOR}"
setenv FEATURE_INPUT_MODE "${FEATURE_INPUT_MODE}"
setenv EXTRA_TAG "${EXTRA_TAG}"
setenv SHUFFLE_SEED "${SHUFFLE_SEED}"
setenv NEURAL_MODEL_TYPES "${MODEL_TYPE}"
setenv SKIP_NEURAL_HPO "${SKIP_NEURAL_HPO}"
setenv NEURAL_HPO_TRIALS "${NEURAL_HPO_TRIALS}"
setenv NEURAL_HPO_EPOCHS "${NEURAL_HPO_EPOCHS}"
setenv NEURAL_HPO_POP "${NEURAL_HPO_POP}"
setenv NEURAL_HPO_NFOLD "${NEURAL_HPO_NFOLD}"
setenv NEURAL_HPO_JOB_NUM "${NEURAL_HPO_JOB_NUM}"
setenv NEURAL_HPO_SAMPLER "${NEURAL_HPO_SAMPLER}"
setenv NEURAL_PATIENCE "${NEURAL_PATIENCE}"
setenv NEURAL_VALIDATION_RATIO "${NEURAL_VALIDATION_RATIO}"
setenv NEURAL_PARAMETER_THRESHOLD "${NEURAL_PARAMETER_THRESHOLD}"
setenv NEURAL_DEVICE "${NEURAL_DEVICE}"
setenv NEURAL_SEQUENCE_CACHE_MB "${NEURAL_SEQUENCE_CACHE_MB}"
setenv NEURAL_MIXED_PRECISION "${NEURAL_MIXED_PRECISION}"
setenv NEURAL_DIRECT_EPOCHS "${NEURAL_DIRECT_EPOCHS}"
setenv NEURAL_DIRECT_PATIENCE "${NEURAL_DIRECT_PATIENCE}"
setenv NEURAL_DIRECT_VALIDATION_RATIO "${NEURAL_DIRECT_VALIDATION_RATIO}"
setenv NEURAL_DIRECT_WEIGHT_DECAY "${NEURAL_DIRECT_WEIGHT_DECAY}"
setenv NEURAL_DIRECT_GRADIENT_CLIP "${NEURAL_DIRECT_GRADIENT_CLIP}"
setenv BNN_BATCH_SIZE "${BNN_BATCH_SIZE}"
setenv BNN_LEARNING_RATE "${BNN_LEARNING_RATE}"
setenv BNN_HIDDEN_SIZE "${BNN_HIDDEN_SIZE}"
setenv BNN_NUM_LAYERS "${BNN_NUM_LAYERS}"
setenv BNN_DROPOUT "${BNN_DROPOUT}"
setenv LSTM_BATCH_SIZE "${LSTM_BATCH_SIZE}"
setenv LSTM_LEARNING_RATE "${LSTM_LEARNING_RATE}"
setenv LSTM_HIDDEN_SIZE "${LSTM_HIDDEN_SIZE}"
setenv LSTM_NUM_LAYERS "${LSTM_NUM_LAYERS}"
setenv LSTM_DROPOUT "${LSTM_DROPOUT}"
setenv LTC_BATCH_SIZE "${LTC_BATCH_SIZE}"
setenv LTC_LEARNING_RATE "${LTC_LEARNING_RATE}"
setenv LTC_HIDDEN_SIZE "${LTC_HIDDEN_SIZE}"
setenv LTC_NUM_LAYERS "${LTC_NUM_LAYERS}"
setenv LTC_DROPOUT "${LTC_DROPOUT}"
csh "${SCRIPT_DIR}/run_timewindow_neural_hpo.csh" "${HW}" "${TARGET_K}" >>& "${JOB_LOG}"
set rc = \$status
echo \$rc > "${MODEL_OUT}/status"
if ( \$rc == 0 && -f "${MODEL_OUT}/complete" ) then
    rm -f "${MODEL_OUT}/fail"
else
    touch "${MODEL_OUT}/fail"
endif
rm -f "${JOB_FILE}.running"
echo "[job] neural ${MODEL_TYPE} HPO/retrain K=${TARGET_K} end rc=\$rc" >>& "${JOB_LOG}"
exit \$rc
EOF
    chmod +x "$JOB_FILE"
    echo "[`date '+%H:%M:%S'`] submit neural ${MODEL_TYPE} HPO/retrain K=${TARGET_K}"
    neural_dsub_submit "$JOB_FILE" "$DSUB_OUT" "$DSUB_ERR"
    if ( $status != 0 ) touch "${MODEL_OUT}/fail"
end

@ RERUN_COUNT = 0
while ( 1 )
    @ DONE_COUNT = 0
    @ FAIL_COUNT = 0
    foreach MODEL_TYPE ( $TYPES )
        if ( "$SKIP_NEURAL_HPO" == "1" ) then
            set MODEL_OUT = "${NEURAL_OUTPUT_ROOT}/step6_neural_retrain/union_K${TARGET_K}_${MODEL_TYPE}_direct"
        else
            set MODEL_OUT = "${NEURAL_OUTPUT_ROOT}/step6_neural_retrain/union_K${TARGET_K}_${MODEL_TYPE}"
        endif
        if ( -f "${MODEL_OUT}/complete" ) @ DONE_COUNT = $DONE_COUNT + 1
        if ( -f "${MODEL_OUT}/fail" ) @ FAIL_COUNT = $FAIL_COUNT + 1
    end
    if ( $FAIL_COUNT > 0 ) then
        if ( $RERUN_COUNT >= $MAX_RERUN ) then
            echo "ERROR: ${FAIL_COUNT} neural task(s) failed after ${MAX_RERUN} reruns"
            exit 1
        endif
        @ RERUN_COUNT = $RERUN_COUNT + 1
        echo "WARNING: rerun failed neural task(s) only (${RERUN_COUNT}/${MAX_RERUN})"
        foreach MODEL_TYPE ( $TYPES )
            if ( "$SKIP_NEURAL_HPO" == "1" ) then
                set MODEL_OUT = "${NEURAL_OUTPUT_ROOT}/step6_neural_retrain/union_K${TARGET_K}_${MODEL_TYPE}_direct"
            else
                set MODEL_OUT = "${NEURAL_OUTPUT_ROOT}/step6_neural_retrain/union_K${TARGET_K}_${MODEL_TYPE}"
            endif
            if ( -f "${MODEL_OUT}/fail" ) then
                set JOB_FILE = "${JOB_DIR}/${MODEL_TYPE}_hpo_retrain.csh"
                set DSUB_OUT = "${LOG_DIR}/${MODEL_TYPE}_hpo_retrain.dsub.out"
                set DSUB_ERR = "${LOG_DIR}/${MODEL_TYPE}_hpo_retrain.dsub.err"
                rm -f "${MODEL_OUT}/fail" "${MODEL_OUT}/status" "${JOB_FILE}.running"
                neural_dsub_submit "$JOB_FILE" "$DSUB_OUT" "$DSUB_ERR"
                if ( $status != 0 ) touch "${MODEL_OUT}/fail"
            endif
        end
        continue
    endif
    if ( $DONE_COUNT == $#TYPES ) break
    echo "[`date '+%H:%M:%S'`] neural progress: ${DONE_COUNT}/$#TYPES; sleep ${POLL_SEC}s"
    sleep $POLL_SEC
end

mkdir -p "${NEURAL_OUTPUT_ROOT}/step5_neural_hpo" "${NEURAL_OUTPUT_ROOT}/step6_neural_retrain"
touch "${NEURAL_OUTPUT_ROOT}/step5_neural_hpo/complete" "${NEURAL_OUTPUT_ROOT}/step6_neural_retrain/complete"
echo "[`date '+%H:%M:%S'`] Neural dsub pipeline done: ${NEURAL_OUTPUT_ROOT}"
