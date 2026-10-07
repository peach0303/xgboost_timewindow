#!/bin/csh -f
# dsub controller for split_after_shuffle. Step2 parallelism is delegated to
# run_timewindow_pipeline.csh prep; Step5/Step6 submit one job per K in parallel.
# By default, outputs are isolated under <source_output_root>/spilt_after_shuffle.
# Usage: csh $0 <source_step1_manifest> <namefile> [source_output_root] [K_csv]

set SCRIPT_DIR = `dirname "$0"`
set SCRIPT_DIR = `cd "$SCRIPT_DIR" && pwd`
if ( $#argv < 2 ) then
    echo "Usage: csh $0 <source_step1_manifest> <namefile> [source_output_root] [K_csv]"
    exit 2
endif
set SOURCE_MANIFEST = "$1"
setenv NAMEFILE "$2"
if ( ! -f "$SOURCE_MANIFEST" ) then
    echo "ERROR: missing source Step1 manifest: $SOURCE_MANIFEST"
    exit 1
endif
set SOURCE_STEP1_DIR = `dirname "$SOURCE_MANIFEST"`
set SOURCE_OUTPUT_ROOT = `dirname "$SOURCE_STEP1_DIR"`
if ( $#argv >= 3 ) set SOURCE_OUTPUT_ROOT = "$3"
if ( ! -d "$SOURCE_OUTPUT_ROOT" ) then
    echo "ERROR: missing source output root: $SOURCE_OUTPUT_ROOT"
    exit 1
endif
set SOURCE_OUTPUT_ROOT = `cd "$SOURCE_OUTPUT_ROOT" && pwd`
setenv OUTPUT_ROOT "${SOURCE_OUTPUT_ROOT}/spilt_after_shuffle"
# Never inherit an output-tag override: Step3 must keep tags such as
# tog_level_15_14 beneath the isolated experiment root.
if ( $?EXTRA_TAG ) unsetenv EXTRA_TAG
if ( ! $?HW ) set HW = boom
source "${SCRIPT_DIR}/timewindow_common.csh"
if ( "$SPLIT_STORAGE_MODE" != "index" && "$SPLIT_STORAGE_MODE" != "materialize" ) then
    echo "ERROR: SPLIT_STORAGE_MODE must be index or materialize"
    exit 2
endif
# The legacy validation side path reads physical train/test filelists directly
# and therefore cannot represent a virtual row-index split.
if ( "$SPLIT_STORAGE_MODE" == "index" ) setenv RUN_STEP2_VALIDATION 0

set TARGET_KS = "$TOP_KS"
if ( $#argv >= 4 ) set TARGET_KS = "$4"
set K_LIST = ( `echo "$TARGET_KS" | tr ',' ' '` )

set STEP1_DIR = "${OUTPUT_ROOT}/step1_shifted"
set EXP_JOB_DIR = "${OUTPUT_ROOT}/jobs_dsub/split_after_shuffle"
set EXP_LOG_DIR = "${OUTPUT_ROOT}/logs_dsub/split_after_shuffle"
mkdir -p "$STEP1_DIR" "$EXP_JOB_DIR" "$EXP_LOG_DIR"
alias split_submit '${DSUB_CMD} -q ${DSUB_QUEUE} -oo \!:2 -eo \!:3 $SPLIT_DSUB_EXTRA_ARGS:q -s \!:1'
alias step5_submit '${DSUB_CMD} -q ${DSUB_QUEUE} -oo \!:2 -eo \!:3 $SPLIT_STEP5_DSUB_EXTRA_ARGS:q -s \!:1'
alias step6_submit '${DSUB_CMD} -q ${DSUB_QUEUE} -oo \!:2 -eo \!:3 $SPLIT_STEP6_DSUB_EXTRA_ARGS:q -s \!:1'

# Step 1 resplit job.
set JOB_FILE = "${EXP_JOB_DIR}/resplit_after_shuffle.csh"
set JOB_LOG = "${EXP_LOG_DIR}/resplit_after_shuffle.log"
set SPLIT_ARGS = ( )
if ( "$SPLIT_TEST_RATIO" != "" ) set SPLIT_ARGS = ( --test-ratio "$SPLIT_TEST_RATIO" )
cat > "$JOB_FILE" << EOF
#!/bin/csh -f
rm -f "${STEP1_DIR}/running" "${STEP1_DIR}/complete" "${STEP1_DIR}/fail"
touch "${STEP1_DIR}/running"
"${PYTHON}" "${TIMEWINDOW_SRC}/resplit_after_shuffle.py" \
  --input-manifest "${SOURCE_MANIFEST}" --output-dir "${STEP1_DIR}" \
  --seed "${SHUFFLE_SEED}" --storage-mode "${SPLIT_STORAGE_MODE}" \
  --index-row-batch-size "${INDEX_ROW_BATCH_SIZE}" \
  --index-row-batch-size-tog "${INDEX_ROW_BATCH_SIZE_TOG}" \
  --index-row-batch-size-level "${INDEX_ROW_BATCH_SIZE_LEVEL}" \
  $SPLIT_ARGS:q --force >>& "${JOB_LOG}"
set rc = \$status
echo \$rc > "${STEP1_DIR}/status"
rm -f "${STEP1_DIR}/running"
if ( \$rc == 0 && -f "${STEP1_DIR}/timewindow_manifest.json" ) then
  touch "${STEP1_DIR}/complete"
  rm -f "${STEP1_DIR}/fail"
else
  touch "${STEP1_DIR}/fail"
endif
exit \$rc
EOF
chmod +x "$JOB_FILE"

@ RERUN_COUNT = 0
while ( ! -f "${STEP1_DIR}/complete" )
    rm -f "${STEP1_DIR}/fail" "${STEP1_DIR}/status"
    split_submit "$JOB_FILE" "${EXP_LOG_DIR}/resplit.dsub.out" "${EXP_LOG_DIR}/resplit.dsub.err"
    if ( $status != 0 ) touch "${STEP1_DIR}/fail"
    while ( ! -f "${STEP1_DIR}/complete" && ! -f "${STEP1_DIR}/fail" )
        sleep "$POLL_SEC"
    end
    if ( -f "${STEP1_DIR}/complete" ) break
    if ( $RERUN_COUNT >= $MAX_RERUN ) then
        echo "ERROR: resplit failed after ${MAX_RERUN} reruns"
        exit 1
    endif
    @ RERUN_COUNT = $RERUN_COUNT + 1
end

# The maintained prep controller submits Step2 selector×combo jobs in parallel,
# then submits the dependent Step3 and Step4 jobs.
setenv TOP_KS "$TARGET_KS"
csh "${SCRIPT_DIR}/run_timewindow_pipeline.csh" "$HW" prep
if ( $status != 0 ) exit 1

# Recompute the same normal Step3 tag used by the main pipeline. No override is
# supplied, so the data types and offsets remain visible in the directory name.
set OUTPUT_TAG = `"$PYTHON" "${TIMEWINDOW_SRC}/build_timewindow_output_tag.py" \
  --tw-manifest "${STEP1_DIR}/timewindow_manifest.json" \
  --offsets "$STEP3_OFFSETS" --data-types "$STEP3_DATA_TYPES" \
  --include-combos "$STEP3_COMBOS"`
if ( $status != 0 || "$OUTPUT_TAG" == "" ) then
    echo "ERROR: failed to resolve split_after_shuffle Step3 output tag"
    exit 1
endif
set TAG_OUTPUT_ROOT = "${OUTPUT_ROOT}/${OUTPUT_TAG}"
set STEP4_ROOT = "${TAG_OUTPUT_ROOT}/step4_hybrid_screen"
set STEP5_ROOT = "${TAG_OUTPUT_ROOT}/step5_hpo"
set STEP6_ROOT = "${TAG_OUTPUT_ROOT}/step6_retrain"

# Generate and submit all Step5 K jobs before waiting.
foreach K ( $K_LIST )
    set UNION_DIR = "${STEP4_ROOT}/union_K${K}"
    set HPO_DIR = "${STEP5_ROOT}/union_K${K}_method${TWO_PHASE_METHOD}"
    set JOB_FILE = "${EXP_JOB_DIR}/step5_K${K}.csh"
    set JOB_LOG = "${EXP_LOG_DIR}/step5_K${K}.log"
    mkdir -p "$HPO_DIR"
    cat > "$JOB_FILE" << EOF
#!/bin/csh -f
rm -f "${HPO_DIR}/complete" "${HPO_DIR}/fail"
touch "${HPO_DIR}/running"
"${PYTHON}" "${TIMEWINDOW_SRC}/step5_hpo_timewindow.py" \
  --train-buffer "${UNION_DIR}/train_union_top${K}.buffer" \
  --test-buffer "${UNION_DIR}/test_union_top${K}.buffer" \
  --feature-names "${UNION_DIR}/feature_names.txt" --output-dir "${HPO_DIR}" \
  --round-num "${HPO_ROUNDS}" --trial-num "${HPO_TRIALS}" --pop-size "${HPO_POP}" \
  --job-num "${HPO_JOB_NUM}" --nfold "${HPO_NFOLD}" \
  --two-phase-method "${TWO_PHASE_METHOD}" --two-phase-cutoff "${TWO_PHASE_CUTOFF}" \
  --two-phase-ratio "${TWO_PHASE_RATIO}" --loss "${XGB_LOSS}" \
  --relative-floor "${XGB_RELATIVE_FLOOR}" --focus-power "${XGB_FOCUS_POWER}" \
  --closeness-metric "${HPO_CLOSENESS_METRIC}" --shuffle-seed "${SHUFFLE_SEED}" >>& "${JOB_LOG}"
set rc = \$status
echo \$rc > "${HPO_DIR}/status"
rm -f "${HPO_DIR}/running"
if ( \$rc == 0 && -f "${HPO_DIR}/best_trials_leaf_1000.csv" ) then
  touch "${HPO_DIR}/complete"
else
  touch "${HPO_DIR}/fail"
endif
exit \$rc
EOF
    chmod +x "$JOB_FILE"
    if ( ! -f "${HPO_DIR}/complete" ) then
        step5_submit "$JOB_FILE" "${EXP_LOG_DIR}/step5_K${K}.dsub.out" "${EXP_LOG_DIR}/step5_K${K}.dsub.err"
        if ( $status != 0 ) touch "${HPO_DIR}/fail"
    endif
end

@ RERUN_COUNT = 0
while ( 1 )
    @ DONE_COUNT = 0
    @ FAIL_COUNT = 0
    foreach K ( $K_LIST )
        set HPO_DIR = "${STEP5_ROOT}/union_K${K}_method${TWO_PHASE_METHOD}"
        if ( -f "${HPO_DIR}/complete" ) @ DONE_COUNT = $DONE_COUNT + 1
        if ( -f "${HPO_DIR}/fail" ) @ FAIL_COUNT = $FAIL_COUNT + 1
    end
    if ( $DONE_COUNT == $#K_LIST ) break
    if ( $FAIL_COUNT > 0 ) then
        if ( $RERUN_COUNT >= $MAX_RERUN ) exit 1
        @ RERUN_COUNT = $RERUN_COUNT + 1
        foreach K ( $K_LIST )
            set HPO_DIR = "${STEP5_ROOT}/union_K${K}_method${TWO_PHASE_METHOD}"
            if ( -f "${HPO_DIR}/fail" ) then
                rm -f "${HPO_DIR}/fail" "${HPO_DIR}/status"
                step5_submit "${EXP_JOB_DIR}/step5_K${K}.csh" "${EXP_LOG_DIR}/step5_K${K}.dsub.out" "${EXP_LOG_DIR}/step5_K${K}.dsub.err"
                if ( $status != 0 ) touch "${HPO_DIR}/fail"
            endif
        end
    endif
    sleep "$POLL_SEC"
end

# Generate and submit all Step6 K jobs before waiting.
foreach K ( $K_LIST )
    set UNION_DIR = "${STEP4_ROOT}/union_K${K}"
    set HPO_DIR = "${STEP5_ROOT}/union_K${K}_method${TWO_PHASE_METHOD}"
    set RETRAIN_DIR = "${STEP6_ROOT}/union_K${K}_method${TWO_PHASE_METHOD}"
    set JOB_FILE = "${EXP_JOB_DIR}/step6_K${K}.csh"
    set JOB_LOG = "${EXP_LOG_DIR}/step6_K${K}.log"
    mkdir -p "$RETRAIN_DIR"
    cat > "$JOB_FILE" << EOF
#!/bin/csh -f
rm -f "${RETRAIN_DIR}/complete" "${RETRAIN_DIR}/fail"
touch "${RETRAIN_DIR}/running"
"${PYTHON}" "${TIMEWINDOW_SRC}/step6_retrain_eval.py" \
  --model-type xgboost --train-buffer "${UNION_DIR}/train_union_top${K}.buffer" \
  --test-buffer "${UNION_DIR}/test_union_top${K}.buffer" \
  --feature-names "${UNION_DIR}/feature_names.txt" \
  --hpo-csv "${HPO_DIR}/best_trials_leaf_1000.csv" --output-dir "${RETRAIN_DIR}" \
  --rounds "${RETRAIN_ROUNDS}" --two-phase-method "${TWO_PHASE_METHOD}" \
  --two-phase-cutoff "${TWO_PHASE_CUTOFF}" --two-phase-ratio "${TWO_PHASE_RATIO}" \
  --loss "${XGB_LOSS}" --relative-floor "${XGB_RELATIVE_FLOOR}" \
  --focus-power "${XGB_FOCUS_POWER}" --shuffle-seed "${SHUFFLE_SEED}" \
  --skip-full-eval >>& "${JOB_LOG}"
set rc = \$status
echo \$rc > "${RETRAIN_DIR}/status"
rm -f "${RETRAIN_DIR}/running"
if ( \$rc == 0 && -f "${RETRAIN_DIR}/eval/metrics.csv" ) then
  touch "${RETRAIN_DIR}/complete"
else
  touch "${RETRAIN_DIR}/fail"
endif
exit \$rc
EOF
    chmod +x "$JOB_FILE"
    if ( ! -f "${RETRAIN_DIR}/complete" ) then
        step6_submit "$JOB_FILE" "${EXP_LOG_DIR}/step6_K${K}.dsub.out" "${EXP_LOG_DIR}/step6_K${K}.dsub.err"
        if ( $status != 0 ) touch "${RETRAIN_DIR}/fail"
    endif
end

@ RERUN_COUNT = 0
while ( 1 )
    @ DONE_COUNT = 0
    @ FAIL_COUNT = 0
    foreach K ( $K_LIST )
        set RETRAIN_DIR = "${STEP6_ROOT}/union_K${K}_method${TWO_PHASE_METHOD}"
        if ( -f "${RETRAIN_DIR}/complete" ) @ DONE_COUNT = $DONE_COUNT + 1
        if ( -f "${RETRAIN_DIR}/fail" ) @ FAIL_COUNT = $FAIL_COUNT + 1
    end
    if ( $DONE_COUNT == $#K_LIST ) break
    if ( $FAIL_COUNT > 0 ) then
        if ( $RERUN_COUNT >= $MAX_RERUN ) exit 1
        @ RERUN_COUNT = $RERUN_COUNT + 1
        foreach K ( $K_LIST )
            set RETRAIN_DIR = "${STEP6_ROOT}/union_K${K}_method${TWO_PHASE_METHOD}"
            if ( -f "${RETRAIN_DIR}/fail" ) then
                rm -f "${RETRAIN_DIR}/fail" "${RETRAIN_DIR}/status"
                step6_submit "${EXP_JOB_DIR}/step6_K${K}.csh" "${EXP_LOG_DIR}/step6_K${K}.dsub.out" "${EXP_LOG_DIR}/step6_K${K}.dsub.err"
                if ( $status != 0 ) touch "${RETRAIN_DIR}/fail"
            endif
        end
    endif
    sleep "$POLL_SEC"
end

touch "${TAG_OUTPUT_ROOT}/complete"
echo "split_after_shuffle dsub pipeline complete: ${TAG_OUTPUT_ROOT}"
