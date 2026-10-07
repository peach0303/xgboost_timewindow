#!/bin/csh -f
# dsub controller for source-window signal -> target-window feature comparison.
# Usage: csh $0 TARGET_TW BASELINE_NAMES SOURCE_PROXIES BASELINE_RESULT OUTPUT_ROOT

set SCRIPT_DIR = `dirname "$0"`
set SCRIPT_DIR = `cd "$SCRIPT_DIR" && pwd`
if ( $#argv < 5 ) then
    echo "Usage: csh $0 <target_tw_manifest> <baseline_feature_names> <source_proxy_features> <baseline_step6> <output_root>"
    exit 2
endif
set TARGET_TW_MANIFEST = "$1"
set BASELINE_FEATURE_NAMES = "$2"
set SOURCE_PROXY_FEATURES = "$3"
set BASELINE_RESULT = "$4"
setenv OUTPUT_ROOT "$5"
if ( ! $?HW ) set HW = boom
source "${SCRIPT_DIR}/timewindow_common.csh"

foreach required ( "$TARGET_TW_MANIFEST" "$BASELINE_FEATURE_NAMES" "$SOURCE_PROXY_FEATURES" )
    if ( ! -f "$required" ) then
        echo "ERROR: missing input: $required"
        exit 1
    endif
end
if ( ! -d "$BASELINE_RESULT" ) then
    echo "ERROR: missing baseline Step6 directory: $BASELINE_RESULT"
    exit 1
endif
set CROSS_ROOT = "${OUTPUT_ROOT}/cross_window_signal_transfer"
set FEATURE_DIR = "${CROSS_ROOT}/features"
set HPO_DIR = "${CROSS_ROOT}/step5_hpo"
set RETRAIN_DIR = "${CROSS_ROOT}/step6_retrain"
set COMPARE_DIR = "${CROSS_ROOT}/comparison"
set JOB_DIR = "${CROSS_ROOT}/jobs_dsub"
set LOG_DIR = "${CROSS_ROOT}/logs_dsub"
mkdir -p "$FEATURE_DIR" "$HPO_DIR" "$RETRAIN_DIR" "$COMPARE_DIR" "$JOB_DIR" "$LOG_DIR"
alias cross_submit '${DSUB_CMD} -q ${DSUB_QUEUE} -oo \!:2 -eo \!:3 $CROSS_DSUB_EXTRA_ARGS:q -s \!:1'
alias cross_hpo_submit '${DSUB_CMD} -q ${DSUB_QUEUE} -oo \!:2 -eo \!:3 $CROSS_HPO_DSUB_EXTRA_ARGS:q -s \!:1'
alias cross_retrain_submit '${DSUB_CMD} -q ${DSUB_QUEUE} -oo \!:2 -eo \!:3 $CROSS_RETRAIN_DSUB_EXTRA_ARGS:q -s \!:1'

set BUILD_JOB = "${JOB_DIR}/build_features.csh"
cat > "$BUILD_JOB" << EOF
#!/bin/csh -f
rm -f "${FEATURE_DIR}/complete" "${FEATURE_DIR}/fail"
touch "${FEATURE_DIR}/running"
"${PYTHON}" "${TIMEWINDOW_SRC}/build_cross_window_features.py" \
  --target-tw-manifest "${TARGET_TW_MANIFEST}" --namefile "${NAMEFILE}" \
  --baseline-feature-names "${BASELINE_FEATURE_NAMES}" \
  --source-proxy-features "${SOURCE_PROXY_FEATURES}" \
  --source-top-signals "${CROSS_SOURCE_TOP_SIGNALS}" \
  --source-offset-min "${CROSS_SOURCE_OFFSET_MIN}" \
  --source-offset-max "${CROSS_SOURCE_OFFSET_MAX}" \
  --output-dir "${FEATURE_DIR}" --force >>& "${LOG_DIR}/build_features.log"
set rc = \$status
echo \$rc > "${FEATURE_DIR}/status"
rm -f "${FEATURE_DIR}/running"
if ( \$rc == 0 && -f "${FEATURE_DIR}/cross_window_union_meta.json" ) then
  touch "${FEATURE_DIR}/complete"
else
  touch "${FEATURE_DIR}/fail"
endif
exit \$rc
EOF

set HPO_JOB = "${JOB_DIR}/step5_hpo.csh"
cat > "$HPO_JOB" << EOF
#!/bin/csh -f
rm -f "${HPO_DIR}/complete" "${HPO_DIR}/fail"
touch "${HPO_DIR}/running"
"${PYTHON}" "${TIMEWINDOW_SRC}/step5_hpo_timewindow.py" \
  --train-buffer "${FEATURE_DIR}/train_cross_window.buffer" \
  --test-buffer "${FEATURE_DIR}/test_cross_window.buffer" \
  --feature-names "${FEATURE_DIR}/feature_names.txt" --output-dir "${HPO_DIR}" \
  --round-num "${HPO_ROUNDS}" --trial-num "${HPO_TRIALS}" --pop-size "${HPO_POP}" \
  --job-num "${HPO_JOB_NUM}" --nfold "${HPO_NFOLD}" \
  --two-phase-method "${TWO_PHASE_METHOD}" --two-phase-cutoff "${TWO_PHASE_CUTOFF}" \
  --two-phase-ratio "${TWO_PHASE_RATIO}" --loss "${XGB_LOSS}" \
  --relative-floor "${XGB_RELATIVE_FLOOR}" --focus-power "${XGB_FOCUS_POWER}" \
  --closeness-metric "${HPO_CLOSENESS_METRIC}" --shuffle-seed "${SHUFFLE_SEED}" >>& "${LOG_DIR}/step5_hpo.log"
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

set RETRAIN_JOB = "${JOB_DIR}/step6_retrain.csh"
cat > "$RETRAIN_JOB" << EOF
#!/bin/csh -f
rm -f "${RETRAIN_DIR}/complete" "${RETRAIN_DIR}/fail"
touch "${RETRAIN_DIR}/running"
"${PYTHON}" "${TIMEWINDOW_SRC}/step6_retrain_eval.py" \
  --model-type xgboost --train-buffer "${FEATURE_DIR}/train_cross_window.buffer" \
  --test-buffer "${FEATURE_DIR}/test_cross_window.buffer" \
  --feature-names "${FEATURE_DIR}/feature_names.txt" \
  --hpo-csv "${HPO_DIR}/best_trials_leaf_1000.csv" --output-dir "${RETRAIN_DIR}" \
  --rounds "${RETRAIN_ROUNDS}" --two-phase-method "${TWO_PHASE_METHOD}" \
  --two-phase-cutoff "${TWO_PHASE_CUTOFF}" --two-phase-ratio "${TWO_PHASE_RATIO}" \
  --loss "${XGB_LOSS}" --relative-floor "${XGB_RELATIVE_FLOOR}" \
  --focus-power "${XGB_FOCUS_POWER}" --shuffle-seed "${SHUFFLE_SEED}" \
  --skip-full-eval >>& "${LOG_DIR}/step6_retrain.log"
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

set COMPARE_JOB = "${JOB_DIR}/compare_results.csh"
cat > "$COMPARE_JOB" << EOF
#!/bin/csh -f
rm -f "${COMPARE_DIR}/complete" "${COMPARE_DIR}/fail"
touch "${COMPARE_DIR}/running"
"${PYTHON}" "${TIMEWINDOW_SRC}/compare_experiment_metrics.py" \
  --baseline "${BASELINE_RESULT}" --candidate "${RETRAIN_DIR}" \
  --baseline-name target_only --candidate-name add_source_window_signals \
  --output-dir "${COMPARE_DIR}" >>& "${LOG_DIR}/compare_results.log"
set rc = \$status
echo \$rc > "${COMPARE_DIR}/status"
rm -f "${COMPARE_DIR}/running"
if ( \$rc == 0 && -f "${COMPARE_DIR}/metrics_comparison.csv" ) then
  touch "${COMPARE_DIR}/complete"
else
  touch "${COMPARE_DIR}/fail"
endif
exit \$rc
EOF
chmod +x "$BUILD_JOB" "$HPO_JOB" "$RETRAIN_JOB" "$COMPARE_JOB"

# Stages are dependent, so submit one stage at a time. Parallelism inside HPO
# is controlled by HPO_JOB_NUM; independent target-window runs can use separate
# controllers and output roots.
foreach STAGE ( build hpo retrain compare )
    switch ( "$STAGE" )
        case build:
            set STAGE_DIR = "$FEATURE_DIR"
            set STAGE_JOB = "$BUILD_JOB"
            breaksw
        case hpo:
            set STAGE_DIR = "$HPO_DIR"
            set STAGE_JOB = "$HPO_JOB"
            breaksw
        case retrain:
            set STAGE_DIR = "$RETRAIN_DIR"
            set STAGE_JOB = "$RETRAIN_JOB"
            breaksw
        case compare:
            set STAGE_DIR = "$COMPARE_DIR"
            set STAGE_JOB = "$COMPARE_JOB"
            breaksw
    endsw
    if ( -f "${STAGE_DIR}/complete" ) continue
    @ RERUN_COUNT = 0
    while ( ! -f "${STAGE_DIR}/complete" )
        rm -f "${STAGE_DIR}/fail" "${STAGE_DIR}/status"
        set SUBMIT_RC = 0
        if ( "$STAGE" == "hpo" ) then
            cross_hpo_submit "$STAGE_JOB" "${LOG_DIR}/${STAGE}.dsub.out" "${LOG_DIR}/${STAGE}.dsub.err"
            set SUBMIT_RC = $status
        else if ( "$STAGE" == "retrain" ) then
            cross_retrain_submit "$STAGE_JOB" "${LOG_DIR}/${STAGE}.dsub.out" "${LOG_DIR}/${STAGE}.dsub.err"
            set SUBMIT_RC = $status
        else
            cross_submit "$STAGE_JOB" "${LOG_DIR}/${STAGE}.dsub.out" "${LOG_DIR}/${STAGE}.dsub.err"
            set SUBMIT_RC = $status
        endif
        if ( $SUBMIT_RC != 0 ) touch "${STAGE_DIR}/fail"
        while ( ! -f "${STAGE_DIR}/complete" && ! -f "${STAGE_DIR}/fail" )
            sleep "$POLL_SEC"
        end
        if ( -f "${STAGE_DIR}/complete" ) break
        if ( $RERUN_COUNT >= $MAX_RERUN ) then
            echo "ERROR: cross-window ${STAGE} failed after ${MAX_RERUN} reruns"
            exit 1
        endif
        @ RERUN_COUNT = $RERUN_COUNT + 1
    end
end

touch "${CROSS_ROOT}/complete"
echo "cross-window dsub pipeline complete: ${CROSS_ROOT}"
