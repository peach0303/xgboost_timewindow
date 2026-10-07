#!/bin/csh -f
# New, standalone hybrid pipeline. Existing pipeline scripts are not changed.
# Usage:
#   csh run_timewindow_hybrid_pipeline.csh TW.json MCP.json XGB.json NAMEFILE OUT [K]
#
# Optional manifest filtering before Hybrid Step 3:
#   setenv STEP3_OFFSETS "12,9,6"
#   setenv STEP3_DATA_TYPES "tog,level"
#   # Outputs go below OUT/tog_level_12_9_6/
# or:
#   setenv STEP3_COMBOS "tog_offset12,level_offset9"


if ( $#argv < 5 ) then
    echo "Usage: csh $0 <tw_manifest> <mcp_manifest> <xgb_manifest> <namefile> <output_root> [final_top_k]"
    exit 2
endif

set SCRIPT_DIR = `dirname "$0"`
set PACKAGE_ROOT = `cd "${SCRIPT_DIR}/.." && pwd`
set TIMEWINDOW_SRC = "${PACKAGE_ROOT}/src/timewindow"
set TW_MANIFEST = "$1"
set MCP_MANIFEST = "$2"
set XGB_MANIFEST = "$3"
set NAMEFILE = "$4"
set OUTPUT_ROOT = "$5"
set FINAL_TOP_K = 1024
if ( $#argv >= 6 ) set FINAL_TOP_K = "$6"


#==========user check args==========
#==============================step 0 args==============================
if ( ! $?PYTHON ) set PYTHON = "/home/data/chenyiheng/conda/envs/pro/bin/python"
if ( ! $?DSUB_CMD ) set DSUB_CMD = ( dsub -A root.ug_dsp.hisdr_HClass )
if ( ! $?DSUB_QUEUE ) set DSUB_QUEUE = normal
if ( ! $?DSUB_EXTRA_ARGS ) set DSUB_EXTRA_ARGS = ( )
if ( ! $?POLL_SEC ) set POLL_SEC = 30
if ( ! $?MAX_RERUN ) set MAX_RERUN = 10
if ( ! $?SHUFFLE_SEED ) set SHUFFLE_SEED = 108
#======================================================================

#==============================step 3 args==============================
if ( ! $?STEP3_OFFSETS ) set STEP3_OFFSETS = ""
if ( ! $?STEP3_DATA_TYPES ) set STEP3_DATA_TYPES = ""
if ( ! $?STEP3_COMBOS ) set STEP3_COMBOS = ""
if ( ! $?EXTRA_TAG ) set EXTRA_TAG = ""
#======================================================================

#==============================step 4 args==============================
if ( ! $?RIDGE_ALPHA ) set RIDGE_ALPHA = 8.0
if ( ! $?SCREEN_XGB_ROUNDS ) set SCREEN_XGB_ROUNDS = 200
if ( ! $?NTHREAD ) set NTHREAD = 32
if ( ! $?XGB_LOSS ) set XGB_LOSS = error_focus
if ( ! $?XGB_RELATIVE_FLOOR ) set XGB_RELATIVE_FLOOR = 0.001
if ( ! $?XGB_FOCUS_POWER ) set XGB_FOCUS_POWER = 3.0
#======================================================================

#==============================step 5 hybrid args==============================
if ( ! $?HPO_ROUNDS ) set HPO_ROUNDS = 50
if ( ! $?HPO_TRIALS ) set HPO_TRIALS = 64
if ( ! $?HPO_POP ) set HPO_POP = 8
if ( ! $?HPO_NFOLD ) set HPO_NFOLD = 3
if ( ! $?HPO_JOB_NUM ) set HPO_JOB_NUM = 1
#======================================================================

#==============================step 5 mlp hpo args==============================
if ( $?SKIP_MLP_TRAIN ) then
    if ( ! $?skip_mlp_train ) set skip_mlp_train = "$SKIP_MLP_TRAIN"
endif
if ( ! $?skip_mlp_train ) set skip_mlp_train = 1
if ( ! $?MLP_HPO_EPOCHS ) set MLP_HPO_EPOCHS = 100
if ( ! $?MLP_HPO_TRIALS ) set MLP_HPO_TRIALS = 40
if ( ! $?MLP_HPO_POP ) set MLP_HPO_POP = 8
if ( ! $?MLP_HPO_JOB_NUM ) set MLP_HPO_JOB_NUM = 1
if ( ! $?MLP_HPO_NFOLD ) set MLP_HPO_NFOLD = 3
if ( ! $?MLP_HPO_SAMPLER ) set MLP_HPO_SAMPLER = NSGAII
if ( ! $?MLP_PATIENCE ) set MLP_PATIENCE = 12
if ( ! $?MLP_VALIDATION_RATIO ) set MLP_VALIDATION_RATIO = 0.15
if ( ! $?MLP_PARAMETER_THRESHOLD ) set MLP_PARAMETER_THRESHOLD = 5000000
#======================================================================

#==============================step 6 mlp train args==============================
if ( ! $?MLP_SKIP_FULL_EVAL ) set MLP_SKIP_FULL_EVAL = 1
if ( ! $?MLP_LEVEL_DIR ) set MLP_LEVEL_DIR = ""
if ( ! $?MLP_TOG_DIR ) set MLP_TOG_DIR = ""
if ( ! $?MLP_LABEL_DIR ) set MLP_LABEL_DIR = ""
if ( ! $?MLP_RAW_NAMEFILE ) set MLP_RAW_NAMEFILE = "$NAMEFILE"
#======================================================================


#=========================
set OUTPUT_TAG = `"$PYTHON" "${TIMEWINDOW_SRC}/build_timewindow_output_tag.py" \
    --tw-manifest "$TW_MANIFEST" --offsets "$STEP3_OFFSETS" \
    --data-types "$STEP3_DATA_TYPES" --include-combos "$STEP3_COMBOS" \
    --override "$EXTRA_TAG"`
if ( $status != 0 || "$OUTPUT_TAG" == "" ) then
    echo "ERROR: failed to build Step3-Step6 output tag"
    exit 2
endif
set TAG_OUTPUT_ROOT = "${OUTPUT_ROOT}/${OUTPUT_TAG}"
set STEP3_DIR = "${TAG_OUTPUT_ROOT}/step3_hybrid_concat"
set STEP4_DIR = "${TAG_OUTPUT_ROOT}/step4_hybrid_screen"
set STEP5_DIR = "${TAG_OUTPUT_ROOT}/step5_hybrid_hpo/union_K${FINAL_TOP_K}"
set INPUT_MANIFEST_DIR = "${TAG_OUTPUT_ROOT}/input_manifests"
set JOB_DIR = "${TAG_OUTPUT_ROOT}/jobs_dsub"
set LOG_DIR = "${TAG_OUTPUT_ROOT}/logs_dsub"

# Effective manifests consumed by Hybrid Step 3. They point to the original
# inputs unless an offset/type or explicit-combo subset is requested.
set MLP_HPO_DIR = "${TAG_OUTPUT_ROOT}/step5_mlp_hpo/union_K${FINAL_TOP_K}"
set MLP_TRAIN_DIR = "${TAG_OUTPUT_ROOT}/step6_mlp_retrain/union_K${FINAL_TOP_K}"
set EFFECTIVE_TW_MANIFEST = "$TW_MANIFEST"
set EFFECTIVE_MCP_MANIFEST = "$MCP_MANIFEST"
set EFFECTIVE_XGB_MANIFEST = "$XGB_MANIFEST"

foreach required ( "$TW_MANIFEST" "$MCP_MANIFEST" "$XGB_MANIFEST" "$NAMEFILE" )
    if ( ! -f "$required" ) then
        echo "ERROR: missing input: $required"
        exit 1
    endif
end
if ( "$skip_mlp_train" != "0" && "$skip_mlp_train" != "1" ) then
    echo "ERROR: skip_mlp_train must be 0 (train) or 1 (skip)"
    exit 2
endif
if ( "$MLP_SKIP_FULL_EVAL" != "0" && "$MLP_SKIP_FULL_EVAL" != "1" ) then
    echo "ERROR: MLP_SKIP_FULL_EVAL must be 0 or 1"
    exit 2
endif
if ( "$skip_mlp_train" == "0" && "$MLP_SKIP_FULL_EVAL" == "0" ) then
    foreach required_dir ( "$MLP_LEVEL_DIR" "$MLP_TOG_DIR" "$MLP_LABEL_DIR" )
        if ( "$required_dir" == "" || ! -d "$required_dir" ) then
            echo "ERROR: full MLP evaluation requires valid MLP_LEVEL_DIR/MLP_TOG_DIR/MLP_LABEL_DIR"
            exit 2
        endif
    end
endif
mkdir -p "$STEP3_DIR" "$STEP4_DIR" "$STEP5_DIR" "$MLP_HPO_DIR" "$MLP_TRAIN_DIR" "$JOB_DIR" "$LOG_DIR"


set FILTER_ENABLED = 0
if ( "$STEP3_COMBOS" != "" ) then
    set FILTER_ENABLED = 1
else if ( "$STEP3_OFFSETS" != "" || "$STEP3_DATA_TYPES" != "" ) then
    if ( "$STEP3_OFFSETS" == "" || "$STEP3_DATA_TYPES" == "" ) then
        echo "ERROR: filtering by offset/type requires both STEP3_OFFSETS and STEP3_DATA_TYPES"
        exit 2
    endif
    set FILTER_ENABLED = 1
endif

if ( $FILTER_ENABLED != 0 ) then
    mkdir -p "$INPUT_MANIFEST_DIR"
    set EFFECTIVE_TW_MANIFEST = "${INPUT_MANIFEST_DIR}/timewindow_manifest.json"
    set EFFECTIVE_MCP_MANIFEST = "${INPUT_MANIFEST_DIR}/mcp_per_offset_manifest.json"
    set EFFECTIVE_XGB_MANIFEST = "${INPUT_MANIFEST_DIR}/xgb_per_offset_manifest.json"

    echo "[`date '+%H:%M:%S'`] ===== Filter manifests for Hybrid Step 3 ====="
    echo "  STEP3_OFFSETS    = ${STEP3_OFFSETS}"
    echo "  STEP3_DATA_TYPES = ${STEP3_DATA_TYPES}"
    echo "  STEP3_COMBOS     = ${STEP3_COMBOS}"
    "$PYTHON" "${TIMEWINDOW_SRC}/filter_timewindow_manifests.py" \
        --tw-manifest-in "$TW_MANIFEST" \
        --mcp-manifest-in "$MCP_MANIFEST" \
        --tw-manifest-out "$EFFECTIVE_TW_MANIFEST" \
        --mcp-manifest-out "$EFFECTIVE_MCP_MANIFEST" \
        --offsets "$STEP3_OFFSETS" --data-types "$STEP3_DATA_TYPES" \
        --include-combos "$STEP3_COMBOS"
    if ( $status != 0 ) exit 1

    # The filter utility accepts the common Step2 manifest schema, so the XGB
    # proxy manifest is passed through its historical --mcp-manifest options.
    "$PYTHON" "${TIMEWINDOW_SRC}/filter_timewindow_manifests.py" \
        --tw-manifest-in "$TW_MANIFEST" \
        --mcp-manifest-in "$XGB_MANIFEST" \
        --tw-manifest-out "${INPUT_MANIFEST_DIR}/timewindow_manifest_xgb.json" \
        --mcp-manifest-out "$EFFECTIVE_XGB_MANIFEST" \
        --offsets "$STEP3_OFFSETS" --data-types "$STEP3_DATA_TYPES" \
        --include-combos "$STEP3_COMBOS"
    if ( $status != 0 ) exit 1
endif

set MLP_EVAL_ARGS = ( --skip-full-eval )
if ( "$MLP_SKIP_FULL_EVAL" == "0" ) then
    set MLP_EVAL_ARGS = ( --tw-manifest "$EFFECTIVE_TW_MANIFEST" \
        --level-dir "$MLP_LEVEL_DIR" --tog-dir "$MLP_TOG_DIR" \
        --label-dir "$MLP_LABEL_DIR" --raw-namefile "$MLP_RAW_NAMEFILE" )
endif

set STEP3_JOB = "${JOB_DIR}/step3_hybrid_concat.csh"
cat > "$STEP3_JOB" << EOF
#!/bin/csh -x
# Hybrid Step 3: concatenate filtered MCP/XGB proxy features.
rm -f "${STEP3_DIR}/complete" "${STEP3_DIR}/fail" "${STEP3_DIR}/status"
touch "${STEP3_DIR}/running"
echo "[job] Hybrid Step 3 start: `date` host=`hostname`"
"$PYTHON" "${TIMEWINDOW_SRC}/step3_hybrid_concat_features.py" \
    --tw-manifest "$EFFECTIVE_TW_MANIFEST" \
    --mcp-manifest "$EFFECTIVE_MCP_MANIFEST" \
    --xgb-manifest "$EFFECTIVE_XGB_MANIFEST" --namefile "$NAMEFILE" \
    --shuffle-seed "$SHUFFLE_SEED" --output-dir "$STEP3_DIR"
set rc = \$status
rm -f "${STEP3_DIR}/running"
if ( \$rc == 0 && -f "${STEP3_DIR}/hybrid_concat_meta.json" ) then
    echo \$rc > "${STEP3_DIR}/status"
    touch "${STEP3_DIR}/complete"
else
    if ( \$rc == 0 ) set rc = 1
    echo \$rc > "${STEP3_DIR}/status"
    touch "${STEP3_DIR}/fail"
endif
exit \$rc
EOF

set STEP4_JOB = "${JOB_DIR}/step4_hybrid_screen.csh"
cat > "$STEP4_JOB" << EOF
#!/bin/csh -x
# Hybrid Step 4: rank MCP/XGB proxies and build the union.
rm -f "${STEP4_DIR}/complete" "${STEP4_DIR}/fail" "${STEP4_DIR}/status"
touch "${STEP4_DIR}/running"
echo "[job] Hybrid Step 4 start: `date` host=`hostname`"
"$PYTHON" "${TIMEWINDOW_SRC}/step4_hybrid_screen_features.py" \
    --hybrid-concat-meta "${STEP3_DIR}/hybrid_concat_meta.json" \
    --final-top-ks "$FINAL_TOP_K" --ridge-alpha "$RIDGE_ALPHA" \
    --xgb-rounds "$SCREEN_XGB_ROUNDS" --nthread "$NTHREAD" \
    --loss "$XGB_LOSS" --relative-floor "$XGB_RELATIVE_FLOOR" \
    --focus-power "$XGB_FOCUS_POWER" --shuffle-seed "$SHUFFLE_SEED" \
    --output-dir "$STEP4_DIR"
set rc = \$status
rm -f "${STEP4_DIR}/running"
if ( \$rc == 0 && -f "${STEP4_DIR}/union_K${FINAL_TOP_K}/union_meta.json" ) then
    echo \$rc > "${STEP4_DIR}/status"
    touch "${STEP4_DIR}/complete"
else
    if ( \$rc == 0 ) set rc = 1
    echo \$rc > "${STEP4_DIR}/status"
    touch "${STEP4_DIR}/fail"
endif
exit \$rc
EOF

set STEP5_JOB = "${JOB_DIR}/step5_hybrid_hpo.csh"
cat > "$STEP5_JOB" << EOF
#!/bin/csh -x
# Hybrid Step 5: tune Ridge plus residual XGBoost.
rm -f "${STEP5_DIR}/complete" "${STEP5_DIR}/fail" "${STEP5_DIR}/status"
touch "${STEP5_DIR}/running"
echo "[job] Hybrid Step 5 start: `date` host=`hostname`"
"$PYTHON" "${TIMEWINDOW_SRC}/step5_hybrid_hpo_timewindow.py" \
    --union-meta "${STEP4_DIR}/union_K${FINAL_TOP_K}/union_meta.json" \
    --output-dir "$STEP5_DIR" --ridge-alpha "$RIDGE_ALPHA" \
    --round-num "$HPO_ROUNDS" --trial-num "$HPO_TRIALS" \
    --pop-size "$HPO_POP" --nfold "$HPO_NFOLD" \
    --job-num "$HPO_JOB_NUM" --nthread "$NTHREAD" \
    --loss "$XGB_LOSS" --relative-floor "$XGB_RELATIVE_FLOOR" \
    --focus-power "$XGB_FOCUS_POWER" --shuffle-seed "$SHUFFLE_SEED"
set rc = \$status
rm -f "${STEP5_DIR}/running"
if ( \$rc == 0 && -f "${STEP5_DIR}/best_hybrid_model_meta.json" ) then
    echo \$rc > "${STEP5_DIR}/status"
    touch "${STEP5_DIR}/complete"
else
    if ( \$rc == 0 ) set rc = 1
    echo \$rc > "${STEP5_DIR}/status"
    touch "${STEP5_DIR}/fail"
endif
exit \$rc
EOF

set MLP_HPO_JOB = "${JOB_DIR}/step5_mlp_hpo.csh"
cat > "$MLP_HPO_JOB" << EOF
#!/bin/csh -x
# MLP branch Step 5: hyperparameter optimization on the Hybrid Step 4 union.
rm -f "${MLP_HPO_DIR}/complete" "${MLP_HPO_DIR}/fail" "${MLP_HPO_DIR}/status"
touch "${MLP_HPO_DIR}/running"
echo "[job] MLP HPO start: `date` host=`hostname`"
"$PYTHON" "${TIMEWINDOW_SRC}/step5_hpo_mlp.py" \
    --train-buffer "${STEP4_DIR}/union_K${FINAL_TOP_K}/train_union_top${FINAL_TOP_K}.buffer" \
    --test-buffer "${STEP4_DIR}/union_K${FINAL_TOP_K}/test_union_top${FINAL_TOP_K}.buffer" \
    --feature-names "${STEP4_DIR}/union_K${FINAL_TOP_K}/feature_names.txt" \
    --output-dir "$MLP_HPO_DIR" --epochs "$MLP_HPO_EPOCHS" \
    --trial-num "$MLP_HPO_TRIALS" --pop-size "$MLP_HPO_POP" \
    --job-num "$MLP_HPO_JOB_NUM" --nfold "$MLP_HPO_NFOLD" \
    --sampler "$MLP_HPO_SAMPLER" --patience "$MLP_PATIENCE" \
    --validation-ratio "$MLP_VALIDATION_RATIO" \
    --parameter-threshold "$MLP_PARAMETER_THRESHOLD" --seed "$SHUFFLE_SEED"
set rc = \$status
rm -f "${MLP_HPO_DIR}/running"
if ( \$rc == 0 && -f "${MLP_HPO_DIR}/best_trial.json" ) then
    echo \$rc > "${MLP_HPO_DIR}/status"
    touch "${MLP_HPO_DIR}/complete"
else
    if ( \$rc == 0 ) set rc = 1
    echo \$rc > "${MLP_HPO_DIR}/status"
    touch "${MLP_HPO_DIR}/fail"
endif
exit \$rc
EOF

set MLP_TRAIN_JOB = "${JOB_DIR}/step6_mlp_train.csh"
cat > "$MLP_TRAIN_JOB" << EOF
#!/bin/csh -x
# MLP branch Step 6: retrain and evaluate the best MLP trial.
rm -f "${MLP_TRAIN_DIR}/complete" "${MLP_TRAIN_DIR}/fail" "${MLP_TRAIN_DIR}/status"
touch "${MLP_TRAIN_DIR}/running"
echo "[job] MLP final train start: `date` host=`hostname`"
"$PYTHON" "${TIMEWINDOW_SRC}/step6_retrain_eval_mlp.py" \
    --train-buffer "${STEP4_DIR}/union_K${FINAL_TOP_K}/train_union_top${FINAL_TOP_K}.buffer" \
    --test-buffer "${STEP4_DIR}/union_K${FINAL_TOP_K}/test_union_top${FINAL_TOP_K}.buffer" \
    --feature-names "${STEP4_DIR}/union_K${FINAL_TOP_K}/feature_names.txt" \
    --hpo-result "${MLP_HPO_DIR}/best_trial.json" \
    --output-dir "$MLP_TRAIN_DIR" --seed "$SHUFFLE_SEED" $MLP_EVAL_ARGS:q
set rc = \$status
rm -f "${MLP_TRAIN_DIR}/running"
if ( \$rc == 0 && -f "${MLP_TRAIN_DIR}/result/mlp_model.joblib" && -f "${MLP_TRAIN_DIR}/eval/metrics.csv" ) then
    echo \$rc > "${MLP_TRAIN_DIR}/status"
    touch "${MLP_TRAIN_DIR}/complete"
else
    if ( \$rc == 0 ) set rc = 1
    echo \$rc > "${MLP_TRAIN_DIR}/status"
    touch "${MLP_TRAIN_DIR}/fail"
endif
exit \$rc
EOF

chmod +x "$STEP3_JOB" "$STEP4_JOB" "$STEP5_JOB" "$MLP_HPO_JOB" "$MLP_TRAIN_JOB"

echo ""
echo "Hybrid pipeline will submit serial dsub stages to queue: $DSUB_QUEUE"
echo "Step3-Step6 output tag: $OUTPUT_TAG"
echo "Tagged output root: $TAG_OUTPUT_ROOT"

set PIPELINE_STAGES = ( step3 step4 step5 )
if ( "$skip_mlp_train" == "0" ) then
    set PIPELINE_STAGES = ( step3 step4 step5 mlp_hpo mlp_train )
    echo "MLP branch: enabled (HPO + final train)"
else
    echo "MLP branch: skipped"
endif
foreach stage ( $PIPELINE_STAGES )
    switch ( "$stage" )
        case step3:
            set stage_name = "Hybrid Step 3"
            set stage_dir = "$STEP3_DIR"
            set job_file = "$STEP3_JOB"
            set expected_file = "${STEP3_DIR}/hybrid_concat_meta.json"
            breaksw
        case step4:
            set stage_name = "Hybrid Step 4"
            set stage_dir = "$STEP4_DIR"
            set job_file = "$STEP4_JOB"
            set expected_file = "${STEP4_DIR}/union_K${FINAL_TOP_K}/union_meta.json"
            breaksw
        case step5:
            set stage_name = "Hybrid Step 5"
            set stage_dir = "$STEP5_DIR"
            set job_file = "$STEP5_JOB"
            set expected_file = "${STEP5_DIR}/best_hybrid_model_meta.json"
            breaksw
        case mlp_hpo:
            set stage_name = "MLP Step 5 HPO"
            set stage_dir = "$MLP_HPO_DIR"
            set job_file = "$MLP_HPO_JOB"
            set expected_file = "${MLP_HPO_DIR}/best_trial.json"
            breaksw
        case mlp_train:
            set stage_name = "MLP Step 6 Train"
            set stage_dir = "$MLP_TRAIN_DIR"
            set job_file = "$MLP_TRAIN_JOB"
            set expected_file = "${MLP_TRAIN_DIR}/result/mlp_model.joblib"
            breaksw
    endsw

    set dsub_out = "${LOG_DIR}/${stage}.dsub.out"
    set dsub_err = "${LOG_DIR}/${stage}.dsub.err"

    if ( -f "$stage_dir/complete" && -f "$expected_file" ) then
        echo "[`date '+%H:%M:%S'`] ${stage_name}: SKIP (complete)"
        continue
    endif

    @ rerun_count = 0
    while ( 1 )
        rm -f "$stage_dir/running" "$stage_dir/complete" "$stage_dir/fail" "$stage_dir/status"
        echo "[`date '+%H:%M:%S'`] submit ${stage_name}; rerun=${rerun_count}/${MAX_RERUN}"
        ${DSUB_CMD} -q ${DSUB_QUEUE} -oo ${dsub_out} -eo ${dsub_err} $DSUB_EXTRA_ARGS:q -s ${job_file}
        set submit_rc = $status
        if ( $submit_rc != 0 ) then
            echo $submit_rc > "$stage_dir/status"
            touch "$stage_dir/fail"
        endif

        while ( ! -f "$stage_dir/complete" && ! -f "$stage_dir/fail" )
            if ( -f "$stage_dir/running" ) then
                echo "  ${stage_name} running; sleep ${POLL_SEC}s"
            else
                echo "  ${stage_name} queued; sleep ${POLL_SEC}s"
            endif
            sleep $POLL_SEC
        end

        if ( -f "$stage_dir/complete" && -f "$expected_file" ) then
            echo "[`date '+%H:%M:%S'`] ${stage_name}: complete"
            break
        endif

        if ( $rerun_count >= $MAX_RERUN ) then
            echo "ERROR: ${stage_name} failed after ${MAX_RERUN} automatic reruns"
            echo "  status: $stage_dir/status"
            echo "  stdout: $dsub_out"
            echo "  stderr: $dsub_err"
            exit 1
        endif

        @ rerun_count = $rerun_count + 1
        echo "WARNING: ${stage_name} failed; automatically rerun failed stage only"
    end
end

echo "[`date '+%H:%M:%S'`] Hybrid pipeline completed: $TAG_OUTPUT_ROOT"
