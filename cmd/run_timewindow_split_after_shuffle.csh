#!/bin/csh -f
# Experimental local runner: reuse Step1, split rows after shuffle, rerun Step2-Step6.
# Compatibility entry; all compute work is submitted by the dsub controller.
# Usage: csh $0 <source_step1_manifest> <namefile> [source_output_root] [K_csv]

# Compatibility entry: all compute work is now submitted through dsub.
set COMPAT_SCRIPT_DIR = `dirname "$0"`
exec csh "${COMPAT_SCRIPT_DIR}/run_timewindow_split_after_shuffle_dsub.csh" $argv:q

set SCRIPT_DIR = `dirname "$0"`
set PACKAGE_ROOT = `cd "${SCRIPT_DIR}/.." && pwd`
set TIMEWINDOW_SRC = "${PACKAGE_ROOT}/src/timewindow"
if ( $#argv < 3 ) then
    echo "Usage: csh $0 <source_step1_manifest> <namefile> <output_root> [K]"
    exit 2
endif
set SOURCE_MANIFEST = "$1"
set NAMEFILE = "$2"
set OUTPUT_ROOT = "$3/split_after_shuffle"
set TARGET_K = 256
if ( $#argv >= 4 ) set TARGET_K = "$4"

if ( ! $?PYTHON ) set PYTHON = "/home/data/chenyiheng/conda/envs/hw-model-train/bin/python"
if ( ! $?SHUFFLE_SEED ) set SHUFFLE_SEED = 108
if ( ! $?SPLIT_TEST_RATIO ) set SPLIT_TEST_RATIO = ""
if ( ! $?MCP_PROXY_TOP_K ) set MCP_PROXY_TOP_K = "$TARGET_K"
if ( ! $?XGB_PROXY_TOP_K ) set XGB_PROXY_TOP_K = "$TARGET_K"
if ( ! $?XGB_PROXY_ROUNDS ) set XGB_PROXY_ROUNDS = 50
if ( ! $?SCREEN_ROUNDS ) set SCREEN_ROUNDS = 200
if ( ! $?HPO_TRIALS ) set HPO_TRIALS = 64
if ( ! $?HPO_ROUNDS ) set HPO_ROUNDS = 50
if ( ! $?HPO_NFOLD ) set HPO_NFOLD = 3
if ( ! $?HPO_JOB_NUM ) set HPO_JOB_NUM = 1
if ( ! $?RETRAIN_ROUNDS ) set RETRAIN_ROUNDS = 50
if ( ! $?NTHREAD ) set NTHREAD = 8

foreach required ( "$SOURCE_MANIFEST" "$NAMEFILE" )
    if ( ! -f "$required" ) then
        echo "ERROR: missing input: $required"
        exit 1
    endif
end

set STEP1_DIR = "${OUTPUT_ROOT}/step1_shifted"
set STEP2_MCP_DIR = "${OUTPUT_ROOT}/step2_mcp"
set STEP2_XGB_DIR = "${OUTPUT_ROOT}/step2_xgb_proxy"
set STEP3_DIR = "${OUTPUT_ROOT}/step3_hybrid_concat"
set STEP4_DIR = "${OUTPUT_ROOT}/step4_hybrid_screen"
set STEP5_DIR = "${OUTPUT_ROOT}/step5_hpo/union_K${TARGET_K}"
set STEP6_DIR = "${OUTPUT_ROOT}/step6_retrain/union_K${TARGET_K}"
mkdir -p "$OUTPUT_ROOT"

set SPLIT_ARGS = ( )
if ( "$SPLIT_TEST_RATIO" != "" ) set SPLIT_ARGS = ( --test-ratio "$SPLIT_TEST_RATIO" )
"$PYTHON" "${TIMEWINDOW_SRC}/resplit_after_shuffle.py" \
    --input-manifest "$SOURCE_MANIFEST" --output-dir "$STEP1_DIR" \
    --seed "$SHUFFLE_SEED" $SPLIT_ARGS:q
if ( $status != 0 ) exit 1
set TW_MANIFEST = "${STEP1_DIR}/timewindow_manifest.json"

if ( ! -f "${STEP2_MCP_DIR}/mcp_per_offset_manifest.json" ) then
    "$PYTHON" "${TIMEWINDOW_SRC}/step2_mcp_per_offset.py" \
        --manifest "$TW_MANIFEST" --namefile "$NAMEFILE" \
        --top-k "$MCP_PROXY_TOP_K" --output-dir "$STEP2_MCP_DIR"
    if ( $status != 0 ) exit 1
else
    echo "Reuse Step2 MCP: ${STEP2_MCP_DIR}"
endif

if ( ! -f "${STEP2_XGB_DIR}/mcp_per_offset_manifest.json" ) then
    "$PYTHON" "${TIMEWINDOW_SRC}/step2_xgb_per_offset.py" \
        --manifest "$TW_MANIFEST" --namefile "$NAMEFILE" \
        --top-k "$XGB_PROXY_TOP_K" --rounds "$XGB_PROXY_ROUNDS" \
        --nthread "$NTHREAD" --output-dir "$STEP2_XGB_DIR"
    if ( $status != 0 ) exit 1
else
    echo "Reuse Step2 XGBoost: ${STEP2_XGB_DIR}"
endif

if ( ! -f "${STEP3_DIR}/hybrid_concat_meta.json" ) then
    "$PYTHON" "${TIMEWINDOW_SRC}/step3_hybrid_concat_features.py" \
        --tw-manifest "$TW_MANIFEST" \
        --mcp-manifest "${STEP2_MCP_DIR}/mcp_per_offset_manifest.json" \
        --xgb-manifest "${STEP2_XGB_DIR}/mcp_per_offset_manifest.json" \
        --namefile "$NAMEFILE" --shuffle-seed "$SHUFFLE_SEED" \
        --output-dir "$STEP3_DIR"
    if ( $status != 0 ) exit 1
else
    echo "Reuse Step3: ${STEP3_DIR}"
endif

set UNION_DIR = "${STEP4_DIR}/union_K${TARGET_K}"
if ( ! -f "${UNION_DIR}/union_meta.json" ) then
    "$PYTHON" "${TIMEWINDOW_SRC}/step4_hybrid_screen_features.py" \
        --hybrid-concat-meta "${STEP3_DIR}/hybrid_concat_meta.json" \
        --final-top-ks "$TARGET_K" --xgb-rounds "$SCREEN_ROUNDS" \
        --nthread "$NTHREAD" --shuffle-seed "$SHUFFLE_SEED" \
        --output-dir "$STEP4_DIR"
    if ( $status != 0 ) exit 1
else
    echo "Reuse Step4: ${UNION_DIR}"
endif

set TRAIN_BUFFER = "${UNION_DIR}/train_union_top${TARGET_K}.buffer"
set TEST_BUFFER = "${UNION_DIR}/test_union_top${TARGET_K}.buffer"
set FEATURE_NAMES = "${UNION_DIR}/feature_names.txt"
set HPO_CSV = "${STEP5_DIR}/best_trials_leaf_1000.csv"
if ( ! -f "$HPO_CSV" ) then
    mkdir -p "$STEP5_DIR"
    "$PYTHON" "${TIMEWINDOW_SRC}/step5_hpo_timewindow.py" \
        --train-buffer "$TRAIN_BUFFER" --test-buffer "$TEST_BUFFER" \
        --feature-names "$FEATURE_NAMES" --output-dir "$STEP5_DIR" \
        --round-num "$HPO_ROUNDS" --trial-num "$HPO_TRIALS" \
        --nfold "$HPO_NFOLD" --job-num "$HPO_JOB_NUM" \
        --shuffle-seed "$SHUFFLE_SEED"
    if ( $status != 0 ) exit 1
else
    echo "Reuse Step5 HPO: ${HPO_CSV}"
endif

if ( ! -f "${STEP6_DIR}/eval/metrics.csv" ) then
    mkdir -p "$STEP6_DIR"
    "$PYTHON" "${TIMEWINDOW_SRC}/step6_retrain_eval.py" \
        --model-type xgboost --train-buffer "$TRAIN_BUFFER" \
        --test-buffer "$TEST_BUFFER" --feature-names "$FEATURE_NAMES" \
        --hpo-csv "$HPO_CSV" --rounds "$RETRAIN_ROUNDS" \
        --output-dir "$STEP6_DIR" --shuffle-seed "$SHUFFLE_SEED" \
        --skip-full-eval
    if ( $status != 0 ) exit 1
else
    echo "Reuse Step6: ${STEP6_DIR}"
endif

touch "${OUTPUT_ROOT}/complete"
echo "split_after_shuffle Step2-Step6 complete: ${OUTPUT_ROOT}"
