#!/bin/csh -f
# Train one or more final-model backends from an existing Step4 screen result.
# Usage: MODEL_TYPES="xgboost,bnn,bnn_fp32,lstm,ltc" csh $0 <hw> <K>
set SCRIPT_DIR = `dirname "$0"`
set SCRIPT_DIR = `cd "$SCRIPT_DIR" && pwd`
if ( $#argv < 2 ) then
    echo "Usage: MODEL_TYPES=xgboost,bnn,bnn_fp32,lstm,ltc csh $0 <hw> <K>"
    exit 2
endif
set HW = "$1"
set TARGET_K = "$2"
source "${SCRIPT_DIR}/timewindow_common.csh"

if ( ! $?MODEL_TYPES ) set MODEL_TYPES = "xgboost,bnn,lstm,ltc"
if ( ! $?NN_EPOCHS ) set NN_EPOCHS = 100
if ( ! $?NN_BATCH_SIZE ) set NN_BATCH_SIZE = 128
if ( ! $?NN_LEARNING_RATE ) set NN_LEARNING_RATE = 0.001
if ( ! $?NN_HIDDEN_SIZE ) set NN_HIDDEN_SIZE = 128
if ( ! $?NN_NUM_LAYERS ) set NN_NUM_LAYERS = 1
if ( ! $?NN_DROPOUT ) set NN_DROPOUT = 0.1
if ( ! $?NN_PATIENCE ) set NN_PATIENCE = 12
if ( ! $?NN_DEVICE ) set NN_DEVICE = auto
if ( ! $?NN_SEQUENCE_CACHE_MB ) set NN_SEQUENCE_CACHE_MB = 512
if ( ! $?NN_MIXED_PRECISION ) set NN_MIXED_PRECISION = auto

set SCREEN_DIR = "${STEP4_DIR}/screen_K${TARGET_K}"
set TRAIN_BUFFER = "${SCREEN_DIR}/train_top${TARGET_K}.buffer"
set TEST_BUFFER = "${SCREEN_DIR}/test_top${TARGET_K}.buffer"
set FEATURE_NAMES = "${SCREEN_DIR}/feature_names.txt"
if ( ! -f "$TRAIN_BUFFER" || ! -f "$TEST_BUFFER" || ! -f "$FEATURE_NAMES" ) then
    echo "ERROR: missing Step4 buffers or feature_names under ${SCREEN_DIR}"
    exit 1
endif

set TYPES = ( `echo "$MODEL_TYPES" | tr ',' ' '` )
foreach MODEL_TYPE ( $TYPES )
    set MODEL_OUT = "${STEP6_DIR}/screen_K${TARGET_K}_${MODEL_TYPE}"
    mkdir -p "$MODEL_OUT"
    set EXTRA_ARGS = ( )
    if ( "$MODEL_TYPE" == "xgboost" ) then
        set HPO_CSV = "${STEP5_DIR}/screen_K${TARGET_K}_method${TWO_PHASE_METHOD}/best_trials_leaf_1000.csv"
        if ( ! -f "$HPO_CSV" ) then
            echo "ERROR: XGBoost requires HPO CSV: ${HPO_CSV}"
            exit 1
        endif
        set EXTRA_ARGS = ( --hpo-csv "$HPO_CSV" --rounds "$RETRAIN_ROUNDS" --two-phase-method "$TWO_PHASE_METHOD" --two-phase-cutoff "$TWO_PHASE_CUTOFF" --two-phase-ratio "$TWO_PHASE_RATIO" )
    else
        set EXTRA_ARGS = ( --epochs "$NN_EPOCHS" --batch-size "$NN_BATCH_SIZE" --learning-rate "$NN_LEARNING_RATE" --hidden-size "$NN_HIDDEN_SIZE" --num-layers "$NN_NUM_LAYERS" --dropout "$NN_DROPOUT" --patience "$NN_PATIENCE" --device "$NN_DEVICE" --sequence-cache-mb "$NN_SEQUENCE_CACHE_MB" --mixed-precision "$NN_MIXED_PRECISION" )
    endif
    echo "[`date '+%H:%M:%S'`] Train ${MODEL_TYPE}, K=${TARGET_K}"
    "$PYTHON" "${TIMEWINDOW_SRC}/step6_retrain_eval.py" \
        --model-type "$MODEL_TYPE" --train-buffer "$TRAIN_BUFFER" \
        --test-buffer "$TEST_BUFFER" --feature-names "$FEATURE_NAMES" \
        --tw-manifest "$STEP3_TW_MANIFEST" --level-dir "$LEVEL_DIR" --tog-dir "$TOG_DIR" \
        --label-dir "$LABEL_DIR" --raw-namefile "$NAMEFILE" \
        --output-dir "$MODEL_OUT" $EXTRA_ARGS:q
    if ( $status != 0 ) exit 1
end
