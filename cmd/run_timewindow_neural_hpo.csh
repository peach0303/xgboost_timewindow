#!/bin/csh -f
# Run optional Optuna HPO and final retraining for BNN/BNN-FP32/LSTM/LTC.
# Set OUTPUT_ROOT to the existing pipeline run before invoking this script.
# Set SKIP_NEURAL_HPO=1 to retrain directly with the model-specific defaults below.
# Usage: SKIP_NEURAL_HPO=0 NEURAL_MODEL_TYPES="bnn,bnn_fp32,lstm,ltc" csh $0 <hw> <K>
set SCRIPT_DIR = `dirname "$0"`
set SCRIPT_DIR = `cd "$SCRIPT_DIR" && pwd`
if ( $#argv < 2 ) then
    echo "Usage: NEURAL_MODEL_TYPES=bnn,bnn_fp32,lstm,ltc csh $0 <hw> <K>"
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

set HPO_ROOT = "${NEURAL_OUTPUT_ROOT}/step5_neural_hpo"
set RETRAIN_ROOT = "${NEURAL_OUTPUT_ROOT}/step6_neural_retrain"
mkdir -p "$RETRAIN_ROOT"
if ( "$SKIP_NEURAL_HPO" == "0" ) mkdir -p "$HPO_ROOT"
set TYPES = ( `echo "$NEURAL_MODEL_TYPES" | tr ',' ' '` )
foreach MODEL_TYPE ( $TYPES )
    if ( "$MODEL_TYPE" != "bnn" && "$MODEL_TYPE" != "bnn_fp32" && "$MODEL_TYPE" != "lstm" && "$MODEL_TYPE" != "ltc" ) then
        echo "ERROR: unsupported neural model: ${MODEL_TYPE}"
        exit 2
    endif
    set HPO_OUT = "${HPO_ROOT}/union_K${TARGET_K}_${MODEL_TYPE}"
    if ( "$SKIP_NEURAL_HPO" == "1" ) then
        set MODEL_OUT = "${RETRAIN_ROOT}/union_K${TARGET_K}_${MODEL_TYPE}_direct"
    else
        set MODEL_OUT = "${RETRAIN_ROOT}/union_K${TARGET_K}_${MODEL_TYPE}"
    endif
    mkdir -p "$MODEL_OUT"

    set RETRAIN_ARGS = ( )
    if ( "$SKIP_NEURAL_HPO" == "0" ) then
        mkdir -p "$HPO_OUT"
        echo "[`date '+%H:%M:%S'`] HPO ${MODEL_TYPE}, K=${TARGET_K}"
        "$NEURAL_PYTHON" "${TIMEWINDOW_SRC}/step5_hpo_neural.py" \
            --model-type "$MODEL_TYPE" --train-buffer "$TRAIN_BUFFER" \
            --test-buffer "$TEST_BUFFER" --feature-names "$FEATURE_NAMES" \
            --output-dir "$HPO_OUT" --epochs "$NEURAL_HPO_EPOCHS" \
            --trial-num "$NEURAL_HPO_TRIALS" --pop-size "$NEURAL_HPO_POP" \
            --job-num "$NEURAL_HPO_JOB_NUM" --nfold "$NEURAL_HPO_NFOLD" \
            --sampler "$NEURAL_HPO_SAMPLER" --patience "$NEURAL_PATIENCE" \
            --validation-ratio "$NEURAL_VALIDATION_RATIO" \
            --parameter-threshold "$NEURAL_PARAMETER_THRESHOLD" \
            --device "$NEURAL_DEVICE" --sequence-cache-mb "$NEURAL_SEQUENCE_CACHE_MB" \
            --mixed-precision "$NEURAL_MIXED_PRECISION" --seed "$SHUFFLE_SEED"
        if ( $status != 0 || ! -f "${HPO_OUT}/best_trial.json" ) exit 1
        set RETRAIN_ARGS = ( --hpo-result "${HPO_OUT}/best_trial.json" )
    else
        if ( "$MODEL_TYPE" == "bnn" || "$MODEL_TYPE" == "bnn_fp32" ) then
            set DIRECT_BATCH_SIZE = "$BNN_BATCH_SIZE"
            set DIRECT_LEARNING_RATE = "$BNN_LEARNING_RATE"
            set DIRECT_HIDDEN_SIZE = "$BNN_HIDDEN_SIZE"
            set DIRECT_NUM_LAYERS = "$BNN_NUM_LAYERS"
            set DIRECT_DROPOUT = "$BNN_DROPOUT"
        else if ( "$MODEL_TYPE" == "lstm" ) then
            set DIRECT_BATCH_SIZE = "$LSTM_BATCH_SIZE"
            set DIRECT_LEARNING_RATE = "$LSTM_LEARNING_RATE"
            set DIRECT_HIDDEN_SIZE = "$LSTM_HIDDEN_SIZE"
            set DIRECT_NUM_LAYERS = "$LSTM_NUM_LAYERS"
            set DIRECT_DROPOUT = "$LSTM_DROPOUT"
        else
            set DIRECT_BATCH_SIZE = "$LTC_BATCH_SIZE"
            set DIRECT_LEARNING_RATE = "$LTC_LEARNING_RATE"
            set DIRECT_HIDDEN_SIZE = "$LTC_HIDDEN_SIZE"
            set DIRECT_NUM_LAYERS = "$LTC_NUM_LAYERS"
            set DIRECT_DROPOUT = "$LTC_DROPOUT"
        endif
        set RETRAIN_ARGS = ( \
            --epochs "$NEURAL_DIRECT_EPOCHS" \
            --batch-size "$DIRECT_BATCH_SIZE" \
            --learning-rate "$DIRECT_LEARNING_RATE" \
            --hidden-size "$DIRECT_HIDDEN_SIZE" \
            --num-layers "$DIRECT_NUM_LAYERS" \
            --dropout "$DIRECT_DROPOUT" \
            --weight-decay "$NEURAL_DIRECT_WEIGHT_DECAY" \
            --patience "$NEURAL_DIRECT_PATIENCE" \
            --validation-ratio "$NEURAL_DIRECT_VALIDATION_RATIO" \
            --gradient-clip "$NEURAL_DIRECT_GRADIENT_CLIP" \
            --sequence-cache-mb "$NEURAL_SEQUENCE_CACHE_MB" \
            --mixed-precision "$NEURAL_MIXED_PRECISION" \
        )
        echo "[`date '+%H:%M:%S'`] Skip HPO; direct ${MODEL_TYPE}: hidden=${DIRECT_HIDDEN_SIZE}, layers=${DIRECT_NUM_LAYERS}, batch=${DIRECT_BATCH_SIZE}, lr=${DIRECT_LEARNING_RATE}, dropout=${DIRECT_DROPOUT}"
    endif

    echo "[`date '+%H:%M:%S'`] Retrain ${MODEL_TYPE}, K=${TARGET_K}"
    "$NEURAL_PYTHON" "${TIMEWINDOW_SRC}/step6_retrain_eval.py" \
        --model-type "$MODEL_TYPE" $RETRAIN_ARGS:q \
        --train-buffer "$TRAIN_BUFFER" --test-buffer "$TEST_BUFFER" \
        --feature-names "$FEATURE_NAMES" --tw-manifest "$STEP3_TW_MANIFEST" \
        --level-dir "$LEVEL_DIR" --tog-dir "$TOG_DIR" --label-dir "$LABEL_DIR" \
        --raw-namefile "$NAMEFILE" --output-dir "$MODEL_OUT" \
        --device "$NEURAL_DEVICE" --seed "$SHUFFLE_SEED"
    if ( $status != 0 ) exit 1
    if ( "$SKIP_NEURAL_HPO" == "0" ) touch "${HPO_OUT}/complete"
    touch "${MODEL_OUT}/complete"
end
