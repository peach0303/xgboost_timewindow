#!/bin/csh -f
# Validate xgboost_timewindow training with the pro conda environment.

set SCRIPT_DIR = `dirname "$0"`
set SCRIPT_DIR = `cd "$SCRIPT_DIR" && pwd`
if ( ! $?PACKAGE_ROOT ) set PACKAGE_ROOT = `cd "${SCRIPT_DIR}/.." && pwd`
if ( ! $?PROJECT_ROOT ) set PROJECT_ROOT = `cd "${PACKAGE_ROOT}/.." && pwd`
if ( ! $?PYTHON ) set PYTHON = "/home/data/chenyiheng/conda/envs/pro/bin/python"
if ( ! $?TIMEWINDOW_DIR ) set TIMEWINDOW_DIR = "${PACKAGE_ROOT}/src/timewindow"

if ( ! $?SOURCE_RUN ) set SOURCE_RUN = "${PROJECT_ROOT}/output/5fsdb_reduction_train_smoke/boom_coremark_5fsdb_submit_20260810_4"
if ( ! $?REDUCED_DIR ) set REDUCED_DIR = "${SOURCE_RUN}/02_reduction/boom_5fsdb"
if ( ! $?LEVEL_DIR ) set LEVEL_DIR = "${REDUCED_DIR}/level"
if ( ! $?TOG_DIR ) set TOG_DIR = "${REDUCED_DIR}/tog"
if ( ! $?LABEL_DIR ) set LABEL_DIR = "${PROJECT_ROOT}/output/timewindow_test/power_labels"
if ( ! $?NAMEFILE ) set NAMEFILE = "${LEVEL_DIR}/after_one.namefile.txt"
if ( ! $?RUN_TAG ) set RUN_TAG = `date '+pro_timewindow_train_verify_%Y%m%d_%H%M%S'`
if ( ! $?OUTPUT_ROOT ) set OUTPUT_ROOT = "${PROJECT_ROOT}/output/timewindow_training_pro_verify/${RUN_TAG}"
if ( ! $?LOG_DIR ) set LOG_DIR = "${OUTPUT_ROOT}/logs"
if ( ! $?STATUS_FILE ) set STATUS_FILE = "${OUTPUT_ROOT}/STATUS.txt"

if ( ! $?OFFSETS ) set OFFSETS = 6
if ( ! $?DATA_TYPES ) set DATA_TYPES = "tog,level"
if ( ! $?STAGE ) set STAGE = after_one
if ( ! $?TEST_COUNT ) set TEST_COUNT = 1
if ( ! $?SAMPLE_ROWS_PER_FILE ) set SAMPLE_ROWS_PER_FILE = 0
if ( ! $?SAMPLE_BINS ) set SAMPLE_BINS = 20
if ( ! $?SAMPLE_MIN_PER_BIN ) set SAMPLE_MIN_PER_BIN = 1
if ( ! $?SAMPLE_SEED ) set SAMPLE_SEED = 108
if ( ! $?STEP1_JOBS ) set STEP1_JOBS = 1
if ( ! $?STEP2_JOBS ) set STEP2_JOBS = 1
if ( ! $?MCP_TOP_K ) set MCP_TOP_K = 32
if ( ! $?MCP_BATCH_SIZE ) set MCP_BATCH_SIZE = 1
if ( ! $?SCREEN_TOP_KS ) set SCREEN_TOP_KS = 16
if ( ! $?DEFAULT_ROUNDS ) set DEFAULT_ROUNDS = 50
if ( ! $?SCREEN_EVAL_ROUNDS ) set SCREEN_EVAL_ROUNDS = "$DEFAULT_ROUNDS"
if ( ! $?SCREEN_ROUNDS ) set SCREEN_ROUNDS = 2
if ( ! $?HPO_TRIALS ) set HPO_TRIALS = 2
if ( ! $?HPO_ROUNDS ) set HPO_ROUNDS = 2
if ( ! $?HPO_POP ) set HPO_POP = 2
if ( ! $?HPO_NFOLD ) set HPO_NFOLD = 2
if ( ! $?RETRAIN_ROUNDS ) set RETRAIN_ROUNDS = 2

mkdir -p "$OUTPUT_ROOT" "$LOG_DIR"
echo "[`date '+%Y-%m-%d %H:%M:%S'`] START output=${OUTPUT_ROOT}" > "$STATUS_FILE"

echo "[`date '+%Y-%m-%d %H:%M:%S'`] Python: ${PYTHON}" | tee -a "${LOG_DIR}/driver.log"
"$PYTHON" -c "import sys,numpy,scipy,sklearn,pandas,xgboost,optuna; print(sys.executable); print('deps ok', 'xgboost', xgboost.__version__, 'optuna', optuna.__version__)"
if ( $status != 0 ) exit 1

set step1_dir = "${OUTPUT_ROOT}/step1_shifted"
echo "[`date '+%Y-%m-%d %H:%M:%S'`] STEP1 timewindow prepare start" | tee -a "$STATUS_FILE"
/usr/bin/time -v -o "${LOG_DIR}/01_step1_prepare.time_v.txt" "$PYTHON" "${TIMEWINDOW_DIR}/step1_prepare_timewindow.py" \
    --level-dir "$LEVEL_DIR" --tog-dir "$TOG_DIR" --label-dir "$LABEL_DIR" \
    --output-dir "$step1_dir" --offsets "$OFFSETS" --data-types "$DATA_TYPES" \
    --test-count "$TEST_COUNT" --stage "$STAGE" \
    --sample-rows-per-file "$SAMPLE_ROWS_PER_FILE" --sample-bins "$SAMPLE_BINS" \
    --sample-min-per-bin "$SAMPLE_MIN_PER_BIN" --sample-seed "$SAMPLE_SEED" --jobs "$STEP1_JOBS" >& "${LOG_DIR}/01_step1_prepare.log"
if ( $status != 0 ) exit 1
echo "[`date '+%Y-%m-%d %H:%M:%S'`] STEP1 timewindow prepare done" | tee -a "$STATUS_FILE"

set step2_dir = "${OUTPUT_ROOT}/step2_mcp"
echo "[`date '+%Y-%m-%d %H:%M:%S'`] STEP2 proxy selection start" | tee -a "$STATUS_FILE"
/usr/bin/time -v -o "${LOG_DIR}/02_step2_mcp.time_v.txt" "$PYTHON" "${TIMEWINDOW_DIR}/step2_mcp_per_offset.py" \
    --manifest "${step1_dir}/timewindow_manifest.json" --namefile "$NAMEFILE" \
    --batch "$MCP_BATCH_SIZE" --top-k "$MCP_TOP_K" --jobs "$STEP2_JOBS" --output-dir "$step2_dir" >& "${LOG_DIR}/02_step2_mcp.log"
if ( $status != 0 ) exit 1
echo "[`date '+%Y-%m-%d %H:%M:%S'`] STEP2 proxy selection done" | tee -a "$STATUS_FILE"

set step3_dir = "${OUTPUT_ROOT}/step3_concat"
echo "[`date '+%Y-%m-%d %H:%M:%S'`] STEP3 concat buffers start" | tee -a "$STATUS_FILE"
/usr/bin/time -v -o "${LOG_DIR}/03_step3_concat.time_v.txt" "$PYTHON" "${TIMEWINDOW_DIR}/step3_concat_features.py" \
    --tw-manifest "${step1_dir}/timewindow_manifest.json" \
    --mcp-manifest "${step2_dir}/mcp_per_offset_manifest.json" \
    --namefile "$NAMEFILE" --output-dir "$step3_dir" >& "${LOG_DIR}/03_step3_concat.log"
if ( $status != 0 ) exit 1
echo "[`date '+%Y-%m-%d %H:%M:%S'`] STEP3 concat buffers done" | tee -a "$STATUS_FILE"

set step4_dir = "${OUTPUT_ROOT}/step4_screening"
echo "[`date '+%Y-%m-%d %H:%M:%S'`] STEP4 feature screening start" | tee -a "$STATUS_FILE"
/usr/bin/time -v -o "${LOG_DIR}/04_step4_screen.time_v.txt" "$PYTHON" "${TIMEWINDOW_DIR}/step4_screen_features.py" \
    --train-buffer "${step3_dir}/train_tw.buffer" --test-buffer "${step3_dir}/test_tw.buffer" \
    --concat-meta "${step3_dir}/concat_meta.json" --top-ks "$SCREEN_TOP_KS" \
    --tw-manifest "${step1_dir}/timewindow_manifest.json" --namefile "$NAMEFILE" \
    --default-rounds "$SCREEN_ROUNDS" --screen-eval-rounds "$SCREEN_EVAL_ROUNDS" \
    --output-dir "$step4_dir" >& "${LOG_DIR}/04_step4_screen.log"
if ( $status != 0 ) exit 1
echo "[`date '+%Y-%m-%d %H:%M:%S'`] STEP4 feature screening done" | tee -a "$STATUS_FILE"

set k = `echo "$SCREEN_TOP_KS" | awk -F, '{print $1}'`
set screen_dir = "${step4_dir}/screen_K${k}"
set step5_dir = "${OUTPUT_ROOT}/step5_hpo/screen_K${k}_single"
echo "[`date '+%Y-%m-%d %H:%M:%S'`] STEP5 HPO start" | tee -a "$STATUS_FILE"
/usr/bin/time -v -o "${LOG_DIR}/05_step5_hpo.time_v.txt" "$PYTHON" "${TIMEWINDOW_DIR}/step5_hpo_timewindow.py" \
    --train-buffer "${screen_dir}/train_top${k}.buffer" --test-buffer "${screen_dir}/test_top${k}.buffer" \
    --feature-names "${screen_dir}/feature_names.txt" --output-dir "$step5_dir" \
    --round-num "$HPO_ROUNDS" --trial-num "$HPO_TRIALS" --pop-size "$HPO_POP" \
    --nfold "$HPO_NFOLD" --two-phase-method "" >& "${LOG_DIR}/05_step5_hpo.log"
if ( $status != 0 ) exit 1
echo "[`date '+%Y-%m-%d %H:%M:%S'`] STEP5 HPO done" | tee -a "$STATUS_FILE"

set step6_dir = "${OUTPUT_ROOT}/step6_retrain/screen_K${k}_single"
echo "[`date '+%Y-%m-%d %H:%M:%S'`] STEP6 retrain eval start" | tee -a "$STATUS_FILE"
/usr/bin/time -v -o "${LOG_DIR}/06_step6_retrain.time_v.txt" "$PYTHON" "${TIMEWINDOW_DIR}/step6_retrain_eval.py" \
    --train-buffer "${screen_dir}/train_top${k}.buffer" --test-buffer "${screen_dir}/test_top${k}.buffer" \
    --hpo-csv "${step5_dir}/best_trials_leaf_1000.csv" --output-dir "$step6_dir" \
    --tw-manifest "${step1_dir}/timewindow_manifest.json" --level-dir "$LEVEL_DIR" --tog-dir "$TOG_DIR" \
    --label-dir "$LABEL_DIR" --raw-namefile "$NAMEFILE" \
    --rounds "$RETRAIN_ROUNDS" --feature-names "${screen_dir}/feature_names.txt" --two-phase-method "" >& "${LOG_DIR}/06_step6_retrain.log"
if ( $status != 0 ) exit 1
echo "[`date '+%Y-%m-%d %H:%M:%S'`] STEP6 retrain eval done" | tee -a "$STATUS_FILE"
echo "[`date '+%Y-%m-%d %H:%M:%S'`] DONE" | tee -a "$STATUS_FILE"
