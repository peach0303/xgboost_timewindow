#!/bin/csh -f
# Step 1: prepare time-window feature/label files.
set SCRIPT_DIR = `dirname "$0"`
set SCRIPT_DIR = `cd "$SCRIPT_DIR" && pwd`
if ( $#argv >= 1 ) then
    set HW = "$1"
else
    set HW = boom
endif
source "${SCRIPT_DIR}/timewindow_common.csh"
mkdir -p "$OUTPUT_ROOT" "$LOG_DIR"

if ( ! -d "$LEVEL_DIR" ) then
    echo "ERROR: LEVEL_DIR not found: ${LEVEL_DIR}"
    exit 1
endif
if ( ! -d "$TOG_DIR" ) then
    echo "ERROR: TOG_DIR not found: ${TOG_DIR}"
    exit 1
endif
if ( ! -d "$LABEL_DIR" ) then
    echo "ERROR: LABEL_DIR not found: ${LABEL_DIR}"
    exit 1
endif

if ( -f "${STEP1_DIR}/timewindow_manifest.json" ) then
    echo "Step 1: SKIP ${STEP1_DIR}/timewindow_manifest.json exists"
    exit 0
endif

set disc = "$BOUNDARY_DISCARD"
if ( "$disc" == "" ) set disc = `echo "$OFFSETS" | tr ',' '\n' | sort -n | tail -1`

echo "[`date '+%H:%M:%S'`] ===== Step 1: Time-Window Prepare ====="
set SAMPLE_DIR = "${STEP1_DIR}/_sampling"
mkdir -p "$SAMPLE_DIR"
"$PYTHON" "${TIMEWINDOW_SRC}/step1_make_global_sample_index.py" \
    --level-dir "$LEVEL_DIR" --tog-dir "$TOG_DIR" --label-dir "$LABEL_DIR" \
    --output-dir "$SAMPLE_DIR" --offsets "$OFFSETS" --boundary-discard "$disc" \
    --test-count "$TEST_COUNT" --stage "$STAGE" --sample-rows-total "$SAMPLE_ROWS_TOTAL" \
    --sample-rows-per-file "$SAMPLE_ROWS_PER_FILE" --sample-power-bins "$SAMPLE_POWER_BINS" \
    --sample-delta-bins "$SAMPLE_DELTA_BINS" --sample-all-threshold "$SAMPLE_ALL_THRESHOLD" \
    --sample-min-per-case "$SAMPLE_MIN_PER_CASE" --sample-test-ratio "$SAMPLE_TEST_RATIO" \
    --weight-exponent "$SAMPLE_WEIGHT_EXPONENT" --weight-min "$SAMPLE_WEIGHT_MIN" \
    --weight-max "$SAMPLE_WEIGHT_MAX" --sample-seed "$SAMPLE_SEED"
if ( $status != 0 ) exit 1
"$PYTHON" "${TIMEWINDOW_SRC}/step1_prepare_timewindow.py" \
    --level-dir "$LEVEL_DIR" --tog-dir "$TOG_DIR" --label-dir "$LABEL_DIR" \
    --output-dir "$STEP1_DIR" --offsets "$OFFSETS" --data-types "$DATA_TYPES" \
    --boundary-discard "$disc" --test-count "$TEST_COUNT" --stage "$STAGE" \
    --sample-index-manifest "${SAMPLE_DIR}/sample_index_manifest.json" --sample-seed "$SAMPLE_SEED" \
    --jobs "$STEP1_JOBS" \
    --compress-output "$STEP1_COMPRESS_OUTPUT" --skip-existing "$STEP1_SKIP_EXISTING"
if ( $status != 0 ) exit 1
