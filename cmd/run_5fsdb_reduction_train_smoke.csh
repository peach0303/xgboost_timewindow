#!/bin/csh -f
# 5-FSDB csh smoke: Raw FSDB -> FSDBAPI cache -> merge -> reduction -> timewindow step1-3.
set SCRIPT_DIR = `dirname "$0"`
set SCRIPT_DIR = `cd "$SCRIPT_DIR" && pwd`
if ( ! $?PACKAGE_ROOT ) set PACKAGE_ROOT = `cd "${SCRIPT_DIR}/.." && pwd`
if ( ! $?PROJECT_ROOT ) set PROJECT_ROOT = `cd "${PACKAGE_ROOT}/.." && pwd`
if ( ! $?PYTHON ) set PYTHON = python3
if ( ! $?FSDBAPI ) set FSDBAPI = "${PACKAGE_ROOT}/src/fsdbapi"
if ( ! $?FSDB_DIR ) set FSDB_DIR = "${PROJECT_ROOT}/mpsoc/core-level/data/RawFSDB_161/boom/coremark_extract"
if ( ! $?LABEL_DIR ) set LABEL_DIR = "${PROJECT_ROOT}/output/timewindow_test/power_labels"
if ( ! $?RUN_TAG ) set RUN_TAG = `date '+boom_coremark_5fsdb_%Y%m%d_%H%M%S'`
if ( ! $?OUTPUT_ROOT ) set OUTPUT_ROOT = "${PROJECT_ROOT}/output/5fsdb_reduction_train_smoke/${RUN_TAG}"
if ( ! $?LOG_DIR ) set LOG_DIR = "${OUTPUT_ROOT}/logs"
if ( ! $?STATUS_FILE ) set STATUS_FILE = "${OUTPUT_ROOT}/STATUS.txt"
if ( ! $?FSDB_IDS_LIST ) set FSDB_IDS_LIST = ( 10 11 12 13 14 )
if ( ! $?STEP ) set STEP = 2
if ( ! $?RAW_COL_BLOCK ) set RAW_COL_BLOCK = 20000
if ( ! $?MERGE_MAX_MEM ) set MERGE_MAX_MEM = 16GB
if ( ! $?MERGE_MEM_FRAC ) set MERGE_MEM_FRAC = 0.40
if ( ! $?REDUCE_MAX_MEM ) set REDUCE_MAX_MEM = 16GB
if ( ! $?REDUCE_TARGET_COLS ) set REDUCE_TARGET_COLS = 50000
if ( ! $?SMOKE_OFFSETS ) set SMOKE_OFFSETS = 6
if ( ! $?SMOKE_DATA_TYPES ) set SMOKE_DATA_TYPES = "tog,level"
if ( ! $?SMOKE_TEST_COUNT ) set SMOKE_TEST_COUNT = 1
if ( ! $?SMOKE_SAMPLE_ROWS_PER_FILE ) set SMOKE_SAMPLE_ROWS_PER_FILE = 0
if ( ! $?SMOKE_SAMPLE_BINS ) set SMOKE_SAMPLE_BINS = 20
if ( ! $?SMOKE_SAMPLE_MIN_PER_BIN ) set SMOKE_SAMPLE_MIN_PER_BIN = 1
if ( ! $?SMOKE_SAMPLE_SEED ) set SMOKE_SAMPLE_SEED = 108
if ( ! $?SMOKE_TOP_K ) set SMOKE_TOP_K = 32
if ( ! $?SMOKE_BATCH_SIZE ) set SMOKE_BATCH_SIZE = 1
set FSDB_BIN = "${FSDBAPI}/build/fsdb_to_tog_cache"
set MERGE_PY = "${FSDBAPI}/bin/merge_tog_cache_manifests.py"
set REDUCE_PY = "${PACKAGE_ROOT}/src/reduction/dimentionality_reduction_csv.py"
set TIMEWINDOW_DIR = "${PACKAGE_ROOT}/src/timewindow"
mkdir -p "$OUTPUT_ROOT" "$LOG_DIR"
echo "[`date '+%Y-%m-%d %H:%M:%S'`] START output=${OUTPUT_ROOT}" > "$STATUS_FILE"

set fsdbapi_out = "${OUTPUT_ROOT}/01_fsdbapi_cache"
set source_root = "${fsdbapi_out}/_source_exports"
mkdir -p "$source_root"
echo "[`date '+%Y-%m-%d %H:%M:%S'`] STEP1 FSDBAPI export start" | tee -a "$STATUS_FILE"
set manifest_args = ""
foreach id ( $FSDB_IDS_LIST )
    set sample = "coremark_${id}"
    set fsdb = "${FSDB_DIR}/${sample}.fsdb"
    set label = "${LABEL_DIR}/${sample}.npy"
    if ( ! -f "$fsdb" ) then
        echo "Missing FSDB: $fsdb"
        exit 2
    endif
    if ( ! -f "$label" ) then
        echo "Missing label: $label"
        exit 2
    endif
    set out_dir = "${source_root}/${sample}"
    set log_file = "${LOG_DIR}/${sample}.fsdbapi.log"
    mkdir -p "$out_dir"
    "$FSDB_BIN" --fsdb "$fsdb" --output-dir "$out_dir" --dataset-name "$sample" --step "$STEP" --raw-col-block "$RAW_COL_BLOCK" --tog-mode sample-xor --end-inclusive --drop-first-tog-row >& "$log_file"
    if ( $status != 0 ) then
        echo "FSDBAPI failed: $sample, see $log_file"
        exit 1
    endif
    set manifest_args = "${manifest_args} --manifest ${out_dir}/_work/tog_cache_manifest.json"
end
echo "[`date '+%Y-%m-%d %H:%M:%S'`] STEP1 FSDBAPI export done" | tee -a "$STATUS_FILE"

set merged_out = "${fsdbapi_out}/merged"
echo "[`date '+%Y-%m-%d %H:%M:%S'`] STEP2 merge start" | tee -a "$STATUS_FILE"
"$PYTHON" "$MERGE_PY" --output-dir "$merged_out" --max-mem "$MERGE_MAX_MEM" --merge-mem-frac "$MERGE_MEM_FRAC" $manifest_args >& "${LOG_DIR}/02_merge.log"
if ( $status != 0 ) exit 1
echo "[`date '+%Y-%m-%d %H:%M:%S'`] STEP2 merge done" | tee -a "$STATUS_FILE"

set merged_manifest = "${merged_out}/_work/tog_cache_manifest.json"
set reduction_base = "${OUTPUT_ROOT}/02_reduction"
set reduced_dir = "${reduction_base}/boom_5fsdb"
mkdir -p "${reduced_dir}/_work" "${reduction_base}/logs"
set level_manifest = "${reduced_dir}/_work/level_cache_manifest.from_level_files.json"
"$PYTHON" - "$merged_manifest" "$level_manifest" <<'INNERPY'
import json, sys
from pathlib import Path
src=Path(sys.argv[1]); dst=Path(sys.argv[2]); m=json.loads(src.read_text())
m['files']=m.get('level_files', m.get('files', []))
if 'level_n_rows' in m: m['n_rows']=m['level_n_rows']
if 'level_nnz' in m: m['nnz']=m['level_nnz']
m['data_type']='level'
dst.write_text(json.dumps(m, indent=2), encoding='utf-8')
INNERPY
if ( $status != 0 ) exit 1

echo "[`date '+%Y-%m-%d %H:%M:%S'`] STEP3 reduction start" | tee -a "$STATUS_FILE"
"$PYTHON" -u "$REDUCE_PY" --input-cache-manifest "$level_manifest" --input-tog-cache-manifest "$merged_manifest" --output-dir "$reduced_dir" --output-meta "${reduced_dir}/meta.json" --output-cols "${reduced_dir}/final_cols.txt" --output-original-headers "${reduced_dir}/original_headers.txt" --max-mem "$REDUCE_MAX_MEM" --target-cols "$REDUCE_TARGET_COLS" --cache-format csc --save-after-one --disable-duplicate-remove --disable-sim-reduce --duplicate-mode hash-only --input-data-type level --snapshot-mem-frac 0.35 >& "${LOG_DIR}/03_reduction.log"
if ( $status != 0 ) exit 1
echo "[`date '+%Y-%m-%d %H:%M:%S'`] STEP3 reduction done" | tee -a "$STATUS_FILE"

echo "[`date '+%Y-%m-%d %H:%M:%S'`] STEP4 timewindow interface smoke start" | tee -a "$STATUS_FILE"
set train_out = "${OUTPUT_ROOT}/03_timewindow_interface_smoke"
mkdir -p "$train_out"
"$PYTHON" "${TIMEWINDOW_DIR}/step1_prepare_timewindow.py" --level-dir "${reduced_dir}/level" --tog-dir "${reduced_dir}/tog" --label-dir "$LABEL_DIR" --output-dir "${train_out}/step1_shifted" --offsets "$SMOKE_OFFSETS" --data-types "$SMOKE_DATA_TYPES" --test-count "$SMOKE_TEST_COUNT" --stage after_one --sample-rows-per-file "$SMOKE_SAMPLE_ROWS_PER_FILE" --sample-bins "$SMOKE_SAMPLE_BINS" --sample-min-per-bin "$SMOKE_SAMPLE_MIN_PER_BIN" --sample-seed "$SMOKE_SAMPLE_SEED" >& "${LOG_DIR}/04_step1_timewindow.log"
if ( $status != 0 ) exit 1
"$PYTHON" "${TIMEWINDOW_DIR}/step2_mcp_per_offset.py" --manifest "${train_out}/step1_shifted/timewindow_manifest.json" --namefile "${reduced_dir}/level/after_one.namefile.txt" --batch "$SMOKE_BATCH_SIZE" --top-k "$SMOKE_TOP_K" --output-dir "${train_out}/step2_mcp" >& "${LOG_DIR}/05_step2_mcp.log"
if ( $status != 0 ) exit 1
"$PYTHON" "${TIMEWINDOW_DIR}/step3_concat_features.py" --tw-manifest "${train_out}/step1_shifted/timewindow_manifest.json" --mcp-manifest "${train_out}/step2_mcp/mcp_per_offset_manifest.json" --namefile "${reduced_dir}/level/after_one.namefile.txt" --output-dir "${train_out}/step3_concat" >& "${LOG_DIR}/06_step3_concat.log"
if ( $status != 0 ) exit 1
echo "[`date '+%Y-%m-%d %H:%M:%S'`] STEP4 timewindow interface smoke done" | tee -a "$STATUS_FILE"
echo "[`date '+%Y-%m-%d %H:%M:%S'`] DONE" | tee -a "$STATUS_FILE"
