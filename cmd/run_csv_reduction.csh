#!/bin/csh -f
# CSV/cache reduction wrapper in csh. Supports cache-manifest mode and simple CSV mode.
set SCRIPT_DIR = `dirname "$0"`
set SCRIPT_DIR = `cd "$SCRIPT_DIR" && pwd`
if ( ! $?PACKAGE_ROOT ) set PACKAGE_ROOT = `cd "${SCRIPT_DIR}/.." && pwd`
if ( ! $?PROJECT_ROOT ) set PROJECT_ROOT = `cd "${PACKAGE_ROOT}/.." && pwd`
if ( ! $?PYTHON ) set PYTHON = python3
if ( ! $?REDUCTION_PY ) set REDUCTION_PY = "${PACKAGE_ROOT}/src/reduction/dimentionality_reduction_csv.py"
if ( ! $?DATASET_DIR ) set DATASET_DIR = "${PROJECT_ROOT}/dataset/RawCsv"
if ( ! $?OUTPUT_BASE_DIR ) set OUTPUT_BASE_DIR = "${PROJECT_ROOT}/output/csv_reduction"
if ( ! $?LOG_DIR ) set LOG_DIR = "${OUTPUT_BASE_DIR}/logs"
if ( ! $?MAX_MEM_INITIAL ) set MAX_MEM_INITIAL = 180GB
if ( ! $?ROW_CHUNKSIZE ) set ROW_CHUNKSIZE = 1024
if ( ! $?TARGET_COLS ) set TARGET_COLS = 500000
if ( ! $?DUPLICATE_MODE ) set DUPLICATE_MODE = hash-only
if ( ! $?INPUT_CACHE_MANIFEST ) set INPUT_CACHE_MANIFEST = ""
if ( ! $?INPUT_TOG_CACHE_MANIFEST ) set INPUT_TOG_CACHE_MANIFEST = ""
if ( ! $?CACHE_INPUT_DATA_TYPE ) set CACHE_INPUT_DATA_TYPE = level
if ( ! $?DISABLE_DUPLICATE_REMOVE ) set DISABLE_DUPLICATE_REMOVE = 0
if ( ! $?DISABLE_SIM_REDUCE ) set DISABLE_SIM_REDUCE = 1
if ( ! $?SAVE_FINAL ) set SAVE_FINAL = 1
if ( ! $?SNAPSHOT_MEM_FRAC ) set SNAPSHOT_MEM_FRAC = 0.40
if ( ! $?SNAPSHOT_ROW_BLOCK_ROWS ) set SNAPSHOT_ROW_BLOCK_ROWS = 0
if ( ! $?FORCE_ROW_SHARD_SAVE ) set FORCE_ROW_SHARD_SAVE = 0
if ( $#argv < 1 ) then
    echo "Usage: csh run_csv_reduction.csh <hardware_name>"
    exit 2
endif
set HW = "$1"
mkdir -p "$OUTPUT_BASE_DIR" "$LOG_DIR"
set output_dir = "${OUTPUT_BASE_DIR}/${HW}"
set log_file = "${LOG_DIR}/${HW}_cache_run.log"
set level_manifest = "$INPUT_CACHE_MANIFEST"
set tog_manifest = "$INPUT_TOG_CACHE_MANIFEST"
if ( "$tog_manifest" == "" ) set tog_manifest = "$INPUT_CACHE_MANIFEST"
if ( "$INPUT_CACHE_MANIFEST" != "" ) then
    if ( ! -f "$INPUT_CACHE_MANIFEST" ) then
        echo "ERROR: INPUT_CACHE_MANIFEST not found: $INPUT_CACHE_MANIFEST"
        exit 2
    endif
    mkdir -p "${output_dir}/_work"
    set derived = "${output_dir}/_work/level_cache_manifest.from_level_files.json"
    "$PYTHON" - "$INPUT_CACHE_MANIFEST" "$derived" <<'INNERPY'
import json, sys
from pathlib import Path
src = Path(sys.argv[1]); dst = Path(sys.argv[2])
m = json.loads(src.read_text())
if m.get('level_files'):
    m['files'] = m.get('level_files', [])
    if 'level_n_rows' in m: m['n_rows'] = m['level_n_rows']
    if 'level_nnz' in m: m['nnz'] = m['level_nnz']
    m['data_type'] = 'level'
    dst.write_text(json.dumps(m, indent=2), encoding='utf-8')
    print(dst)
else:
    print(src)
INNERPY
    if ( $status != 0 ) exit 1
    set maybe_level = `tail -1 "$derived" 2>/dev/null`
    if ( "$maybe_level" != "" ) set level_manifest = "$maybe_level"
    set args = ( --input-cache-manifest "$level_manifest" --input-tog-cache-manifest "$tog_manifest" --output-dir "$output_dir" --output-meta "${output_dir}/meta.json" --output-cols "${output_dir}/final_cols.txt" --output-original-headers "${output_dir}/original_headers.txt" --max-mem "$MAX_MEM_INITIAL" --target-cols "$TARGET_COLS" --row-chunksize "$ROW_CHUNKSIZE" --cache-format csc --save-after-one --duplicate-mode "$DUPLICATE_MODE" --input-data-type "$CACHE_INPUT_DATA_TYPE" --snapshot-mem-frac "$SNAPSHOT_MEM_FRAC" )
    if ( "$DISABLE_DUPLICATE_REMOVE" == "1" ) set args = ( $args --disable-duplicate-remove )
    if ( "$DISABLE_SIM_REDUCE" == "1" ) set args = ( $args --disable-sim-reduce )
    if ( "$SAVE_FINAL" == "1" ) set args = ( $args --save-final )
    if ( "$FORCE_ROW_SHARD_SAVE" == "1" ) set args = ( $args --force-row-shard-save )
    if ( "$SNAPSHOT_ROW_BLOCK_ROWS" != "0" ) set args = ( $args --snapshot-row-block-rows "$SNAPSHOT_ROW_BLOCK_ROWS" )
    echo "Running cache reduction -> $output_dir"
    "$PYTHON" -u "$REDUCTION_PY" $args >& "$log_file"
    exit $status
endif
set hw_dir = "${DATASET_DIR}/${HW}"
if ( ! -d "$hw_dir" ) then
    echo "ERROR: hardware directory not found: $hw_dir"
    exit 2
endif
set filelist = "${output_dir}/filelist.txt"
mkdir -p "$output_dir"
find "$hw_dir" -type f \( -name "*.csv" -o -name "*.csv.tgz" \) | sort > "$filelist"
"$PYTHON" -u "$REDUCTION_PY" --input-filelist "$filelist" --output-dir "$output_dir" --output-meta "${output_dir}/meta.json" --output-cols "${output_dir}/final_cols.txt" --output-original-headers "${output_dir}/original_headers.txt" --max-mem "$MAX_MEM_INITIAL" --target-cols "$TARGET_COLS" --row-chunksize "$ROW_CHUNKSIZE" --duplicate-mode "$DUPLICATE_MODE" >& "$log_file"
exit $status
