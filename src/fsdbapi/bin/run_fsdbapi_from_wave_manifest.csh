#!/bin/csh -f
# Minimal csh FSDBAPI runner from wave_manifest.tsv.
set SCRIPT_DIR = `dirname "$0"`
set SCRIPT_DIR = `cd "$SCRIPT_DIR" && pwd`
if ( ! $?FSDBAPI ) set FSDBAPI = `cd "${SCRIPT_DIR}/.." && pwd`
if ( ! $?PYTHON ) set PYTHON = python3
if ( ! $?DESIGN_NAME ) set DESIGN_NAME = boom
if ( ! $?WAVE_MANIFEST ) set WAVE_MANIFEST = "${FSDBAPI}/wave_manifest.tsv"
if ( ! $?OUT ) set OUT = "${FSDBAPI}/out/${DESIGN_NAME}_fsdbapi_cache"
if ( ! $?SOURCE_ROOT ) set SOURCE_ROOT = "${OUT}/_source_exports"
if ( ! $?MERGED_OUT ) set MERGED_OUT = "${OUT}/merged"
if ( ! $?LOG_DIR ) set LOG_DIR = "${OUT}/log"
if ( ! $?PLAN_TSV ) set PLAN_TSV = "${OUT}/fsdbapi_plan.tsv"
if ( ! $?FSDBAPI_MANIFEST ) set FSDBAPI_MANIFEST = "${OUT}/fsdbapi_manifest.tsv"
if ( ! $?STEP ) set STEP = 2
if ( ! $?RAW_COL_BLOCK ) set RAW_COL_BLOCK = 100000
if ( ! $?TOG_MODE ) set TOG_MODE = sample-xor
if ( ! $?INCLUDE_END_SAMPLE ) set INCLUDE_END_SAMPLE = 1
if ( ! $?DROP_FIRST_TOG_ROW ) set DROP_FIRST_TOG_ROW = 1
if ( ! $?NO_LEVEL_CACHE ) set NO_LEVEL_CACHE = 0
if ( ! $?MERGE_MANIFESTS ) set MERGE_MANIFESTS = 1
if ( ! $?CONTINUE_ON_ERROR ) set CONTINUE_ON_ERROR = 0
if ( ! $?REQUIRE_INPUT_EXISTS ) set REQUIRE_INPUT_EXISTS = 1
if ( ! $?MERGE_MAX_MEM ) set MERGE_MAX_MEM = 28GB
mkdir -p "$SOURCE_ROOT" "$MERGED_OUT" "$LOG_DIR"
set FSDB_BIN = "${FSDBAPI}/build/fsdb_to_tog_cache"
set MERGE_PY = "${FSDBAPI}/bin/merge_tog_cache_manifests.py"
"$PYTHON" - "$WAVE_MANIFEST" "$PLAN_TSV" "$SOURCE_ROOT" "$LOG_DIR" "$DESIGN_NAME" "$REQUIRE_INPUT_EXISTS" <<'INNERPY'
import csv, re, sys
from pathlib import Path
wm, plan, source_root, log_dir, design, require_exists = sys.argv[1:]
require_exists = require_exists == '1'
def clean(v):
    return re.sub(r'[^A-Za-z0-9_.+-]+', '_', str(v).strip()).strip('_') or 'sample'
rows=[]
with open(wm, newline='', encoding='utf-8') as f:
    for r in csv.DictReader(f, delimiter='\t'):
        if r.get('design') and r.get('design') != design:
            continue
        fsdb = r.get('wave_path','').strip()
        cid = clean(r.get('sample_id') or Path(fsdb).stem)
        start = r.get('start_ns','auto') or 'auto'
        end = r.get('end_ns','auto') or 'auto'
        status = 'ok'
        msg = ''
        if require_exists and not Path(fsdb).is_file():
            status='missing_wave'; msg=f'missing {fsdb}'
        out = Path(source_root)/cid
        rows.append([cid, fsdb, start, end, str(out), str(out/'_work/tog_cache_manifest.json'), str(Path(log_dir)/'fsdbapi'/f'{cid}.log'), status, msg])
Path(plan).parent.mkdir(parents=True, exist_ok=True)
with open(plan,'w',encoding='utf-8') as f:
    f.write('cache_id\tfsdb_path\tstart_ns\tend_ns\toutput_dir\tcache_manifest\tlog_path\tinput_status\tinput_message\n')
    for row in rows: f.write('\t'.join(row)+'\n')
INNERPY
if ( $status != 0 ) exit 1
echo "cache_id	status	cache_manifest	message" > "$FSDBAPI_MANIFEST"
set manifest_args = ""
foreach line ( `tail -n +2 "$PLAN_TSV"` )
    set cache_id = `echo "$line" | awk -F'\t' '{print $1}'`
    set fsdb = `echo "$line" | awk -F'\t' '{print $2}'`
    set start_ns = `echo "$line" | awk -F'\t' '{print $3}'`
    set end_ns = `echo "$line" | awk -F'\t' '{print $4}'`
    set output_dir = `echo "$line" | awk -F'\t' '{print $5}'`
    set cache_manifest = `echo "$line" | awk -F'\t' '{print $6}'`
    set log_path = `echo "$line" | awk -F'\t' '{print $7}'`
    set input_status = `echo "$line" | awk -F'\t' '{print $8}'`
    mkdir -p "$output_dir" `dirname "$log_path"`
    if ( "$input_status" != "ok" ) then
        echo "${cache_id}	failed	${cache_manifest}	${input_status}" >> "$FSDBAPI_MANIFEST"
        if ( "$CONTINUE_ON_ERROR" != "1" ) exit 1
        continue
    endif
    set args = ( --fsdb "$fsdb" --output-dir "$output_dir" --dataset-name "$cache_id" --step "$STEP" --raw-col-block "$RAW_COL_BLOCK" --tog-mode "$TOG_MODE" )
    if ( "$start_ns" != "auto" ) set args = ( $args --start "$start_ns" )
    if ( "$end_ns" != "auto" ) set args = ( $args --end "$end_ns" )
    if ( "$INCLUDE_END_SAMPLE" == "1" ) set args = ( $args --end-inclusive )
    if ( "$DROP_FIRST_TOG_ROW" == "1" ) set args = ( $args --drop-first-tog-row )
    if ( "$NO_LEVEL_CACHE" == "1" ) set args = ( $args --no-level-cache )
    "$FSDB_BIN" $args >& "$log_path"
    if ( $status == 0 ) then
        echo "${cache_id}	success	${cache_manifest}	" >> "$FSDBAPI_MANIFEST"
        set manifest_args = "${manifest_args} --manifest ${cache_manifest}"
    else
        echo "${cache_id}	failed	${cache_manifest}	see ${log_path}" >> "$FSDBAPI_MANIFEST"
        if ( "$CONTINUE_ON_ERROR" != "1" ) exit 1
    endif
end
if ( "$MERGE_MANIFESTS" == "1" ) then
    "$PYTHON" "$MERGE_PY" --output-dir "$MERGED_OUT" --max-mem "$MERGE_MAX_MEM" $manifest_args
    if ( $status != 0 ) exit 1
endif
