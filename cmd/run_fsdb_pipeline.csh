#!/bin/csh -f
#===============================================================================
# FSDB → Level+TOG 降维 全流程脚本
# 用法: csh run_fsdb_pipeline.csh <hardware_name>
# 例:   csh run_fsdb_pipeline.csh boom
#===============================================================================

# ==================== 基础路径（可按需修改） ====================
set SCRIPT_DIR  = `dirname "$0"`
set PACKAGE_ROOT = `cd "${SCRIPT_DIR}/.." && pwd`
set PROJECT_ROOT = `cd "${PACKAGE_ROOT}/.." && pwd`
set TOOLS_ROOT  = "${PACKAGE_ROOT}"
set FSDB_DIR    = "${PROJECT_ROOT}/mpsoc/core-level/data/RawFSDB_161"
set OUTPUT_BASE = "${PROJECT_ROOT}/output/xgboost_timewindow_fsdb_pipeline"
set WORK_DIR    = "${PROJECT_ROOT}/tmp/xgboost_timewindow_fsdb_work"

# ==================== 待修改: FSDB 文件列表 ====================
# 按 workload_name fsdb_path start_ns end_ns 格式填写，每行一个
# workload_name: 自定义名称，作为 cache_id 和输出前缀
# fsdb_path:     FSDB 文件绝对路径
# start_ns:      采样起始时间（FSDB tag 单位）
# end_ns:        采样结束时间（FSDB tag 单位）
set fsdb_list = ( \
    "coremark_1  ${FSDB_DIR}/boom/coremark_extract/coremark_1.fsdb  0  50000" \
    "coremark_2  ${FSDB_DIR}/boom/coremark_extract/coremark_2.fsdb  0  50000" \
)
# ==================== 待修改: Power FSDB 配置（可选） ====================
# 格式: cache_id  power_fsdb  power_signal
# cache_id 需与上方 fsdb_list 同名，确保特征与标签一一对应
# 时间窗自动从 fsdb_list 对齐（按 cache_id 匹配 start/end）
# 不需要功耗则留空 power_list
set power_list = ( \
    "coremark_1  ${FSDB_DIR}/power/coremark_1.fsdb  top.chip.power" \
    "coremark_2  ${FSDB_DIR}/power/coremark_2.fsdb  top.chip.power" \
)

# ==================== 待修改: 采样参数 ====================
set STEP       = 2           # 采样间隔（FSDB tag 单位）
set END_INCL   = 1           # 是否包含 end 采样点 (1=是 0=否)
set DROP_FIRST = 1           # TOG 丢弃首行 (1=是 0=否)
set RAW_COL    = 20000       # 每块原始列数（内存/速度平衡）
set REVERSE_BUS_BITS = 0     # 若 debug 发现 bus bit 顺序与 Verdi 相反，设为 1
set VIEW_START = ""          # 可选：提前 FSDB load window 起点，用于保证 start 处初始电平正确；留空=使用 start

# ==================== 待修改: 层级过滤（可选） ====================
set SIGNAL_PREFIX = ""       # 留空=不过滤；各前缀用空格分隔
# set SIGNAL_PREFIX = ( "boom_tile.core." "boom_tile.lsu." )

# ==================== 降维参数 ====================
set MAX_MEM    = "180GB"
set TARGET_COLS = 500000
set MAX_CYCLES_PER_FILE = 1000  # 每个 NPZ 文件最大周期数（0=不限制）
set FSDB_CHUNK_CYCLES = 1000  # 已弃用：保留用于旧 time-window split 配置，不再用于 Step2

# ==================== LSF/bsub 参数 ====================
set BSUB_QUEUE = ""          # 留空=使用默认队列；例如: normal
set BSUB_EXTRA_ARGS = ( )     # 例如: ( -R "rusage[mem=32000]" -M 32000 )
set POLL_SEC = 30            # 主流程轮询 complete/failed 的间隔秒数

#===============================================================================
# 初始化
#===============================================================================
echo "============================================================"
echo "FSDB Pipeline: ${1}"
echo "Start: `date`"
echo "============================================================"

if ( $#argv < 1 ) then
    echo "Usage: csh run_fsdb_pipeline.csh <hardware_name>"
    exit 1
endif
set HW = $1

# 可选激活环境：如果包根目录存在 activate.csh，则加载；否则使用当前 shell 环境。
if ( -f ${TOOLS_ROOT}/activate.csh ) then
    source ${TOOLS_ROOT}/activate.csh
endif

set CACHE_DIR  = "${WORK_DIR}/cache_${HW}"
set MERGED_DIR = "${OUTPUT_BASE}/${HW}_cache"
set REDUCE_DIR = "${OUTPUT_BASE}/${HW}_reduced"
set LOG_DIR    = "${OUTPUT_BASE}/logs"

mkdir -p ${WORK_DIR} ${OUTPUT_BASE} ${LOG_DIR} ${REDUCE_DIR}
rm -rf ${CACHE_DIR}
mkdir -p ${CACHE_DIR}
rm -f ${MERGED_DIR}/complete ${MERGED_DIR}/failed ${REDUCE_DIR}/complete ${REDUCE_DIR}/failed ${OUTPUT_BASE}/complete

# FSDBAPI 二进制
set FSDBAPI_ROOT = "${TOOLS_ROOT}/src/fsdbapi"
set FSDB_BIN   = "${FSDBAPI_ROOT}/build/fsdb_to_tog_cache"
set MERGE_PY   = "${FSDBAPI_ROOT}/bin/merge_tog_cache_manifests.py"
set CONCAT_TIME_PY = "${FSDBAPI_ROOT}/bin/concat_tog_cache_time_windows.py"
set CONCAT_BLOCK_PY = "${FSDBAPI_ROOT}/bin/concat_tog_cache_blocks.py"
set REDUCE_PY  = "${TOOLS_ROOT}/src/reduction/dimentionality_reduction_csv.py"

#===============================================================================
# Step 1: 信号过滤（如果指定了前缀）
#===============================================================================
set SIGNAL_LIST_ARG = ""
set SIG_FILE = "${WORK_DIR}/${HW}_signal_filter.txt"
rm -f ${SIG_FILE}

# 判断是否配置了前缀过滤: 变量存在且非空字符串
set DO_FILTER = 0
if ( $?SIGNAL_PREFIX ) then
    # 数组: 检查第一个元素是否非空
    if ( "$SIGNAL_PREFIX" != "" ) set DO_FILTER = 1
endif

if ( ${DO_FILTER} == 1 ) then
    echo ""
    echo "=== Step 1: 按前缀过滤信号 ==="
    # 从第一个 FSDB 获取信号列表
    set first_line = "$fsdb_list[1]"
    set first_fsdb = `echo "${first_line}" | awk '{print $2}'`
    echo "  读取信号: ${first_fsdb}"
    ${FSDB_BIN} --fsdb ${first_fsdb} --list-signals ${WORK_DIR}/all_signals.txt

    if ( ! -f ${WORK_DIR}/all_signals.txt || `wc -l < ${WORK_DIR}/all_signals.txt` == 0 ) then
        echo "  ❌ --list-signals 失败，检查 FSDB 路径和 LD_LIBRARY_PATH"
        exit 1
    endif
    echo "  总信号数: `wc -l < ${WORK_DIR}/all_signals.txt`"

    # 按前缀筛选
    foreach prefix ( $SIGNAL_PREFIX:q )
        grep "${prefix}" ${WORK_DIR}/all_signals.txt | awk '{print $1}' >> ${SIG_FILE}
    end
    set sig_count = `wc -l < ${SIG_FILE}`
    if ( ${sig_count} == 0 ) then
        echo "  ❌ 未匹配到任何信号，检查 SIGNAL_PREFIX 是否正确"
        echo "  查看信号列表: head -20 ${WORK_DIR}/all_signals.txt"
        exit 1
    endif
    echo "  过滤后: ${sig_count} 信号"
    set SIGNAL_LIST_ARG = "--signal-list ${SIG_FILE}"
else
    echo ""
    echo "=== Step 1: 跳过（未配置前缀过滤） ==="
endif

#===============================================================================
# Step 2: FSDB → Cache（每个 FSDB 按 block 并行导出）
#===============================================================================
echo ""
echo "=== Step 2: FSDB → Cache（每 case 按 block bsub 并行提交） ==="
set MANIFEST_ARGS = ""
set OK_COUNT = 0
set FSDB_COUNT = $#fsdb_list
set BLOCK_TASK_COUNT = 0

foreach line ( $fsdb_list:q )
    set cache_id = `echo "$line" | awk '{print $1}'`
    set fsdb     = `echo "$line" | awk '{print $2}'`
    set start_ns = `echo "$line" | awk '{print $3}'`
    set end_ns   = `echo "$line" | awk '{print $4}'`
    echo "  [${cache_id}] parsed fsdb=${fsdb} start=${start_ns} end=${end_ns} step=${STEP}"
    python3 -c 'import sys; s=sys.argv[1]; e=sys.argv[2]; int(s,0); int(e,0); assert int(e,0) > int(s,0), f"end must be larger than start: start={s} end={e}"' ${start_ns} ${end_ns}
    if ( $status != 0 ) then
        echo "ERROR: ${cache_id} 时间窗非法，请检查 fsdb_list 第3/4列: start=${start_ns} end=${end_ns}"
        touch ${CACHE_DIR}/failed
        exit 1
    endif

    set case_dir = "${CACHE_DIR}/${cache_id}"
    set block_root = "${case_dir}/blocks"
    set block_table = "${case_dir}/block_list.tsv"
    mkdir -p ${case_dir} ${block_root}
    rm -f ${case_dir}/complete ${case_dir}/failed ${case_dir}/status ${block_table}

    ${FSDB_BIN} \
        --fsdb ${fsdb} \
        --output-dir ${case_dir}/_block_list_probe \
        --dataset-name ${cache_id} \
        --raw-col-block ${RAW_COL} \
        ${SIGNAL_LIST_ARG} \
        --list-blocks ${block_table} >& ${LOG_DIR}/${cache_id}.list_blocks.log
    if ( $status != 0 || ! -f ${block_table} ) then
        echo "ERROR: ${cache_id} 生成 block 列表失败，查看 ${LOG_DIR}/${cache_id}.list_blocks.log"
        touch ${case_dir}/failed
        exit 1
    endif

    set block_ids = (`awk 'NR>1 {print $1}' ${block_table}`)
    echo "  [${cache_id}] split by block: $#block_ids blocks"

    foreach block_id ( $block_ids )
        set block_tag = `printf "%05d" ${block_id}`
        set block_signals = `awk -v id=${block_id} 'NR>1 && $1==id {print $4}' ${block_table}`
        set block_bits = `awk -v id=${block_id} 'NR>1 && $1==id {print $5}' ${block_table}`
        set out_dir  = "${block_root}/block_${block_tag}"
        mkdir -p ${out_dir}
        set log_file = "${LOG_DIR}/${cache_id}.block${block_tag}.log"
        set job_file = "${out_dir}/run_fsdb_to_tog_cache.csh"
        set bsub_out = "${LOG_DIR}/${cache_id}.block${block_tag}.bsub.out"
        set bsub_err = "${LOG_DIR}/${cache_id}.block${block_tag}.bsub.err"
        rm -f ${out_dir}/complete ${out_dir}/failed ${out_dir}/status ${job_file}

        echo "  [${cache_id}/block${block_tag}] bsub submit signals=${block_signals} bits=${block_bits}"

        set cmd = "${FSDB_BIN} \
            --fsdb ${fsdb} \
            --output-dir ${out_dir} \
            --dataset-name ${cache_id}_block${block_tag} \
            --step ${STEP} \
            --start ${start_ns} \
            --end ${end_ns} \
            --raw-col-block ${RAW_COL} \
            --only-block-id ${block_id} \
            --tog-mode sample-xor \
            --max-cycles-per-file ${MAX_CYCLES_PER_FILE} \
            ${SIGNAL_LIST_ARG}"

        if ( ${END_INCL} == 1 )   set cmd = "${cmd} --end-inclusive"
        if ( ${DROP_FIRST} == 1 ) set cmd = "${cmd} --drop-first-tog-row"
        if ( ${REVERSE_BUS_BITS} == 1 ) set cmd = "${cmd} --reverse-bus-bits"
        if ( "${VIEW_START}" != "" ) set cmd = "${cmd} --view-start ${VIEW_START}"

        cat > ${job_file} << EOF
#!/bin/csh -f
echo "[job] start `date` host=`hostname` pwd=`pwd`" >& ${log_file}
echo "[job] case=${cache_id} block=${block_id} signals=${block_signals} bits=${block_bits}" >>& ${log_file}
echo "[job] command: ${cmd}" >>& ${log_file}
if ( -f ${TOOLS_ROOT}/activate.csh ) then
    echo "[job] source ${TOOLS_ROOT}/activate.csh" >>& ${log_file}
    source ${TOOLS_ROOT}/activate.csh >>& ${log_file}
endif
${cmd} >>& ${log_file}
set rc = \$status
echo \${rc} > ${out_dir}/status
if ( \${rc} == 0 && -f ${out_dir}/_work/tog_cache_manifest.json ) then
    touch ${out_dir}/complete
else
    touch ${out_dir}/failed
endif
exit \${rc}
EOF

        if ( "${BSUB_QUEUE}" != "" ) then
            if ( $#BSUB_EXTRA_ARGS > 0 ) then
                bsub -q ${BSUB_QUEUE} -J fsdb_${HW}_${cache_id}_b${block_id} -o ${bsub_out} -e ${bsub_err} $BSUB_EXTRA_ARGS:q csh -f ${job_file}
            else
                bsub -q ${BSUB_QUEUE} -J fsdb_${HW}_${cache_id}_b${block_id} -o ${bsub_out} -e ${bsub_err} csh -f ${job_file}
            endif
        else
            if ( $#BSUB_EXTRA_ARGS > 0 ) then
                bsub -J fsdb_${HW}_${cache_id}_b${block_id} -o ${bsub_out} -e ${bsub_err} $BSUB_EXTRA_ARGS:q csh -f ${job_file}
            else
                bsub -J fsdb_${HW}_${cache_id}_b${block_id} -o ${bsub_out} -e ${bsub_err} csh -f ${job_file}
            endif
        endif
        if ( $status != 0 ) then
            echo "  [${cache_id}/block${block_tag}] bsub 提交失败"
            touch ${out_dir}/failed
        endif
        @ BLOCK_TASK_COUNT = $BLOCK_TASK_COUNT + 1
    end
end

echo "  监测所有 FSDB cache block 子任务 complete/failed..."
while ( 1 )
    set DONE_COUNT = 0
    set FAIL_COUNT = 0
    foreach line ( $fsdb_list:q )
        set cache_id = `echo "$line" | awk '{print $1}'`
        set case_dir = "${CACHE_DIR}/${cache_id}"
        set block_root = "${case_dir}/blocks"
        set block_table = "${case_dir}/block_list.tsv"
        set block_ids = (`awk 'NR>1 {print $1}' ${block_table}`)
        foreach block_id ( $block_ids )
            set block_tag = `printf "%05d" ${block_id}`
            set out_dir = "${block_root}/block_${block_tag}"
            if ( -f ${out_dir}/complete ) @ DONE_COUNT = $DONE_COUNT + 1
            if ( -f ${out_dir}/failed ) @ FAIL_COUNT = $FAIL_COUNT + 1
        end
    end
    if ( ${FAIL_COUNT} > 0 ) then
        echo "ERROR: 有 ${FAIL_COUNT} 个 FSDB cache block 子任务失败"
        touch ${CACHE_DIR}/failed
        exit 1
    endif
    if ( ${DONE_COUNT} == ${BLOCK_TASK_COUNT} ) break
    echo "  FSDB cache block progress: ${DONE_COUNT}/${BLOCK_TASK_COUNT}; sleep ${POLL_SEC}s"
    sleep ${POLL_SEC}
end

echo "  拼接每个 case 的 block manifest..."
foreach line ( $fsdb_list:q )
    set cache_id = `echo "$line" | awk '{print $1}'`
    set case_dir = "${CACHE_DIR}/${cache_id}"
    set block_root = "${case_dir}/blocks"
    set block_table = "${case_dir}/block_list.tsv"
    set block_ids = (`awk 'NR>1 {print $1}' ${block_table}`)
    set CONCAT_ARGS = ""
    foreach block_id ( $block_ids )
        set block_tag = `printf "%05d" ${block_id}`
        set manifest_file = "${block_root}/block_${block_tag}/_work/tog_cache_manifest.json"
        if ( ! -f ${manifest_file} ) then
            echo "ERROR: 缺失 block manifest: ${manifest_file}"
            touch ${case_dir}/failed
            exit 1
        endif
        set CONCAT_ARGS = "${CONCAT_ARGS} --manifest ${manifest_file}"
    end

    python3 ${CONCAT_BLOCK_PY} --output-dir ${case_dir} --dataset-name ${cache_id} ${CONCAT_ARGS} >& ${LOG_DIR}/${cache_id}.concat_blocks.log
    if ( $status != 0 || ! -f ${case_dir}/_work/tog_cache_manifest.json ) then
        echo "ERROR: ${cache_id} block manifest 拼接失败，查看 ${LOG_DIR}/${cache_id}.concat_blocks.log"
        touch ${case_dir}/failed
        exit 1
    endif
    touch ${case_dir}/complete
end

foreach line ( $fsdb_list:q )
    set cache_id = `echo "$line" | awk '{print $1}'`
    set case_dir = "${CACHE_DIR}/${cache_id}"
    set manifest_file = "${case_dir}/_work/tog_cache_manifest.json"

    if ( -f ${case_dir}/complete && -f ${manifest_file} ) then
        set MANIFEST_ARGS = "${MANIFEST_ARGS} --manifest ${manifest_file}"
        @ OK_COUNT = $OK_COUNT + 1
        echo "  [${cache_id}] done: blocks concatenated"
    else
        echo "  [${cache_id}] FAILED, check ${LOG_DIR}/${cache_id}.concat_blocks.log"
    endif
end

if ( ${OK_COUNT} != ${FSDB_COUNT} ) then
    echo "ERROR: 部分 FSDB 导出/拼接失败 (${OK_COUNT}/${FSDB_COUNT})"
    touch ${CACHE_DIR}/failed
    exit 1
endif

touch ${CACHE_DIR}/complete
echo "  Step 2 complete: ${CACHE_DIR}/complete (${OK_COUNT}/${FSDB_COUNT}, block_tasks=${BLOCK_TASK_COUNT})"

#===============================================================================
# Step 3: 功耗提取（Power FSDB → npy）
#===============================================================================
echo ""
echo "=== Step 3: 功耗提取（bsub 并行提交每个 Power FSDB） ==="

set POWER_BIN  = "${FSDBAPI_ROOT}/build/fsdb_to_power_label"
set POWER_DIR  = "${OUTPUT_BASE}/${HW}_power"
mkdir -p ${POWER_DIR}
rm -f ${POWER_DIR}/complete ${POWER_DIR}/failed

if ( "${power_list}" != "" ) then
    set POWER_COUNT = $#power_list
    foreach line ( $power_list:q )
        set cache_id    = `echo "$line" | awk '{print $1}'`
        set power_fsdb  = `echo "$line" | awk '{print $2}'`
        set power_sig   = `echo "$line" | awk '{print $3}'`
        set job_file = "${POWER_DIR}/${cache_id}.power_job.csh"
        set power_log = "${LOG_DIR}/${cache_id}.power.log"
        set npy_log = "${LOG_DIR}/${cache_id}.power.npy.log"
        set bsub_out = "${LOG_DIR}/${cache_id}.power.bsub.out"
        set bsub_err = "${LOG_DIR}/${cache_id}.power.bsub.err"
        rm -f ${POWER_DIR}/${cache_id}.complete ${POWER_DIR}/${cache_id}.failed ${POWER_DIR}/${cache_id}.status ${job_file}

        # 从 fsdb_list 查找匹配的时间窗
        set pwr_start = ""
        set pwr_end = ""
        foreach fline ( $fsdb_list:q )
            set fid = `echo "$fline" | awk '{print $1}'`
            if ( "${fid}" == "${cache_id}" ) then
                set pwr_start = `echo "$fline" | awk '{print $3}'`
                set pwr_end   = `echo "$fline" | awk '{print $4}'`
                break
            endif
        end

        if ( "${pwr_start}" == "" || "${pwr_end}" == "" ) then
            echo "  [${cache_id}] 未在 fsdb_list 中找到时间窗"
            touch ${POWER_DIR}/${cache_id}.failed
            continue
        endif

        echo "  [${cache_id}] bsub submit ${power_fsdb} → ${power_sig} (${pwr_start}-${pwr_end})"

        cat > ${job_file} << EOF
#!/bin/csh -f
echo "[job] start `date` host=`hostname` pwd=`pwd`" >& ${power_log}
echo "[job] power fsdb=${power_fsdb} signal=${power_sig}" >>& ${power_log}
if ( -f ${TOOLS_ROOT}/activate.csh ) then
    echo "[job] source ${TOOLS_ROOT}/activate.csh" >>& ${power_log}
    source ${TOOLS_ROOT}/activate.csh >>& ${power_log}
endif
${POWER_BIN} \
    --fsdb ${power_fsdb} \
    --power-signal ${power_sig} \
    --output-tsv ${POWER_DIR}/${cache_id}.power.tsv \
    --output-meta ${POWER_DIR}/${cache_id}.power.meta.json \
    --step ${STEP} \
    --start ${pwr_start} --end ${pwr_end} >>& ${power_log}
set rc = \$status
if ( \${rc} == 0 ) then
    python3 -c "
import numpy as np, csv
with open('${POWER_DIR}/${cache_id}.power.tsv') as f:
    vals = [float(r['power']) for r in csv.DictReader(f, delimiter='\t')]
if ${DROP_FIRST} == 1 and len(vals) > 1:
    vals = vals[1:]
np.save('${POWER_DIR}/${cache_id}.npy', np.array(vals, dtype='float32').reshape(-1,1))
print(f'power: {len(vals)} rows → ${cache_id}.npy')
" >& ${npy_log}
    set rc = \$status
endif
echo \${rc} > ${POWER_DIR}/${cache_id}.status
if ( \${rc} == 0 && -f ${POWER_DIR}/${cache_id}.npy ) then
    touch ${POWER_DIR}/${cache_id}.complete
else
    touch ${POWER_DIR}/${cache_id}.failed
endif
exit \${rc}
EOF

        if ( "${BSUB_QUEUE}" != "" ) then
            if ( $#BSUB_EXTRA_ARGS > 0 ) then
                bsub -q ${BSUB_QUEUE} -J power_${HW}_${cache_id} -o ${bsub_out} -e ${bsub_err} $BSUB_EXTRA_ARGS:q csh -f ${job_file}
            else
                bsub -q ${BSUB_QUEUE} -J power_${HW}_${cache_id} -o ${bsub_out} -e ${bsub_err} csh -f ${job_file}
            endif
        else
            if ( $#BSUB_EXTRA_ARGS > 0 ) then
                bsub -J power_${HW}_${cache_id} -o ${bsub_out} -e ${bsub_err} $BSUB_EXTRA_ARGS:q csh -f ${job_file}
            else
                bsub -J power_${HW}_${cache_id} -o ${bsub_out} -e ${bsub_err} csh -f ${job_file}
            endif
        endif
        if ( $status != 0 ) then
            echo "  [${cache_id}] bsub 提交失败"
            touch ${POWER_DIR}/${cache_id}.failed
        endif
    end

    echo "  监测所有 power 子任务 complete/failed..."
    while ( 1 )
        set DONE_COUNT = 0
        set FAIL_COUNT = 0
        foreach line ( $power_list:q )
            set cache_id = `echo "$line" | awk '{print $1}'`
            if ( -f ${POWER_DIR}/${cache_id}.complete ) @ DONE_COUNT = $DONE_COUNT + 1
            if ( -f ${POWER_DIR}/${cache_id}.failed ) @ FAIL_COUNT = $FAIL_COUNT + 1
        end
        if ( ${FAIL_COUNT} > 0 ) then
            echo "ERROR: 有 ${FAIL_COUNT} 个 power 子任务失败"
            touch ${POWER_DIR}/failed
            exit 1
        endif
        if ( ${DONE_COUNT} == ${POWER_COUNT} ) break
        echo "  power progress: ${DONE_COUNT}/${POWER_COUNT}; sleep ${POLL_SEC}s"
        sleep ${POLL_SEC}
    end

    touch ${POWER_DIR}/complete
    echo "  Step 3 complete: ${POWER_DIR}/complete (${POWER_COUNT}/${POWER_COUNT})"
else
    echo "  跳过（power_list 为空）"
    touch ${POWER_DIR}/complete
endif

#===============================================================================
# Step 4: 合并多个 Cache Manifest
#===============================================================================
echo ""
echo "=== Step 4: 合并 Cache Manifest ==="

while ( ! -f ${CACHE_DIR}/complete )
    echo "  等待 Step 2 complete: ${CACHE_DIR}/complete"
    sleep ${POLL_SEC}
end

if ( ${OK_COUNT} == 1 ) then
    echo "  仅 1 个 cache，跳过合并"
    mkdir -p ${MERGED_DIR}
    set single_cache_dir = `echo ${CACHE_DIR}/*`
    cp -r ${single_cache_dir}/* ${MERGED_DIR}/
else
    mkdir -p ${MERGED_DIR}
    set MERGE_STATUS_DIR = "${MERGED_DIR}/_work/merge_block_status"
    mkdir -p ${MERGE_STATUS_DIR}
    set first_line = "$fsdb_list[1]"
    set first_cache_id = `echo "${first_line}" | awk '{print $1}'`
    set first_manifest = "${CACHE_DIR}/${first_cache_id}/_work/tog_cache_manifest.json"
    set BLOCK_IDS = (`python3 -c 'import json,sys; m=json.load(open(sys.argv[1])); print(" ".join(str(int(b["block_id"])) for b in m["blocks"]))' ${first_manifest}`)
    set BLOCK_COUNT = $#BLOCK_IDS

    echo "  bsub 并行提交 ${BLOCK_COUNT} 个 merge block"
    foreach block_id ( $BLOCK_IDS )
        set block_tag = `printf "%05d" ${block_id}`
        set checkpoint_file = "${MERGED_DIR}/_work/tog_cache_block_manifests/block${block_tag}.json"
        set job_file = "${MERGE_STATUS_DIR}/merge.block${block_tag}.csh"
        set merge_log = "${LOG_DIR}/merge.block${block_id}.log"
        set bsub_out = "${LOG_DIR}/merge.block${block_id}.bsub.out"
        set bsub_err = "${LOG_DIR}/merge.block${block_id}.bsub.err"
        rm -f ${MERGE_STATUS_DIR}/block${block_id}.complete ${MERGE_STATUS_DIR}/block${block_id}.failed ${MERGE_STATUS_DIR}/block${block_id}.status ${job_file}

        cat > ${job_file} << EOF
#!/bin/csh -f
echo "[job] start `date` host=`hostname` pwd=`pwd`" >& ${merge_log}
echo "[job] merge block=${block_id}" >>& ${merge_log}
if ( -f ${TOOLS_ROOT}/activate.csh ) then
    echo "[job] source ${TOOLS_ROOT}/activate.csh" >>& ${merge_log}
    source ${TOOLS_ROOT}/activate.csh >>& ${merge_log}
endif
python3 ${MERGE_PY} --output-dir ${MERGED_DIR} --max-mem ${MAX_MEM} --max-cycles-per-file ${MAX_CYCLES_PER_FILE} --only-block-id ${block_id} ${MANIFEST_ARGS} >>& ${merge_log}
set rc = \$status
echo \${rc} > ${MERGE_STATUS_DIR}/block${block_id}.status
if ( \${rc} == 0 && -f ${checkpoint_file} ) then
    touch ${MERGE_STATUS_DIR}/block${block_id}.complete
else
    touch ${MERGE_STATUS_DIR}/block${block_id}.failed
endif
exit \${rc}
EOF

        if ( "${BSUB_QUEUE}" != "" ) then
            if ( $#BSUB_EXTRA_ARGS > 0 ) then
                bsub -q ${BSUB_QUEUE} -J merge_${HW}_b${block_id} -o ${bsub_out} -e ${bsub_err} $BSUB_EXTRA_ARGS:q csh -f ${job_file}
            else
                bsub -q ${BSUB_QUEUE} -J merge_${HW}_b${block_id} -o ${bsub_out} -e ${bsub_err} csh -f ${job_file}
            endif
        else
            if ( $#BSUB_EXTRA_ARGS > 0 ) then
                bsub -J merge_${HW}_b${block_id} -o ${bsub_out} -e ${bsub_err} $BSUB_EXTRA_ARGS:q csh -f ${job_file}
            else
                bsub -J merge_${HW}_b${block_id} -o ${bsub_out} -e ${bsub_err} csh -f ${job_file}
            endif
        endif
        if ( $status != 0 ) then
            echo "  [block ${block_id}] bsub 提交失败"
            touch ${MERGE_STATUS_DIR}/block${block_id}.failed
        endif
    end

    echo "  监测所有 merge block 子任务 complete/failed..."
    while ( 1 )
        set MERGE_OK_COUNT = 0
        set MERGE_FAIL_COUNT = 0
        foreach block_id ( $BLOCK_IDS )
            if ( -f ${MERGE_STATUS_DIR}/block${block_id}.complete ) @ MERGE_OK_COUNT = $MERGE_OK_COUNT + 1
            if ( -f ${MERGE_STATUS_DIR}/block${block_id}.failed ) @ MERGE_FAIL_COUNT = $MERGE_FAIL_COUNT + 1
        end
        if ( ${MERGE_FAIL_COUNT} > 0 ) then
            echo "ERROR: 有 ${MERGE_FAIL_COUNT} 个 merge block 子任务失败"
            touch ${MERGED_DIR}/failed
            exit 1
        endif
        if ( ${MERGE_OK_COUNT} == ${BLOCK_COUNT} ) break
        echo "  merge block progress: ${MERGE_OK_COUNT}/${BLOCK_COUNT}; sleep ${POLL_SEC}s"
        sleep ${POLL_SEC}
    end

    foreach block_id ( $BLOCK_IDS )
        echo "  [block ${block_id}] done"
    end

    python3 ${MERGE_PY} --output-dir ${MERGED_DIR} --max-mem ${MAX_MEM} --max-cycles-per-file ${MAX_CYCLES_PER_FILE} --finalize-only ${MANIFEST_ARGS} >& ${LOG_DIR}/merge.finalize.log
    if ( $status != 0 ) then
        echo "ERROR: merge finalize 失败，查看 ${LOG_DIR}/merge.finalize.log"
        touch ${MERGED_DIR}/failed
        exit 1
    endif
endif
if ( ! -f ${MERGED_DIR}/_work/tog_cache_manifest.json ) then
    echo "ERROR: 合并后缺失 manifest: ${MERGED_DIR}/_work/tog_cache_manifest.json"
    touch ${MERGED_DIR}/failed
    exit 1
endif
touch ${MERGED_DIR}/complete
echo "  ✅ Cache: ${MERGED_DIR}/_work/tog_cache_manifest.json"
echo "  Step 4 complete: ${MERGED_DIR}/complete"

#===============================================================================
# Step 5: 降维
#===============================================================================
echo ""
echo "=== Step 5: 降维（Level → Level+TOG 双输出） ==="

while ( ! -f ${MERGED_DIR}/complete )
    echo "  等待 Step 4 complete: ${MERGED_DIR}/complete"
    sleep ${POLL_SEC}
end

# 待修改/确认: 降维输入 manifest
# - 新版 merge 输出: level_cache_mode=initial_plus_tog，直接使用同一个 TOG manifest；
#   降维脚本会在清理全0/全1/重复列之后，根据 initial_level_files + TOG 还原 level。
# - 单个 cache 跳过 merge 或旧 cache: manifest 里可能只有 level_files，下面会自动派生一个
#   level_input_manifest，把 block.files 指向 level_files，避免误把 TOG 当 level 读取。
set TOG_MANIFEST = "${MERGED_DIR}/_work/tog_cache_manifest.json"
set LEVEL_INPUT_MANIFEST = "${REDUCE_DIR}/_work/level_input_manifest.json"
mkdir -p "${REDUCE_DIR}/_work"
python3 - "${TOG_MANIFEST}" "${LEVEL_INPUT_MANIFEST}" << INNERPY
import json, sys
from pathlib import Path
src = Path(sys.argv[1])
dst = Path(sys.argv[2])
m = json.loads(src.read_text(encoding='utf-8'))
blocks = m.get('blocks', [])
has_initial = m.get('level_cache_mode') == 'initial_plus_tog' or any(b.get('initial_level_files') for b in blocks)
has_level_files = any(b.get('level_files') for b in blocks)
if has_initial:
    out = m
    out['level_input_note'] = 'initial_plus_tog: reduction restores level from initial_level_files + TOG manifest'
elif has_level_files:
    out = json.loads(json.dumps(m))
    for b in out.get('blocks', []):
        if b.get('level_files'):
            b['files'] = b.get('level_files', {})
            if 'level_n_rows' in b:
                b['n_rows'] = b['level_n_rows']
            if 'level_nnz' in b:
                b['nnz'] = b['level_nnz']
    out['data_type'] = 'level'
    out['level_input_note'] = 'derived from level_files for full level cache reduction'
else:
    raise SystemExit('manifest has neither initial_level_files nor level_files; cannot produce level output safely')
dst.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding='utf-8')
print(dst)
INNERPY
if ( $status != 0 ) then
    echo "ERROR: 生成 level 输入 manifest 失败"
    exit 1
endif

rm -f ${REDUCE_DIR}/_work/reduction.status ${REDUCE_DIR}/complete ${REDUCE_DIR}/failed
set REDUCE_JOB = "${REDUCE_DIR}/_work/reduction_job.csh"
set REDUCE_BSUB_OUT = "${LOG_DIR}/reduction.bsub.out"
set REDUCE_BSUB_ERR = "${LOG_DIR}/reduction.bsub.err"
cat > ${REDUCE_JOB} << EOF
#!/bin/csh -f
echo "[job] start `date` host=`hostname` pwd=`pwd`" >& ${LOG_DIR}/reduction.log
echo "[job] reduction manifest=${LEVEL_INPUT_MANIFEST}" >>& ${LOG_DIR}/reduction.log
if ( -f ${TOOLS_ROOT}/activate.csh ) then
    echo "[job] source ${TOOLS_ROOT}/activate.csh" >>& ${LOG_DIR}/reduction.log
    source ${TOOLS_ROOT}/activate.csh >>& ${LOG_DIR}/reduction.log
endif
python3 -u ${REDUCE_PY} \
    --input-cache-manifest ${LEVEL_INPUT_MANIFEST} \
    --input-tog-cache-manifest ${TOG_MANIFEST} \
    --output-dir ${REDUCE_DIR} \
    --output-meta ${REDUCE_DIR}/meta.json \
    --output-cols ${REDUCE_DIR}/final_cols.txt \
    --output-original-headers ${REDUCE_DIR}/original_headers.txt \
    --max-mem ${MAX_MEM} \
    --target-cols ${TARGET_COLS} \
    --cache-format csc \
    --save-after-xz-clean \
    --save-after-one \
    --save-final \
    --disable-sim-reduce \
    --duplicate-mode hash-only \
    --input-data-type level \
    --force-row-shard-save \
    --snapshot-row-block-rows ${MAX_CYCLES_PER_FILE} >>& ${LOG_DIR}/reduction.log
set rc = \$status
echo \${rc} > ${REDUCE_DIR}/_work/reduction.status
if ( \${rc} == 0 ) then
    touch ${REDUCE_DIR}/complete
    touch ${OUTPUT_BASE}/complete
else
    touch ${REDUCE_DIR}/failed
endif
exit \${rc}
EOF

echo "  bsub submit reduction"
if ( "${BSUB_QUEUE}" != "" ) then
    if ( $#BSUB_EXTRA_ARGS > 0 ) then
        bsub -q ${BSUB_QUEUE} -J reduce_${HW} -o ${REDUCE_BSUB_OUT} -e ${REDUCE_BSUB_ERR} $BSUB_EXTRA_ARGS:q csh -f ${REDUCE_JOB}
    else
        bsub -q ${BSUB_QUEUE} -J reduce_${HW} -o ${REDUCE_BSUB_OUT} -e ${REDUCE_BSUB_ERR} csh -f ${REDUCE_JOB}
    endif
else
    if ( $#BSUB_EXTRA_ARGS > 0 ) then
        bsub -J reduce_${HW} -o ${REDUCE_BSUB_OUT} -e ${REDUCE_BSUB_ERR} $BSUB_EXTRA_ARGS:q csh -f ${REDUCE_JOB}
    else
        bsub -J reduce_${HW} -o ${REDUCE_BSUB_OUT} -e ${REDUCE_BSUB_ERR} csh -f ${REDUCE_JOB}
    endif
endif
if ( $status != 0 ) then
    echo "ERROR: reduction bsub 提交失败"
    touch ${REDUCE_DIR}/failed
    exit 1
endif

echo "  监测 reduction complete/failed..."
while ( 1 )
    if ( -f ${REDUCE_DIR}/failed ) then
        echo "ERROR: 降维失败，查看 ${LOG_DIR}/reduction.log"
        exit 1
    endif
    if ( -f ${REDUCE_DIR}/complete ) break
    echo "  reduction running; sleep ${POLL_SEC}s"
    sleep ${POLL_SEC}
end

#===============================================================================
# 完成
#===============================================================================
echo ""
echo "============================================================"
echo "FSDB Pipeline 完成"
echo "  输出: ${REDUCE_DIR}"
echo "    level/: Level 稀疏 NPZ + signal names"
echo "    tog/:   TOG 稀疏 NPZ + signal names"
echo "  结束时间: `date`"
echo "============================================================"
