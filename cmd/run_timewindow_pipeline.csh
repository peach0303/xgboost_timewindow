#!/bin/csh -f
#===============================================================================
# Time-Window Power Modeling Pipeline（所有计算任务均通过 dsub 提交）
#
# 主 pipeline 与分支开关：
#
#   [Step 1: 全局采样 + time-window 数据]
#                       |
#                       v
#   [Step 1b: 检查/回填权重建议文件]
#                       |
#                       v
#        +--------------+--------------+
#        |                             |
#        v                             v
#   [Step 2 MCP]                  [Step 2 XGBoost]
#        +--------------+--------------+
#                       |
#                       v
#   [生成 OUTPUT_TAG] ----------------> [${OUTPUT_ROOT}/${OUTPUT_TAG}]
#   STEP3_* 自动生成；EXTRA_TAG 覆盖                 |
#   STEP2_SELECTOR_MODE=hybrid（默认）: MCP + XGBoost |
#   STEP2_SELECTOR_MODE=single: 仅 PROXY_SELECTOR     |
#                                                  v
#   [Step 3: MCP/XGB 特征拼接] -> [Step 4: 混合筛选 Top-K]
#                       |                         |
#                       | PIPELINE_MODE=prep      +--> 结束（仅准备数据）
#                       | FEATURE_INPUT_MODE=proxy_columns（默认）: 代理列直入
#                       | FEATURE_INPUT_MODE=signal_all_windows: 信号展开至全部时间窗
#                       | TWO_PHASE_METHOD=""（默认）: 直接 HPO/训练
#                       | TWO_PHASE_METHOD=A: Two-phase Method A HPO/训练
#                       v
#   [Step 5: XGBoost HPO，每个 K 并行]
#                       |
#                       v
#   [Step 6: XGBoost 重训练/评估，每个 K 并行] ----> 主 pipeline 完成
#                       |
#                       | skip_mlp_train=0
#                       v
#   [MLP Step 5 HPO + Step 6 重训练/评估] --------> MLP 可选分支
#
#   PIPELINE_MODE=all                 : 从 Step 1 跑到 Step 6（默认）
#   PIPELINE_MODE=prep                : 只跑 Step 1~4
#   PIPELINE_MODE=train_k + TARGET_K  : 复用已有 Step 4，只训练指定 K
#   skip_mlp_train=1                  : 跳过 MLP 分支（默认）
#   skip_mlp_train=0                  : 在 XGBoost 主线后追加 MLP 分支
#   RUN_STEP2_VALIDATION=0/1          : 关闭/开启 Step 2 legacy 验证旁路
#   STEP1_PARALLEL_DSUB=0/1           : Step 1 单任务/按 case×combo 拆分
#   Step 2 固定按 selector×combo 拆分 dsub；level/tog 可独立配置参数与资源
#   TWO_PHASE_METHOD=""|A             : Direct XGBoost/Two-phase Method A
#   FEATURE_INPUT_MODE=proxy_columns  : Step 5/6 直接使用筛选后的代理列（默认）
#   FEATURE_INPUT_MODE=signal_all_windows : 将代理信号展开到全部可用时间窗口
#   STEP3_OFFSETS/DATA_TYPES/COMBOS    : 改变 Step 3–6 自动 tag 与数据子集
#   EXTRA_TAG=<name>                   : 覆盖自动 tag，切换/命名另一套下游 pipeline
#
# Usage:
#   csh xgboost_timewindow/cmd/run_timewindow_pipeline.csh <hardware_name> [all|prep|train_k] [K]
#===============================================================================

#==============================step 0 args==============================
# 基础路径、运行环境与 dsub 调度参数。
set SCRIPT_DIR  = `dirname "$0"`
set PACKAGE_ROOT = `cd "${SCRIPT_DIR}/.." && pwd`
set PROJECT_ROOT = `cd "${PACKAGE_ROOT}/.." && pwd`
set TOOLS_ROOT  = "${PACKAGE_ROOT}"
if ( ! $?TIMEWINDOW_SRC ) set TIMEWINDOW_SRC = "${TOOLS_ROOT}/src/timewindow"
if ( ! $?REDUCTION_BASE ) set REDUCTION_BASE = "${PROJECT_ROOT}/output/xgboost_timewindow_fsdb_pipeline"
if ( ! $?OUTPUT_BASE ) set OUTPUT_BASE = "${PROJECT_ROOT}/output/xgboost_timewindow_train"
if ( ! $?WORK_DIR ) set WORK_DIR = "${PROJECT_ROOT}/tmp/xgboost_timewindow_train_work"

if ( ! $?PYTHON ) set PYTHON = "/home/data/chenyiheng/conda/envs/pro/bin/python"
if ( ! $?DSUB_CMD ) set DSUB_CMD = ( dsub -A root.ug_dsp.hisdr_HClass )
if ( ! $?DSUB_QUEUE ) set DSUB_QUEUE = normal
if ( ! $?DSUB_EXTRA_ARGS ) set DSUB_EXTRA_ARGS = ( )
if ( ! $?POLL_SEC ) set POLL_SEC = 30
if ( ! $?MAX_RERUN ) set MAX_RERUN = 10
if ( ! $?SHUFFLE_SEED ) set SHUFFLE_SEED = 108
# 所有提交统一展开为：
# ${DSUB_CMD} -q ${DSUB_QUEUE} -oo <stdout> -eo <stderr> ${DSUB_EXTRA_ARGS} -s <job.csh>
alias dsub_submit '${DSUB_CMD} -q ${DSUB_QUEUE} -oo \!:2 -eo \!:3 $DSUB_EXTRA_ARGS:q -s \!:1'
#======================================================================

#==============================step 1 args==============================
if ( ! $?STAGE ) set STAGE = after_one
if ( ! $?OFFSETS ) set OFFSETS = "12,11,10,9,8,7,6"
if ( ! $?DATA_TYPES ) set DATA_TYPES = "tog,level"
if ( ! $?BOUNDARY_DISCARD ) set BOUNDARY_DISCARD = ""
if ( ! $?TEST_COUNT ) set TEST_COUNT = 1
if ( ! $?SAMPLE_ROWS_PER_FILE ) set SAMPLE_ROWS_PER_FILE = 0
if ( ! $?SAMPLE_ROWS_TOTAL ) set SAMPLE_ROWS_TOTAL = 0
if ( ! $?SAMPLE_BINS ) set SAMPLE_BINS = 20
if ( ! $?SAMPLE_POWER_BINS ) set SAMPLE_POWER_BINS = 12
if ( ! $?SAMPLE_DELTA_BINS ) set SAMPLE_DELTA_BINS = 12
if ( ! $?SAMPLE_ALL_THRESHOLD ) set SAMPLE_ALL_THRESHOLD = 32
if ( ! $?SAMPLE_WEIGHT_EXPONENT ) set SAMPLE_WEIGHT_EXPONENT = 0.5
if ( ! $?SAMPLE_WEIGHT_MIN ) set SAMPLE_WEIGHT_MIN = 0.25
if ( ! $?SAMPLE_WEIGHT_MAX ) set SAMPLE_WEIGHT_MAX = 8.0
if ( ! $?SAMPLE_MIN_PER_BIN ) set SAMPLE_MIN_PER_BIN = 1
if ( ! $?SAMPLE_MIN_PER_CASE ) set SAMPLE_MIN_PER_CASE = 1
if ( ! $?SAMPLE_TEST_RATIO ) set SAMPLE_TEST_RATIO = -1
if ( ! $?SAMPLE_SEED ) set SAMPLE_SEED = 108

if ( ! $?STEP1_JOBS ) set STEP1_JOBS = 1
if ( ! $?STEP1_COMPRESS_OUTPUT ) set STEP1_COMPRESS_OUTPUT = 0
if ( ! $?STEP1_SKIP_EXISTING ) set STEP1_SKIP_EXISTING = 1
if ( ! $?STEP1_PARALLEL_DSUB ) set STEP1_PARALLEL_DSUB = 1
if ( ! $?STEP1_DSUB_EXTRA_ARGS ) set STEP1_DSUB_EXTRA_ARGS = ( $DSUB_EXTRA_ARGS )
# WEIGHT_RECOMMENDATION_FILE 可指定回填 JSON；默认路径在运行目录初始化后确定。
#======================================================================

#==============================step 2 args==============================
if ( ! $?ALPHA_MCP ) set ALPHA_MCP = 50.0
if ( ! $?ALPHA_MCP_TOG ) set ALPHA_MCP_TOG = ${ALPHA_MCP}
if ( ! $?ALPHA_MCP_LEVEL ) set ALPHA_MCP_LEVEL = ${ALPHA_MCP}
if ( ! $?GAMMA ) set GAMMA = 3.0
if ( ! $?ALPHA_L2 ) set ALPHA_L2 = 8.0
if ( ! $?BATCH_SIZE ) set BATCH_SIZE = 4
if ( ! $?BATCH_SIZE_TOG ) set BATCH_SIZE_TOG = ${BATCH_SIZE}
if ( ! $?BATCH_SIZE_LEVEL ) set BATCH_SIZE_LEVEL = ${BATCH_SIZE}
if ( ! $?STEP2_SELECTOR_MODE ) set STEP2_SELECTOR_MODE = hybrid
if ( ! $?PROXY_SELECTOR ) set PROXY_SELECTOR = mcp
if ( ! $?RUN_STEP2_VALIDATION ) set RUN_STEP2_VALIDATION = 1
if ( ! $?STEP2_VALIDATION_CHUNK_SIZE ) set STEP2_VALIDATION_CHUNK_SIZE = 200
if ( ! $?MCP_PROXY_TOP_K ) set MCP_PROXY_TOP_K = 200
if ( ! $?XGB_PROXY_TOP_K ) set XGB_PROXY_TOP_K = 200
if ( ! $?XGB_PROXY_ROUNDS ) set XGB_PROXY_ROUNDS = 50
if ( ! $?XGB_PROXY_MAX_DEPTH ) set XGB_PROXY_MAX_DEPTH = 4
if ( ! $?XGB_PROXY_ETA ) set XGB_PROXY_ETA = 0.08
if ( ! $?XGB_PROXY_SUBSAMPLE ) set XGB_PROXY_SUBSAMPLE = 0.9
if ( ! $?XGB_PROXY_COLSAMPLE ) set XGB_PROXY_COLSAMPLE = 0.8
if ( ! $?XGB_PROXY_NTHREAD ) set XGB_PROXY_NTHREAD = 1
if ( ! $?XGB_LOSS ) set XGB_LOSS = relative_error_focus
if ( ! $?XGB_RELATIVE_FLOOR ) set XGB_RELATIVE_FLOOR = 0.001
if ( ! $?XGB_FOCUS_POWER ) set XGB_FOCUS_POWER = 3.0
if ( ! $?STEP2_TOG_DSUB_EXTRA_ARGS ) set STEP2_TOG_DSUB_EXTRA_ARGS = ( $DSUB_EXTRA_ARGS )
if ( ! $?STEP2_LEVEL_DSUB_EXTRA_ARGS ) set STEP2_LEVEL_DSUB_EXTRA_ARGS = ( $DSUB_EXTRA_ARGS )
if ( ! $?STEP2_MERGE_DSUB_EXTRA_ARGS ) set STEP2_MERGE_DSUB_EXTRA_ARGS = ( $DSUB_EXTRA_ARGS )
if ( ! $?STEP2_VALIDATION_DSUB_EXTRA_ARGS ) set STEP2_VALIDATION_DSUB_EXTRA_ARGS = ( $DSUB_EXTRA_ARGS )
# STEP2_VALIDATION_COMBOS 可选；未设置时按 OFFSETS×DATA_TYPES 自动生成。
#======================================================================

#==============================step 3 args==============================
if ( ! $?STEP3_OFFSETS ) set STEP3_OFFSETS = "$OFFSETS"
if ( ! $?STEP3_DATA_TYPES ) set STEP3_DATA_TYPES = "$DATA_TYPES"
if ( ! $?STEP3_COMBOS ) set STEP3_COMBOS = ""
if ( ! $?EXTRA_TAG ) set EXTRA_TAG = ""
if ( ! $?STEP3_DSUB_EXTRA_ARGS ) set STEP3_DSUB_EXTRA_ARGS = ( $DSUB_EXTRA_ARGS )
#======================================================================

#==============================step 4 args==============================
if ( ! $?TOP_KS ) set TOP_KS = "256,512,1024,2048,4096"
if ( ! $?HYBRID_SCREEN_XGB_ROUNDS ) set HYBRID_SCREEN_XGB_ROUNDS = 200
if ( ! $?FEATURE_INPUT_MODE ) set FEATURE_INPUT_MODE = proxy_columns
if ( ! $?STEP4_DSUB_EXTRA_ARGS ) set STEP4_DSUB_EXTRA_ARGS = ( $DSUB_EXTRA_ARGS )
# Step 4 同时复用 ALPHA_L2、XGB_PROXY_NTHREAD 与 XGB loss 参数。
#======================================================================

#==============================step 5 xgb hpo args==============================
if ( ! $?HPO_TRIALS ) set HPO_TRIALS = 64
if ( ! $?HPO_ROUNDS ) set HPO_ROUNDS = 50
if ( ! $?HPO_POP ) set HPO_POP = 8
if ( ! $?HPO_NFOLD ) set HPO_NFOLD = 3
if ( ! $?HPO_JOB_NUM ) set HPO_JOB_NUM = 1
if ( ! $?HPO_CLOSENESS_METRIC ) set HPO_CLOSENESS_METRIC = tail_mean_ape
if ( ! $?TWO_PHASE_METHOD ) set TWO_PHASE_METHOD = ""
# 默认调用语义为 --two-phase-method ""；仅显式设置 A 时启用 Method A。
if ( ! $?TWO_PHASE_CUTOFF ) set TWO_PHASE_CUTOFF = 9
if ( ! $?TWO_PHASE_RATIO ) set TWO_PHASE_RATIO = 0.8
if ( ! $?STEP5_DSUB_EXTRA_ARGS ) set STEP5_DSUB_EXTRA_ARGS = ( $DSUB_EXTRA_ARGS )

#======================================================================

#==============================step 6 xgb train args==============================
if ( ! $?RETRAIN_ROUNDS ) set RETRAIN_ROUNDS = 50
if ( ! $?STEP6_DSUB_EXTRA_ARGS ) set STEP6_DSUB_EXTRA_ARGS = ( $DSUB_EXTRA_ARGS )
if ( ! $?FULL_EVAL_DSUB_EXTRA_ARGS ) set FULL_EVAL_DSUB_EXTRA_ARGS = ( $STEP6_DSUB_EXTRA_ARGS )
if ( ! $?FULL_EVAL_REDUCE_DSUB_EXTRA_ARGS ) set FULL_EVAL_REDUCE_DSUB_EXTRA_ARGS = ( $STEP6_DSUB_EXTRA_ARGS )
if ( ! $?FULL_EVAL_CHUNK_ROWS ) set FULL_EVAL_CHUNK_ROWS = 250000

#======================================================================

#==============================step 5 mlp hpo args==============================
# 小写开关是主接口；同时接受大写别名以保持环境变量风格兼容。
if ( $?SKIP_MLP_TRAIN ) then
    if ( ! $?skip_mlp_train ) set skip_mlp_train = "$SKIP_MLP_TRAIN"
endif
if ( ! $?skip_mlp_train ) set skip_mlp_train = 1
if ( "$skip_mlp_train" != "0" && "$skip_mlp_train" != "1" ) then
    echo "ERROR: skip_mlp_train must be 0 (train) or 1 (skip)"
    exit 2
endif
if ( ! $?MLP_HPO_TRIALS ) set MLP_HPO_TRIALS = 40
if ( ! $?MLP_HPO_EPOCHS ) set MLP_HPO_EPOCHS = 100
if ( ! $?MLP_HPO_POP ) set MLP_HPO_POP = 8
if ( ! $?MLP_HPO_NFOLD ) set MLP_HPO_NFOLD = 3
if ( ! $?MLP_HPO_JOB_NUM ) set MLP_HPO_JOB_NUM = 1
if ( ! $?MLP_HPO_SAMPLER ) set MLP_HPO_SAMPLER = NSGAII
if ( ! $?MLP_PATIENCE ) set MLP_PATIENCE = 12
if ( ! $?MLP_VALIDATION_RATIO ) set MLP_VALIDATION_RATIO = 0.15
if ( ! $?MLP_PARAMETER_THRESHOLD ) set MLP_PARAMETER_THRESHOLD = 5000000
if ( ! $?MLP_DSUB_EXTRA_ARGS ) set MLP_DSUB_EXTRA_ARGS = ( $DSUB_EXTRA_ARGS )
if ( ! $?MLP_FULL_EVAL_DSUB_EXTRA_ARGS ) set MLP_FULL_EVAL_DSUB_EXTRA_ARGS = ( $MLP_DSUB_EXTRA_ARGS )
if ( ! $?MLP_FULL_EVAL_REDUCE_DSUB_EXTRA_ARGS ) set MLP_FULL_EVAL_REDUCE_DSUB_EXTRA_ARGS = ( $MLP_DSUB_EXTRA_ARGS )
if ( ! $?MLP_FULL_EVAL_CHUNK_ROWS ) set MLP_FULL_EVAL_CHUNK_ROWS = ${FULL_EVAL_CHUNK_ROWS}
#======================================================================

#==============================step 6 mlp train args==============================
# MLP Step 6 复用 Step 0 的 SHUFFLE_SEED，以及运行时生成的输入/输出路径。
#======================================================================

#===============================================================================
# 初始化与参数
#===============================================================================
if ( $#argv >= 1 ) then
    set HW = "$1"
else
    set HW = gemmini
endif
set PIPELINE_MODE = all
if ( $#argv >= 2 ) set PIPELINE_MODE = "$2"
set TARGET_K = ""
if ( $#argv >= 3 ) set TARGET_K = "$3"

if ( "$PIPELINE_MODE" != "all" && "$PIPELINE_MODE" != "prep" && "$PIPELINE_MODE" != "train_k" ) then
    echo "ERROR: PIPELINE_MODE must be one of: all, prep, train_k"
    exit 2
endif
if ( "$PIPELINE_MODE" == "train_k" && "$TARGET_K" == "" ) then
    echo "ERROR: train_k mode requires K, e.g. csh $0 ${HW} train_k 1024"
    exit 2
endif
if ( "$STEP2_SELECTOR_MODE" != "hybrid" && "$STEP2_SELECTOR_MODE" != "single" ) then
    echo "ERROR: STEP2_SELECTOR_MODE must be hybrid or single"
    exit 2
endif
if ( "$PROXY_SELECTOR" != "mcp" && "$PROXY_SELECTOR" != "xgb" ) then
    echo "ERROR: PROXY_SELECTOR must be mcp or xgb"
    exit 2
endif
if ( "$FEATURE_INPUT_MODE" != "proxy_columns" && "$FEATURE_INPUT_MODE" != "signal_all_windows" ) then
    echo "ERROR: FEATURE_INPUT_MODE must be proxy_columns or signal_all_windows"
    exit 2
endif
if ( "$TWO_PHASE_METHOD" != "" && "$TWO_PHASE_METHOD" != "A" ) then
    echo 'ERROR: TWO_PHASE_METHOD must be empty ("") or A'
    exit 2
endif

if ( ! $?LEVEL_DIR ) set LEVEL_DIR = "${REDUCTION_BASE}/${HW}_reduced/level"
if ( ! $?TOG_DIR ) set TOG_DIR = "${REDUCTION_BASE}/${HW}_reduced/tog"
if ( ! $?LABEL_DIR ) set LABEL_DIR = "${REDUCTION_BASE}/${HW}_power"
if ( ! $?NAMEFILE ) set NAMEFILE = "${LEVEL_DIR}/${STAGE}.namefile.txt"
if ( ! $?RUN_TAG ) set RUN_TAG = `date '+%Y%m%d_%H%M%S'`
if ( ! $?OUTPUT_ROOT ) set OUTPUT_ROOT = "${OUTPUT_BASE}/${HW}_${STAGE}_${RUN_TAG}"

set STEP1_DIR = "${OUTPUT_ROOT}/step1_shifted"
if ( ${STEP1_PARALLEL_DSUB} != 0 ) then
    set STEP1_SAMPLE_DIR = "${STEP1_DIR}/sample_index"
else
    set STEP1_SAMPLE_DIR = "${STEP1_DIR}/_sampling"
endif
if ( ! $?WEIGHT_RECOMMENDATION_FILE ) then
    set WEIGHT_RECOMMENDATION_FILE = "${STEP1_SAMPLE_DIR}/sampling_distribution_and_weight_recommendation.json"
endif
set WEIGHT_RECOMMENDATION_STATUS_DIR = "${STEP1_DIR}/weight_recommendation_status"
set STEP2_DIR = "${OUTPUT_ROOT}/step2_hybrid"
if ( "$STEP2_SELECTOR_MODE" == "single" ) set STEP2_DIR = "${OUTPUT_ROOT}/step2_single_${PROXY_SELECTOR}"
set STEP2_MCP_DIR = "${OUTPUT_ROOT}/step2_mcp"
set STEP2_XGB_DIR = "${OUTPUT_ROOT}/step2_xgb_proxy"
# 默认展开等价于：foreach selector ( mcp xgb )
set STEP2_SELECTOR_LIST = ( mcp xgb )
if ( "$STEP2_SELECTOR_MODE" == "single" ) set STEP2_SELECTOR_LIST = ( ${PROXY_SELECTOR} )
set STEP2_VALIDATION_DIR = "${OUTPUT_ROOT}/step2_validation_legacy_xgb"
if ( "$STEP2_SELECTOR_MODE" == "single" ) then
    set STEP2_VALIDATION_DIR = "${OUTPUT_ROOT}/step2_validation_legacy_${PROXY_SELECTOR}"
endif
set LOG_DIR = "${OUTPUT_ROOT}/logs_dsub"
set JOB_DIR = "${OUTPUT_ROOT}/jobs_dsub"

mkdir -p ${OUTPUT_ROOT} ${WORK_DIR} ${LOG_DIR} ${JOB_DIR}

if ( ! -d ${LEVEL_DIR} ) then
    echo "ERROR: LEVEL_DIR 不存在: ${LEVEL_DIR}"
    exit 1
endif
if ( ! -d ${TOG_DIR} ) then
    echo "ERROR: TOG_DIR 不存在: ${TOG_DIR}"
    exit 1
endif
if ( ! -d ${LABEL_DIR} ) then
    echo "ERROR: LABEL_DIR 不存在: ${LABEL_DIR}"
    exit 1
endif
if ( ! -f ${NAMEFILE} ) then
    echo "ERROR: NAMEFILE 不存在: ${NAMEFILE}"
    exit 1
endif

set K_LIST = ( `echo ${TOP_KS} | tr ',' ' '` )
if ( "$TARGET_K" != "" ) set K_LIST = ( ${TARGET_K} )
set OFFSET_LIST = ( `echo ${OFFSETS} | tr ',' ' '` )
set TYPE_LIST = ( `echo ${DATA_TYPES} | tr ',' ' '` )
set COMBO_LIST = ( )
set COMBO_ORDER = ""
foreach offset_k ( ${OFFSET_LIST} )
    foreach dtype ( ${TYPE_LIST} )
        set combo_key = "${dtype}_offset${offset_k}"
        set COMBO_LIST = ( ${COMBO_LIST} ${combo_key} )
        if ( "${COMBO_ORDER}" == "" ) then
            set COMBO_ORDER = "${combo_key}"
        else
            set COMBO_ORDER = "${COMBO_ORDER},${combo_key}"
        endif
    end
end

# COMBO_ORDER was built with OFFSETS outer and DATA_TYPES inner.
if ( ! $?STEP2_VALIDATION_COMBOS ) set STEP2_VALIDATION_COMBOS = "$COMBO_ORDER"
set DISCARD = "${BOUNDARY_DISCARD}"
if ( "${DISCARD}" == "" ) set DISCARD = `echo ${OFFSETS} | tr ',' '\n' | sort -n | tail -1`

echo "============================================================"
echo "Time-Window Training Pipeline (dsub)"
echo "  HW          = ${HW}"
echo "  LEVEL_DIR   = ${LEVEL_DIR}"
echo "  TOG_DIR     = ${TOG_DIR}"
echo "  LABEL_DIR   = ${LABEL_DIR}"
echo "  NAMEFILE    = ${NAMEFILE}"
echo "  OUTPUT_ROOT = ${OUTPUT_ROOT}"
echo "  MODE        = ${PIPELINE_MODE}"
echo "  STAGE       = ${STAGE}"
echo "  DATA_TYPES  = ${DATA_TYPES}"
echo "  OFFSETS     = ${OFFSETS}"
echo "  STEP3_TYPES = ${STEP3_DATA_TYPES}"
echo "  STEP3_OFFS  = ${STEP3_OFFSETS}"
if ( "${STEP3_COMBOS}" != "" ) echo "  STEP3_COMBOS= ${STEP3_COMBOS}"
echo "  PROXY       = hybrid (MCP + XGB, K_MCP:K_XGB=1:1 before dedup)"
echo "  MCP_TOP_K   = ${MCP_PROXY_TOP_K}"
echo "  XGB_TOP_K   = ${XGB_PROXY_TOP_K}"
echo "  SAMPLE_TOTAL= ${SAMPLE_ROWS_TOTAL}"
echo "  WEIGHT_REPORT= ${WEIGHT_RECOMMENDATION_FILE}"
echo "  SAMPLE_TEST = ${SAMPLE_TEST_RATIO}"
echo "  STEP1_JOBS  = ${STEP1_JOBS}"
echo "  STEP1_COMP  = ${STEP1_COMPRESS_OUTPUT}"
echo "  STEP1_REUSE = ${STEP1_SKIP_EXISTING}"
echo "  STEP2_DSUB  = one dsub per selector/combo"
echo "  MCP_TOG     = alpha=${ALPHA_MCP_TOG}, batch=${BATCH_SIZE_TOG}"
echo "  MCP_LEVEL   = alpha=${ALPHA_MCP_LEVEL}, batch=${BATCH_SIZE_LEVEL}"
echo "  STEP2_SELECT= ${STEP2_SELECTOR_MODE} (${STEP2_SELECTOR_LIST})"
echo "  STEP1_DSUB  = ${STEP1_PARALLEL_DSUB}"
echo "  STEP2_MODE  = selector×combo dsub jobs"
echo "  STEP2_VALID = ${RUN_STEP2_VALIDATION}"
echo "  VALID_COMBOS= ${STEP2_VALIDATION_COMBOS}"
echo "  VALID_CHUNK = ${STEP2_VALIDATION_CHUNK_SIZE}"
echo "  COMBOS      = ${COMBO_LIST}"
echo "  K_LIST      = ${K_LIST}"
echo "  FEATURE_MODE= ${FEATURE_INPUT_MODE}"
if ( "$TWO_PHASE_METHOD" == "A" ) then
    echo "  XGB_TRAIN   = Two-phase A, cutoff=${TWO_PHASE_CUTOFF}, phase1_ratio=${TWO_PHASE_RATIO}"
else
    echo "  XGB_TRAIN   = direct"
endif
echo "  SKIP_MLP    = ${skip_mlp_train} (0=train, 1=skip)"
if ( "${skip_mlp_train}" == "0" ) then
    echo "  MLP_HPO     = trials=${MLP_HPO_TRIALS}, epochs=${MLP_HPO_EPOCHS}, folds=${MLP_HPO_NFOLD}"
    echo "  MLP_SEARCH  = sampler=${MLP_HPO_SAMPLER}, pop=${MLP_HPO_POP}, jobs=${MLP_HPO_JOB_NUM}"
endif
echo "  DSUB_CMD    = ${DSUB_CMD}"
echo "============================================================"

#===============================================================================
# ▶ Step 1 控制流：[采样索引] -> [case×combo 并行处理] -> [manifest 合并]
# 关键协议：每个任务通过 running/complete/fail 文件驱动等待与失败重提。
#===============================================================================
if ( "$PIPELINE_MODE" != "train_k" ) then
    set DSUB_EXTRA_ARGS = ( $STEP1_DSUB_EXTRA_ARGS )

    if ( ! -f ${STEP1_DIR}/complete ) then
        mkdir -p ${STEP1_DIR}
        rm -f ${STEP1_DIR}/fail ${STEP1_DIR}/status

        if ( ${STEP1_PARALLEL_DSUB} != 0 ) then
            set STEP1_PART_DIR = "${STEP1_DIR}/_parts"
            mkdir -p ${STEP1_SAMPLE_DIR} ${STEP1_PART_DIR}

            if ( ! -f ${STEP1_SAMPLE_DIR}/complete ) then
                rm -f ${STEP1_SAMPLE_DIR}/fail ${STEP1_SAMPLE_DIR}/status
                set job_file = "${JOB_DIR}/step1_sample_index.csh"
                set job_log = "${LOG_DIR}/step1_sample_index.log"
                set dsub_out = "${LOG_DIR}/step1_sample_index.dsub.out"
                set dsub_err = "${LOG_DIR}/step1_sample_index.dsub.err"
                cat > ${job_file} << EOF
#!/bin/csh -x
# 任务说明：执行主 pipeline 生成的单个计算阶段。
# 参数说明：输入、输出及训练参数均已由主脚本展开。
# dsub 状态协议：running -> complete 或 fail。
rm -f "${job_file}.running"
touch "${job_file}.running"
echo "[job] step1 global sample index start `date` host=`hostname`" >& ${job_log}
${PYTHON} ${TIMEWINDOW_SRC}/step1_make_global_sample_index.py --level-dir ${LEVEL_DIR} --tog-dir ${TOG_DIR} --label-dir ${LABEL_DIR} --output-dir ${STEP1_SAMPLE_DIR} --offsets ${OFFSETS} --boundary-discard ${DISCARD} --test-count ${TEST_COUNT} --stage ${STAGE} --sample-rows-total ${SAMPLE_ROWS_TOTAL} --sample-rows-per-file ${SAMPLE_ROWS_PER_FILE} --sample-power-bins ${SAMPLE_POWER_BINS} --sample-delta-bins ${SAMPLE_DELTA_BINS} --sample-all-threshold ${SAMPLE_ALL_THRESHOLD} --sample-min-per-case ${SAMPLE_MIN_PER_CASE} --sample-test-ratio ${SAMPLE_TEST_RATIO} --weight-exponent ${SAMPLE_WEIGHT_EXPONENT} --weight-min ${SAMPLE_WEIGHT_MIN} --weight-max ${SAMPLE_WEIGHT_MAX} --sample-seed ${SAMPLE_SEED} >>& ${job_log}
set rc = \$status
echo \${rc} > ${STEP1_SAMPLE_DIR}/status
if ( \${rc} == 0 && -f ${STEP1_SAMPLE_DIR}/sample_index_manifest.json ) then
    touch ${STEP1_SAMPLE_DIR}/complete
else
    touch ${STEP1_SAMPLE_DIR}/fail
endif
rm -f "${job_file}.running"
exit \${rc}
EOF
                echo "[`date '+%H:%M:%S'`] submit Step 1 global sample index"
                chmod +x "${job_file}"
                dsub_submit "${job_file}" "${dsub_out}" "${dsub_err}"
                if ( $status != 0 ) then
                    echo "ERROR: Step 1 sample index dsub 提交失败"
                    touch ${STEP1_SAMPLE_DIR}/fail
                endif
            else
                echo "Step 1 sample index: SKIP"
            endif

            @ rerun_count = 0
            while ( ! -f ${STEP1_SAMPLE_DIR}/complete )
                if ( -f ${STEP1_SAMPLE_DIR}/fail ) then
                    if ( $rerun_count >= $MAX_RERUN ) then
                        echo "ERROR: Step 1 sample index failed after ${MAX_RERUN} automatic reruns"
                        touch ${STEP1_DIR}/fail
                        exit 1
                    endif
                    @ rerun_count = $rerun_count + 1
                    echo "WARNING: Step 1 sample index failed; automatically rerun failed task only (${rerun_count}/${MAX_RERUN})"
                    rm -f ${STEP1_SAMPLE_DIR}/fail ${STEP1_SAMPLE_DIR}/status "${job_file}.running"
                    dsub_submit "${job_file}" "${dsub_out}" "${dsub_err}"
                    if ( $status != 0 ) touch ${STEP1_SAMPLE_DIR}/fail
                endif
                echo "  waiting Step 1 sample index complete; sleep ${POLL_SEC}s"
                sleep ${POLL_SEC}
            end

            # 新版采样 manifest 是 stem 列表的权威来源；对已有输出自动回填
            # legacy *_stems.txt，避免仅有 complete/manifest 时 cat 失败。
            if ( ! -f ${STEP1_SAMPLE_DIR}/train_stems.txt || ! -f ${STEP1_SAMPLE_DIR}/test_stems.txt ) then
                echo "Step 1 stem lists missing; backfill from sample_index_manifest.json"
                ${PYTHON} ${TIMEWINDOW_SRC}/utils/manifest_stems.py --manifest ${STEP1_SAMPLE_DIR}/sample_index_manifest.json --output-dir ${STEP1_SAMPLE_DIR}
                if ( $status != 0 ) then
                    echo "ERROR: failed to backfill Step 1 stem lists"
                    touch ${STEP1_DIR}/fail
                    exit 1
                endif
            endif
            set STEM_LIST = ( `cat ${STEP1_SAMPLE_DIR}/train_stems.txt ${STEP1_SAMPLE_DIR}/test_stems.txt` )
            if ( $#STEM_LIST == 0 ) then
                echo "ERROR: Step 1 manifest contains no train/test stems"
                touch ${STEP1_DIR}/fail
                exit 1
            endif
            echo "[`date '+%H:%M:%S'`] submit Step 1 case/combo jobs: $#STEM_LIST stems x $#COMBO_LIST combos"
            set STEP1_TASK_ID = 0
            foreach stem ( ${STEM_LIST} )
                foreach offset_k ( ${OFFSET_LIST} )
                    foreach dtype ( ${TYPE_LIST} )
                        set combo_key = "${dtype}_offset${offset_k}"
                        set part_dir = "${STEP1_PART_DIR}/${stem}/${combo_key}"
                        @ STEP1_TASK_ID = ${STEP1_TASK_ID} + 1
                        mkdir -p ${part_dir}
                        if ( -f ${part_dir}/complete ) then
                            echo "Step 1 ${stem}/${combo_key}: SKIP"
                            continue
                        endif
                        rm -f ${part_dir}/fail ${part_dir}/status
                        set task_tag = "t${STEP1_TASK_ID}_${combo_key}"
                        set job_file = "${JOB_DIR}/step1_${task_tag}.csh"
                        set job_log = "${LOG_DIR}/step1_${task_tag}.log"
                        set dsub_out = "${LOG_DIR}/step1_${task_tag}.dsub.out"
                        set dsub_err = "${LOG_DIR}/step1_${task_tag}.dsub.err"
                        cat > ${job_file} << EOF
#!/bin/csh -x
# 任务说明：执行主 pipeline 生成的单个计算阶段。
# 参数说明：输入、输出及训练参数均已由主脚本展开。
# dsub 状态协议：running -> complete 或 fail。
rm -f "${job_file}.running"
touch "${job_file}.running"
echo "[job] step1 ${stem}/${combo_key} start `date` host=`hostname`" >& ${job_log}
${PYTHON} ${TIMEWINDOW_SRC}/step1_prepare_timewindow.py --level-dir ${LEVEL_DIR} --tog-dir ${TOG_DIR} --label-dir ${LABEL_DIR} --output-dir ${part_dir} --offsets ${offset_k} --data-types ${dtype} --boundary-discard ${DISCARD} --test-count ${TEST_COUNT} --stage ${STAGE} --sample-rows-per-file 0 --sample-bins ${SAMPLE_BINS} --sample-min-per-bin ${SAMPLE_MIN_PER_BIN} --sample-seed ${SAMPLE_SEED} --jobs 1 --include-stems ${stem} --sample-index-manifest ${STEP1_SAMPLE_DIR}/sample_index_manifest.json --compress-output ${STEP1_COMPRESS_OUTPUT} --skip-existing ${STEP1_SKIP_EXISTING} >>& ${job_log}
set rc = \$status
echo \${rc} > ${part_dir}/status
if ( \${rc} == 0 && -f ${part_dir}/timewindow_manifest.json ) then
    touch ${part_dir}/complete
else
    touch ${part_dir}/fail
endif
rm -f "${job_file}.running"
exit \${rc}
EOF
                        chmod +x "${job_file}"
                        dsub_submit "${job_file}" "${dsub_out}" "${dsub_err}"
                        if ( $status != 0 ) then
                            echo "ERROR: Step 1 ${stem}/${combo_key} dsub 提交失败"
                            touch ${part_dir}/fail
                        endif
                    end
                end
            end

            @ rerun_count = 0
            while ( 1 )
                set DONE_COUNT = 0
                set FAIL_COUNT = 0
                foreach stem ( ${STEM_LIST} )
                    foreach combo_key ( ${COMBO_LIST} )
                        set part_dir = "${STEP1_PART_DIR}/${stem}/${combo_key}"
                        if ( -f ${part_dir}/complete ) @ DONE_COUNT = ${DONE_COUNT} + 1
                        if ( -f ${part_dir}/fail ) @ FAIL_COUNT = ${FAIL_COUNT} + 1
                    end
                end
                @ TOTAL_COUNT = $#STEM_LIST * $#COMBO_LIST
                if ( ${FAIL_COUNT} > 0 ) then
                    if ( $rerun_count >= $MAX_RERUN ) then
                        echo "ERROR: ${FAIL_COUNT} 个 Step 1 子任务在 ${MAX_RERUN} 次自动重提后仍失败"
                        touch ${STEP1_DIR}/fail
                        exit 1
                    endif
                    @ rerun_count = $rerun_count + 1
                    echo "WARNING: automatically rerun failed Step 1 tasks only (${rerun_count}/${MAX_RERUN})"
                    set STEP1_TASK_ID = 0
                    foreach stem ( ${STEM_LIST} )
                        foreach offset_k ( ${OFFSET_LIST} )
                            foreach dtype ( ${TYPE_LIST} )
                                @ STEP1_TASK_ID = ${STEP1_TASK_ID} + 1
                                set combo_key = "${dtype}_offset${offset_k}"
                                set task_tag = "t${STEP1_TASK_ID}_${combo_key}"
                                set part_dir = "${STEP1_PART_DIR}/${stem}/${combo_key}"
                                if ( -f ${part_dir}/fail ) then
                                    set job_file = "${JOB_DIR}/step1_${task_tag}.csh"
                                    set dsub_out = "${LOG_DIR}/step1_${task_tag}.dsub.out"
                                    set dsub_err = "${LOG_DIR}/step1_${task_tag}.dsub.err"
                                    rm -f ${part_dir}/fail ${part_dir}/status "${job_file}.running"
                                    dsub_submit "${job_file}" "${dsub_out}" "${dsub_err}"
                                    if ( $status != 0 ) touch ${part_dir}/fail
                                endif
                            end
                        end
                    end
                    continue
                endif
                if ( ${DONE_COUNT} == ${TOTAL_COUNT} ) break
                echo "  Step 1 progress: ${DONE_COUNT}/${TOTAL_COUNT}; sleep ${POLL_SEC}s"
                sleep ${POLL_SEC}
            end

            set STEP1_MANIFESTS = ( )
            foreach stem ( ${STEM_LIST} )
                foreach combo_key ( ${COMBO_LIST} )
                    set STEP1_MANIFESTS = ( ${STEP1_MANIFESTS} "${STEP1_PART_DIR}/${stem}/${combo_key}/timewindow_manifest.json" )
                end
            end
            set job_file = "${JOB_DIR}/step1_merge.csh"
            set dsub_out = "${LOG_DIR}/step1_merge.dsub.out"
            set dsub_err = "${LOG_DIR}/step1_merge.dsub.err"
            cat > ${job_file} << EOF
#!/bin/csh -x
# Step 1 merge：合并已完成的 case×combo manifests。
# 参数说明：输入 manifest 列表及输出路径均由主脚本展开。
rm -f ${STEP1_DIR}/complete ${STEP1_DIR}/fail ${STEP1_DIR}/status "${job_file}.running"
touch "${job_file}.running"
${PYTHON} ${TIMEWINDOW_SRC}/merge_timewindow_manifests.py --merge-step1-filelists --output ${STEP1_DIR}/timewindow_manifest.json --offsets ${OFFSETS} --data-types ${DATA_TYPES} --combo-order ${COMBO_ORDER} ${STEP1_MANIFESTS}
set rc = \$status
rm -f "${job_file}.running"
echo \${rc} > ${STEP1_DIR}/status
if ( \${rc} == 0 && -f ${STEP1_DIR}/timewindow_manifest.json ) then
    touch ${STEP1_DIR}/complete
else
    if ( \${rc} == 0 ) set rc = 1
    touch ${STEP1_DIR}/fail
endif
exit \${rc}
EOF
            chmod +x "${job_file}"
            @ rerun_count = 0
            while ( ! -f ${STEP1_DIR}/complete )
                rm -f ${STEP1_DIR}/fail ${STEP1_DIR}/status "${job_file}.running"
                echo "[`date '+%H:%M:%S'`] submit Step 1 merge; rerun=${rerun_count}/${MAX_RERUN}"
                dsub_submit "${job_file}" "${dsub_out}" "${dsub_err}"
                if ( $status != 0 ) touch ${STEP1_DIR}/fail
                while ( ! -f ${STEP1_DIR}/complete && ! -f ${STEP1_DIR}/fail )
                    sleep ${POLL_SEC}
                end
                if ( -f ${STEP1_DIR}/complete ) break
                if ( $rerun_count >= $MAX_RERUN ) then
                    echo "ERROR: Step 1 merge failed after ${MAX_RERUN} automatic reruns"
                    exit 1
                endif
                @ rerun_count = $rerun_count + 1
                echo "WARNING: automatically rerun failed Step 1 merge task only"
            end
        else
            set job_file = "${JOB_DIR}/step1_prepare.csh"
            set job_log = "${LOG_DIR}/step1_prepare.log"
            set dsub_out = "${LOG_DIR}/step1_prepare.dsub.out"
            set dsub_err = "${LOG_DIR}/step1_prepare.dsub.err"
            cat > ${job_file} << EOF
#!/bin/csh -x
# 任务说明：执行主 pipeline 生成的单个计算阶段。
# 参数说明：输入、输出及训练参数均已由主脚本展开。
# dsub 状态协议：running -> complete 或 fail。
rm -f "${job_file}.running"
touch "${job_file}.running"
echo "[job] step1 start `date` host=`hostname`" >& ${job_log}
mkdir -p ${STEP1_SAMPLE_DIR}
${PYTHON} ${TIMEWINDOW_SRC}/step1_make_global_sample_index.py --level-dir ${LEVEL_DIR} --tog-dir ${TOG_DIR} --label-dir ${LABEL_DIR} --output-dir ${STEP1_SAMPLE_DIR} --offsets ${OFFSETS} --boundary-discard ${DISCARD} --test-count ${TEST_COUNT} --stage ${STAGE} --sample-rows-total ${SAMPLE_ROWS_TOTAL} --sample-rows-per-file ${SAMPLE_ROWS_PER_FILE} --sample-power-bins ${SAMPLE_POWER_BINS} --sample-delta-bins ${SAMPLE_DELTA_BINS} --sample-all-threshold ${SAMPLE_ALL_THRESHOLD} --sample-min-per-case ${SAMPLE_MIN_PER_CASE} --sample-test-ratio ${SAMPLE_TEST_RATIO} --weight-exponent ${SAMPLE_WEIGHT_EXPONENT} --weight-min ${SAMPLE_WEIGHT_MIN} --weight-max ${SAMPLE_WEIGHT_MAX} --sample-seed ${SAMPLE_SEED} >>& ${job_log}
if ( \$status != 0 ) then
    set rc = \$status
    echo \${rc} > ${STEP1_DIR}/status
    rm -f "${job_file}.running"
    touch ${STEP1_DIR}/fail
    exit \${rc}
endif
${PYTHON} ${TIMEWINDOW_SRC}/step1_prepare_timewindow.py --level-dir ${LEVEL_DIR} --tog-dir ${TOG_DIR} --label-dir ${LABEL_DIR} --output-dir ${STEP1_DIR} --offsets ${OFFSETS} --data-types ${DATA_TYPES} --boundary-discard ${DISCARD} --test-count ${TEST_COUNT} --stage ${STAGE} --sample-index-manifest ${STEP1_SAMPLE_DIR}/sample_index_manifest.json --sample-seed ${SAMPLE_SEED} --jobs ${STEP1_JOBS} --compress-output ${STEP1_COMPRESS_OUTPUT} --skip-existing ${STEP1_SKIP_EXISTING} >>& ${job_log}
set rc = \$status
echo \${rc} > ${STEP1_DIR}/status
if ( \${rc} == 0 && -f ${STEP1_DIR}/timewindow_manifest.json ) then
    touch ${STEP1_DIR}/complete
else
    touch ${STEP1_DIR}/fail
endif
rm -f "${job_file}.running"
exit \${rc}
EOF
            echo "[`date '+%H:%M:%S'`] submit Step 1"
            chmod +x "${job_file}"
            dsub_submit "${job_file}" "${dsub_out}" "${dsub_err}"
            if ( $status != 0 ) then
                echo "ERROR: Step 1 dsub 提交失败"
                touch ${STEP1_DIR}/fail
            endif
        endif
    else
        echo "Step 1: SKIP"
    endif

    @ rerun_count = 0
    while ( ! -f ${STEP1_DIR}/complete )
        if ( -f ${STEP1_DIR}/fail ) then
            if ( ${STEP1_PARALLEL_DSUB} != 0 || $rerun_count >= $MAX_RERUN ) then
                echo "ERROR: Step 1 failed, see ${LOG_DIR}/step1*.log"
                exit 1
            endif
            @ rerun_count = $rerun_count + 1
            set job_file = "${JOB_DIR}/step1_prepare.csh"
            set dsub_out = "${LOG_DIR}/step1_prepare.dsub.out"
            set dsub_err = "${LOG_DIR}/step1_prepare.dsub.err"
            echo "WARNING: automatically rerun failed Step 1 task only (${rerun_count}/${MAX_RERUN})"
            rm -f ${STEP1_DIR}/fail ${STEP1_DIR}/status "${job_file}.running"
            dsub_submit "${job_file}" "${dsub_out}" "${dsub_err}"
            if ( $status != 0 ) touch ${STEP1_DIR}/fail
        endif
        echo "  waiting Step 1 complete; sleep ${POLL_SEC}s"
        sleep ${POLL_SEC}
    end

    # Older Step 1 outputs may have sampled labels but no recommendation JSON.
    # Backfill from the already sampled time-window labels without resampling.
    if ( -f "${WEIGHT_RECOMMENDATION_FILE}" ) then
        echo "Step 1 weight recommendation: SKIP (${WEIGHT_RECOMMENDATION_FILE})"
    else
        mkdir -p "${WEIGHT_RECOMMENDATION_STATUS_DIR}"
        set job_file = "${JOB_DIR}/step1_weight_recommendation.csh"
        set dsub_out = "${LOG_DIR}/step1_weight_recommendation.dsub.out"
        set dsub_err = "${LOG_DIR}/step1_weight_recommendation.dsub.err"
        cat > ${job_file} << EOF
#!/bin/csh -x
# 任务说明：旧版 Step 1 缺少权重建议时，基于已采样 labels 回填建议 JSON。
# 参数说明：读取 timewindow manifest，输出到 WEIGHT_RECOMMENDATION_FILE 指定路径。
rm -f "${WEIGHT_RECOMMENDATION_STATUS_DIR}/running" "${WEIGHT_RECOMMENDATION_STATUS_DIR}/complete" "${WEIGHT_RECOMMENDATION_STATUS_DIR}/fail" "${WEIGHT_RECOMMENDATION_STATUS_DIR}/status"
touch "${WEIGHT_RECOMMENDATION_STATUS_DIR}/running"
${PYTHON} ${TIMEWINDOW_SRC}/generate_sample_weight_recommendation.py \
    --tw-manifest "${STEP1_DIR}/timewindow_manifest.json" \
    --output-file "${WEIGHT_RECOMMENDATION_FILE}" \
    --power-bins ${SAMPLE_POWER_BINS} \
    --delta-bins ${SAMPLE_DELTA_BINS} \
    --rare-threshold ${SAMPLE_ALL_THRESHOLD} \
    --weight-exponent ${SAMPLE_WEIGHT_EXPONENT} \
    --weight-min ${SAMPLE_WEIGHT_MIN} \
    --weight-max ${SAMPLE_WEIGHT_MAX}
set rc = \$status
rm -f "${WEIGHT_RECOMMENDATION_STATUS_DIR}/running"
echo \${rc} > "${WEIGHT_RECOMMENDATION_STATUS_DIR}/status"
if ( \${rc} == 0 && -f "${WEIGHT_RECOMMENDATION_FILE}" ) then
    touch "${WEIGHT_RECOMMENDATION_STATUS_DIR}/complete"
else
    if ( \${rc} == 0 ) set rc = 1
    touch "${WEIGHT_RECOMMENDATION_STATUS_DIR}/fail"
endif
exit \${rc}
EOF
        chmod +x "${job_file}"
        @ rerun_count = 0
        while ( ! -f "${WEIGHT_RECOMMENDATION_FILE}" )
            rm -f "${WEIGHT_RECOMMENDATION_STATUS_DIR}/running" "${WEIGHT_RECOMMENDATION_STATUS_DIR}/complete" "${WEIGHT_RECOMMENDATION_STATUS_DIR}/fail" "${WEIGHT_RECOMMENDATION_STATUS_DIR}/status"
            echo "[`date '+%H:%M:%S'`] submit Step 1 weight recommendation; rerun=${rerun_count}/${MAX_RERUN}"
            dsub_submit "${job_file}" "${dsub_out}" "${dsub_err}"
            if ( $status != 0 ) touch "${WEIGHT_RECOMMENDATION_STATUS_DIR}/fail"

            while ( ! -f "${WEIGHT_RECOMMENDATION_FILE}" && ! -f "${WEIGHT_RECOMMENDATION_STATUS_DIR}/fail" )
                if ( -f "${WEIGHT_RECOMMENDATION_STATUS_DIR}/running" ) then
                    echo "  Step 1 weight recommendation running; sleep ${POLL_SEC}s"
                else
                    echo "  Step 1 weight recommendation queued; sleep ${POLL_SEC}s"
                endif
                sleep ${POLL_SEC}
            end
            if ( -f "${WEIGHT_RECOMMENDATION_FILE}" ) break
            if ( $rerun_count >= $MAX_RERUN ) then
                echo "ERROR: Step 1 weight recommendation failed after ${MAX_RERUN} automatic reruns"
                echo "  stdout: ${dsub_out}"
                echo "  stderr: ${dsub_err}"
                exit 1
            endif
            @ rerun_count = $rerun_count + 1
            echo "WARNING: automatically rerun failed Step 1 weight recommendation only"
        end
    endif

    #===============================================================================
    # ▶ Step 2 控制流：[selector] -> [combo 并行任务] -> [selector manifest 合并]
    # 每个 combo 单独 dsub；MCP 参数和 dsub 资源按 combo_key 中的 level/tog 选择。
    #===============================================================================
    # STEP2_DIR 保存整个 Step 2 的聚合状态，与 selector 输出目录相互独立。
    mkdir -p ${STEP2_DIR}
    if ( -f ${STEP2_DIR}/complete ) then
        foreach selector ( ${STEP2_SELECTOR_LIST} )
            if ( "${selector}" == "mcp" ) then
                set selector_dir = "${STEP2_MCP_DIR}"
            else
                set selector_dir = "${STEP2_XGB_DIR}"
            endif
            if ( ! -f ${selector_dir}/mcp_per_offset_manifest.json ) rm -f ${STEP2_DIR}/complete
        end
    endif
    if ( ! -f ${STEP2_DIR}/complete ) then
        foreach selector ( ${STEP2_SELECTOR_LIST} )
            if ( "${selector}" == "mcp" ) then
                set selector_dir = "${STEP2_MCP_DIR}"
                set selector_program = "${TIMEWINDOW_SRC}/step2_mcp_per_offset.py"
            else
                set selector_dir = "${STEP2_XGB_DIR}"
                set selector_program = "${TIMEWINDOW_SRC}/step2_xgb_per_offset.py"
                set selector_args = ( --top-k ${XGB_PROXY_TOP_K} --rounds ${XGB_PROXY_ROUNDS} --max-depth ${XGB_PROXY_MAX_DEPTH} --eta ${XGB_PROXY_ETA} --subsample ${XGB_PROXY_SUBSAMPLE} --colsample-bytree ${XGB_PROXY_COLSAMPLE} --nthread ${XGB_PROXY_NTHREAD} )
            endif
            set selector_part_dir = "${selector_dir}/_combo_parts"
            mkdir -p ${selector_dir} ${selector_part_dir}
            rm -f ${selector_dir}/complete ${selector_dir}/fail ${selector_dir}/status
            foreach combo_key ( ${COMBO_LIST} )
                # combo_key 含 level 时使用 level 专用 MCP 参数和调度资源，否则使用 tog。
                set alpha_mcp = ${ALPHA_MCP_TOG}
                set batch_size = ${BATCH_SIZE_TOG}
                set DSUB_EXTRA_ARGS = ( $STEP2_TOG_DSUB_EXTRA_ARGS )
                if ( "${combo_key}" =~ *level* ) then
                    set alpha_mcp = ${ALPHA_MCP_LEVEL}
                    set batch_size = ${BATCH_SIZE_LEVEL}
                    set DSUB_EXTRA_ARGS = ( $STEP2_LEVEL_DSUB_EXTRA_ARGS )
                endif
                if ( "${selector}" == "mcp" ) then
                    set selector_args = ( --alpha ${alpha_mcp} --gamma ${GAMMA} --alpha-l2 ${ALPHA_L2} --batch ${batch_size} --top-k ${MCP_PROXY_TOP_K} )
                endif
                set part_dir = "${selector_part_dir}/${combo_key}"
                mkdir -p ${part_dir}
                if ( -f ${part_dir}/complete && -f ${part_dir}/mcp_per_offset_manifest.json ) continue
                rm -f ${part_dir}/complete ${part_dir}/fail ${part_dir}/status
                set job_file = "${JOB_DIR}/step2_${selector}_${combo_key}.csh"
                set job_log = "${LOG_DIR}/step2_${selector}_${combo_key}.log"
                set dsub_out = "${LOG_DIR}/step2_${selector}_${combo_key}.dsub.out"
                set dsub_err = "${LOG_DIR}/step2_${selector}_${combo_key}.dsub.err"
                cat > ${job_file} << EOF
#!/bin/csh -x
# 任务说明：处理 Step 2 ${selector} 的一个 data_type×offset 组合。
# dsub 状态协议：running -> complete 或 fail。
rm -f "${job_file}.running"
touch "${job_file}.running"
${PYTHON} ${selector_program} --manifest ${STEP1_DIR}/timewindow_manifest.json --namefile ${NAMEFILE} $selector_args:q --jobs 1 --include-combos ${combo_key} --output-dir ${part_dir} >>& ${job_log}
set rc = \$status
echo \${rc} > ${part_dir}/status
if ( \${rc} == 0 && -f ${part_dir}/mcp_per_offset_manifest.json ) then
    touch ${part_dir}/complete
else
    touch ${part_dir}/fail
endif
rm -f "${job_file}.running"
exit \${rc}
EOF
                chmod +x "${job_file}"
                dsub_submit "${job_file}" "${dsub_out}" "${dsub_err}"
                if ( $status != 0 ) touch ${part_dir}/fail
            end
            @ rerun_count = 0
            while ( 1 )
                set DONE_COUNT = 0
                set FAIL_COUNT = 0
                foreach combo_key ( ${COMBO_LIST} )
                    set part_dir = "${selector_part_dir}/${combo_key}"
                    if ( -f ${part_dir}/complete ) @ DONE_COUNT = ${DONE_COUNT} + 1
                    if ( -f ${part_dir}/fail ) @ FAIL_COUNT = ${FAIL_COUNT} + 1
                end
                if ( ${FAIL_COUNT} > 0 ) then
                    if ( $rerun_count >= $MAX_RERUN ) then
                        echo "ERROR: Step 2 ${selector} combo tasks failed after ${MAX_RERUN} reruns"
                        touch ${selector_dir}/fail ${STEP2_DIR}/fail
                        exit 1
                    endif
                    @ rerun_count = $rerun_count + 1
                    foreach combo_key ( ${COMBO_LIST} )
                        set DSUB_EXTRA_ARGS = ( $STEP2_TOG_DSUB_EXTRA_ARGS )
                        if ( "${combo_key}" =~ *level* ) then
                            set DSUB_EXTRA_ARGS = ( $STEP2_LEVEL_DSUB_EXTRA_ARGS )
                        endif
                        set part_dir = "${selector_part_dir}/${combo_key}"
                        if ( -f ${part_dir}/fail ) then
                            set job_file = "${JOB_DIR}/step2_${selector}_${combo_key}.csh"
                            set dsub_out = "${LOG_DIR}/step2_${selector}_${combo_key}.dsub.out"
                            set dsub_err = "${LOG_DIR}/step2_${selector}_${combo_key}.dsub.err"
                            rm -f ${part_dir}/fail ${part_dir}/status "${job_file}.running"
                            dsub_submit "${job_file}" "${dsub_out}" "${dsub_err}"
                            if ( $status != 0 ) touch ${part_dir}/fail
                        endif
                    end
                    continue
                endif
                if ( ${DONE_COUNT} == $#COMBO_LIST ) break
                echo "  Step 2 ${selector} combo progress: ${DONE_COUNT}/$#COMBO_LIST; sleep ${POLL_SEC}s"
                sleep ${POLL_SEC}
            end

            set STEP2_PART_MANIFESTS = ( )
            foreach combo_key ( ${COMBO_LIST} )
                set STEP2_PART_MANIFESTS = ( ${STEP2_PART_MANIFESTS} "${selector_part_dir}/${combo_key}/mcp_per_offset_manifest.json" )
            end
            set job_file = "${JOB_DIR}/step2_${selector}_merge.csh"
            set job_log = "${LOG_DIR}/step2_${selector}_merge.log"
            set dsub_out = "${LOG_DIR}/step2_${selector}_merge.dsub.out"
            set dsub_err = "${LOG_DIR}/step2_${selector}_merge.dsub.err"
            cat > ${job_file} << EOF
#!/bin/csh -x
# 任务说明：合并 Step 2 ${selector} 的逐组合 manifest。
# dsub 状态协议：running -> complete 或 fail。
rm -f "${job_file}.running"
touch "${job_file}.running"
${PYTHON} ${TIMEWINDOW_SRC}/merge_timewindow_manifests.py --output ${selector_dir}/mcp_per_offset_manifest.json --combo-order ${COMBO_ORDER} ${STEP2_PART_MANIFESTS} >>& ${job_log}
set rc = \$status
echo \${rc} > ${selector_dir}/status
if ( \${rc} == 0 && -f ${selector_dir}/mcp_per_offset_manifest.json ) then
    touch ${selector_dir}/complete
else
    touch ${selector_dir}/fail
endif
rm -f "${job_file}.running"
exit \${rc}
EOF
            @ rerun_count = 0
            # selector 合并任务不区分 level/tog，使用独立的轻量资源预留。
            set DSUB_EXTRA_ARGS = ( $STEP2_MERGE_DSUB_EXTRA_ARGS )
            while ( ! -f ${selector_dir}/complete )
                rm -f ${selector_dir}/fail ${selector_dir}/status "${job_file}.running"
                chmod +x "${job_file}"
                dsub_submit "${job_file}" "${dsub_out}" "${dsub_err}"
                if ( $status != 0 ) touch ${selector_dir}/fail
                while ( ! -f ${selector_dir}/complete && ! -f ${selector_dir}/fail )
                    sleep ${POLL_SEC}
                end
                if ( -f ${selector_dir}/complete ) break
                if ( $rerun_count >= $MAX_RERUN ) then
                    echo "ERROR: Step 2 ${selector} manifest merge failed after ${MAX_RERUN} reruns"
                    touch ${STEP2_DIR}/fail
                    exit 1
                endif
                @ rerun_count = $rerun_count + 1
            end
        end
        echo 0 > ${STEP2_DIR}/status
        touch ${STEP2_DIR}/complete
    else
        echo "Step 2 ${STEP2_SELECTOR_MODE}: SKIP"
    endif
    set STEP2_VALIDATION_MANIFEST = "${STEP2_XGB_DIR}/mcp_per_offset_manifest.json"
    if ( "$STEP2_SELECTOR_MODE" == "single" && "$PROXY_SELECTOR" == "mcp" ) then
        set STEP2_VALIDATION_MANIFEST = "${STEP2_MCP_DIR}/mcp_per_offset_manifest.json"
    endif
    if ( ${RUN_STEP2_VALIDATION} != 0 ) then
        # validation 是 Step 2 的独立旁路，不复用 level/tog combo 资源。
        set DSUB_EXTRA_ARGS = ( $STEP2_VALIDATION_DSUB_EXTRA_ARGS )
        set validation_summary = "${STEP2_VALIDATION_DIR}/legacy_xgb_validation/tile_reset_domain-bc6/eval/evaluation_summary.json"
        if ( ! -f ${STEP2_VALIDATION_DIR}/complete ) then
            mkdir -p ${STEP2_VALIDATION_DIR}
            rm -f ${STEP2_VALIDATION_DIR}/fail ${STEP2_VALIDATION_DIR}/status
            set job_file = "${JOB_DIR}/step2_validation_legacy_xgb.csh"
            set job_log = "${LOG_DIR}/step2_validation_legacy_xgb.log"
            set dsub_out = "${LOG_DIR}/step2_validation_legacy_xgb.dsub.out"
            set dsub_err = "${LOG_DIR}/step2_validation_legacy_xgb.dsub.err"
            cat > ${job_file} << EOF
#!/bin/csh -x
# 任务说明：执行主 pipeline 生成的单个计算阶段。
# 参数说明：输入、输出及训练参数均已由主脚本展开。
# dsub 状态协议：running -> complete 或 fail。
rm -f "${job_file}.running"
touch "${job_file}.running"
echo "[job] post-Step2 legacy XGBoost validation start `date` host=`hostname`" >& ${job_log}
cd ${STEP2_VALIDATION_DIR}
${PYTHON} ${TIMEWINDOW_SRC}/step2_validation/xgb_power_v2_3.py --tw-manifest ${STEP1_DIR}/timewindow_manifest.json --proxy-manifest ${STEP2_VALIDATION_MANIFEST} --namefile ${NAMEFILE} --combos "${STEP2_VALIDATION_COMBOS}" --cache-dir ${STEP2_VALIDATION_DIR}/dataset_cache --build-block-size ${STEP2_VALIDATION_CHUNK_SIZE} --hpotag legacy_xgb_validation >>& ${job_log}
set rc = \$status
echo \${rc} > ${STEP2_VALIDATION_DIR}/status
if ( \${rc} == 0 && -f ${STEP2_VALIDATION_DIR}/legacy_xgb_validation/tile_reset_domain-bc6/eval/evaluation_summary.json ) then
    touch ${STEP2_VALIDATION_DIR}/complete
else
    touch ${STEP2_VALIDATION_DIR}/fail
endif
rm -f "${job_file}.running"
exit \${rc}
EOF
            echo "[`date '+%H:%M:%S'`] submit post-Step2 legacy XGBoost validation"
            chmod +x "${job_file}"
            dsub_submit "${job_file}" "${dsub_out}" "${dsub_err}"
            if ( $status != 0 ) then
                echo "ERROR: post-Step2 legacy XGBoost validation dsub 提交失败"
                touch ${STEP2_VALIDATION_DIR}/fail
            endif
        else
            echo "Post-Step2 legacy XGBoost validation: SKIP"
        endif

        @ rerun_count = 0
        while ( ! -f ${STEP2_VALIDATION_DIR}/complete )
            if ( -f ${STEP2_VALIDATION_DIR}/fail ) then
                if ( $rerun_count >= $MAX_RERUN ) then
                    echo "ERROR: post-Step2 validation failed after ${MAX_RERUN} automatic reruns"
                    exit 1
                endif
                @ rerun_count = $rerun_count + 1
                echo "WARNING: automatically rerun failed post-Step2 validation only (${rerun_count}/${MAX_RERUN})"
                rm -f ${STEP2_VALIDATION_DIR}/fail ${STEP2_VALIDATION_DIR}/status "${job_file}.running"
                dsub_submit "${job_file}" "${dsub_out}" "${dsub_err}"
                if ( $status != 0 ) touch ${STEP2_VALIDATION_DIR}/fail
            endif
            echo "  waiting post-Step2 validation complete; sleep ${POLL_SEC}s"
            sleep ${POLL_SEC}
        end
        if ( ! -f ${validation_summary} ) then
            echo "ERROR: missing validation summary: ${validation_summary}"
            exit 1
        endif
    else
        echo "Post-Step2 legacy XGBoost validation: disabled"
    endif
endif

# Step 1/2 是共享基础结果；Step 3–6 按所选组合或 EXTRA_TAG 隔离输出。
if ( ! -f "${STEP1_DIR}/timewindow_manifest.json" ) then
    echo "ERROR: missing ${STEP1_DIR}/timewindow_manifest.json; run all/prep mode first"
    exit 1
endif
set OUTPUT_TAG = `"$PYTHON" "${TIMEWINDOW_SRC}/build_timewindow_output_tag.py" \
    --tw-manifest "${STEP1_DIR}/timewindow_manifest.json" \
    --offsets "$STEP3_OFFSETS" \
    --data-types "$STEP3_DATA_TYPES" \
    --include-combos "$STEP3_COMBOS" \
    --override "$EXTRA_TAG"`
if ( $status != 0 || "$OUTPUT_TAG" == "" ) then
    echo "ERROR: failed to build Step3-Step6 output tag"
    exit 2
endif

set TAG_OUTPUT_ROOT = "${OUTPUT_ROOT}/${OUTPUT_TAG}"
set STEP3_DIR = "${TAG_OUTPUT_ROOT}/step3_hybrid_concat"
set STEP4_DIR = "${TAG_OUTPUT_ROOT}/step4_hybrid_screen"
set STEP5_DIR = "${TAG_OUTPUT_ROOT}/step5_hpo"
set STEP6_DIR = "${TAG_OUTPUT_ROOT}/step6_retrain"
set MLP_STEP5_DIR = "${TAG_OUTPUT_ROOT}/step5_mlp_hpo"
set MLP_STEP6_DIR = "${TAG_OUTPUT_ROOT}/step6_mlp_retrain"
set STEP3_INPUT_DIR = "${TAG_OUTPUT_ROOT}/input_manifests"
set STEP3_TW_MANIFEST = "${STEP1_DIR}/timewindow_manifest.json"
set STEP3_MCP_MANIFEST = "${STEP2_MCP_DIR}/mcp_per_offset_manifest.json"
set STEP3_XGB_MANIFEST = "${STEP2_XGB_DIR}/mcp_per_offset_manifest.json"
set STEP3_XGB_TW_MANIFEST = "${STEP3_INPUT_DIR}/timewindow_manifest_xgb.json"
if ( "$STEP2_SELECTOR_MODE" == "single" ) then
    set STEP3_DIR = "${TAG_OUTPUT_ROOT}/step3_single_${PROXY_SELECTOR}_concat"
    set STEP4_DIR = "${TAG_OUTPUT_ROOT}/step4_single_${PROXY_SELECTOR}_screen"
endif
if ( "$FEATURE_INPUT_MODE" == "signal_all_windows" ) then
    set STEP4_DIR = "${STEP4_DIR}_signal_all_windows"
endif
set LOG_DIR = "${TAG_OUTPUT_ROOT}/logs_dsub"
set JOB_DIR = "${TAG_OUTPUT_ROOT}/jobs_dsub"
if ( "$STEP2_SELECTOR_MODE" != "hybrid" || "$FEATURE_INPUT_MODE" != "proxy_columns" ) then
    set DOWNSTREAM_VARIANT = "${STEP2_SELECTOR_MODE}_${PROXY_SELECTOR}_${FEATURE_INPUT_MODE}"
    set STEP5_DIR = "${TAG_OUTPUT_ROOT}/step5_hpo_${DOWNSTREAM_VARIANT}"
    set STEP6_DIR = "${TAG_OUTPUT_ROOT}/step6_retrain_${DOWNSTREAM_VARIANT}"
    set MLP_STEP5_DIR = "${TAG_OUTPUT_ROOT}/step5_mlp_hpo_${DOWNSTREAM_VARIANT}"
    set MLP_STEP6_DIR = "${TAG_OUTPUT_ROOT}/step6_mlp_retrain_${DOWNSTREAM_VARIANT}"
    set LOG_DIR = "${TAG_OUTPUT_ROOT}/logs_dsub_${DOWNSTREAM_VARIANT}"
    set JOB_DIR = "${TAG_OUTPUT_ROOT}/jobs_dsub_${DOWNSTREAM_VARIANT}"
endif
if ( "$STEP3_COMBOS" != "" || "$STEP3_OFFSETS" != "$OFFSETS" || "$STEP3_DATA_TYPES" != "$DATA_TYPES" ) then
    set STEP3_TW_MANIFEST = "${STEP3_INPUT_DIR}/timewindow_manifest.json"
    set STEP3_MCP_MANIFEST = "${STEP3_INPUT_DIR}/mcp_per_offset_manifest.json"
    set STEP3_XGB_MANIFEST = "${STEP3_INPUT_DIR}/xgb_per_offset_manifest.json"
    if ( "$STEP2_SELECTOR_MODE" == "single" && "$PROXY_SELECTOR" == "xgb" ) then
        set STEP3_XGB_TW_MANIFEST = "${STEP3_TW_MANIFEST}"
    endif
endif
mkdir -p ${TAG_OUTPUT_ROOT} ${LOG_DIR} ${JOB_DIR}

echo "Step3-Step6 output tag: ${OUTPUT_TAG}"
echo "Tagged output root: ${TAG_OUTPUT_ROOT}"

#===============================================================================
# ▶ Step 3 控制流：[按需过滤 manifest] -> [MCP/XGB 特征拼接] -> [等待/失败重提]
# 关键分支：single 模式只拼接指定 selector，hybrid 模式合并 MCP 与 XGB。
#===============================================================================
if ( "$PIPELINE_MODE" != "train_k" ) then
    set DSUB_EXTRA_ARGS = ( $STEP3_DSUB_EXTRA_ARGS )

    if ( ! -f ${STEP3_DIR}/complete ) then
        mkdir -p ${STEP3_DIR}
        rm -f ${STEP3_DIR}/fail ${STEP3_DIR}/status
        set job_file = "${JOB_DIR}/step3_concat.csh"
        set job_log = "${LOG_DIR}/step3_concat.log"
        set dsub_out = "${LOG_DIR}/step3_concat.dsub.out"
        set dsub_err = "${LOG_DIR}/step3_concat.dsub.err"
        cat > ${job_file} << EOF
#!/bin/csh -x
# 任务说明：执行主 pipeline 生成的单个计算阶段。
# 参数说明：输入、输出及训练参数均已由主脚本展开。
# dsub 状态协议：running -> complete 或 fail。
rm -f "${job_file}.running"
touch "${job_file}.running"
echo "[job] step3 start `date` host=`hostname`" >& ${job_log}
if ( "${STEP3_TW_MANIFEST}" != "${STEP1_DIR}/timewindow_manifest.json" ) then
    mkdir -p ${STEP3_INPUT_DIR}
    if ( "${STEP2_SELECTOR_MODE}" == "hybrid" || "${PROXY_SELECTOR}" == "mcp" ) then
        ${PYTHON} ${TIMEWINDOW_SRC}/filter_timewindow_manifests.py --tw-manifest-in ${STEP1_DIR}/timewindow_manifest.json --mcp-manifest-in ${STEP2_MCP_DIR}/mcp_per_offset_manifest.json --tw-manifest-out ${STEP3_TW_MANIFEST} --mcp-manifest-out ${STEP3_MCP_MANIFEST} --offsets ${STEP3_OFFSETS} --data-types ${STEP3_DATA_TYPES} --include-combos "${STEP3_COMBOS}" >>& ${job_log}
        if ( \$status != 0 ) then
            echo \$status > ${STEP3_DIR}/status
            rm -f "${job_file}.running"
            touch ${STEP3_DIR}/fail
            exit 1
        endif
    endif
    if ( "${STEP2_SELECTOR_MODE}" == "hybrid" || "${PROXY_SELECTOR}" == "xgb" ) then
        ${PYTHON} ${TIMEWINDOW_SRC}/filter_timewindow_manifests.py --tw-manifest-in ${STEP1_DIR}/timewindow_manifest.json --mcp-manifest-in ${STEP2_XGB_DIR}/mcp_per_offset_manifest.json --tw-manifest-out ${STEP3_XGB_TW_MANIFEST} --mcp-manifest-out ${STEP3_XGB_MANIFEST} --offsets ${STEP3_OFFSETS} --data-types ${STEP3_DATA_TYPES} --include-combos "${STEP3_COMBOS}" >>& ${job_log}
        if ( \$status != 0 ) then
            echo \$status > ${STEP3_DIR}/status
            rm -f "${job_file}.running"
            touch ${STEP3_DIR}/fail
            exit 1
        endif
    endif
endif
if ( "${STEP2_SELECTOR_MODE}" == "single" ) then
    if ( "${PROXY_SELECTOR}" == "mcp" ) then
        ${PYTHON} ${TIMEWINDOW_SRC}/step3_hybrid_concat_features.py --tw-manifest ${STEP3_TW_MANIFEST} --mcp-manifest ${STEP3_MCP_MANIFEST} --selectors mcp --namefile ${NAMEFILE} --shuffle-seed ${SHUFFLE_SEED} --output-dir ${STEP3_DIR} >>& ${job_log}
    else
        ${PYTHON} ${TIMEWINDOW_SRC}/step3_hybrid_concat_features.py --tw-manifest ${STEP3_TW_MANIFEST} --xgb-manifest ${STEP3_XGB_MANIFEST} --selectors xgb --namefile ${NAMEFILE} --shuffle-seed ${SHUFFLE_SEED} --output-dir ${STEP3_DIR} >>& ${job_log}
    endif
else
    ${PYTHON} ${TIMEWINDOW_SRC}/step3_hybrid_concat_features.py --tw-manifest ${STEP3_TW_MANIFEST} --mcp-manifest ${STEP3_MCP_MANIFEST} --xgb-manifest ${STEP3_XGB_MANIFEST} --selectors mcp,xgb --namefile ${NAMEFILE} --shuffle-seed ${SHUFFLE_SEED} --output-dir ${STEP3_DIR} >>& ${job_log}
endif
set rc = \$status
echo \${rc} > ${STEP3_DIR}/status
if ( \${rc} == 0 && -f ${STEP3_DIR}/hybrid_concat_meta.json ) then
    touch ${STEP3_DIR}/complete
else
    touch ${STEP3_DIR}/fail
endif
rm -f "${job_file}.running"
exit \${rc}
EOF
        echo "[`date '+%H:%M:%S'`] submit Step 3"
        chmod +x "${job_file}"
        dsub_submit "${job_file}" "${dsub_out}" "${dsub_err}"
        if ( $status != 0 ) then
            echo "ERROR: Step 3 dsub 提交失败"
            touch ${STEP3_DIR}/fail
        endif
    else
        echo "Step 3: SKIP"
    endif

    @ rerun_count = 0
    while ( ! -f ${STEP3_DIR}/complete )
        if ( -f ${STEP3_DIR}/fail ) then
            if ( $rerun_count >= $MAX_RERUN ) then
                echo "ERROR: Step 3 failed after ${MAX_RERUN} automatic reruns; see ${LOG_DIR}/step3_concat.log"
                exit 1
            endif
            @ rerun_count = $rerun_count + 1
            echo "WARNING: automatically rerun failed Step 3 only (${rerun_count}/${MAX_RERUN})"
            rm -f ${STEP3_DIR}/fail ${STEP3_DIR}/status "${job_file}.running"
            dsub_submit "${job_file}" "${dsub_out}" "${dsub_err}"
            if ( $status != 0 ) touch ${STEP3_DIR}/fail
        endif
        echo "  waiting Step 3 complete; sleep ${POLL_SEC}s"
        sleep ${POLL_SEC}
    end

    #===============================================================================
    # ▶ Step 4 控制流：[读取拼接特征] -> [混合筛选 Top-K] -> [等待/失败重提]
    # 该阶段生成 Step 5/6 按 K 并行训练所需的 buffer 和 feature_names。
    #===============================================================================
    set DSUB_EXTRA_ARGS = ( $STEP4_DSUB_EXTRA_ARGS )
    set first_k = `echo ${TOP_KS} | awk -F, '{print $1}'`
    if ( ! -f ${STEP4_DIR}/complete ) then
        mkdir -p ${STEP4_DIR}
        rm -f ${STEP4_DIR}/fail ${STEP4_DIR}/status
        set job_file = "${JOB_DIR}/step4_screen.csh"
        set job_log = "${LOG_DIR}/step4_screen.log"
        set dsub_out = "${LOG_DIR}/step4_screen.dsub.out"
        set dsub_err = "${LOG_DIR}/step4_screen.dsub.err"
        cat > ${job_file} << EOF
#!/bin/csh -x
# 任务说明：执行主 pipeline 生成的单个计算阶段。
# 参数说明：输入、输出及训练参数均已由主脚本展开。
# dsub 状态协议：running -> complete 或 fail。
rm -f "${job_file}.running"
touch "${job_file}.running"
echo "[job] step4 start `date` host=`hostname`" >& ${job_log}
${PYTHON} ${TIMEWINDOW_SRC}/step4_hybrid_screen_features.py --hybrid-concat-meta ${STEP3_DIR}/hybrid_concat_meta.json --final-top-ks ${TOP_KS} --feature-input-mode ${FEATURE_INPUT_MODE} --tw-manifest ${STEP3_TW_MANIFEST} --namefile ${NAMEFILE} --ridge-alpha ${ALPHA_L2} --xgb-rounds ${HYBRID_SCREEN_XGB_ROUNDS} --nthread ${XGB_PROXY_NTHREAD} --loss ${XGB_LOSS} --relative-floor ${XGB_RELATIVE_FLOOR} --focus-power ${XGB_FOCUS_POWER} --shuffle-seed ${SHUFFLE_SEED} --output-dir ${STEP4_DIR} >>& ${job_log}
set rc = \$status
echo \${rc} > ${STEP4_DIR}/status
if ( \${rc} == 0 && -f ${STEP4_DIR}/union_K${first_k}/train_union_top${first_k}.buffer ) then
    touch ${STEP4_DIR}/complete
else
    touch ${STEP4_DIR}/fail
endif
rm -f "${job_file}.running"
exit \${rc}
EOF
        echo "[`date '+%H:%M:%S'`] submit Step 4"
        chmod +x "${job_file}"
        dsub_submit "${job_file}" "${dsub_out}" "${dsub_err}"
        if ( $status != 0 ) then
            echo "ERROR: Step 4 dsub 提交失败"
            touch ${STEP4_DIR}/fail
        endif
    else
        echo "Step 4: SKIP"
    endif

    @ rerun_count = 0
    while ( ! -f ${STEP4_DIR}/complete )
        if ( -f ${STEP4_DIR}/fail ) then
            if ( $rerun_count >= $MAX_RERUN ) then
                echo "ERROR: Step 4 failed after ${MAX_RERUN} automatic reruns; see ${LOG_DIR}/step4_screen.log"
                exit 1
            endif
            @ rerun_count = $rerun_count + 1
            echo "WARNING: automatically rerun failed Step 4 only (${rerun_count}/${MAX_RERUN})"
            rm -f ${STEP4_DIR}/fail ${STEP4_DIR}/status "${job_file}.running"
            dsub_submit "${job_file}" "${dsub_out}" "${dsub_err}"
            if ( $status != 0 ) touch ${STEP4_DIR}/fail
        endif
        echo "  waiting Step 4 complete; sleep ${POLL_SEC}s"
        sleep ${POLL_SEC}
    end
endif

if ( "$PIPELINE_MODE" == "prep" ) then
    echo "[`date '+%H:%M:%S'`] PREP DONE. Tagged features under ${TAG_OUTPUT_ROOT}"
    touch ${TAG_OUTPUT_ROOT}/complete
    exit 0
endif

if ( ! -f ${STEP4_DIR}/complete ) then
    echo "ERROR: missing ${STEP4_DIR}/complete; run prep mode first"
    exit 1
endif
if ( "$PIPELINE_MODE" == "train_k" && ! -f ${STEP4_DIR}/union_K${TARGET_K}/feature_names.txt ) then
    echo "ERROR: missing ${STEP4_DIR}/union_K${TARGET_K}/feature_names.txt; run prep mode first"
    exit 1
endif

#===============================================================================
# ▶ Step 5 控制流：[K 并行 HPO] -> [轮询 complete/fail] -> [仅重提失败 K]
# 每个 K 使用同一组 Step 5 独立 dsub 资源预留。
#===============================================================================
set DSUB_EXTRA_ARGS = ( $STEP5_DSUB_EXTRA_ARGS )
mkdir -p ${STEP5_DIR}
set XGB_TRAIN_TAG = direct_xgb
if ( "$TWO_PHASE_METHOD" == "A" ) set XGB_TRAIN_TAG = "methodA_c${TWO_PHASE_CUTOFF}_r${TWO_PHASE_RATIO}"
set STEP5_TASK_COUNT = 0
foreach k ( ${K_LIST} )
    set k_dir = "${STEP4_DIR}/union_K${k}"
    set hpo_out = "${STEP5_DIR}/union_K${k}_${XGB_TRAIN_TAG}"
    mkdir -p ${hpo_out}
    if ( -f ${hpo_out}/complete ) then
        echo "Step 5 HPO K=${k}: SKIP"
        continue
    endif
    rm -f ${hpo_out}/fail ${hpo_out}/status
    set job_file = "${JOB_DIR}/step5_hpo_K${k}.csh"
    set job_log = "${LOG_DIR}/step5_hpo_K${k}.log"
    set dsub_out = "${LOG_DIR}/step5_hpo_K${k}.dsub.out"
    set dsub_err = "${LOG_DIR}/step5_hpo_K${k}.dsub.err"
    cat > ${job_file} << EOF
#!/bin/csh -x
# 任务说明：执行主 pipeline 生成的单个计算阶段。
# 参数说明：输入、输出及训练参数均已由主脚本展开。
# dsub 状态协议：running -> complete 或 fail。
rm -f "${job_file}.running"
touch "${job_file}.running"
echo "[job] step5 K=${k} start `date` host=`hostname`" >& ${job_log}
if ( ! -f ${k_dir}/train_union_top${k}.buffer ) then
    echo "ERROR: missing ${k_dir}/train_union_top${k}.buffer" >>& ${job_log}
    rm -f "${job_file}.running"
    touch ${hpo_out}/fail
    exit 1
endif
if ( -f ${hpo_out}/best_trials_leaf_1000.csv ) then
    rm -f "${job_file}.running"
    touch ${hpo_out}/complete
    exit 0
endif
${PYTHON} ${TIMEWINDOW_SRC}/step5_hpo_timewindow.py --train-buffer ${k_dir}/train_union_top${k}.buffer --test-buffer ${k_dir}/test_union_top${k}.buffer --feature-names ${k_dir}/feature_names.txt --output-dir ${hpo_out} --round-num ${HPO_ROUNDS} --trial-num ${HPO_TRIALS} --pop-size ${HPO_POP} --job-num ${HPO_JOB_NUM} --nfold ${HPO_NFOLD} --two-phase-method "${TWO_PHASE_METHOD}" --two-phase-cutoff ${TWO_PHASE_CUTOFF} --two-phase-ratio ${TWO_PHASE_RATIO} --loss ${XGB_LOSS} --relative-floor ${XGB_RELATIVE_FLOOR} --focus-power ${XGB_FOCUS_POWER} --closeness-metric ${HPO_CLOSENESS_METRIC} --shuffle-seed ${SHUFFLE_SEED} >>& ${job_log}
set rc = \$status
echo \${rc} > ${hpo_out}/status
if ( \${rc} == 0 && -f ${hpo_out}/best_trials_leaf_1000.csv ) then
    touch ${hpo_out}/complete
else
    touch ${hpo_out}/fail
endif
rm -f "${job_file}.running"
exit \${rc}
EOF
    echo "[`date '+%H:%M:%S'`] submit Step 5 K=${k}"
    chmod +x "${job_file}"
    dsub_submit "${job_file}" "${dsub_out}" "${dsub_err}"
    if ( $status != 0 ) then
        echo "ERROR: Step 5 K=${k} dsub 提交失败"
        touch ${hpo_out}/fail
    endif
    @ STEP5_TASK_COUNT = ${STEP5_TASK_COUNT} + 1
end

@ rerun_count = 0
while ( 1 )
    set DONE_COUNT = 0
    set FAIL_COUNT = 0
    foreach k ( ${K_LIST} )
        set hpo_out = "${STEP5_DIR}/union_K${k}_${XGB_TRAIN_TAG}"
        if ( -f ${hpo_out}/complete ) @ DONE_COUNT = ${DONE_COUNT} + 1
        if ( -f ${hpo_out}/fail ) @ FAIL_COUNT = ${FAIL_COUNT} + 1
    end
    if ( ${FAIL_COUNT} > 0 ) then
        if ( $rerun_count >= $MAX_RERUN ) then
            echo "ERROR: ${FAIL_COUNT} 个 Step 5 HPO 任务在 ${MAX_RERUN} 次自动重提后仍失败"
            touch ${STEP5_DIR}/fail
            exit 1
        endif
        @ rerun_count = $rerun_count + 1
        echo "WARNING: automatically rerun failed Step 5 HPO tasks only (${rerun_count}/${MAX_RERUN})"
        foreach k ( ${K_LIST} )
            set hpo_out = "${STEP5_DIR}/union_K${k}_${XGB_TRAIN_TAG}"
            if ( -f ${hpo_out}/fail ) then
                set job_file = "${JOB_DIR}/step5_hpo_K${k}.csh"
                set dsub_out = "${LOG_DIR}/step5_hpo_K${k}.dsub.out"
                set dsub_err = "${LOG_DIR}/step5_hpo_K${k}.dsub.err"
                rm -f ${hpo_out}/fail ${hpo_out}/status "${job_file}.running"
                dsub_submit "${job_file}" "${dsub_out}" "${dsub_err}"
                if ( $status != 0 ) touch ${hpo_out}/fail
            endif
        end
        continue
    endif
    if ( ${DONE_COUNT} == $#K_LIST ) break
    echo "  Step 5 HPO progress: ${DONE_COUNT}/$#K_LIST; sleep ${POLL_SEC}s"
    sleep ${POLL_SEC}
end
touch ${STEP5_DIR}/complete

#===============================================================================
# ▶ Step 6 控制流：[K 并行重训练] -> [每 case 一个 full-eval dsub] -> [reduce dsub]
# full-eval 由主流水线直接编排，所有输入与当前 Step 3/4/6 变量保持一致。
#===============================================================================
set DSUB_EXTRA_ARGS = ( $STEP6_DSUB_EXTRA_ARGS )
mkdir -p ${STEP6_DIR}
foreach k ( ${K_LIST} )
    set hpo_out = "${STEP5_DIR}/union_K${k}_${XGB_TRAIN_TAG}"
    set retrain_out = "${STEP6_DIR}/union_K${k}_${XGB_TRAIN_TAG}"
    mkdir -p ${retrain_out}
    if ( -f ${retrain_out}/complete && -f ${retrain_out}/eval/full_unsampled/evaluation_summary.json ) then
        echo "Step 6 Retrain K=${k}: SKIP"
        touch ${retrain_out}/train_complete
        continue
    endif
    rm -f ${retrain_out}/fail ${retrain_out}/status
    set job_file = "${JOB_DIR}/step6_retrain_K${k}.csh"
    set job_log = "${LOG_DIR}/step6_retrain_K${k}.log"
    set dsub_out = "${LOG_DIR}/step6_retrain_K${k}.dsub.out"
    set dsub_err = "${LOG_DIR}/step6_retrain_K${k}.dsub.err"
    cat > ${job_file} << EOF
#!/bin/csh -x
# 任务说明：执行主 pipeline 生成的单个计算阶段。
# 参数说明：输入、输出及训练参数均已由主脚本展开。
# dsub 状态协议：running -> complete 或 fail。
rm -f "${job_file}.running"
touch "${job_file}.running"
echo "[job] step6 K=${k} start `date` host=`hostname`" >& ${job_log}
if ( ! -f ${hpo_out}/best_trials_leaf_1000.csv ) then
    echo "ERROR: missing ${hpo_out}/best_trials_leaf_1000.csv" >>& ${job_log}
    rm -f "${job_file}.running"
    touch ${retrain_out}/fail
    exit 1
endif
if ( -f ${retrain_out}/train_complete && -f ${retrain_out}/eval/metrics.csv ) then
    rm -f "${job_file}.running"
    exit 0
endif
${PYTHON} ${TIMEWINDOW_SRC}/step6_retrain_eval.py --train-buffer ${STEP4_DIR}/union_K${k}/train_union_top${k}.buffer --test-buffer ${STEP4_DIR}/union_K${k}/test_union_top${k}.buffer --hpo-csv ${hpo_out}/best_trials_leaf_1000.csv --output-dir ${retrain_out} --rounds ${RETRAIN_ROUNDS} --feature-names ${STEP4_DIR}/union_K${k}/feature_names.txt --tw-manifest ${STEP3_TW_MANIFEST} --level-dir ${LEVEL_DIR} --tog-dir ${TOG_DIR} --label-dir ${LABEL_DIR} --raw-namefile ${NAMEFILE} --two-phase-method "${TWO_PHASE_METHOD}" --two-phase-cutoff ${TWO_PHASE_CUTOFF} --two-phase-ratio ${TWO_PHASE_RATIO} --loss ${XGB_LOSS} --relative-floor ${XGB_RELATIVE_FLOOR} --focus-power ${XGB_FOCUS_POWER} --feature-input-mode ${FEATURE_INPUT_MODE} --shuffle-seed ${SHUFFLE_SEED} --skip-full-eval >>& ${job_log}
set rc = \$status
echo \${rc} > ${retrain_out}/status
if ( \${rc} == 0 && -f ${retrain_out}/eval/metrics.csv ) then
    touch ${retrain_out}/train_complete
else
    touch ${retrain_out}/fail
endif
rm -f "${job_file}.running"
exit \${rc}
EOF
    echo "[`date '+%H:%M:%S'`] submit Step 6 K=${k}"
    chmod +x "${job_file}"
    dsub_submit "${job_file}" "${dsub_out}" "${dsub_err}"
    if ( $status != 0 ) then
        echo "ERROR: Step 6 K=${k} dsub 提交失败"
        touch ${retrain_out}/fail
    endif
end

@ rerun_count = 0
while ( 1 )
    set DONE_COUNT = 0
    set FAIL_COUNT = 0
    foreach k ( ${K_LIST} )
        set retrain_out = "${STEP6_DIR}/union_K${k}_${XGB_TRAIN_TAG}"
        if ( -f ${retrain_out}/train_complete ) @ DONE_COUNT = ${DONE_COUNT} + 1
        if ( -f ${retrain_out}/fail ) @ FAIL_COUNT = ${FAIL_COUNT} + 1
    end
    if ( ${FAIL_COUNT} > 0 ) then
        if ( $rerun_count >= $MAX_RERUN ) then
            echo "ERROR: ${FAIL_COUNT} 个 Step 6 任务在 ${MAX_RERUN} 次自动重提后仍失败"
            touch ${STEP6_DIR}/fail
            exit 1
        endif
        @ rerun_count = $rerun_count + 1
        echo "WARNING: automatically rerun failed Step 6 tasks only (${rerun_count}/${MAX_RERUN})"
        foreach k ( ${K_LIST} )
            set retrain_out = "${STEP6_DIR}/union_K${k}_${XGB_TRAIN_TAG}"
            if ( -f ${retrain_out}/fail ) then
                set job_file = "${JOB_DIR}/step6_retrain_K${k}.csh"
                set dsub_out = "${LOG_DIR}/step6_retrain_K${k}.dsub.out"
                set dsub_err = "${LOG_DIR}/step6_retrain_K${k}.dsub.err"
                rm -f ${retrain_out}/fail ${retrain_out}/status "${job_file}.running"
                dsub_submit "${job_file}" "${dsub_out}" "${dsub_err}"
                if ( $status != 0 ) touch ${retrain_out}/fail
            endif
        end
        continue
    endif
    if ( ${DONE_COUNT} == $#K_LIST ) break
    echo "  Step 6 model retrain progress: ${DONE_COUNT}/$#K_LIST; sleep ${POLL_SEC}s"
    sleep ${POLL_SEC}
end

alias full_eval_case_submit '${DSUB_CMD} -q ${DSUB_QUEUE} -oo \!:2 -eo \!:3 $FULL_EVAL_DSUB_EXTRA_ARGS:q -s \!:1'
alias full_eval_reduce_submit '${DSUB_CMD} -q ${DSUB_QUEUE} -oo \!:2 -eo \!:3 $FULL_EVAL_REDUCE_DSUB_EXTRA_ARGS:q -s \!:1'
foreach k ( ${K_LIST} )
    set retrain_out = "${STEP6_DIR}/union_K${k}_${XGB_TRAIN_TAG}"
    set full_out = "${retrain_out}/eval/full_unsampled"
    if ( -f ${full_out}/evaluation_summary.json ) then
        touch ${retrain_out}/complete
        continue
    endif

    set full_job_dir = "${JOB_DIR}/step6_full_eval_K${k}"
    set full_log_dir = "${LOG_DIR}/step6_full_eval_K${k}"
    set full_task_manifest = "${full_out}/case_tasks.json"
    set full_task_tsv = "${full_out}/case_tasks.tsv"
    set full_feature_names = "${STEP4_DIR}/union_K${k}/feature_names.txt"
    set full_result_dir = "${retrain_out}/result"
    mkdir -p "${full_out}" "${full_job_dir}" "${full_log_dir}"

    echo "[`date '+%Y-%m-%d %H:%M:%S'`] Step 6 full-eval K=${k} prepare start"
    ${PYTHON} ${TIMEWINDOW_SRC}/full_unsampled_eval.py prepare \
        --tw-manifest "${STEP3_TW_MANIFEST}" --output-dir "${full_out}"
    if ( $status != 0 || ! -f "${full_task_tsv}" ) then
        echo "ERROR: Step 6 full-eval K=${k} prepare failed"
        touch ${retrain_out}/fail
        exit 1
    endif
    set full_case_ids = ( `awk 'NR > 1 {print $1}' "${full_task_tsv}"` )
    if ( $#full_case_ids == 0 ) then
        echo "ERROR: no Step 6 full-eval cases in ${full_task_manifest}"
        touch ${retrain_out}/fail
        exit 1
    endif

    foreach case_id ( ${full_case_ids} )
        set full_case_dir = "${full_out}/cases/case_`printf '%05d' ${case_id}`"
        if ( -f "${full_case_dir}/complete" ) continue
        mkdir -p "${full_case_dir}"
        rm -f "${full_case_dir}/fail" "${full_case_dir}/status"
        set full_case_job = "${full_job_dir}/case_${case_id}.csh"
        set full_case_log = "${full_log_dir}/case_${case_id}.log"
        set full_case_dsub_out = "${full_log_dir}/case_${case_id}.dsub.out"
        set full_case_dsub_err = "${full_log_dir}/case_${case_id}.dsub.err"
        cat > "${full_case_job}" << EOF
#!/bin/csh -f
echo "[job] full-eval K=${k} case=${case_id} start \`date '+%Y-%m-%d %H:%M:%S'\` host=\`hostname\`" >& "${full_case_log}"
${PYTHON} ${TIMEWINDOW_SRC}/full_unsampled_eval.py case --task-manifest "${full_task_manifest}" --task-index ${case_id} --level-dir "${LEVEL_DIR}" --tog-dir "${TOG_DIR}" --label-dir "${LABEL_DIR}" --raw-namefile "${NAMEFILE}" --feature-names "${full_feature_names}" --result-dir "${full_result_dir}" --output-dir "${full_out}" --two-phase-method "${TWO_PHASE_METHOD}" --two-phase-cutoff ${TWO_PHASE_CUTOFF} --chunk-rows ${FULL_EVAL_CHUNK_ROWS} >>& "${full_case_log}"
set rc = \$status
echo \$rc > "${full_case_dir}/status"
if ( \$rc != 0 ) touch "${full_case_dir}/fail"
echo "[job] full-eval K=${k} case=${case_id} end rc=\$rc \`date '+%Y-%m-%d %H:%M:%S'\`" >>& "${full_case_log}"
exit \$rc
EOF
        chmod +x "${full_case_job}"
        echo "[`date '+%Y-%m-%d %H:%M:%S'`] submit Step 6 full-eval K=${k} case=${case_id}"
        full_eval_case_submit "${full_case_job}" "${full_case_dsub_out}" "${full_case_dsub_err}"
        if ( $status != 0 ) touch "${full_case_dir}/fail"
    end

    @ full_rerun_count = 0
    while ( 1 )
        set full_done_count = 0
        set full_fail_count = 0
        foreach case_id ( ${full_case_ids} )
            set full_case_dir = "${full_out}/cases/case_`printf '%05d' ${case_id}`"
            if ( -f "${full_case_dir}/complete" ) @ full_done_count = ${full_done_count} + 1
            if ( -f "${full_case_dir}/fail" ) @ full_fail_count = ${full_fail_count} + 1
        end
        if ( ${full_fail_count} > 0 ) then
            if ( ${full_rerun_count} >= ${MAX_RERUN} ) then
                echo "ERROR: ${full_fail_count} Step 6 full-eval K=${k} case tasks failed after ${MAX_RERUN} reruns"
                touch ${retrain_out}/fail
                exit 1
            endif
            @ full_rerun_count = ${full_rerun_count} + 1
            echo "[`date '+%Y-%m-%d %H:%M:%S'`] rerun failed Step 6 full-eval K=${k} cases ${full_rerun_count}/${MAX_RERUN}"
            foreach case_id ( ${full_case_ids} )
                set full_case_dir = "${full_out}/cases/case_`printf '%05d' ${case_id}`"
                if ( -f "${full_case_dir}/fail" ) then
                    rm -f "${full_case_dir}/fail" "${full_case_dir}/status"
                    full_eval_case_submit "${full_job_dir}/case_${case_id}.csh" "${full_log_dir}/case_${case_id}.dsub.out" "${full_log_dir}/case_${case_id}.dsub.err"
                    if ( $status != 0 ) touch "${full_case_dir}/fail"
                endif
            end
            continue
        endif
        if ( ${full_done_count} == $#full_case_ids ) break
        echo "[`date '+%Y-%m-%d %H:%M:%S'`] Step 6 full-eval K=${k} case progress ${full_done_count}/$#full_case_ids; sleep ${POLL_SEC}s"
        sleep ${POLL_SEC}
    end

    set full_reduce_job = "${full_job_dir}/reduce.csh"
    set full_reduce_log = "${full_log_dir}/reduce.log"
    set full_reduce_out = "${full_log_dir}/reduce.dsub.out"
    set full_reduce_err = "${full_log_dir}/reduce.dsub.err"
    cat > "${full_reduce_job}" << EOF
#!/bin/csh -f
echo "[job] full-eval K=${k} reduce start \`date '+%Y-%m-%d %H:%M:%S'\` host=\`hostname\`" >& "${full_reduce_log}"
${PYTHON} ${TIMEWINDOW_SRC}/full_unsampled_eval.py reduce --task-manifest "${full_task_manifest}" --output-dir "${full_out}" >>& "${full_reduce_log}"
set rc = \$status
echo \$rc > "${full_out}/reduce.status"
if ( \$rc == 0 && -f "${full_out}/evaluation_summary.json" ) then
    touch "${full_out}/complete"
else
    touch "${full_out}/reduce.fail"
endif
echo "[job] full-eval K=${k} reduce end rc=\$rc \`date '+%Y-%m-%d %H:%M:%S'\`" >>& "${full_reduce_log}"
exit \$rc
EOF
    chmod +x "${full_reduce_job}"

    @ full_rerun_count = 0
    while ( ! -f "${full_out}/complete" )
        rm -f "${full_out}/reduce.fail" "${full_out}/reduce.status"
        echo "[`date '+%Y-%m-%d %H:%M:%S'`] submit Step 6 full-eval K=${k} reduce"
        full_eval_reduce_submit "${full_reduce_job}" "${full_reduce_out}" "${full_reduce_err}"
        if ( $status != 0 ) touch "${full_out}/reduce.fail"
        while ( ! -f "${full_out}/complete" && ! -f "${full_out}/reduce.fail" )
            echo "[`date '+%Y-%m-%d %H:%M:%S'`] Step 6 full-eval K=${k} reduce running; sleep ${POLL_SEC}s"
            sleep ${POLL_SEC}
        end
        if ( -f "${full_out}/complete" ) break
        if ( ${full_rerun_count} >= ${MAX_RERUN} ) then
            echo "ERROR: Step 6 full-eval K=${k} reduce failed after ${MAX_RERUN} reruns"
            touch ${retrain_out}/fail
            exit 1
        endif
        @ full_rerun_count = ${full_rerun_count} + 1
    end

    touch ${retrain_out}/complete
end
touch ${STEP6_DIR}/complete

#===============================================================================
# ▶ MLP Step 5/6 控制流：[K 并行可续跑 HPO/重训练] -> [每 case full-eval] -> [reduce]
# HPO 和重训练共用 MLP 训练资源；full-eval 直接由主流水线按 case 编排。
#===============================================================================
if ( "$skip_mlp_train" == "0" ) then
    set DSUB_EXTRA_ARGS = ( $MLP_DSUB_EXTRA_ARGS )
    mkdir -p ${MLP_STEP5_DIR} ${MLP_STEP6_DIR}
    foreach k ( ${K_LIST} )
        set k_dir = "${STEP4_DIR}/union_K${k}"
        set mlp_hpo_out = "${MLP_STEP5_DIR}/union_K${k}_mlp"
        set mlp_retrain_out = "${MLP_STEP6_DIR}/union_K${k}_mlp"
        mkdir -p ${mlp_hpo_out} ${mlp_retrain_out}
        if ( -f ${mlp_retrain_out}/complete && -f ${mlp_retrain_out}/eval/full_unsampled/evaluation_summary.json ) then
            echo "MLP HPO/Retrain/Full-eval K=${k}: SKIP"
            touch ${mlp_retrain_out}/train_complete
            continue
        endif
        if ( -f ${mlp_retrain_out}/eval/metrics.csv && -f ${mlp_retrain_out}/result/mlp_model.joblib ) then
            echo "MLP HPO/Retrain K=${k}: model already trained; continue full-eval"
            touch ${mlp_retrain_out}/train_complete
            continue
        endif
        rm -f ${mlp_retrain_out}/fail ${mlp_retrain_out}/status
        set job_file = "${JOB_DIR}/mlp_train_K${k}.csh"
        set job_log = "${LOG_DIR}/mlp_train_K${k}.log"
        set dsub_out = "${LOG_DIR}/mlp_train_K${k}.dsub.out"
        set dsub_err = "${LOG_DIR}/mlp_train_K${k}.dsub.err"
        cat > ${job_file} << EOF
#!/bin/csh -x
rm -f "${job_file}.running"
touch "${job_file}.running"
echo "[job] MLP HPO/retrain K=${k} start \`date '+%Y-%m-%d %H:%M:%S'\` host=\`hostname\`" >& ${job_log}
if ( ! -f ${k_dir}/train_union_top${k}.buffer || ! -f ${k_dir}/test_union_top${k}.buffer || ! -f ${k_dir}/feature_names.txt ) then
    echo "ERROR: missing MLP input under ${k_dir}" >>& ${job_log}
    rm -f "${job_file}.running"
    touch ${mlp_retrain_out}/fail
    exit 1
endif
rm -f ${mlp_hpo_out}/complete
${PYTHON} ${TIMEWINDOW_SRC}/step5_hpo_mlp.py --train-buffer ${k_dir}/train_union_top${k}.buffer --test-buffer ${k_dir}/test_union_top${k}.buffer --feature-names ${k_dir}/feature_names.txt --output-dir ${mlp_hpo_out} --epochs ${MLP_HPO_EPOCHS} --trial-num ${MLP_HPO_TRIALS} --pop-size ${MLP_HPO_POP} --job-num ${MLP_HPO_JOB_NUM} --nfold ${MLP_HPO_NFOLD} --sampler ${MLP_HPO_SAMPLER} --patience ${MLP_PATIENCE} --validation-ratio ${MLP_VALIDATION_RATIO} --parameter-threshold ${MLP_PARAMETER_THRESHOLD} --seed ${SHUFFLE_SEED} >>& ${job_log}
if ( \$status != 0 || ! -f ${mlp_hpo_out}/best_trial.json || ! -f ${mlp_hpo_out}/hpo_progress.json ) then
    echo "ERROR: MLP HPO failed" >>& ${job_log}
    rm -f "${job_file}.running"
    touch ${mlp_retrain_out}/fail
    exit 1
endif
touch ${mlp_hpo_out}/complete
if ( -f ${mlp_retrain_out}/train_complete && -f ${mlp_retrain_out}/eval/metrics.csv && -f ${mlp_retrain_out}/result/mlp_model.joblib ) then
    rm -f "${job_file}.running"
    exit 0
endif
${PYTHON} ${TIMEWINDOW_SRC}/step6_retrain_eval_mlp.py --train-buffer ${k_dir}/train_union_top${k}.buffer --test-buffer ${k_dir}/test_union_top${k}.buffer --hpo-result ${mlp_hpo_out}/best_trial.json --output-dir ${mlp_retrain_out} --feature-names ${k_dir}/feature_names.txt --tw-manifest ${STEP3_TW_MANIFEST} --level-dir ${LEVEL_DIR} --tog-dir ${TOG_DIR} --label-dir ${LABEL_DIR} --raw-namefile ${NAMEFILE} --feature-input-mode ${FEATURE_INPUT_MODE} --shuffle-seed ${SHUFFLE_SEED} --seed ${SHUFFLE_SEED} --skip-full-eval >>& ${job_log}
set rc = \$status
echo \${rc} > ${mlp_retrain_out}/status
if ( \${rc} == 0 && -f ${mlp_retrain_out}/eval/metrics.csv && -f ${mlp_retrain_out}/result/mlp_model.joblib ) then
    touch ${mlp_retrain_out}/train_complete
else
    touch ${mlp_retrain_out}/fail
endif
rm -f "${job_file}.running"
echo "[job] MLP HPO/retrain K=${k} end rc=\${rc} \`date '+%Y-%m-%d %H:%M:%S'\`" >>& ${job_log}
exit \${rc}
EOF
        echo "[`date '+%H:%M:%S'`] submit MLP HPO/Retrain K=${k}"
        chmod +x "${job_file}"
        dsub_submit "${job_file}" "${dsub_out}" "${dsub_err}"
        if ( $status != 0 ) then
            echo "ERROR: MLP K=${k} dsub 提交失败"
            touch ${mlp_retrain_out}/fail
        endif
    end

    @ rerun_count = 0
    while ( 1 )
        set DONE_COUNT = 0
        set FAIL_COUNT = 0
        foreach k ( ${K_LIST} )
            set mlp_retrain_out = "${MLP_STEP6_DIR}/union_K${k}_mlp"
            if ( -f ${mlp_retrain_out}/train_complete ) @ DONE_COUNT = ${DONE_COUNT} + 1
            if ( -f ${mlp_retrain_out}/fail ) @ FAIL_COUNT = ${FAIL_COUNT} + 1
        end
        if ( ${FAIL_COUNT} > 0 ) then
            if ( $rerun_count >= $MAX_RERUN ) then
                echo "ERROR: ${FAIL_COUNT} 个 MLP 任务在 ${MAX_RERUN} 次自动重提后仍失败"
                touch ${MLP_STEP6_DIR}/fail
                exit 1
            endif
            @ rerun_count = $rerun_count + 1
            echo "WARNING: automatically rerun failed MLP tasks only (${rerun_count}/${MAX_RERUN})"
            foreach k ( ${K_LIST} )
                set mlp_retrain_out = "${MLP_STEP6_DIR}/union_K${k}_mlp"
                if ( -f ${mlp_retrain_out}/fail ) then
                    set job_file = "${JOB_DIR}/mlp_train_K${k}.csh"
                    set dsub_out = "${LOG_DIR}/mlp_train_K${k}.dsub.out"
                    set dsub_err = "${LOG_DIR}/mlp_train_K${k}.dsub.err"
                    rm -f ${mlp_retrain_out}/fail ${mlp_retrain_out}/status "${job_file}.running"
                    dsub_submit "${job_file}" "${dsub_out}" "${dsub_err}"
                    if ( $status != 0 ) touch ${mlp_retrain_out}/fail
                endif
            end
            continue
        endif
        if ( ${DONE_COUNT} == $#K_LIST ) break
        echo "  MLP HPO/retrain progress: ${DONE_COUNT}/$#K_LIST; sleep ${POLL_SEC}s"
        sleep ${POLL_SEC}
    end

    alias mlp_full_eval_case_submit '${DSUB_CMD} -q ${DSUB_QUEUE} -oo \!:2 -eo \!:3 $MLP_FULL_EVAL_DSUB_EXTRA_ARGS:q -s \!:1'
    alias mlp_full_eval_reduce_submit '${DSUB_CMD} -q ${DSUB_QUEUE} -oo \!:2 -eo \!:3 $MLP_FULL_EVAL_REDUCE_DSUB_EXTRA_ARGS:q -s \!:1'
    foreach k ( ${K_LIST} )
        set mlp_retrain_out = "${MLP_STEP6_DIR}/union_K${k}_mlp"
        set mlp_full_out = "${mlp_retrain_out}/eval/full_unsampled"
        if ( -f ${mlp_full_out}/evaluation_summary.json ) then
            touch ${mlp_retrain_out}/complete
            continue
        endif

        set mlp_full_job_dir = "${JOB_DIR}/mlp_full_eval_K${k}"
        set mlp_full_log_dir = "${LOG_DIR}/mlp_full_eval_K${k}"
        set mlp_full_task_manifest = "${mlp_full_out}/case_tasks.json"
        set mlp_full_task_tsv = "${mlp_full_out}/case_tasks.tsv"
        set mlp_full_feature_names = "${STEP4_DIR}/union_K${k}/feature_names.txt"
        set mlp_full_result_dir = "${mlp_retrain_out}/result"
        mkdir -p "${mlp_full_out}" "${mlp_full_job_dir}" "${mlp_full_log_dir}"

        echo "[`date '+%Y-%m-%d %H:%M:%S'`] MLP full-eval K=${k} prepare start"
        ${PYTHON} ${TIMEWINDOW_SRC}/full_unsampled_eval.py prepare \
            --tw-manifest "${STEP3_TW_MANIFEST}" --output-dir "${mlp_full_out}"
        if ( $status != 0 || ! -f "${mlp_full_task_tsv}" ) then
            echo "ERROR: MLP full-eval K=${k} prepare failed"
            touch ${mlp_retrain_out}/fail
            exit 1
        endif
        set mlp_full_case_ids = ( `awk 'NR > 1 {print $1}' "${mlp_full_task_tsv}"` )
        if ( $#mlp_full_case_ids == 0 ) then
            echo "ERROR: no MLP full-eval cases in ${mlp_full_task_manifest}"
            touch ${mlp_retrain_out}/fail
            exit 1
        endif

        foreach case_id ( ${mlp_full_case_ids} )
            set mlp_full_case_dir = "${mlp_full_out}/cases/case_`printf '%05d' ${case_id}`"
            if ( -f "${mlp_full_case_dir}/complete" ) continue
            mkdir -p "${mlp_full_case_dir}"
            rm -f "${mlp_full_case_dir}/fail" "${mlp_full_case_dir}/status"
            set mlp_full_case_job = "${mlp_full_job_dir}/case_${case_id}.csh"
            set mlp_full_case_log = "${mlp_full_log_dir}/case_${case_id}.log"
            set mlp_full_case_dsub_out = "${mlp_full_log_dir}/case_${case_id}.dsub.out"
            set mlp_full_case_dsub_err = "${mlp_full_log_dir}/case_${case_id}.dsub.err"
            cat > "${mlp_full_case_job}" << EOF
#!/bin/csh -f
echo "[job] MLP full-eval K=${k} case=${case_id} start \`date '+%Y-%m-%d %H:%M:%S'\` host=\`hostname\`" >& "${mlp_full_case_log}"
${PYTHON} ${TIMEWINDOW_SRC}/full_unsampled_eval.py case --task-manifest "${mlp_full_task_manifest}" --task-index ${case_id} --level-dir "${LEVEL_DIR}" --tog-dir "${TOG_DIR}" --label-dir "${LABEL_DIR}" --raw-namefile "${NAMEFILE}" --feature-names "${mlp_full_feature_names}" --result-dir "${mlp_full_result_dir}" --output-dir "${mlp_full_out}" --model-type mlp --chunk-rows ${MLP_FULL_EVAL_CHUNK_ROWS} >>& "${mlp_full_case_log}"
set rc = \$status
echo \$rc > "${mlp_full_case_dir}/status"
if ( \$rc != 0 ) touch "${mlp_full_case_dir}/fail"
echo "[job] MLP full-eval K=${k} case=${case_id} end rc=\$rc \`date '+%Y-%m-%d %H:%M:%S'\`" >>& "${mlp_full_case_log}"
exit \$rc
EOF
            chmod +x "${mlp_full_case_job}"
            echo "[`date '+%Y-%m-%d %H:%M:%S'`] submit MLP full-eval K=${k} case=${case_id}"
            mlp_full_eval_case_submit "${mlp_full_case_job}" "${mlp_full_case_dsub_out}" "${mlp_full_case_dsub_err}"
            if ( $status != 0 ) touch "${mlp_full_case_dir}/fail"
        end

        @ mlp_full_rerun_count = 0
        while ( 1 )
            set mlp_full_done_count = 0
            set mlp_full_fail_count = 0
            foreach case_id ( ${mlp_full_case_ids} )
                set mlp_full_case_dir = "${mlp_full_out}/cases/case_`printf '%05d' ${case_id}`"
                if ( -f "${mlp_full_case_dir}/complete" ) @ mlp_full_done_count = ${mlp_full_done_count} + 1
                if ( -f "${mlp_full_case_dir}/fail" ) @ mlp_full_fail_count = ${mlp_full_fail_count} + 1
            end
            if ( ${mlp_full_fail_count} > 0 ) then
                if ( ${mlp_full_rerun_count} >= ${MAX_RERUN} ) then
                    echo "ERROR: ${mlp_full_fail_count} MLP full-eval K=${k} case tasks failed after ${MAX_RERUN} reruns"
                    touch ${mlp_retrain_out}/fail
                    exit 1
                endif
                @ mlp_full_rerun_count = ${mlp_full_rerun_count} + 1
                echo "[`date '+%Y-%m-%d %H:%M:%S'`] rerun failed MLP full-eval K=${k} cases ${mlp_full_rerun_count}/${MAX_RERUN}"
                foreach case_id ( ${mlp_full_case_ids} )
                    set mlp_full_case_dir = "${mlp_full_out}/cases/case_`printf '%05d' ${case_id}`"
                    if ( -f "${mlp_full_case_dir}/fail" ) then
                        rm -f "${mlp_full_case_dir}/fail" "${mlp_full_case_dir}/status"
                        mlp_full_eval_case_submit "${mlp_full_job_dir}/case_${case_id}.csh" "${mlp_full_log_dir}/case_${case_id}.dsub.out" "${mlp_full_log_dir}/case_${case_id}.dsub.err"
                        if ( $status != 0 ) touch "${mlp_full_case_dir}/fail"
                    endif
                end
                continue
            endif
            if ( ${mlp_full_done_count} == $#mlp_full_case_ids ) break
            echo "[`date '+%Y-%m-%d %H:%M:%S'`] MLP full-eval K=${k} case progress ${mlp_full_done_count}/$#mlp_full_case_ids; sleep ${POLL_SEC}s"
            sleep ${POLL_SEC}
        end

        set mlp_full_reduce_job = "${mlp_full_job_dir}/reduce.csh"
        set mlp_full_reduce_log = "${mlp_full_log_dir}/reduce.log"
        set mlp_full_reduce_out = "${mlp_full_log_dir}/reduce.dsub.out"
        set mlp_full_reduce_err = "${mlp_full_log_dir}/reduce.dsub.err"
        cat > "${mlp_full_reduce_job}" << EOF
#!/bin/csh -f
echo "[job] MLP full-eval K=${k} reduce start \`date '+%Y-%m-%d %H:%M:%S'\` host=\`hostname\`" >& "${mlp_full_reduce_log}"
${PYTHON} ${TIMEWINDOW_SRC}/full_unsampled_eval.py reduce --task-manifest "${mlp_full_task_manifest}" --output-dir "${mlp_full_out}" >>& "${mlp_full_reduce_log}"
set rc = \$status
echo \$rc > "${mlp_full_out}/reduce.status"
if ( \$rc == 0 && -f "${mlp_full_out}/evaluation_summary.json" ) then
    touch "${mlp_full_out}/complete"
else
    touch "${mlp_full_out}/reduce.fail"
endif
echo "[job] MLP full-eval K=${k} reduce end rc=\$rc \`date '+%Y-%m-%d %H:%M:%S'\`" >>& "${mlp_full_reduce_log}"
exit \$rc
EOF
        chmod +x "${mlp_full_reduce_job}"

        @ mlp_full_rerun_count = 0
        while ( ! -f "${mlp_full_out}/complete" )
            rm -f "${mlp_full_out}/reduce.fail" "${mlp_full_out}/reduce.status"
            echo "[`date '+%Y-%m-%d %H:%M:%S'`] submit MLP full-eval K=${k} reduce"
            mlp_full_eval_reduce_submit "${mlp_full_reduce_job}" "${mlp_full_reduce_out}" "${mlp_full_reduce_err}"
            if ( $status != 0 ) touch "${mlp_full_out}/reduce.fail"
            while ( ! -f "${mlp_full_out}/complete" && ! -f "${mlp_full_out}/reduce.fail" )
                echo "[`date '+%Y-%m-%d %H:%M:%S'`] MLP full-eval K=${k} reduce running; sleep ${POLL_SEC}s"
                sleep ${POLL_SEC}
            end
            if ( -f "${mlp_full_out}/complete" ) break
            if ( ${mlp_full_rerun_count} >= ${MAX_RERUN} ) then
                echo "ERROR: MLP full-eval K=${k} reduce failed after ${MAX_RERUN} reruns"
                touch ${mlp_retrain_out}/fail
                exit 1
            endif
            @ mlp_full_rerun_count = ${mlp_full_rerun_count} + 1
        end
        touch ${mlp_retrain_out}/complete
    end
    touch ${MLP_STEP5_DIR}/complete ${MLP_STEP6_DIR}/complete
else
    echo "MLP branch: SKIP (skip_mlp_train=${skip_mlp_train})"
endif

touch ${TAG_OUTPUT_ROOT}/complete

echo "[`date '+%H:%M:%S'`] ALL DONE. Results under ${TAG_OUTPUT_ROOT}"
