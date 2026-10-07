# Time-Window Power Modeling 训练流程说明

> **Opt1 主流水线说明（优先于下文的历史分步示例）**
>
> `cmd/run_timewindow_pipeline.csh` 现在固定并发运行 MCP 与 XGBoost 两套 Step2，
> 不再读取 `PROXY_SELECTOR` 做二选一。Step3 使用
> `step3_hybrid_concat_features.py`，Step4 使用
> `step4_hybrid_screen_features.py`，按 `K_MCP:K_XGB=1:1` 分别取 Top-K 后对同名
> 代理去重求并集。Step5/Step6 将并集直接交给标准单阶段 XGBoost HPO 和重训。
> 单路 Step2 入口仍可用于独立调试，但不会改变主流水线的双路行为。

Step2 完成后，流水线默认运行旧版 XGBoost 功耗预测验证。原脚本已复制到以下目录，适配修改只发生在副本：

```text
xgboost_timewindow/src/timewindow/step2_validation/xgb_power_v2_3.py
xgboost_timewindow/src/timewindow/step2_validation/xgb_power_eval_from_prediction_csv.py
```

副本直接读取 Step1 `timewindow_manifest.json` 与 Step2 `mcp_per_offset_manifest.json`：按 manifest 中的 combo 顺序读取各 combo 的 `selected_indices_file`，对每个 case 执行 `X_combo[:, selected_indices]`，再横向拼成训练输入。Step1 的 train stems 只用于 train/val，全部 test stems 固定进入 test；评价沿用旧脚本的点误差、跳变误差、阈值统计和直方图体系。模型与训练参数保持旧脚本默认值。

独立运行（MCP 与 XGBoost proxy 的 Step2 manifest 均适用）：

```bash
OUTPUT_ROOT=/path/to/a/timewindow/run
PYTHON=/home/data/chenyiheng/conda/envs/pro/bin/python
mkdir -p "${OUTPUT_ROOT}/step2_validation_legacy_xgb"
cd "${OUTPUT_ROOT}/step2_validation_legacy_xgb"
"${PYTHON}" /home/data/chenyiheng/workspace/2026-dac/xgboost_timewindow/src/timewindow/step2_validation/xgb_power_v2_3.py \
  --tw-manifest "${OUTPUT_ROOT}/step1_shifted/timewindow_manifest.json" \
  --proxy-manifest "${OUTPUT_ROOT}/step2_mcp/mcp_per_offset_manifest.json" \
  --combos "tog_offset12,level_offset12,tog_offset11" \
  --namefile /path/to/after_one.namefile.txt \
  --cache-dir "${OUTPUT_ROOT}/step2_validation_legacy_xgb/dataset_cache" \
  --build-block-size 200 \
  --hpotag legacy_xgb_validation
```


`--combos` 按逗号分隔指定验证使用的 combo，名称必须与 Step1/Step2 manifest 中的键一致，且给定顺序就是最终特征横向拼接顺序；不传或传空字符串时使用 Step1 的全部 combo。

`--build-block-size` 控制组合代理数据集构建时每个缓存块的行数，默认值为 `200`。减小该值（例如 `100` 或 `50`）会限制各 combo 代理特征横向拼接时的内存，但会产生更多缓存文件并增加磁盘 I/O；设为 `0` 或负数时，每个 case 只生成一个块。

注意：Step1 特征使用压缩 CSR `.npz`，`scipy.load_npz` 仍需一次加载一个完整 combo，因此该参数不能降低这部分内存下限。若峰值由单个 Step1 combo 主导，需要进一步让 Step1 按行分片输出或改用支持分片读取的存储格式。

也可运行包装入口：

```bash
OFFSETS="12,11" \
DATA_TYPES="tog,level" \
STEP2_VALIDATION_CHUNK_SIZE=100 \
OUTPUT_ROOT=/path/to/a/timewindow/run \
csh xgboost_timewindow/cmd/run_timewindow_step2_validation.csh boom
```

C shell 会按 `OFFSETS` 外层、`DATA_TYPES` 内层自动生成 `tog_offset12,level_offset12,tog_offset11,level_offset11`。若显式设置 `STEP2_VALIDATION_COMBOS`，则使用该值覆盖自动生成结果。

验证输出位于 `${OUTPUT_ROOT}/step2_validation_legacy_xgb/`。完整流水线默认执行该阶段；设置 `RUN_STEP2_VALIDATION=0` 可跳过。


本文档整理当前 `src_timewindow` 训练链路所需脚本、输入数据、运行方式、参数配置，以及从 FSDB 到最终模型的数据流。默认工作区为：

```bash
/home/data/chenyiheng/workspace/2026-dac
```

## 1. 环境

推荐使用已经验证通过的 conda 环境：

```bash
PYTHON=/home/data/chenyiheng/conda/envs/pro/bin/python
```

已验证依赖：

```text
xgboost 2.0.1
optuna 3.5.0
```

快速检查：

```bash
/home/data/chenyiheng/conda/envs/pro/bin/python - <<'PY'
import numpy, scipy, sklearn, pandas, xgboost, optuna
print('deps ok', xgboost.__version__, optuna.__version__)
PY
```

## 2. 训练所需脚本

核心训练目录：

```text
xgboost_timewindow/src/timewindow/
```

FSDBAPI 接口目录：

```text
xgboost_timewindow/src/fsdbapi/
```

降维脚本目录：

```text
xgboost_timewindow/src/reduction/
```

命令入口目录：

```text
xgboost_timewindow/cmd/
```

脚本职责：

```text
run_timewindow_pipeline.csh      完整训练 wrapper，串联 step1 -> step6
run_timewindow_pipeline_xgb_proxy.csh
                                历史兼容入口；主流水线已固定采用 MCP+XGB
step1_make_global_sample_index.py
                                分别在 train/test 的 power label 池中做全局采样，输出每个 case 的 row index
step1_prepare_timewindow.py     读取降维后的 TOG/level cache，对齐 power label，生成 time-window 数据
step2_mcp_per_offset.py         对每个 (data_type, offset) 做 MCP/variance proxy feature selection
step2_xgb_per_offset.py         对每个 (data_type, offset) 用 XGBoost importance 直接筛 top-K proxy features
merge_timewindow_manifests.py   合并 Step1/Step2 拆分 dsub 子任务产出的 per-combo manifest
step3_hybrid_concat_features.py 分别拼接 MCP/XGB 的 selected features
step4_hybrid_screen_features.py Ridge 权重/XGB gain 双排序，各取 K 后去重求并集
step5_hpo_timewindow.py         Optuna HPO
step6_retrain_eval.py           使用 best HPO params 直接训练单一 XGBoost 并评估
utils/two_phase_trainer.py      two-phase 训练辅助逻辑
```

辅助验证脚本：

```text
xgboost_timewindow/cmd/validate_timewindow_training_pro.csh
```

它固定使用 `pro` 环境，基于已有降维 cache 跑一个小配置 step1 -> step6，用于快速验证训练链路是否可用。

端到端 FSDB smoke 脚本：

```text
xgboost_timewindow/cmd/run_5fsdb_reduction_train_smoke.csh
xgboost_timewindow/cmd/run_5fsdb_reduction_train_smoke_xgb_proxy.csh
```

第一条使用 MCP/variance Step2，第二条使用 XGBoost top-K Step2。二者都串联：FSDBAPI 导出 -> merge -> 降维 -> time-window 接口 smoke。

## 3. 输入数据

训练入口需要三类输入：

```text
LEVEL_DIR   降维后的 level cache 目录
TOG_DIR     降维后的 tog cache 目录
LABEL_DIR   power label .npy 目录
NAMEFILE    特征名文件，通常是 ${LEVEL_DIR}/after_one.namefile.txt
```

本次验证使用的默认路径：

```bash
SOURCE_RUN=/home/data/chenyiheng/workspace/2026-dac/output/5fsdb_reduction_train_smoke/boom_coremark_5fsdb_submit_20260810_4
REDUCED_DIR=${SOURCE_RUN}/02_reduction/boom_5fsdb
LEVEL_DIR=${REDUCED_DIR}/level
TOG_DIR=${REDUCED_DIR}/tog
LABEL_DIR=/home/data/chenyiheng/workspace/2026-dac/output/timewindow_test/power_labels
NAMEFILE=${LEVEL_DIR}/after_one.namefile.txt
```

power label 示例：

```text
coremark_10.npy shape=(1000, 1)
coremark_11.npy shape=(1000, 1)
...
coremark_14.npy shape=(1000, 1)
```

注意：FSDBAPI 不生成 power label。FSDBAPI 只从 FSDB 生成信号 `level/tog` cache；power label 是外部已有功耗序列，训练时通过文件名匹配，例如：

```text
m000_coremark_10 -> coremark_10.npy
m001_coremark_11 -> coremark_11.npy
```

## 4. 从已有降维 cache 快速验证训练

最推荐先跑这个，确认训练环境和数据接口可用。

不采样，小配置验证：

```bash
cd /home/data/chenyiheng/workspace/2026-dac
RUN_TAG=pro_timewindow_train_verify \
PYTHON=/home/data/chenyiheng/conda/envs/pro/bin/python \
csh xgboost_timewindow/cmd/validate_timewindow_training_pro.csh
```

开启 power 分层采样，每个 case 保留 128 个 power 样本点：

```bash
cd /home/data/chenyiheng/workspace/2026-dac
RUN_TAG=pro_timewindow_train_sample128 \
SAMPLE_ROWS_PER_FILE=128 \
SAMPLE_BINS=16 \
SAMPLE_MIN_PER_BIN=2 \
PYTHON=/home/data/chenyiheng/conda/envs/pro/bin/python \
csh xgboost_timewindow/cmd/validate_timewindow_training_pro.csh
```

输出目录默认：

```text
output/timewindow_training_pro_verify/${RUN_TAG}
```

关键结果文件：

```text
STATUS.txt
logs/*.log
step1_shifted/timewindow_manifest.json
step2_mcp/mcp_per_offset_manifest.json
step2_xgb_proxy/mcp_per_offset_manifest.json
step3_hybrid_concat/hybrid_concat_meta.json
step4_hybrid_screen/union_K*/train_union_top*.buffer
step4_hybrid_screen/union_K*/test_union_top*.buffer
step5_hpo/union_K*_direct_xgb/best_trials_leaf_1000.csv
step6_retrain/union_K*_direct_xgb/result/xgbmodel.json
step6_retrain/*/eval/metrics.csv
```

## 5. 使用完整训练 wrapper

如果输入目录已经准备好，可以直接使用官方 wrapper：

```bash
cd /home/data/chenyiheng/workspace/2026-dac
PYTHON=/home/data/chenyiheng/conda/envs/pro/bin/python \
PROJECT_ROOT=/home/data/chenyiheng/workspace/2026-dac \
LEVEL_DIR=/path/to/reduced/level \
TOG_DIR=/path/to/reduced/tog \
LABEL_DIR=/path/to/power_labels \
NAMEFILE=/path/to/reduced/level/after_one.namefile.txt \
OUTPUT_ROOT=/path/to/output/timewindow_train \
OFFSETS=12,11,10,9,8,7,6 \
DATA_TYPES=tog,level \
STAGE=after_one \
SAMPLE_ROWS_PER_FILE=0 \
TOP_KS=256,512,1024 \
HPO_TRIALS=64 \
HPO_ROUNDS=50 \
RETRAIN_ROUNDS=50 \
csh xgboost_timewindow/cmd/run_timewindow_pipeline.csh
```

主流水线会自动同时提交 MCP 和 XGBoost 两个 Step2 dsub 作业。可分别调整：

```bash
cd /home/data/chenyiheng/workspace/2026-dac
PYTHON=/home/data/chenyiheng/conda/envs/pro/bin/python \
MCP_PROXY_TOP_K=200 \
XGB_PROXY_TOP_K=200 \
XGB_PROXY_ROUNDS=50 \
XGB_PROXY_MAX_DEPTH=4 \
XGB_PROXY_ETA=0.08 \
csh xgboost_timewindow_opt1/cmd/run_timewindow_pipeline.csh
```

这里的 `TOP_KS` 表示 Step4 从 MCP 排名和 XGB 排名中各取 K 个代理；同名代理只保留
一次，所以最终并集列数位于 K 到 2K 之间。

如果要控制内存和时间，建议先采样：

```bash
SAMPLE_ROWS_PER_FILE=512 SAMPLE_BINS=32 SAMPLE_MIN_PER_BIN=4 \
TOP_KS=128,256 HPO_TRIALS=16 HPO_ROUNDS=20 RETRAIN_ROUNDS=20 \
csh xgboost_timewindow/cmd/run_timewindow_pipeline.csh
```

## 6. 参数配置说明

### 6.1 路径参数

```text
PYTHON       Python 解释器，建议 /home/data/chenyiheng/conda/envs/pro/bin/python
LEVEL_DIR    降维后的 level cache 目录
TOG_DIR      降维后的 tog cache 目录
LABEL_DIR    power label .npy 目录
NAMEFILE     特征名文件
OUTPUT_ROOT  训练输出目录
STAGE        降维阶段名，当前常用 after_one
```

### 6.2 time-window 参数

```text
OFFSETS            使用哪些历史 offset，例如 12,11,10,9,8,7,6
DATA_TYPES         使用 tog、level 或二者，例如 tog,level
BOUNDARY_DISCARD   丢弃前多少个 label，默认 max(OFFSETS)
TEST_COUNT         最后多少个 case 作为测试集，默认 1
```

对齐规则：

```text
LEVEL[t+1] <-> TOG[t] <-> label[t]
```

对于 `label[i]` 和 `offset=k`：

```text
TOG row   = i - k
LEVEL row = i + 1 - k
label     = i
```

例如 `offset=6`，`boundary_discard=6`：

```text
label 使用 y[6:]
TOG 使用 rows 0..993
LEVEL 使用 rows 1..994
```

因此原始 label 为 `(1000,1)` 时，每个 case 得到 `994` 个有效样本点。

### 6.3 power 分层采样参数

采样在 `step1_prepare_timewindow.py` 中执行。先根据 power label 选行，再裁剪 TOG/level 特征和 label。

```text
SAMPLE_ROWS_PER_FILE   每个 case 保留多少行；0 表示不采样
SAMPLE_BINS            power 分位数分箱数
SAMPLE_MIN_PER_BIN     每个非空 bin 至少随机抽多少行
SAMPLE_SEED            随机种子
```

策略：

```text
1. 保留首尾点
2. 保留 power 最小值和最大值点
3. 按 power 分位数分箱，每个 bin 抽样
4. 名额不足时从剩余点随机补齐
5. 最后按原始行号排序，保留时间顺序
```

示例：

```bash
SAMPLE_ROWS_PER_FILE=128 SAMPLE_BINS=16 SAMPLE_MIN_PER_BIN=2
```

本次验证结果：

```text
每个 case: 994 -> 128 rows
train: 4 * 128 = 512 rows
test:  1 * 128 = 128 rows
```

### 6.4 代理筛选流程总览

Step2 当前有两套代理筛选实现：

```text
MCP 代理筛选:
  step2_mcp_per_offset.py
  默认输出目录: step2_mcp
  默认入口: run_timewindow_pipeline.csh

XGBoost 代理筛选:
  step2_xgb_per_offset.py
  默认输出目录: step2_xgb_proxy
  推荐入口: run_timewindow_pipeline_xgb_proxy.csh
```

两套流程的下游契约保持一致：Step2 都会为每个 `(data_type, offset)` 写出 `selected_indices.txt`，并汇总成 `mcp_per_offset_manifest.json`。因此 Step3 仍然使用同一个参数名：

```bash
--mcp-manifest <step2_dir>/mcp_per_offset_manifest.json
```

后续 Step3 -> Step4 -> Step5 -> Step6 不需要区分代理来自 MCP 还是 XGBoost。

主 pipeline 中 Step1/Step2 默认使用 dsub 并行：

```text
STEP1_PARALLEL_DSUB=1   Step1 先全局采样，再按 case x combo 提交 dsub
STEP2_PARALLEL_DSUB=1   Step2 每个 combo 一个 dsub，完成后合并 mcp_per_offset_manifest.json
```

Step1 的采样不再是每个 case 单独采样，而是先由 `step1_make_global_sample_index.py` 分别读取 train/test 的 power label，在各自 split 内做全局分层抽样，再把选中的 split-local row index 映射回各个 case。这样 train/test 都覆盖自己的 power 分布，测试集不会被训练集的大样本量挤到单一区间：

```text
step1_shifted/sample_index/sample_index_manifest.json
step1_shifted/sample_index/indices/<stem>.sample_idx.npy
step1_shifted/sample_index/train_stems.txt
step1_shifted/sample_index/test_stems.txt
```

Step1 矩阵切片随后按 `stem x (offset, data_type)` 全并行提交，拆分后的中间结果在：

```text
step1_shifted/_parts/<stem>/<data_type>_offset<k>/
step2_mcp/_parts/<data_type>_offset<k>/
step2_xgb_proxy/_parts/<data_type>_offset<k>/
```

`SAMPLE_ROWS_TOTAL` 控制 train+test 总共采多少行；当显式设置它时，`SAMPLE_TEST_RATIO` 控制分给 test 的比例，默认 `-1` 表示按 `test_case_count / valid_case_count` 自动分配。若 `SAMPLE_ROWS_TOTAL=0` 且 `SAMPLE_ROWS_PER_FILE>0`，则按 split 分别使用 `SAMPLE_ROWS_PER_FILE * train_case_count` 和 `SAMPLE_ROWS_PER_FILE * test_case_count`，兼容旧配置的样本量级。`SAMPLE_MIN_PER_CASE` 默认 1，用于避免某个 case 在 split 内全局抽样后完全没有样本。

合并后的最终文件名保持不变，所以 Step3 以后无需改动。如果需要回退到旧的单 dsub 作业，可以设置：

```bash
STEP1_PARALLEL_DSUB=0 STEP2_PARALLEL_DSUB=0 csh xgboost_timewindow/cmd/run_timewindow_pipeline.csh boom prep
```


### 6.4.1 Step1 稠密 level 数据优化

Step1 输出格式保持不变，仍然是每个 combo 下的 `*.feature.npz`、`*.label.npy` 和四个 filelist。为降低较稠密 level 数据的内存和时间开销，新增两个运行参数：

```text
STEP1_COMPRESS_OUTPUT=0   默认关闭 Step1 中间 npz 压缩；load_npz 仍兼容读取
STEP1_SKIP_EXISTING=1     默认复用已存在的 feature.npz/label.npy，便于失败后续跑
```

关闭压缩只改变 npz 内部是否 zip 压缩，不改变 scipy sparse npz 的读取接口；下游 Step2/Step3 仍然用 `load_npz`，所以可以和已经跑出的压缩文件混用。代价是磁盘占用会变大，但对稠密 level 通常能明显减少 CPU 时间和压缩阶段的内存峰值。

如果输入是 row-shard 格式，Step1 现在会按目标行窗口逐 shard 读取和切片，再 `vstack` 需要的片段，避免先把所有 shard 合成一个完整大矩阵。

已经跑了一部分时，不需要删除已有 `_parts`：

```text
已有 complete 的 part 目录: pipeline 直接跳过
已有 feature.npz/label.npy 但没有 complete: Step1 子任务复用文件并重建 filelist/manifest
```

### 6.5 MCP 代理筛选参数与逻辑

```text
ALPHA_MCP / --alpha       MCPRegression alpha
GAMMA / --gamma           MCPRegression gamma
ALPHA_L2 / --alpha-l2     Ridge refinement alpha
BATCH_SIZE / --batch      每次加载多少个 train case
MCP_TOP_K / --top-k       仅在没有 skglm、走 variance fallback 时生效
--include-combos          只处理指定 combo；主 pipeline 的 split dsub 子任务使用
```

默认参数：

```text
ALPHA_MCP=50.0
GAMMA=3.0
ALPHA_L2=8.0
BATCH_SIZE=4
```

筛选逻辑：

```text
1. 对每个 (data_type, offset) 单独处理。
2. 读取 test set；train set 按 BATCH_SIZE 切成若干批。
3. 如果有 skglm，使用 MCPRegression(alpha, gamma, positive=True)。
4. MCP 选中特征 = coef_ 非 0 的列。
5. 如果没有 skglm，fallback 为 sparse variance top-K。
6. 每个 batch 用筛出的列训练 Ridge(alpha_l2)，在完整 test set 上算 MAPE。
7. 保留 best MAPE 对应 batch 的 selected_indices。
```

注意：`pro` 环境安装了 `skglm`，因此 step2 会走真实 `MCPRegression`，`--top-k` 不限制 MCP 选出的特征数。MCP 最终选出多少列由 `alpha/gamma` 和数据决定。

已有 `output/timewindow_test/step2_mcp` 的参考结果：

```text
alpha=50.0, gamma=3.0, alpha_l2=8.0
14 个组合，Step2 合计选出 8021 个代理列

tog_offset12     470
level_offset12   689
tog_offset11     450
level_offset11   730
tog_offset10     435
level_offset10   717
tog_offset9      438
level_offset9    706
tog_offset8      419
level_offset8    684
tog_offset7      442
level_offset7    715
tog_offset6      441
level_offset6    685
```

### 6.6 XGBoost 代理筛选参数与逻辑

```text
MCP_PROXY_TOP_K / --top-k      MCP 不可用回退到 variance 时的每 combo 代理上限
XGB_PROXY_TOP_K / --top-k      每个 (data_type, offset) 导出的代理列数
XGB_PROXY_ROUNDS / --rounds    Step2 小模型 boosting rounds
XGB_PROXY_MAX_DEPTH            Step2 小模型 max_depth
XGB_PROXY_ETA                  Step2 小模型 eta
XGB_PROXY_SUBSAMPLE            Step2 小模型 subsample
XGB_PROXY_COLSAMPLE            Step2 小模型 colsample_bytree
XGB_PROXY_NTHREAD              Step2 小模型线程数
BATCH_SIZE / --batch           每次加载多少个 train case
```

默认参数：

```text
XGB_PROXY_TOP_K=200
XGB_PROXY_ROUNDS=50
XGB_PROXY_MAX_DEPTH=4
XGB_PROXY_ETA=0.08
XGB_PROXY_SUBSAMPLE=0.9
XGB_PROXY_COLSAMPLE=0.8
XGB_PROXY_NTHREAD=1
BATCH_SIZE=4
```

筛选逻辑：

```text
1. 对每个 (data_type, offset) 单独处理。
2. 读取 test set；train set 按 BATCH_SIZE 切成若干批。
3. 每个 batch 训练一个 XGBoost reg:squarederror 小模型。
4. 从模型导出 gain importance，按 gain 降序取 top-K 列。
5. 如果树实际 split 使用的列少于 top-K，用 0 importance 的剩余列补齐到 min(topK, 总列数)。
6. 用该 batch 的 XGBoost 模型在完整 test set 上算 MAPE。
7. 保留 best MAPE 对应 batch 的 selected_indices。
```

XGBoost Step2 的输出 manifest 文件名仍然是 `mcp_per_offset_manifest.json`，这是为了兼容 Step3 的现有接口。manifest 内部会标记：

```json
{
  "selector": "xgboost"
}
```

### 6.7 两套代理筛选的差异

```text
MCP:
  优点: 稀疏回归直接给非 0 系数，代理数量由正则强度自动决定。
  控制量: alpha/gamma；top-k 只影响没有 skglm 时的 variance fallback。
  输出数量: 每个组合不固定，可能几百列，也可能更多或更少。

XGBoost:
  优点: 明确每个组合导出 top-K，数量可控；重要性来自树模型 split gain。
  控制量: XGB_PROXY_TOP_K、rounds、max_depth、eta、采样参数。
  输出数量: 每个组合约等于 top-K，最多不超过该组合总列数。
```

两者都只是 Step2 的第一层 proxy 降维。Step4 仍会在 concat 后的完整代理特征上再训练默认 XGBoost，并按 signal-level importance 做第二层 top-K screening。

### 6.8 Screening 参数

```text
TOP_KS / --top-ks             signal-level top-K 列表，例如 256,512,1024
DEFAULT_ROUNDS                默认 XGBoost 训练轮数，用于 feature importance
```

step4 会先用完整 concat 特征训练默认 XGBoost，再把 column importance 聚合到 signal level，筛选 top-K signals。

### 6.9 HPO 参数

```text
HPO_TRIALS       Optuna trial 数
HPO_ROUNDS       每个 trial 的 boosting rounds
HPO_POP          NSGAII/NSGAIII population size
HPO_NFOLD        CV fold 数
TWO_PHASE_METHOD ""、A 或 B
TWO_PHASE_CUTOFF offset 分组 cutoff
TWO_PHASE_RATIO  two-phase 第一阶段轮数比例
```

小规模验证建议：

```text
HPO_TRIALS=2 HPO_ROUNDS=2 HPO_NFOLD=2
```

正式训练可逐步提高：

```text
HPO_TRIALS=64 HPO_ROUNDS=50 HPO_NFOLD=3
```

### 6.10 可选 MLP 分支参数

```text
skip_mlp_train          1=跳过（默认），0=运行 MLP HPO 和 retrain
SKIP_MLP_TRAIN          大写兼容别名；小写变量优先
MLP_HPO_TRIALS          Optuna trial 数，默认 40
MLP_HPO_EPOCHS          每折最大训练 epoch，默认 100
MLP_HPO_POP             NSGAII/NSGAIII population size，默认 8
MLP_HPO_NFOLD           交叉验证折数，默认 3
MLP_HPO_JOB_NUM         单个 K 内并行 trial 数，默认 1
MLP_HPO_SAMPLER         NSGAII、NSGAIII、TPE 或 Random
MLP_PATIENCE            early stopping patience，默认 12
MLP_VALIDATION_RATIO    内部验证集比例，默认 0.15
MLP_PARAMETER_THRESHOLD Pareto 候选的最大参数量，默认 5000000
MLP_FULL_EVAL_CHUNK_ROWS 每个 MLP full-eval case 的推理分块，默认复用 FULL_EVAL_CHUNK_ROWS
MLP_FULL_EVAL_DSUB_EXTRA_ARGS        MLP full-eval case dsub 资源参数
MLP_FULL_EVAL_REDUCE_DSUB_EXTRA_ARGS MLP full-eval reduce dsub 资源参数
```

开启方式：

```bash
skip_mlp_train=0 \
MLP_HPO_TRIALS=40 MLP_HPO_EPOCHS=100 \
csh xgboost_timewindow/cmd/run_timewindow_pipeline.csh boom all
```

每个 K 的 MLP 作业先执行 HPO，再按最优配置重训；重训练使用 `--skip-full-eval`，
随后主流水线按 case 独立提交 MLP full-sample 推理并在全部 case 完成后 reduce。主流水线等待
所有 K 的 full-eval 完成后才写 `OUTPUT_ROOT/complete`。`prep` 模式只生成共享的
Step1--4 数据，不运行任何模型训练。

MLP HPO 使用输出目录内的 SQLite study 持久化每个 trial。目标 trial 数每完成约 10%
（间隔为 `ceil(MLP_HPO_TRIALS / 10)`）会原子更新试验表、Pareto 表、当前 best 和进度文件。
任务中断后以相同数据及 HPO 配置重提时，只补齐未完成 trial；如果提高
`MLP_HPO_TRIALS`，则从已有 study 继续追加。数据路径、shape、CV、epoch、sampler 等关键
配置与已有 study 不一致时会拒绝混用。

MLP 产物：

```text
step5_mlp_hpo/union_K*/optuna_study.db
step5_mlp_hpo/union_K*/hpo_progress.json
step5_mlp_hpo/union_K*/best_trial.{json,csv}
step5_mlp_hpo/union_K*/mlp_trials.csv
step5_mlp_hpo/union_K*/pareto_trials.csv
step6_mlp_retrain/union_K*/result/mlp_model.joblib
step6_mlp_retrain/union_K*/result/feature_importance.csv
step6_mlp_retrain/union_K*/eval/metrics.csv
step6_mlp_retrain/union_K*/pred/true_vs_pred_scatter.png
step6_mlp_retrain/union_K*/eval/full_unsampled/evaluation_summary.json
```

### 6.11 Retrain 参数

```text
RETRAIN_ROUNDS                    最终重训 rounds
FULL_EVAL_CHUNK_ROWS              每个 case 构建特征和推理时的行分块，默认 250000
FULL_EVAL_DSUB_EXTRA_ARGS         每个 full-eval case dsub 的资源参数
FULL_EVAL_REDUCE_DSUB_EXTRA_ARGS  full-eval reduce dsub 的资源参数
```

最终模型输出：

```text
step6_retrain/.../result/xgbmodel.json
step6_retrain/.../eval/metrics.csv
step6_retrain/.../pred/predictions.txt
step6_retrain/.../pred/true_vs_pred_scatter.png
```

`true_vs_pred_scatter.png` 横轴为真实 power、纵轴为预测 power，包含 `y=x` 参考线，
标题显示测试集 R²。点数超过 50000 时按原始顺序等距抽样绘图，指标仍使用全部样本计算。

### 6.12 训练后 power / delta power 评估

Step6 在提供 Step1 manifest、原始 reduction cache 和 power label 路径时，会同时生成两套结果：

```text
eval/sampled_delta_power/
  prediction_samples.csv
  jump_detail.csv
  jump_threshold_summary*.csv

eval/full_unsampled/
  prediction_samples.csv
  true_vs_pred_scatter.png
  power_metrics_{train,test,overall}.csv
  power_and_delta_detail/point_metrics*.csv
  power_and_delta_detail/jump_detail.csv
  power_and_delta_detail/jump_threshold_summary*.csv
```

主流水线的 XGBoost 和 MLP full-unsampled 评价都分为三阶段：Step6 先使用
`--skip-full-eval` 完成模型重训；随后主流水线直接按 `timewindow_manifest.json`
中的 train/test stem 为每个 case 提交一个独立 dsub；全部 case 完成后再提交
一个 reduce dsub。full eval 不调用第二层 csh，其 manifest、原始数据路径、特征名、
模型目录和训练参数均直接来自当前主流水线变量。每个 case 只加载一次实际使用的 LEVEL/TOG
原始行窗口，再依据最终特征名中的 signal index 和 offset 按 chunk 构建模型输入。

中间结果和日志位于：

```text
eval/full_unsampled/case_tasks.{json,tsv}
eval/full_unsampled/cases/case_*/
logs_dsub/step6_full_eval_K*/case_*.log
logs_dsub/step6_full_eval_K*/reduce.log
logs_dsub/mlp_full_eval_K*/case_*.log
logs_dsub/mlp_full_eval_K*/reduce.log
```

case 日志会分别打印 `load_metadata_and_labels`、`load_level_source`、
`load_tog_source`、`load_model`、每个 `predict_chunk`、`predict_and_write`、
`build_and_write_jump_detail` 和 `case_total` 的耗时；reduce 日志打印预测合并、功耗
指标、jump 明细合并、jump summary 和总耗时。`complete`/`fail` 文件支持仅重提失败
case 和断点续跑。
采样集的 delta power 使用 sample index 定位当前周期，并把同一 case 的前一周期真实 power 当作无误差预测值。全量评估直接从未采样 reduction cache 重建最终筛选列；delta power 只在同一 case 内对连续周期做差，每个 case 的首个有效样本不会与前一个 case 相连。


## 7. 数据 shape 示例

以 5 个 `coremark_10~14`、`offset=6` 为例。

power label：

```text
coremark_10.npy ... coremark_14.npy: (1000, 1)
```

FSDBAPI / merge 后统一列空间约：

```text
TOG cache:   每个 case 约 (1000, 689630)
level cache: 每个 case 约 (1001, 689630)
```

降维 after_one 后：

```text
TOG:   每个 case (1000, 163379)
level: 每个 case (1001, 163379)
```

不采样训练验证：

```text
每个 case 有效行数: 994
train rows: 3976
test rows: 994
step3 concat: X_train=(3976,676), X_test=(994,676)
```

采样 `SAMPLE_ROWS_PER_FILE=128` 后：

```text
每个 case: 128 rows
train rows: 512
test rows: 128
step3 concat: X_train=(512,149), X_test=(128,149)
```

## 8. 是否 shuffle

当前训练流程没有对最终训练样本整体 shuffle。

顺序是：

```text
train = case0 全部行 + case1 全部行 + case2 全部行 + ...
test  = 最后 TEST_COUNT 个 case
```

如果启用 power 采样，采样内部有随机抽样，但最终行号会排序，因此每个 case 内仍保持原始时间顺序。

随机性主要来自：

```text
power 分层采样的随机补点
Optuna sampler
XGBoost 的 subsample/colsample_bytree
Two-phase CV 中 KFold(shuffle=True)
```

## 9. 从 FSDB 到模型的数据流

```text
Raw FSDB
  -> FSDBAPI export
      level cache: 信号电平
      tog cache:   相邻采样点 XOR / 翻转
  -> merge cache manifests
      多个 case 对齐到统一列空间
  -> csv reduction
      去全 0、保留全 1、可选 duplicate/sim reduction
      输出 after_one/final 等阶段 cache
  -> step1_make_global_sample_index
      在 train/test 各自的 power label 池中全局采样
      输出每个 case 的 sample index
  -> step1_prepare_timewindow
      按 offset 对齐 TOG/level 和 power label
      根据 sample index 对每个 case 裁剪行
      run_timewindow_pipeline.csh 默认按 case x offset/type 拆成多个 dsub 子任务
      merge_timewindow_manifests.py 合并最终 timewindow_manifest.json
  -> step2_mcp_per_offset + step2_xgb_per_offset
      两个 dsub 作业并发运行，每个 offset/type 独立选 proxy features
      分别输出 step2_mcp 与 step2_xgb_proxy manifest
      MCP: 非 0 系数自动决定数量
      XGBoost: gain importance 直接导出 top-K
  -> step3_hybrid_concat_features
      分别组织 MCP 与 XGB 代理矩阵
  -> step4_hybrid_screen_features
      MCP 侧按 Ridge 权重、XGB 侧按 gain 排序
      每侧各取 K（1:1），同名代理去重后生成 union DMatrix buffer
  -> step5_hpo_timewindow
      在 union buffer 上为单阶段直接 XGBoost 搜索超参
  -> step6_retrain_eval
      直接拟合原始 power，评估并保存单一 xgbmodel.json
```

## 10. 推荐运行配置

### 快速功能验证

```bash
RUN_TAG=quick_verify \
SAMPLE_ROWS_PER_FILE=128 SAMPLE_BINS=16 SAMPLE_MIN_PER_BIN=2 \
SCREEN_TOP_KS=16 SCREEN_ROUNDS=2 \
HPO_TRIALS=2 HPO_ROUNDS=2 HPO_POP=2 HPO_NFOLD=2 \
RETRAIN_ROUNDS=2 \
csh xgboost_timewindow/cmd/validate_timewindow_training_pro.csh
```

### 中等规模验证

```bash
SAMPLE_ROWS_PER_FILE=512 SAMPLE_BINS=32 SAMPLE_MIN_PER_BIN=4 \
TOP_KS=128,256 DEFAULT_ROUNDS=20 \
HPO_TRIALS=16 HPO_ROUNDS=20 HPO_NFOLD=3 \
RETRAIN_ROUNDS=20 \
csh xgboost_timewindow/cmd/run_timewindow_pipeline.csh
```

XGBoost 代理筛选版本：

```bash
SAMPLE_ROWS_PER_FILE=512 SAMPLE_BINS=32 SAMPLE_MIN_PER_BIN=4 \
XGB_PROXY_TOP_K=200 XGB_PROXY_ROUNDS=50 \
TOP_KS=128,256 DEFAULT_ROUNDS=20 \
HPO_TRIALS=16 HPO_ROUNDS=20 HPO_NFOLD=3 \
RETRAIN_ROUNDS=20 \
csh xgboost_timewindow/cmd/run_timewindow_pipeline_xgb_proxy.csh
```

### 正式训练起点

```bash
SAMPLE_ROWS_PER_FILE=0 \
OFFSETS=12,11,10,9,8,7,6 \
TOP_KS=256,512,1024,2048 \
DEFAULT_ROUNDS=50 \
HPO_TRIALS=64 HPO_ROUNDS=50 HPO_NFOLD=3 \
RETRAIN_ROUNDS=50 \
csh xgboost_timewindow/cmd/run_timewindow_pipeline.csh
```

如果样本周期数很长或内存紧张，优先启用 `SAMPLE_ROWS_PER_FILE`。
