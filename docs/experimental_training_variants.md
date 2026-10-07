# Time-window 对比实验旁路

本文档覆盖三个不会覆盖既有结果的实验：按行重划分、BNN 全精度对照，以及跨窗口信号迁移。
所有输出都应使用独立目录或下述固定 tag。计算任务统一由 dsub 提交；控制脚本只在登录节点
生成作业、提交、轮询和失败重提。公共变量优先由 `cmd/timewindow_common.csh` 提供。

## 1. split_after_shuffle

原始 Step1 按 testcase 划分 train/test。dsub 控制器复用现有 Step1 矩阵，将原 train/test
样本合并，以固定种子全局打乱后再按行切分，并生成与 Step2 兼容的新 manifest。
随后 Step2 按 selector×combo 并行提交，Step5/Step6 按 K 并行提交：

```csh
setenv HW boom
setenv PYTHON /home/data/chenyiheng/conda/envs/hw-model-train/bin/python
setenv SHUFFLE_SEED 108
setenv SPLIT_STORAGE_MODE index
setenv INDEX_ROW_BATCH_SIZE_TOG 50000
setenv INDEX_ROW_BATCH_SIZE_LEVEL 5000
csh cmd/run_timewindow_split_after_shuffle_dsub.csh \
  /path/to/existing/step1_shifted/timewindow_manifest.json \
  /path/to/after_one.namefile.txt
```

默认从源 manifest 的位置推导原输出根目录。例如源 manifest 位于
`<original_root>/step1_shifted/timewindow_manifest.json`，新实验会完整写入：

```text
<original_root>/spilt_after_shuffle/
  step1_shifted/                   # index 模式仅保存 manifest 和行索引
  step2_mcp/                       # 重新进行 MCP 代理筛选
  step2_xgb_proxy/                 # 重新进行 XGBoost 代理筛选
  tog_level_15_14/                 # 正常保留 data type/offset 标签
    step3_hybrid_concat/
    step4_hybrid_screen/
    step5_hpo/
    step6_retrain/
```

如需显式指定原输出根目录和 K：

```csh
csh cmd/run_timewindow_split_after_shuffle_dsub.csh \
  /path/to/existing/step1_shifted/timewindow_manifest.json \
  /path/to/after_one.namefile.txt /path/to/original_root 256,512,1024
```

未设置 `SPLIT_TEST_RATIO` 时保留原始测试样本占总样本的比例；需要固定比例时：

```csh
setenv SPLIT_TEST_RATIO 0.2
```

`SPLIT_STORAGE_MODE` 支持两种模式：

- `index`（默认）：只保存 `train_indices.npy`、`test_indices.npy` 和新 manifest，
  Step2/Step3 读取原 Step1 数据并按索引切片，不复制完整特征矩阵；
- `materialize`：保存新的 feature、label、weight 文件，磁盘占用更高，但读取更直接。

索引模式会关闭尚不支持虚拟行索引的 legacy Step2 validation 旁路；MCP、XGBoost 代理筛选、
Step3 拼接及后续训练仍使用新的随机划分。

index 模式默认分别使用：

```csh
setenv INDEX_ROW_BATCH_SIZE_TOG 50000
setenv INDEX_ROW_BATCH_SIZE_LEVEL 5000
```

`INDEX_ROW_BATCH_SIZE=200000` 保留为其他 data type 及旧 manifest 的兼容回退。每个 combo
根据自身 `data_type` 选择批大小：

- MCP 对每个行块独立筛选代理，最后取代理并集；
- XGBoost 通过 `QuantileDMatrix + DataIter` 流式消费行块；
- Step3 按行块读取最终入选列，再合并为下游 buffer。

该值越小，原始特征读取峰值内存越低，但会增加分片重复读取和调度开销；它不影响
BNN/LSTM/LTC 的训练 mini-batch。

切换模式、随机种子或测试集比例时应使用新的 `spilt_after_shuffle` 输出目录，避免复用旧的
Step2 完成标记。

脚本不设置 `EXTRA_TAG`，因此不会覆盖 `tog_level_15_14` 等正常组合标签。输出固定写入
`<original_root>/spilt_after_shuffle/`。旧入口
`run_timewindow_split_after_shuffle.csh` 仅作兼容转发，也会进入 dsub 控制器。

注意：这种随机行划分会让同一个 testcase 的相邻周期同时进入训练和测试，适合判断
case domain shift 对指标的影响，但其测试指标可能因时间相关性泄漏而偏乐观，不能代替
按 testcase 留出的最终泛化评价。新 manifest 因此显式记录
`supports_case_level_full_evaluation=false`。

## 2. BNN 的 FP32 对照

新增模型类型 `bnn_fp32`。它与 BNN 都是固定双隐藏层，并保留相同的 LayerNorm、
Dropout 和回归头；区别仅为：

- `bnn`：两层隐藏权重和激活在前向时取 `-1/+1`；
- `bnn_fp32`：隐藏层使用 FP32 `Linear + Hardtanh`。

并排直接训练：

```csh
setenv OUTPUT_ROOT /path/to/existing/pipeline_run
setenv SKIP_NEURAL_HPO 1
setenv NEURAL_MODEL_TYPES bnn,bnn_fp32
setenv NEURAL_DEVICE cuda
csh cmd/run_timewindow_neural_hpo_dsub.csh boom 1024
```

神经网络作业默认使用 `NEURAL_PYTHON=/home/data/chenyiheng/conda/envs/hw-model-train/bin/python`；
需要更换环境时再显式覆盖。输出目录分别带 `_bnn_direct` 和 `_bnn_fp32_direct`，不会互相覆盖。
两种模型也都支持 HPO。

## 3. 将 5–0 代理信号映射到 12–6/12–3 特征

该实验只迁移信号身份，不会把 5–0 的特征值直接拼入模型。dsub 控制器依次提交特征构建、
HPO、重训练和结果比较作业；公共 HPO/训练参数来自 `timewindow_common.csh`：

```csh
setenv HW boom
setenv NAMEFILE /path/to/after_one.namefile.txt
setenv CROSS_SOURCE_TOP_SIGNALS 1024
setenv CROSS_SOURCE_OFFSET_MIN 0
setenv CROSS_SOURCE_OFFSET_MAX 5
csh cmd/run_timewindow_cross_window_dsub.csh \
  /path/to/target_12_6/timewindow_manifest.json \
  /path/to/12_6/feature_names.txt \
  /path/to/5_0/top1024_xgb_proxies.txt \
  /path/to/12_6/baseline_step6 \
  /path/to/experiment_root
```

目标 manifest 决定最终允许的 offset。比较 12–6 与 12–3 时，应分别传入对应的过滤后
manifest，并使用不同 `OUTPUT_ROOT`。已有近窗代理筛选和远窗基线特征均会复用。

控制器默认只接受来源 offset 0–5；即使代理排名文件混入 6–12 特征，也会在信号去重和
`CROSS_SOURCE_TOP_SIGNALS` 截断前过滤。可通过 `CROSS_SOURCE_OFFSET_MIN/MAX` 覆盖范围，
过滤数量记录在 `features/cross_window_union_meta.json`。

输出 `metrics_comparison.csv/json`，其中 MAPE/RMSE 等越低越好，R2 越高越好，并给出
`candidate_minus_baseline` 和 `improved`。

## 4. LSTM/LTC 加速开关

默认新增两项加速：

- 将逐时间点稀疏切片改为一次向量化列提取；
- 当预计内存不超过 512 MiB 时缓存序列张量，避免每个 epoch 重复稀疏转稠密；
- CUDA 上对 LSTM/LTC 默认启用自动混合精度。

可按机器内存和数值稳定性覆盖：

```csh
setenv NEURAL_SEQUENCE_CACHE_MB 1024
setenv NEURAL_MIXED_PRECISION auto   # auto / on / off
setenv NEURAL_MODEL_TYPES lstm,ltc
csh cmd/run_timewindow_neural_hpo_dsub.csh boom 1024
```

CPU 训练时混合精度自动关闭；将缓存上限设为 `0` 可禁用缓存。日志和
`model_meta.json` 会记录实际是否启用 AMP 及缓存上限。

## 5. R2 散点图

现有公共评价代码已经输出 `pred/true_vs_pred_scatter.png`：横轴为真实 power，纵轴为预测
power，包含 `y=x` 参考线以及标题中的 R²。该路径覆盖 XGBoost、BNN、BNN-FP32、LSTM、
LTC、MLP 以及 full-unsampled 汇总评价。
