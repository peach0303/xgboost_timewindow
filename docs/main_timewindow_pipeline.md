# Time-Window 主 Pipeline 说明

本文说明 `cmd/run_timewindow_pipeline.csh` 当前主流水线中各 Step 使用的特征、模型、训练损失和评价体系。默认主线为：

```text
原始 reduced LEVEL/TOG + 功耗标签
  -> Step 1 全局采样与时间窗数据
  -> Step 2 MCP 与 XGBoost 双路代理筛选
  -> Step 3 拼接代理特征
  -> Step 4 二次排序并生成各 K 的训练特征
  -> Step 5 XGBoost HPO
  -> Step 6 重训练与采样/全量评价
```

默认关闭 MLP 分支和 Two-phase Method A。所有实际计算阶段均由主脚本生成作业脚本后通过 dsub 提交。

## 1. 默认配置与主要分支

| 配置 | 默认值 | 含义 |
|---|---:|---|
| `OFFSETS` | `12,11,10,9,8,7,6` | 使用的时间窗口偏移 |
| `DATA_TYPES` | `tog,level` | 使用 TOG 和 LEVEL 两类 reduced 特征 |
| `STEP2_SELECTOR_MODE` | `hybrid` | Step 2 同时运行 MCP 与 XGBoost 选择器 |
| `STEP2_PARALLEL_DSUB` | `0` | 为 `1` 时，每个 selector/combo 单独提交 dsub |
| `FEATURE_INPUT_MODE` | `proxy_columns` | Step 5/6 直接使用选出的代理列 |
| `TWO_PHASE_METHOD` | 空字符串 | 关闭 Two-phase，直接训练一个 XGBoost |
| `XGB_LOSS` | `relative_error_focus` | Step 4 XGBoost、Step 5 和 Step 6 的默认训练损失 |
| `HPO_CLOSENESS_METRIC` | `tail_mean_ape` | Step 5 HPO 的精度目标 |
| `skip_mlp_train` | `1` | 默认不运行可选 MLP 分支 |

两个 Step 5/6 特征模式为：

- `proxy_columns`：直接使用 Step 4 选中的 `type_o<offset>_<signal>` 代理列。这是新版默认方式。
- `signal_all_windows`：先从代理列还原原始 signal 名，再取这些 signal 在当前 manifest 中所有可用 `type × offset` 的列。这是旧版特征方式。

## 2. Step 1：全局采样与时间窗数据准备

### 输入特征

- reduced LEVEL 稀疏矩阵；
- reduced TOG 稀疏矩阵；
- 每个 case 的逐周期功耗标签；
- 默认构造 `offset=12..6` 下的 LEVEL/TOG 数据组合。

Step 1 本身不训练预测模型。它负责对齐 feature、label、case 和时间窗口，并产生后续步骤读取的稀疏矩阵、标签、样本权重、filelist 与 `timewindow_manifest.json`。

### 采样规则

采样只依据标签分布，不依据输入 feature：

1. 在训练集功耗 `power` 和同一 case 内的相邻功耗变化 `delta_power` 各自的最小值到最大值之间生成等宽数值边界，默认各 12 个 bin。
2. 将样本放入 `power × delta_power` 二维联合桶。
3. 联合桶样本数不超过 `SAMPLE_ALL_THRESHOLD=32` 时全部保留，以保护稀有区间和跳变样本。
4. 其余稠密桶使用 water-fill 方式平衡分配剩余采样预算。
5. 默认保证每个 case 至少保留 1 行。采样上限是软上限；如果稀有桶样本已经超过预算，仍全部保留。

训练/测试按照 case 划分，默认排序后的最后 `TEST_COUNT=1` 个 case 作为测试集。

### 样本权重

按采样后训练集的二维联合桶频率生成逆频率权重：

```text
weight(bin) = (median_nonzero_count / bin_count) ^ exponent
```

默认 `exponent=0.5`，先裁剪到 `[0.25, 8.0]`，再归一化，使采样训练行的平均权重为 1。测试样本根据同一权重表映射。已有旧版采样结果缺少权重建议文件时，pipeline 会从已采样 label 回填建议文件，不重新采样。

### 模型、损失与评价

- 模型：无。
- 损失函数：无。
- 评价：输出采样前后 power、delta_power 和二维联合桶的计数、采样率、稀有桶以及建议权重；它属于数据分布审计，不是模型精度评价。

## 3. Step 2：逐 data type / offset 的代理特征选择

默认 hybrid 模式对每个 `tog|level × offset` 组合分别运行 MCP 和轻量 XGBoost。两路看到的都是该组合下的原始 signal 列，不会在 Step 2 混合不同 offset。

### 3.1 MCP 路

- 特征：单个组合的全部原始 signal 列。
- 选择模型：有 `skglm` 时使用允许正负系数的 `MCPRegression(alpha=50, gamma=3)`；按文件 batch 拟合，每个 batch 的非零系数列取并集。
- 降级策略：没有 `skglm` 时按特征方差选择 Top-K。
- 验证模型：对并集特征拟合 `Ridge(alpha=8)`。
- 损失：MCP 使用其非凸惩罚回归目标；验证 Ridge 使用 L2 正则的平方误差目标。
- 排名/选择：MCP 系数非零即入选；不是按 Ridge 分数在本 Step 再截断。
- 评价：Ridge 测试集的 MAPE、RMSE、NRMSE、R²，同时保存预测和模型。

注意：Step 2 MCP 的选择与 Ridge 验证当前不读取 Step 1 样本权重。

### 3.2 XGBoost 路

- 特征：单个组合的全部原始 signal 列。
- 模型：轻量 XGBoost `gbtree`，默认 50 rounds、`max_depth=4`、`eta=0.08`。
- 损失：`reg:squarederror`。
- 模型参数中的 `eval_metric`：RMSE；完成选择后另行计算测试集指标。
- 排名/选择：优先按 feature gain 取 Top-K；模型没有 gain 时退化为 split weight；不足 K 时按原列顺序补足。
- 评价：测试集 MAPE、MAE、MaxAPE、MSE、RMSE、NRMSE、R²。

Step 2 XGBoost 当前也不读取 Step 1 样本权重。

### Step 2 验证旁路

`RUN_STEP2_VALIDATION=1` 时还会运行 legacy XGBoost 验证。这是代理选择结果的附加检查，不参与主线 Step 3–6 的模型选优。

## 4. Step 3：代理特征拼接

### 使用的特征

- hybrid 默认：分别拼接所有组合下 MCP 代理列和 XGBoost 代理列；两套矩阵保持分离。
- single 模式：只拼接 `PROXY_SELECTOR=mcp|xgb` 指定的一路。
- 特征名统一为 `type_o<offset>_<signal>`，例如 `tog_o12_clock_enable`。

Step 3 会从 Step 1 生成的 `.weight.npy` 重新加载样本权重，并写入 XGBoost buffer；如果权重文件不存在则使用全 1。

### 模型、损失与评价

- 模型：无，仅执行稀疏矩阵拼接和确定性 shuffle。
- 损失函数：无。
- 评价：检查不同 selector 的 train/test label 完全一致，并记录各组合列范围、矩阵 shape、feature names 和权重。

## 5. Step 4：混合二次筛选与最终特征生成

Step 4 不直接产生最终功耗模型，而是分别给两路代理特征重新排序，再生成每个 K 的 Step 5/6 输入。

### MCP 代理的二次排序

- 模型：带样本权重的 Ridge，默认 `alpha=8`。
- 损失：加权平方误差 + L2 正则。
- 排名：按 `abs(Ridge coefficient)` 降序。
- 评价：训练集和测试集的 MAE、MAPE、RMSE、R²。

### XGBoost 代理的二次排序

- 模型：XGBoost `gbtree`，默认 200 rounds。
- 损失：默认 `relative_error_focus`，并使用 Step 1 样本权重。
- 排名：按 feature gain 降序。
- 评价：训练集和测试集的 MAE、MAPE、RMSE、R²。

默认 `relative_error_focus` 先按标签幅值归一化误差：

```text
normalized_error = (prediction - label) / max(abs(label), relative_floor)
loss trend ≈ abs(normalized_error) ^ focus_power
```

默认 `relative_floor=0.001`、`focus_power=3`。实现通过自定义 gradient/hessian 训练，并把样本权重同时乘到 gradient 和 hessian 上，因此会同时强化相对误差较大的样本和 Step 1 判定为稀有的样本。

### 最终 K 特征

对每个 `TOP_KS`：

1. MCP 路取前 K，XGBoost 路取前 K；
2. 按特征名去重，MCP 优先；hybrid 模式最终列数最多约为 `2K`，而不是全局恰好 K；
3. 根据 `FEATURE_INPUT_MODE` 输出代理列本身，或将代理 signal 展开到所有可用时间窗；
4. 保存带 label 和 sample weight 的 train/test XGBoost buffer。

## 6. Step 5：XGBoost 超参数搜索

### 特征与模型

- 特征：Step 4 为每个 K 生成的最终特征 buffer。
- 模型：XGBoost `gbtree`，`grow_policy=lossguide`。
- 搜索参数：L1/L2、max leaves、max depth、eta、gamma、subsample、column sampling、min child weight 等。
- 搜索器：默认 Optuna NSGA-II；默认 64 trials、3-fold CV。
- HPO 只在训练 buffer 上做交叉验证，测试 buffer 不参与参数选择。

### 损失函数

默认继续使用 Step 4 相同的 `relative_error_focus`，并继承 buffer 中的样本权重。也支持 `squared_error`、Pseudo-Huber、absolute error、relative squared 和 error focus。

### HPO 评价体系

HPO 是双目标最小化：

1. 精度目标：默认 `tail_mean_ape`，即验证集相对绝对误差最高 5% 样本的平均 APE；也可切换为 MAPE、RMSE、P95 APE 或 MaxAPE。
2. 复杂度目标：叶节点数量或其代码中的复杂度代理。

最终从 Pareto trials 中保留复杂度阈值以内的候选，输出 `best_trials_leaf_1000.csv`，供 Step 6 使用。

### 可选 Two-phase Method A

默认 `TWO_PHASE_METHOD=""`，直接使用全部特征训练。设为 `A` 后：

1. 按特征名中的 offset 分组；默认 `offset > 9` 为 far，`offset <= 9` 为 near。
2. Phase 1 将 near 列清零，只用 far 信息训练默认 80% rounds。
3. Phase 2 恢复全部列，在 Phase 1 模型上继续训练剩余 20% rounds。

HPO 的每个 fold 都按上述两阶段方式训练。Method A 不改变损失函数和评价指标，只改变特征进入模型的时间顺序。

## 7. Step 6：最终重训练与评价

### 特征、模型和损失

- 特征：与对应 K 的 Step 5 完全一致。
- 模型：从 HPO CSV 中选择精度最优候选，重训练一个 XGBoost `gbtree`；开启 Method A 时按相同两阶段规则训练。
- 损失：默认 `relative_error_focus`，与 Step 5 保持一致。
- 样本权重：训练时使用 Step 4 buffer 中保存的权重。

### 最终评价体系

最终评价不再按样本权重加权，所有样本等权报告：

- MAPE：平均绝对百分比误差，输出为百分数；
- Max Error：最大绝对相对误差；
- MSE、RMSE；
- NRMSE：`RMSE / mean(label)`；
- NMAE：`sum(abs_error) / sum(label)`；
- R²；
- 标签均值、标准差、最小值、最大值和相对误差方差。

评价分为三层：

1. 采样 train/test：分别输出训练集和测试集指标及预测。
2. sampled delta-power：每个采样点使用真实前一周期功耗作 anchor，评价预测功耗变化量。
3. full unsampled：从 reduced cache 重建筛选后的特征，对所有未采样有效行推理；分别输出 train、test、overall 功耗指标。

功耗跳变评价仅使用同一文件内连续周期。以该分组最大真实跳变的 10%～90% 为阈值，统计超过阈值的样本数、平均/中位/P90/P95/P99/最大相对跳变误差，以及平均/最大绝对跳变误差；结果按 overall、split 和 file 汇总。

## 8. 可选 MLP 分支

设置 `skip_mlp_train=0` 后，MLP 与 XGBoost 使用同一份 Step 4 最终特征，但训练体系独立：

- 模型：scikit-learn `MLPRegressor`，输入使用 `StandardScaler(with_mean=False)`，标签也做标准化；Adam 优化器，自适应学习率。
- 网络搜索：1～4 个等宽隐藏层，每层宽度从 32/64/128/256/512 中选择，激活函数为 ReLU 或 tanh。
- 训练损失：MLPRegressor 的平方误差回归损失，并带 `alpha` L2 正则。
- HPO：训练集 K-fold，双目标最小化验证 MAPE 和参数量；测试 buffer 只做形状审计，不参与 HPO。
- 最终训练：先用 early stopping 决定 epoch 数，再使用全部训练行按该 epoch 数重新拟合。
- 最终评价：与 XGBoost Step 6 相同的功耗指标、sampled delta-power 和 full-unsampled 评价。

当前 MLP 数据加载只读取 Step 4 buffer 的 feature 和 label，不使用其中的 sample weight。

## 9. 指标使用原则

阅读实验结果时应区分三类量：

| 类别 | 默认值 | 用途 |
|---|---|---|
| 训练损失 | `relative_error_focus` | 决定 XGBoost 每棵树如何更新 |
| HPO 精度目标 | `tail_mean_ape` | 选择更关注最差 5% 相对误差的参数 |
| 最终报告指标 | MAPE、RMSE、NRMSE、NMAE、R²、跳变分位指标 | 全面比较模型效果，不参与反向训练 |

因此，“训练损失更小”不必然意味着最终 MAPE 或 R²一定更好；正式比较应优先查看 Step 6 的 full-unsampled 指标和功耗跳变指标，同时结合采样测试集结果判断分布偏差。
