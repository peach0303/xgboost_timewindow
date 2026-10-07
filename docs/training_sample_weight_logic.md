# 训练阶段样本加权逻辑说明

本文说明 `src/timewindow` 中的训练样本权重如何生成、传递并最终作用于损失函数。结论基于当前代码（2026-09-16）逐行核对。

## 1. 结论摘要

默认流水线中的样本加权链路是：

```text
Step 1 依据训练集的 (power, delta_power) 二维分桶频次计算权重
  -> 为每一行保存 *.weight.npy
  -> Step 3/4 将权重写进 XGBoost DMatrix buffer
  -> Step 4 特征筛选训练使用权重
  -> Step 5 XGBoost HPO 使用权重
  -> Step 6 XGBoost 最终重训练使用权重
```

需要明确区分“生成/搬运权重”和“训练时真正加权”：

| 阶段 | 是否使用样本权重 | 作用 |
| --- | --- | --- |
| Step 1 | 是，生成权重 | 根据训练样本分布计算每个二维桶、每一行的权重 |
| Step 2 MCP/XGBoost proxy | 否 | 当前 Step 2 代理特征选择模型没有传入样本权重 |
| Step 3 | 是，传递权重 | 读取 `*.weight.npy`，与特征、标签同步打乱，并写入 `DMatrix` |
| Step 4 | 是，训练并传递 | Ridge 使用 `sample_weight`；XGBoost 从 `DMatrix` 读取权重；Top-K buffer 继续保存权重 |
| Step 5 XGBoost HPO | 是，训练加权 | 每个 CV fold 的训练数据保留权重，内置或自定义目标函数使用权重 |
| Step 6 XGBoost | 是，训练加权 | 单阶段和两阶段最终训练均保留权重 |
| Step 5/6 MLP、BNN、LSTM、LTC | **否** | 当前实现丢弃/未传入 buffer 中的样本权重；`weight_decay` 不是样本权重 |
| 项目自定义评估指标 | **否** | MAPE、RMSE、tail mean APE 等按普通样本均值计算 |

因此，“哪一步进行了加权”有两个答案：

1. **权重在 Step 1 被计算。**
2. **权重在模型求解时生效：主流水线为 Step 4、Step 5 和 Step 6 的 Ridge/XGBoost 训练；Step 3 只负责封装和传递。**

## 2. Step 1：权重如何计算

### 2.1 二维分桶

权重不是仅按功耗 `power` 计算，而是按以下二维联合分布计算：

- 当前行功耗：`power`
- 相邻行功耗变化：`delta_power`

`delta_power` 在每个 case/file 内计算，不跨文件边界。相关代码：

- [`utils/sampling_policy.py:10`](../src/timewindow/utils/sampling_policy.py#L10)：计算 `delta_power`。
- [`utils/sampling_policy.py:33`](../src/timewindow/utils/sampling_policy.py#L33)：在训练集最小值到最大值之间生成等宽数值桶边界。
- [`utils/sampling_policy.py:46`](../src/timewindow/utils/sampling_policy.py#L46)：将 `(power, delta_power)` 映射为联合桶编号。
- [`step1_make_global_sample_index.py:127`](../src/timewindow/step1_make_global_sample_index.py#L127)：读取所有 case 的功耗并分别计算变化量。
- [`step1_make_global_sample_index.py:129`](../src/timewindow/step1_make_global_sample_index.py#L129)：拼接 train/test 数据。
- [`step1_make_global_sample_index.py:135`](../src/timewindow/step1_make_global_sample_index.py#L135)：**只用训练集**计算 `power` 和 `delta_power` 的等宽数值桶边界。

这里的“等宽”是按物理数值范围划分，而不是按分位数划分。例如功耗训练范围为 `0~100`、配置 5 个桶时，边界固定为 `0, 20, 40, 60, 80, 100`。因此每个桶表达稳定的功耗/跳变区间，桶内样本数允许不均衡；测试集继续复用训练集边界，低于或高于训练范围的测试值会落入首桶或末桶。

### 2.2 逆频率权重公式

核心计算位于 [`utils/sampling_policy.py:134`](../src/timewindow/utils/sampling_policy.py#L134)：

设联合桶 `b` 中的已选训练样本数为 `c_b`，所有非空桶计数的中位数为 `m`，指数为 `alpha`，则初始桶权重为：

```text
raw_weight_b = (m / c_b) ^ alpha
```

随后执行：

```text
clipped_weight_b = clip(raw_weight_b, weight_min, weight_max)
weight_b = clipped_weight_b / 所有训练行的加权平均值
```

对应位置：

- [`utils/sampling_policy.py:142`](../src/timewindow/utils/sampling_policy.py#L142)：统计每个联合桶的样本数。
- [`utils/sampling_policy.py:147`](../src/timewindow/utils/sampling_policy.py#L147)：以非空桶计数中位数作为参考频次。
- [`utils/sampling_policy.py:148`](../src/timewindow/utils/sampling_policy.py#L148)：计算逆频率幂次权重。
- [`utils/sampling_policy.py:149`](../src/timewindow/utils/sampling_policy.py#L149)：按上下限截断。
- [`utils/sampling_policy.py:150`](../src/timewindow/utils/sampling_policy.py#L150)：按样本数计算行权重均值。
- [`utils/sampling_policy.py:152`](../src/timewindow/utils/sampling_policy.py#L152)：归一化，使训练样本的平均权重约等于 1。

默认参数由主流水线给出：

```text
alpha      = 0.5
weight_min = 0.25
weight_max = 8.0
```

见 [`cmd/run_timewindow_pipeline.csh:95`](../cmd/run_timewindow_pipeline.csh#L95) 和 [`cmd/run_timewindow_pipeline.csh:384`](../cmd/run_timewindow_pipeline.csh#L384)。`alpha=0.5` 相当于使用逆频率的平方根，强度比完全逆频率 `alpha=1.0` 温和。

注意：上下限截断发生在均值归一化之前，所以归一化后的最终权重不保证仍严格落在 `[0.25, 8.0]` 内。

### 2.3 从桶权重映射到每一行

默认全局采样路径先根据**已选训练行**统计桶频次，再将同一张训练集权重表应用到训练集和测试集：

- [`step1_make_global_sample_index.py:162`](../src/timewindow/step1_make_global_sample_index.py#L162)：取得已选训练样本的联合桶。
- [`step1_make_global_sample_index.py:166`](../src/timewindow/step1_make_global_sample_index.py#L166)：计算桶权重表。
- [`step1_make_global_sample_index.py:169`](../src/timewindow/step1_make_global_sample_index.py#L169)：训练集中未出现的桶用权重 `1.0` 兜底。
- [`utils/sampling_policy.py:181`](../src/timewindow/utils/sampling_policy.py#L181)：按每一行所属联合桶查表，得到行权重。
- [`step1_make_global_sample_index.py:183`](../src/timewindow/step1_make_global_sample_index.py#L183)：对 train/test 的每个 stem 分别保存索引与权重。
- [`step1_make_global_sample_index.py:191`](../src/timewindow/step1_make_global_sample_index.py#L191)：权重文件名为 `*.sample_weight.npy`。
- [`step1_make_global_sample_index.py:193`](../src/timewindow/step1_make_global_sample_index.py#L193)：写入每一行对应的权重。

这里的关键防泄漏点是：**桶边界和权重表都由训练集确定，测试集只查表，不参与权重拟合。**

### 2.4 Step 1 输出训练可消费的权重文件

[`step1_prepare_timewindow.py:193`](../src/timewindow/step1_prepare_timewindow.py#L193) 从全局采样 manifest 载入索引和权重；[`step1_prepare_timewindow.py:368`](../src/timewindow/step1_prepare_timewindow.py#L368) 将权重保存为与标签同名的 `*.weight.npy`：

```text
case.label.npy
case.weight.npy
```

若权重不存在，则 [`step1_prepare_timewindow.py:369`](../src/timewindow/step1_prepare_timewindow.py#L369) 使用全 1 权重，此时等价于不做样本重加权。代码还在 [`step1_prepare_timewindow.py:371`](../src/timewindow/step1_prepare_timewindow.py#L371) 检查权重行数必须与标签行数一致，并在 [`step1_prepare_timewindow.py:385`](../src/timewindow/step1_prepare_timewindow.py#L385) 生成 `train_weight.f`、`test_weight.f` 文件列表。

## 3. Step 3：权重与训练数据绑定

主线和 hybrid 线的逻辑相同：

- [`step3_concat_features.py:175`](../src/timewindow/step3_concat_features.py#L175)：读取标签。
- [`step3_concat_features.py:176`](../src/timewindow/step3_concat_features.py#L176)：从标签路径推导相邻的 `*.weight.npy`。
- [`step3_concat_features.py:177`](../src/timewindow/step3_concat_features.py#L177)：读取权重；文件不存在时回退为全 1。
- [`step3_concat_features.py:208`](../src/timewindow/step3_concat_features.py#L208)：拼接各 case 的权重。
- [`step3_concat_features.py:210`](../src/timewindow/step3_concat_features.py#L210)：特征、标签和权重按同一个顺序同步打乱，避免错位。
- [`step3_concat_features.py:221`](../src/timewindow/step3_concat_features.py#L221)：通过 `weight=w_train` 将权重写入训练 `DMatrix`。
- [`step3_concat_features.py:222`](../src/timewindow/step3_concat_features.py#L222)：测试权重也写入测试 `DMatrix`。

Hybrid 对应位置为：

- [`step3_hybrid_concat_features.py:161`](../src/timewindow/step3_hybrid_concat_features.py#L161)：读取 `*.weight.npy`。
- [`step3_hybrid_concat_features.py:188`](../src/timewindow/step3_hybrid_concat_features.py#L188)：写入训练 `DMatrix`。
- [`step3_hybrid_concat_features.py:189`](../src/timewindow/step3_hybrid_concat_features.py#L189)：写入测试 `DMatrix`。

Step 3 本身不优化模型，它的职责是保证权重与行一一对应，并把权重持久化进后续使用的二进制 buffer。

## 4. 权重在训练中如何生效

### 4.1 XGBoost 内置损失

当使用 `squared_error`、`pseudo_huber` 或 `absolute_error` 时，[`utils/training_policy.py:78`](../src/timewindow/utils/training_policy.py#L78) 配置 XGBoost 内置 objective。训练调用接收带权重的 `DMatrix`，XGBoost 在每行梯度和 Hessian 的统计中使用该权重。

可将加权目标理解为：

```text
L = sum_i weight_i * loss(y_i, prediction_i)
```

权重较大的稀有桶样本，对分裂选择和叶子值的影响更大。

### 4.2 自定义损失中的显式加权

默认流水线使用 `relative_error_focus`。自定义目标函数在 [`utils/training_policy.py:41`](../src/timewindow/utils/training_policy.py#L41) 定义：

- [`utils/training_policy.py:48`](../src/timewindow/utils/training_policy.py#L48)：读取标签并计算预测误差。
- [`utils/training_policy.py:50`](../src/timewindow/utils/training_policy.py#L50)：相对误差类损失使用 `max(abs(label), relative_floor)` 缩放误差。
- [`utils/training_policy.py:52`](../src/timewindow/utils/training_policy.py#L52)：`error_focus` 类损失通过更高次幂强调大误差样本。
- [`utils/training_policy.py:59`](../src/timewindow/utils/training_policy.py#L59)：从 `DMatrix` 读取样本权重。
- [`utils/training_policy.py:61`](../src/timewindow/utils/training_policy.py#L61)：梯度乘以样本权重。
- [`utils/training_policy.py:62`](../src/timewindow/utils/training_policy.py#L62)：Hessian 乘以样本权重。

因此默认自定义损失同时包含两层强调：

```text
最终影响 = 二维分桶样本权重 × 相对误差缩放 × 大误差 focus
```

样本权重解决“某种功耗/变化模式出现得少”的问题；`relative_error_focus` 解决“相对误差大或个体误差大”的问题，两者不是同一个概念。

### 4.3 Step 4：特征筛选训练

Step 4 已经发生真实的带权训练，不只是保存 buffer：

- 普通筛选线在 [`step4_screen_features.py:399`](../src/timewindow/step4_screen_features.py#L399) 构造带权 `DMatrix`，在 [`step4_screen_features.py:404`](../src/timewindow/step4_screen_features.py#L404) 训练用于特征重要度的 XGBoost。
- Hybrid MCP 线在 [`step4_hybrid_screen_features.py:277`](../src/timewindow/step4_hybrid_screen_features.py#L277) 将 `sample_weight` 传给 Ridge。
- Hybrid XGBoost 线在 [`step4_hybrid_screen_features.py:299`](../src/timewindow/step4_hybrid_screen_features.py#L299) 构造带权 `DMatrix`，在 [`step4_hybrid_screen_features.py:301`](../src/timewindow/step4_hybrid_screen_features.py#L301) 训练模型。
- 筛选后输出的 union buffer 在 [`step4_hybrid_screen_features.py:149`](../src/timewindow/step4_hybrid_screen_features.py#L149) 继续保存训练权重，供 Step 5/6 使用。

这意味着权重不仅影响最终模型，还会影响 Ridge 系数/XGBoost gain 排名，进而影响被选入 Top-K 的特征。

### 4.4 Step 5：HPO 训练

主线 XGBoost HPO 直接读取 Step 4 buffer：

- [`step5_hpo_timewindow.py:86`](../src/timewindow/step5_hpo_timewindow.py#L86)：读取含权重的训练 `DMatrix`。
- [`step5_hpo_timewindow.py:88`](../src/timewindow/step5_hpo_timewindow.py#L88)：使用 `DMatrix.slice()` 打乱行；切片会保留对应行的权重。
- [`step5_hpo_timewindow.py:152`](../src/timewindow/step5_hpo_timewindow.py#L152)：单阶段交叉验证使用带权 `dtrain`。
- [`step5_hpo_timewindow.py:176`](../src/timewindow/step5_hpo_timewindow.py#L176)：两阶段手工 CV 通过 `DMatrix.slice()` 构造训练/验证 fold，同样保留权重。

Hybrid Ridge + residual XGBoost 路径更加显式：

- [`step5_hybrid_hpo_timewindow.py:109`](../src/timewindow/step5_hybrid_hpo_timewindow.py#L109)：fold 内 Ridge 使用训练权重。
- [`step5_hybrid_hpo_timewindow.py:113`](../src/timewindow/step5_hybrid_hpo_timewindow.py#L113)：残差 XGBoost 训练集带权。
- [`step5_hybrid_hpo_timewindow.py:116`](../src/timewindow/step5_hybrid_hpo_timewindow.py#L116)：验证集也带权，用于 XGBoost 的验证指标和 early stopping。
- [`step5_hybrid_hpo_timewindow.py:147`](../src/timewindow/step5_hybrid_hpo_timewindow.py#L147)：全训练集重拟合 Ridge 时继续使用权重。
- [`step5_hybrid_hpo_timewindow.py:150`](../src/timewindow/step5_hybrid_hpo_timewindow.py#L150)：全训练集残差 XGBoost 继续使用权重。

但 Optuna 用于选择 trial 的项目自定义分数没有加权：主线路径的 [`step5_hpo_timewindow.py:146`](../src/timewindow/step5_hpo_timewindow.py#L146) 只将标签、预测传入 `closeness_score`；Hybrid 路径的 [`step5_hybrid_hpo_timewindow.py:223`](../src/timewindow/step5_hybrid_hpo_timewindow.py#L223) 也直接计算普通 tail relative error 均值。因此当前行为是“模型拟合带权，HPO 排名指标不带权”。

### 4.5 Step 6：最终 XGBoost 重训练

- [`step6_retrain_eval.py:136`](../src/timewindow/step6_retrain_eval.py#L136)：重新读取含权重的 train/test buffer。
- [`step6_retrain_eval.py:138`](../src/timewindow/step6_retrain_eval.py#L138)：训练集切片打乱时保留权重。
- [`step6_retrain_eval.py:223`](../src/timewindow/step6_retrain_eval.py#L223)：单阶段最终训练使用带权 `dtrain`。
- [`step6_retrain_eval.py:238`](../src/timewindow/step6_retrain_eval.py#L238)：两阶段方法 A 接收原始带权 `DMatrix`。
- [`step6_retrain_eval.py:248`](../src/timewindow/step6_retrain_eval.py#L248)：两阶段方法 B 接收原始带权 `DMatrix`。

两阶段训练器在重建中间 `DMatrix` 时显式复制权重：

- 方法 A：[`utils/two_phase_trainer.py:83`](../src/timewindow/utils/two_phase_trainer.py#L83)。
- 方法 B 第一阶段：[`utils/two_phase_trainer.py:129`](../src/timewindow/utils/two_phase_trainer.py#L129)。
- 方法 B 残差阶段：[`utils/two_phase_trainer.py:140`](../src/timewindow/utils/two_phase_trainer.py#L140)。

所以无论单阶段、两阶段 A，还是两阶段 B，样本权重都会进入最终 XGBoost 模型训练。

## 5. 当前没有使用样本权重的路径

### 5.1 Step 2 代理特征选择

Step 2 的 MCP、Ridge 和 XGBoost proxy 训练调用没有读取 `*.weight.npy` 或向模型传递 `sample_weight`/`DMatrix.weight`。因此样本权重不会影响 Step 2 的初始代理特征选择；它从 Step 3/4 开始影响下游筛选和训练。

### 5.2 sklearn MLP

[`mlp_core.py:19`](../src/timewindow/mlp_core.py#L19) 从 `DMatrix` 只取特征和标签，没有调用 `get_weight()`；[`mlp_core.py:96`](../src/timewindow/mlp_core.py#L96) 调用 `model.fit(x_scaled, y_scaled)` 时也没有传入样本权重。Step 5 MLP HPO 和 [`step6_retrain_eval_mlp.py:90`](../src/timewindow/step6_retrain_eval_mlp.py#L90) 的最终重训练都复用这条无样本权重路径。

### 5.3 BNN、LSTM、LTC

PyTorch 路径在 [`utils/neural_models.py:240`](../src/timewindow/utils/neural_models.py#L240) 使用默认 `SmoothL1Loss()`，其 batch loss 在 [`utils/neural_models.py:267`](../src/timewindow/utils/neural_models.py#L267) 直接计算，没有逐样本权重张量。调用方 [`step6_retrain_eval.py:211`](../src/timewindow/step6_retrain_eval.py#L211) 也只传入特征和标签。

这些模型配置中的 `weight_decay` 是对**模型参数**的正则化，见 [`utils/neural_models.py:237`](../src/timewindow/utils/neural_models.py#L237)，不是本文讨论的训练样本权重。

## 6. 测试集权重和评估指标

测试权重被写入 `DMatrix`，主要有两个作用：

1. XGBoost 自带的 `eval_metric` 在验证/测试 `DMatrix` 上评估时可以读取权重。
2. early stopping 使用 XGBoost 自带指标时会受测试/验证权重影响。

但是仓库中最终落盘的 MAPE、RMSE、R2、tail mean APE 等自定义指标，当前均直接对误差数组求普通均值，没有乘测试权重。例如：

- [`step4_hybrid_screen_features.py:47`](../src/timewindow/step4_hybrid_screen_features.py#L47)
- [`step5_hpo_timewindow.py:48`](../src/timewindow/step5_hpo_timewindow.py#L48)
- [`step6_retrain_eval.py:269`](../src/timewindow/step6_retrain_eval.py#L269)

因此报告中的业务指标表示原始样本分布上的未加权性能；这通常便于与真实数据分布对照，但它与训练时优化的加权目标并不完全相同。

## 7. 容易混淆的三个 `weight`

| 名称 | 含义 | 是否为样本加权 |
| --- | --- | --- |
| `DMatrix(..., weight=...)` / `sample_weight=...` | 每条训练样本对损失的贡献系数 | 是 |
| XGBoost `min_child_weight` | 子节点允许继续分裂所需的最小 Hessian 总量 | 否 |
| 神经网络 `weight_decay` | 模型参数的 L2/AdamW 正则化 | 否 |

另外，BNN 中的 binary weight 和特征重要度中的 path weight 都指模型参数，不是行级样本权重。

## 8. 兼容和补写工具

### `analyze_sample_distribution.py`

该工具可对已有 Step 1 标签重新分析分布，并在显式指定 `--write-weights` 时写入 Step 3 可消费的 `*.weight.npy`：

- [`analyze_sample_distribution.py:170`](../src/timewindow/analyze_sample_distribution.py#L170)：只用训练集拟合桶边界和桶权重。
- [`analyze_sample_distribution.py:181`](../src/timewindow/analyze_sample_distribution.py#L181)：测试集只映射训练权重表。
- [`analyze_sample_distribution.py:270`](../src/timewindow/analyze_sample_distribution.py#L270)：只有启用 `--write-weights` 才实际写文件。
- [`analyze_sample_distribution.py:279`](../src/timewindow/analyze_sample_distribution.py#L279)：默认拒绝覆盖已有权重，需额外指定 `--overwrite`。

### `generate_sample_weight_recommendation.py`

[`generate_sample_weight_recommendation.py:123`](../src/timewindow/generate_sample_weight_recommendation.py#L123) 只为旧版输出补写权重**建议报告 JSON**，不会更新 `*.weight.npy`，所以单独运行它不会改变训练权重。主流水线的兼容调用见 [`cmd/run_timewindow_pipeline.csh:629`](../cmd/run_timewindow_pipeline.csh#L629)。

## 9. 快速检查实际产物

可用下面的 Python 片段确认某个训练 buffer 是否携带权重：

```python
import numpy as np
import xgboost as xgb

matrix = xgb.DMatrix("train_union_top1024.buffer")
weight = matrix.get_weight()

print("rows:", matrix.num_row())
print("weight rows:", weight.size)
print("min/mean/max:", weight.min(), weight.mean(), weight.max())
print("all ones:", np.allclose(weight, 1.0))
```

判断方式：

- `weight.size == matrix.num_row()`：buffer 中存在逐行权重。
- `np.allclose(weight, 1.0)`：虽然存在权重字段，但训练效果等价于未加权。
- 权重非全 1：XGBoost/Ridge 路径会进行实际的样本重加权。

## 10. 已有自动化验证

[`tests/test_opt1_sampling_and_loss.py`](../tests/test_opt1_sampling_and_loss.py) 已覆盖以下关键约束：

- 稀有联合桶样本被保留，权重的训练行均值约为 1。
- 特征、标签、权重在打乱后仍保持行对齐。
- 自定义 `error_focus` 目标函数会读取样本权重并增强相应梯度/Hessian。
- 全局采样器输出的索引与权重长度一致，且权重均为正数。

[`tests/test_pipeline_feature_modes.py`](../tests/test_pipeline_feature_modes.py) 还验证了 Step 4 不同特征输入模式生成的 buffer 能保留指定权重。
