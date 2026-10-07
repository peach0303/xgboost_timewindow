# xgboost_timewindow_opt1

这是针对功耗长尾样本优化的独立版本，原目录不受影响。

## Opt1 核心变化

1. Step1 在 `(power, delta_power)` 二维空间使用训练集分位边界分桶。联合桶样本数不超过
   `SAMPLE_ALL_THRESHOLD` 时全部保留，其余桶在剩余预算内做均衡抽样。测试集只复用训练集
   边界，避免数据泄漏。
2. 输出 `sampling_distribution_and_weight_recommendation.json`，包含 power、delta power
   的采样前后边缘分布、二维联合计数、采样率、稀疏桶列表和联合桶建议权重。
3. 权重采用采样后联合桶频率的逆平方根（指数可配），截断后归一化为训练行均值 1。
   Step1 将每行权重写入 `*.weight.npy`；MCP/Ridge、XGB 代理筛选、HPO 与最终重训均使用它。
4. 组织完训练矩阵后，Step2 代理筛选、Step3 buffer、Step4 筛选、Step5 HPO 和 Step6
   重训均使用固定种子 shuffle，保证可复现。
5. 主流水线的 Step2 固定并发执行 MCP 与 XGBoost 两套代理筛选。Step3 分别组织两套
   代理矩阵；Step4 按现有混合流程采用 `K_MCP:K_XGB=1:1` 的配额独立排序，再对同名
   代理去重求并集。Step5/Step6 使用该并集直接训练单阶段 XGBoost，不再使用
   Ridge + 残差 XGBoost。

## 关键参数

- `SAMPLE_POWER_BINS=12`、`SAMPLE_DELTA_BINS=12`
- `SAMPLE_ALL_THRESHOLD=32`
- `SAMPLE_WEIGHT_EXPONENT=0.5`、`SAMPLE_WEIGHT_MIN=0.25`、`SAMPLE_WEIGHT_MAX=8`
- `SHUFFLE_SEED=108`
- `MCP_PROXY_TOP_K=200`、`XGB_PROXY_TOP_K=200`
- `TOP_KS=256,512,1024,2048,4096`（每种选择器各取 K 个，去重后并集为 K 到 2K 个）
- `XGB_LOSS=relative_error_focus`
- `XGB_RELATIVE_FLOOR=0.001`
- `XGB_FOCUS_POWER=3.0`
- `HPO_CLOSENESS_METRIC=tail_mean_ape`
- `skip_mlp_train=1`（默认跳过 MLP；设为 `0` 时开启 MLP HPO + retrain）
- `MLP_HPO_TRIALS=40`、`MLP_HPO_EPOCHS=100`、`MLP_HPO_NFOLD=3`
- `MLP_HPO_SAMPLER=NSGAII`、`MLP_PARAMETER_THRESHOLD=5000000`

主流水线与分步脚本均接受这些设置。示例：

```csh
setenv SAMPLE_ROWS_TOTAL 200000
setenv SAMPLE_ALL_THRESHOLD 32
setenv XGB_LOSS relative_error_focus
setenv XGB_FOCUS_POWER 3.0
csh xgboost_timewindow_opt1/cmd/run_timewindow_pipeline.csh boom all
```

主流水线输出的 XGBoost 最终模型为
`step6_retrain/union_K<k>_direct_xgb/result/xgbmodel.json`。设置
`skip_mlp_train=0` 后，还会生成：

```text
step5_mlp_hpo/union_K<k>_mlp/best_trial.json
step6_mlp_retrain/union_K<k>_mlp/result/mlp_model.joblib
step6_mlp_retrain/union_K<k>_mlp/eval/metrics.csv
```

其中 MLP 和 XGBoost 使用相同的 Step4 union buffer，两个模型的输出互不覆盖。MCP/XGB 仅负责从
不同角度筛选代理，最终功耗预测只使用一个直接拟合原始 power 标签的 XGBoost 模型。

独立的历史混合 Ridge + 残差 XGB 流水线仍保留，默认使用 `error_focus`。残差模型不建议使用
`relative_error_focus`，因为它的标签是 Ridge 残差而不是原始 power。

## XGBoost 损失

- `squared_error`：传统平方误差，对照基线。
- `pseudo_huber`：平滑稳健损失。
- `absolute_error`：绝对误差。
- `relative_squared`：逐样本相对平方误差。
- `error_focus`：平滑高阶误差，`focus_power>2` 时重点压低大误差样本。
- `relative_error_focus`：相对误差与大误差强化结合，适合直接预测原始 power。

`tail_mean_ape` 是默认 HPO 精度目标：计算相对误差最差 5% 样本的平均值（CVaR 风格），
比平均 MAPE 更难通过牺牲一小部分样本来换取总体均值；也可用更激进但更易受噪声影响的
`max_ape`。

## 原项目说明

独立整理后的 time-window XGBoost 训练脚本包。

- `src/timewindow/`: Python 训练数据处理和 XGBoost 训练脚本
- `src/reduction/`: 降维脚本
- `src/fsdbapi/`: FSDBAPI 接口脚本、C++ 源码、Makefile 和当前 build 产物
- `cmd/`: 可执行 shell wrapper
- `docs/`: 训练流程与参数说明

入口文档：`docs/timewindow_training_workflow.md`

新增代理筛选算法（Stability Selection、mRMR、互信息、Elastic Net、Extra-Trees）
以及 BNN/LSTM/LTC 训练后端的设计、参数和运行示例见
`docs/proxy_and_neural_model_exploration.md`。

默认情况下，`cmd/*.csh` 会把 `PROJECT_ROOT` 定位为本目录的上一级，也就是当前工作区根目录 `/home/data/chenyiheng/workspace/2026-dac`。如需迁移到其他位置，可显式设置 `PROJECT_ROOT=/path/to/workspace`。

## 神经网络 HPO

BNN、LSTM 和 LTC 可通过 `src/timewindow/step5_hpo_neural.py` 进行 Optuna
多目标 HPO（训练集 K 折 MAPE + 参数量），再由
`step6_retrain_eval.py --hpo-result best_trial.json` 使用最佳参数重训练。
已有 Step4 结果可通过 `cmd/run_timewindow_neural_hpo.csh <hw> <K>` 一键执行；
运行前应将 `OUTPUT_ROOT` 指向已有流水线输出目录。

设置 `SKIP_NEURAL_HPO=1` 可跳过搜索，使用 BNN/LSTM/LTC 的平衡型默认参数直接执行
Step6；直训结果写入 `step6_neural_retrain/union_K<k>_<model>_direct/`。具体参数和
覆盖变量见 `docs/proxy_and_neural_model_exploration.md`。

按行 shuffle 后重划分、`bnn_fp32` 对照、5–0 代理信号映射到 12–6/12–3，以及
LSTM/LTC 加速开关见 `docs/experimental_training_variants.md`。
