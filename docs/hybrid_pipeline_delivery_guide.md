# Hybrid Time-window 流水线使用说明

## 1. 主入口

```csh
csh cmd/run_timewindow_hybrid_pipeline.csh \
  <tw_manifest> <mcp_manifest> <xgb_manifest> <namefile> <output_root> [final_top_k]
```

流水线先按需过滤 offset/combo，再依次通过 dsub 执行 Hybrid Step 3、Step 4、
Step 5。每个任务目录使用 `running`、`complete`、`fail`、`status` 标记状态；
失败时只重提当前阶段，默认最多重试 10 次。

Step 3–6 的输出会统一增加组合 tag 层级。例如设置 `STEP3_OFFSETS="12,11,10"` 和
`STEP3_DATA_TYPES="level,tog"` 后，输出写入 `OUTPUT_ROOT/tog_level_12_11_10/`。
未显式设置筛选参数时，tag 根据 manifest 的全部 combination 自动生成；也可通过
`setenv EXTRA_TAG experiment_a` 覆盖，覆盖值只能是安全的单级目录名。

## 2. 样本加权

先扫描标签分布，只生成报告、不修改输入数据：

```bash
python src/timewindow/analyze_sample_distribution.py \
  --tw-manifest /path/to/timewindow_manifest.json \
  --output-dir /path/to/distribution_audit
```

主要输出：

- `sample_distribution_and_weight_recommendation.json`
- `joint_bin_weight_recommendation.csv`
- `label_file_inventory.csv`

默认建议在训练集 `(power, delta_power)` 二维分位桶中使用逆频率平方根权重，
将权重截断在 `[0.25, 8.0]`，然后归一化到训练样本平均权重为 1。

确认建议后，生成 Step 3 可直接读取的权重：

```bash
python src/timewindow/analyze_sample_distribution.py \
  --tw-manifest /path/to/timewindow_manifest.json \
  --output-dir /path/to/distribution_audit \
  --write-weights
```

已有权重时默认拒绝覆盖；确认替换时添加 `--overwrite`。

若标签为：

```text
/path/case001.label.npy
```

权重必须写为：

```text
/path/case001.weight.npy
```

权重是与标签长度、行序完全一致的一维 NumPy 数组，要求值有限且大于 0。
生成权重后必须从 Step 3 重新运行，旧 DMatrix buffer 不会自动更新。

当前权重覆盖情况：

- Step 3：读取相邻 `*.weight.npy`，嵌入 DMatrix buffer；缺失时使用全 1。
- Step 4：Ridge/XGBoost 筛选使用权重，并传递到 union buffer。
- Hybrid Step 5：Ridge 和残差 XGBoost 训练使用权重。
- MLP Step 5/6：当前基于 sklearn 1.3 的 `MLPRegressor`，尚未使用样本权重。
- 常规 MAPE/RMSE/R2 以及 Hybrid HPO 的 Optuna MAPE 仍为未加权统计口径。

## 3. 开启 MLP HPO 和训练

MLP 分支默认关闭。运行主脚本前设置：

```csh
setenv skip_mlp_train 0
```

也可使用大写别名：

```csh
setenv SKIP_MLP_TRAIN 0
```

示例：

```csh
setenv skip_mlp_train 0
setenv MLP_HPO_TRIALS 40
setenv MLP_HPO_EPOCHS 100
setenv MLP_HPO_NFOLD 3
setenv MLP_HPO_SAMPLER NSGAII

csh cmd/run_timewindow_hybrid_pipeline.csh \
  TW.json MCP.json XGB.json NAMEFILE OUTPUT_ROOT 1024
```

MLP HPO 使用 Step 4 的 union train/test buffer 和 `feature_names.txt`。主要输出为：

```text
OUTPUT_ROOT/<tag>/step5_mlp_hpo/union_K1024/best_trial.json
OUTPUT_ROOT/<tag>/step6_mlp_retrain/union_K1024/result/mlp_model.joblib
OUTPUT_ROOT/<tag>/step6_mlp_retrain/union_K1024/eval/metrics.csv
```

MLP 默认只评估 buffer 中的 train/test 数据。如需执行完整未采样数据评估：

```csh
setenv MLP_SKIP_FULL_EVAL 0
setenv MLP_LEVEL_DIR /path/to/level
setenv MLP_TOG_DIR /path/to/tog
setenv MLP_LABEL_DIR /path/to/label
setenv MLP_RAW_NAMEFILE /path/to/raw_namefile
```

## 4. 可选 manifest 过滤

按 offset 和数据类型过滤：

```csh
setenv STEP3_OFFSETS "12,9,6"
setenv STEP3_DATA_TYPES "tog,level"
```

或直接指定组合：

```csh
setenv STEP3_COMBOS "tog_offset12,level_offset9"
```
