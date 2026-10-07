# MCP + XGBoost 混合代理框架

本框架完全由新增脚本组成，不修改原有 Step1～Step6。它接受同一份 Step1
manifest，以及分别由 MCP、XGBoost Step2 训练得到的两份
`mcp_per_offset_manifest.json`。

## 数据流

1. `step3_hybrid_concat_features.py` 分别合并 MCP、XGB 所有 combo 的代理列，输出
   `mcp/` 和 `xgb/` 两组训练/测试 buffer。
2. `step4_hybrid_screen_features.py` 在 MCP 合并矩阵上训练岭回归，按系数绝对值排序；
   在 XGB 合并矩阵上用与 `step2_validation/xgb_power_v2_3.py` 一致的默认参数训练
   XGBoost，按 gain 排序。两边均取 `final_top_k`，即
   `K_MCP = K_XGB = final_top_k`，再按 `type + offset + signal` 名称去重生成并集。
3. `step5_hybrid_hpo_timewindow.py` 在每个交叉验证训练折内，先从 MCP 排名中取前
   `floor(K_MCP × ratio)` 个代理训练岭回归，再用完整并集训练 XGBoost 拟合岭回归
   残差。`ratio` 是 Optuna 离散参数，候选为 `1/8, 2/8, ..., 8/8`。

Step5 的验证预测是：

```text
y_hat = Ridge(X_MCP_top_ratio) + XGBoost_residual(X_union)
```

岭回归只在每折训练数据上拟合，验证数据不会参与代理权重或线性参数估计。

## 一条命令运行

```csh
csh xgboost_timewindow/cmd/run_timewindow_hybrid_pipeline.csh \
  /path/to/step1/timewindow_manifest.json \
  /path/to/step2_mcp/mcp_per_offset_manifest.json \
  /path/to/step2_xgb_proxy/mcp_per_offset_manifest.json \
  /path/to/after_one.namefile.txt \
  /path/to/hybrid_output \
  1024
```

可用环境变量覆盖 `PYTHON`、`RIDGE_ALPHA`、`SCREEN_XGB_ROUNDS`、
`HPO_ROUNDS`、`HPO_TRIALS`、`HPO_POP`、`HPO_NFOLD`、`HPO_JOB_NUM` 和
`NTHREAD`。

## 主要产物

- `step4_hybrid_screen/mcp_ridge_importance.csv`：所有 MCP 候选代理及绝对权重。
- `step4_hybrid_screen/xgb_gain_importance.csv`：所有 XGB 候选代理及 gain。
- `step4_hybrid_screen/union_K<K>/union_meta.json`：并集 buffer、MCP 排名和列映射。
- `step5_hybrid_hpo/union_K<K>/hybrid_trials.csv`：全部 HPO trial。
- `step5_hybrid_hpo/union_K<K>/best_hybrid_model_meta.json`：最佳 ratio、线性代理、
  CV/测试指标和模型参数。
- `best_ridge_model.joblib` 与 `best_residual_xgb_model.json`：最佳混合模型的两部分。

