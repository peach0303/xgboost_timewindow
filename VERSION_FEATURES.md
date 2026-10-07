# xgboost_timewindow_opt1 版本特性

## 数据与采样

- 在 `(power, delta_power)` 二维空间按训练集分位点分桶。
- 稀疏桶全量保留，其余桶在预算内均衡采样；测试集复用训练集桶边界。
- 输出采样前后分布、稀疏桶及权重建议。
- 采用联合桶频率的逆平方根生成样本权重，并用于代理筛选、HPO 和最终训练。
- Step2 至 Step6 在训练前使用固定种子 shuffle，结果可复现。

## 代理选择与训练

- Step2 并发运行 MCP 与 XGBoost 两套代理选择流程。
- Step3 分别组织 MCP/XGB 代理矩阵。
- Step4 对 MCP 使用 Ridge 权重排序、对 XGB 使用 gain 排序；两侧各取 K（1:1），同名代理去重后形成并集。
- Step5 在混合代理并集上执行标准 XGBoost HPO。
- Step6 直接拟合原始 power，仅输出单一 XGBoost 模型，不使用 Ridge + 残差模型。

## 损失函数

XGBoost 支持：

- `squared_error`
- `pseudo_huber`
- `absolute_error`
- `relative_squared`
- `error_focus`
- `relative_error_focus`

默认使用 `relative_error_focus`，HPO 默认关注误差最差 5% 样本的平均相对误差。

## 主要输出

```text
step2_mcp/
step2_xgb_proxy/
step3_hybrid_concat/
step4_hybrid_screen/union_K*/
step5_hpo/union_K*_direct_xgb/
step6_retrain/union_K*_direct_xgb/result/xgbmodel.json
```
