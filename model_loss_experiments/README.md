# Step4 后模型 × 损失实验框架

该目录绕过 Step5 HPO：直接读取 Step4 的固定特征 buffer，用固定配置训练多个“模型 × 损失函数”组合。当前已接通 PyTorch MLP；新增模型只需在 `model_loss/models.py` 注册构造函数，新增损失只需在 `model_loss/losses.py` 注册逐样本损失。

## 数据边界

- 输入：Step4 标准 `screen_K*` 或混合 `union_K*` 目录中的 train/test DMatrix buffer。
- 划分：仅从 train buffer 内按固定随机种子划出验证集，用于 early stopping。
- 测试：test buffer 不参与训练和选 epoch，只在恢复最佳验证 checkpoint 后评估。
- 权重：训练损失和加权指标均使用 buffer 内的样本权重。
- 稀疏特征：全量矩阵始终保持 CSR，只在 mini-batch 内转为稠密 Tensor。

## 运行 MLP

推荐使用项目已有的 `hw-model-train` 环境：

```bash
cd /home/data/chenyiheng/workspace/2026-dac/xgboost_timewindow
/home/data/chenyiheng/conda/envs/hw-model-train/bin/python \
  model_loss_experiments/run_experiment.py \
  --step4-dir /path/to/pipeline/step4_screen \
  --k 512 \
  --output-dir /path/to/experiments/mlp_k512
```

若直接传入叶子目录，不需要 `--k`：

```bash
/home/data/chenyiheng/conda/envs/hw-model-train/bin/python \
  model_loss_experiments/run_experiment.py \
  --step4-dir /path/to/step4_hybrid_screen/union_K512 \
  --losses mse,huber,relative_mae \
  --output-dir /path/to/experiments/mlp_k512
```

也可用 `--train-buffer`、`--test-buffer` 和可选的 `--feature-names` 显式指定输入。默认固定配置位于 `configs/mlp.json`；命令行支持覆盖 `--epochs`、`--device`、`--seed`。这不是 HPO：每个损失只训练一次相同的 MLP 配置。

## 输出

根目录中的 `comparison.csv/json` 汇总各损失的测试指标。每个 `mlp__<loss>/` 包含：

- `best_model.pt`：模型参数、特征缩放、标签缩放和元数据；
- `metadata.json`：配置、随机种子、最佳 epoch、参数量和耗时；
- `history.csv`：逐 epoch 训练/验证损失；
- `metrics_train.json`、`metrics_test.json`：统一的加权/非加权指标；
- `predictions_test.csv`：测试标签、预测、误差和样本权重。

当前损失：`mse`、`mae`、`huber`、`relative_mae`、`relative_mse`、`log_cosh`。其中相对误差分母使用配置中的 `relative_floor` 防止零值发散。

## 验证

```bash
cd model_loss_experiments
/home/data/chenyiheng/conda/envs/hw-model-train/bin/python -m pytest -q
```
