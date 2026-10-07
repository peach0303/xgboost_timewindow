# 代理筛选与复杂时序模型探索

## 1. 已落地接口

`step2_proxy_select.py` 保持现有 Step2/Step3 契约：每个组合输出
`selected_indices.txt`，根目录输出 `mcp_per_offset_manifest.json`。因此 Step3 无需修改。
主 pipeline 的 `PROXY_SELECTOR` 现在除 `mcp/xgb/xgboost` 外，还接受：

| selector | 思路 | 当前场景的定位 | 主要代价 |
|---|---|---|---|
| `f_regression` | 稀疏单变量 F 统计 | 最快基线、百万列粗筛 | 只识别边际线性关系，忽略冗余 |
| `elastic_net` | SGD Elastic Net 系数 | 适合超高维稀疏矩阵，同时处理一定共线性 | 对强非线性关系不敏感 |
| `stability` | 多次子采样 Elastic Net，按入选频率排序 | 推荐的稳健默认方案，降低 case/采样扰动 | 训练次数为 `stability_runs` |
| `mutual_info` | 互信息相关性 | 捕捉非线性单变量依赖 | 需先方差预筛并稠密化 |
| `mrmr` | 最大相关、最小冗余贪心选择 | 信号高度相关时比纯 MI 更合适 | 候选池内相关矩阵计算较贵 |
| `extra_trees` | 极随机树 impurity importance | 与 XGBoost 不同的随机非线性树基线 | 需预筛、稠密化，importance 有偏差风险 |

实现采用“按 case 均衡抽取训练行”的有界训练池。特征选择完全不读取 test label，
选完后才用 Ridge probe 在测试集报告一次指标。manifest 中会写入
`selection_uses_test_labels=false`。这也修正了旧 Step2 以 test MAPE 选择最佳 batch
所带来的评估泄漏风险。

示例：

```bash
PROXY_SELECTOR=stability \
PROXY_TOP_K=200 PROXY_MAX_TRAIN_ROWS=10000 PROXY_PREFILTER_K=5000 \
csh xgboost_timewindow/cmd/run_timewindow_pipeline.csh boom prep
```

快速基线与非线性对照：

```bash
PROXY_SELECTOR=f_regression PROXY_TOP_K=200 csh xgboost_timewindow/cmd/run_timewindow_pipeline.csh boom prep
PROXY_SELECTOR=mrmr PROXY_TOP_K=200 csh xgboost_timewindow/cmd/run_timewindow_pipeline.csh boom prep
PROXY_SELECTOR=extra_trees PROXY_TOP_K=200 csh xgboost_timewindow/cmd/run_timewindow_pipeline.csh boom prep
```

`mutual_info/mrmr/extra_trees` 会稠密化预筛后的候选池，默认以
`--max-dense-elements=50000000` 做硬保护。大数据首先降低 `PROXY_PREFILTER_K`。

## 2. 调研结论与建议优先级

1. **Stability Selection + Elastic Net：第一优先级。** 原论文将子采样与高维选择器结合，
   目标是提高变量选择稳定性并控制误选；本场景样本 case 少、列极多且信号共线，契合度最高。
2. **mRMR：第二优先级。** mRMR 同时考虑目标相关性和已选特征冗余，适合作为 TOG/level
   高相关列的非线性对照。工程上采用“方差预筛 -> MI 候选池 -> 相关冗余惩罚”的近似版本。
3. **Extra-Trees：第三优先级。** 极随机树对交互和非线性敏感，可作为不同于 boosting 的
   工业成熟树模型基线；但需关注 impurity importance 对连续值/高基数特征的偏好。
4. **HSIC-Lasso：研究候选。** 它能做模型无关的非线性且去冗余选择，但原始核矩阵方案对当前
   十万至百万列代价过高。建议只在预筛到 500--2000 列后实验，或采用 Nyström/RFF 近似，
   因而本次没有伪装成可扩展实现直接接入生产入口。
5. **跨 offset Group Lasso：下一阶段。** 当前 Step2 是每个 `(type, offset)` 独立筛选，无法
   显式奖励同一 signal 跨 offset 的一致性。若要做 group sparsity，应将 Step2 改为联合 combo
   训练；这会改变内存模型，但不必改变最终 manifest 契约。

主要资料：

- Meinshausen & Bühlmann, *Stability Selection*, JRSS-B 2010：
  https://doi.org/10.1111/j.1467-9868.2010.00740.x
- Peng, Long & Ding, *Feature Selection Based on Mutual Information: mRMR*, TPAMI 2005：
  https://doi.org/10.1109/TPAMI.2005.159
- Geurts, Ernst & Wehenkel, *Extremely Randomized Trees*, Machine Learning 2006：
  https://doi.org/10.1007/s10994-006-6226-1
- Freidling et al., *Post-selection inference with HSIC-Lasso*, ICML 2021：
  https://proceedings.mlr.press/v139/freidling21a.html
- scikit-learn Feature Selection 文档：
  https://scikit-learn.org/stable/modules/feature_selection.html

## 3. Step6 多模型训练

`step6_retrain_eval.py --model-type` 支持：

- `xgboost`：保持原 HPO 和 two-phase 行为，必须提供 `--hpo-csv`；
- `bnn`：隐藏层权重和激活均二值化，使用 Straight-Through Estimator，回归头保持浮点；
- `lstm`：按特征名解析 offset，将扁平 buffer 重建为“旧 offset -> 新 offset”的序列；
- `ltc`：使用可学习、随输入和状态变化的 time constant，并用 Euler 步进的液态循环基线。

LSTM/LTC 不会把列顺序直接当时间。适配器以 `(data_type, signal)` 为 channel，以 offset
为时间轴，缺失的 `(offset, channel)` 槽位补零；具体映射会保存到 `model_meta.json` 和
PyTorch checkpoint。

安装可选依赖：

```bash
/home/data/chenyiheng/conda/envs/pro/bin/pip install -r xgboost_timewindow/requirements-neural.txt
```

从已有 Step4 结果训练四种模型：

```bash
MODEL_TYPES=xgboost,bnn,lstm,ltc \
NN_EPOCHS=100 NN_BATCH_SIZE=128 NN_HIDDEN_SIZE=128 NN_DEVICE=auto \
csh xgboost_timewindow/cmd/run_timewindow_step6_models.csh boom 1024
```

神经模型输出仍沿用 `result/`、`pred/`、`eval/metrics.csv` 目录契约，模型文件分别为
`bnn_model.pt`、`lstm_model.pt`、`ltc_model.pt`。

模型依据：BNN 使用二值权重/激活训练思想；LSTM 用门控记忆处理跨 offset 依赖；LTC
用于连续时间动态的轻量探索。本实现的 LTC 是工程基线，不等同于论文完整的神经电路和
高阶 ODE 求解器，正式对比可进一步接入官方 `ncps` 的 LTC/CfC。

- Courbariaux et al., *Binarized Neural Networks*, 2016：https://arxiv.org/abs/1602.02830
- Hochreiter & Schmidhuber, *Long Short-Term Memory*, 1997：
  https://doi.org/10.1162/neco.1997.9.8.1735
- Hasani et al., *Liquid Time-constant Networks*, 2020：https://arxiv.org/abs/2006.04439
- Hasani et al., *Closed-form Continuous-time Neural Models*, 2021：
  https://arxiv.org/abs/2106.13898

## 4. 公平实验设计

- 固定 Step1 train/test case 划分和采样索引，不以测试集指标调 selector 或网络超参；
- 在训练 case 内再划验证集，至少报告 5 个随机种子的均值和标准差；
- 同时报告 MAPE、MAE、RMSE、NRMSE、R2、筛选列数、峰值内存、训练/推理时间；
- 先比较同一 Step4 K 下 XGBoost/BNN/LSTM/LTC，再比较不同 Step2 selector；
- 对 LSTM/LTC 增加一个“打乱 offset 顺序”的消融，确认收益确实来自时序结构。

## 5. BNN/LSTM/LTC 超参数优化

`step5_hpo_neural.py` 对 BNN、LSTM 和 LTC 使用训练集 K 折交叉验证。测试 buffer
只检查列数兼容性，不参与 trial 评分。默认采用 NSGA-II，同时最小化验证集 MAPE 和
模型参数量，并按 `NEURAL_PARAMETER_THRESHOLD` 从 Pareto 候选中选择最低 MAPE 配置。

直接运行单个模型：

```bash
python xgboost_timewindow/src/timewindow/step5_hpo_neural.py \
  --model-type lstm \
  --train-buffer /path/to/train.buffer \
  --test-buffer /path/to/test.buffer \
  --feature-names /path/to/feature_names.txt \
  --output-dir /path/to/lstm_hpo \
  --trial-num 40 --nfold 3 --epochs 100 --device auto
```

HPO 生成 `best_trial.json`、`best_trial.csv` 和 `neural_trials.csv`。最终重训练可通过
`step6_retrain_eval.py --hpo-result /path/to/best_trial.json` 读取最佳参数。

从已有 Step4 结果执行多模型 HPO 和重训练：

```csh
setenv OUTPUT_ROOT /path/to/existing/pipeline_run
setenv NEURAL_MODEL_TYPES bnn,lstm,ltc
setenv NEURAL_HPO_TRIALS 40
setenv NEURAL_HPO_NFOLD 3
setenv NEURAL_DEVICE auto
csh xgboost_timewindow/cmd/run_timewindow_neural_hpo.csh boom 1024
```

GPU HPO 建议保持 `NEURAL_HPO_JOB_NUM=1`，避免多个 trial 同时占用同一块显存。

### 5.1 不经过 HPO 的典型参数

为了快速比较模型，建议先使用下表的平衡型配置。三种模型共享
`epochs=100`、`patience=12`、`validation_ratio=0.15`、
`weight_decay=1e-5`、`gradient_clip=1.0` 和同一个随机种子。

| 模型 | hidden size | 层数 | batch size | learning rate | dropout |
|---|---:|---:|---:|---:|---:|
| BNN | 128 | 固定双隐藏层 | 256 | 1e-3 | 0.1 |
| LSTM | 128 | 2 | 128 | 5e-4 | 0.1 |
| LTC | 128 | 1 | 128 | 1e-3 | 0.1 |

LSTM 使用略低学习率以提高循环网络训练稳定性；BNN 使用较大 batch 以降低二值梯度噪声。
这些参数适合建立基线，不代表每个数据集上的最优值。比较时除精度指标外，还应同时报告
`model_meta.json` 中的参数量，以及训练时间和推理时间。

不经过 HPO 直接调用 Step6 的示例（将路径替换为同一个 Step4 K 的产物）：

```bash
PYTHON=/home/data/chenyiheng/conda/envs/hw-model-train/bin/python
TRAIN=/path/to/train_union_top1024.buffer
TEST=/path/to/test_union_top1024.buffer
NAMES=/path/to/feature_names.txt

$PYTHON xgboost_timewindow/src/timewindow/step6_retrain_eval.py \
  --model-type bnn --train-buffer "$TRAIN" --test-buffer "$TEST" \
  --feature-names "$NAMES" --output-dir /path/to/bnn_direct \
  --epochs 100 --batch-size 256 --learning-rate 0.001 \
  --hidden-size 128 --num-layers 2 --dropout 0.1 \
  --weight-decay 0.00001 --patience 12 --validation-ratio 0.15 \
  --gradient-clip 1.0 --device auto --skip-full-eval

$PYTHON xgboost_timewindow/src/timewindow/step6_retrain_eval.py \
  --model-type lstm --train-buffer "$TRAIN" --test-buffer "$TEST" \
  --feature-names "$NAMES" --output-dir /path/to/lstm_direct \
  --epochs 100 --batch-size 128 --learning-rate 0.0005 \
  --hidden-size 128 --num-layers 2 --dropout 0.1 \
  --weight-decay 0.00001 --patience 12 --validation-ratio 0.15 \
  --gradient-clip 1.0 --device auto --skip-full-eval

$PYTHON xgboost_timewindow/src/timewindow/step6_retrain_eval.py \
  --model-type ltc --train-buffer "$TRAIN" --test-buffer "$TEST" \
  --feature-names "$NAMES" --output-dir /path/to/ltc_direct \
  --epochs 100 --batch-size 128 --learning-rate 0.001 \
  --hidden-size 128 --num-layers 1 --dropout 0.1 \
  --weight-decay 0.00001 --patience 12 --validation-ratio 0.15 \
  --gradient-clip 1.0 --device auto --skip-full-eval
```

一键脚本默认执行 HPO。设置 `SKIP_NEURAL_HPO=1` 后，将使用上述参数直接训练，
并把结果写入带 `_direct` 后缀的目录：

```csh
setenv OUTPUT_ROOT /path/to/existing/pipeline_run
setenv SKIP_NEURAL_HPO 1
setenv NEURAL_MODEL_TYPES bnn,lstm,ltc
setenv NEURAL_DEVICE auto
csh xgboost_timewindow/cmd/run_timewindow_neural_hpo.csh boom 1024
```

所有典型参数均可通过环境变量覆盖，例如：

```csh
setenv LSTM_HIDDEN_SIZE 256
setenv LSTM_NUM_LAYERS 2
setenv LSTM_LEARNING_RATE 0.0003
setenv NEURAL_DIRECT_EPOCHS 150
```
