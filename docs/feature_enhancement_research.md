# `xgboost_timewindow` 特征增强调研

更新时间：2026-09-08

## 1. 结论摘要

当前流水线已经把每个信号在多个历史 offset 上的 `TOG`、`level` 原始值交给 XGBoost，但还没有显式构造“一个信号在整段窗口中的活动形态”和“一个模块在当前窗口中的整体活动”。最值得尝试的特征依次是：

1. 每信号多尺度窗口统计：`TOG` 的 sum/mean/max/active-count/EWMA，以及首尾差、短窗/长窗比值；
2. 由 `TOG + level` 分解出的 0→1、1→0 方向性翻转；
3. 由层次化信号名生成的模块活动特征：模块 toggle 总量、活跃信号数、level 占空、模块活动占全芯片比例；
4. 若能取得扇出、负载电容或 cell 信息，增加 `fanout × toggle`、`capacitance × toggle` 等结构加权活动；
5. 对高度相关信号做训练集内聚类或 mRMR/stability selection，并增加簇级活动统计，减少冗余。

不过，任何增强实验之前都应先修正评估边界。旧版 `step2_xgb_per_offset.py` 和 `step2_mcp_per_offset.py` 会用测试集 MAPE 选择最佳训练 batch，`step4_screen_features.py` 又在测试集上反复评估不同 K。测试集因此参与了模型/特征选择。仓库里的 `step2_proxy_select.py` 已做到只在训练集选特征，可用作新实验基线；Step4 的 K 选择和 Step5 的 CV 仍需改成按 case 分组、仅使用训练 case。

## 2. 当前流水线判断

### 2.1 已有输入

- 对齐规则：`LEVEL[t+1] ↔ TOG[t] ↔ label[t]`。
- 对 `label[i]`，offset 为 `k` 时使用 `TOG[i-k]` 和 `LEVEL[i+1-k]`。
- Step2 对每个 `(data_type, offset)` 独立筛代理信号。
- Step3 横向拼接各 offset/type 的入选列。
- Step4 把列 importance 按原始 signal 聚合，默认取跨 offset/type 的最大 gain，然后把入选 signal 的全部 Step1 列展开回来。
- 当前数据按 case 切分，默认最后若干 case 为测试集；case 内样本仍有很强时间自相关。

因此，XGBoost 当前可以自行学习原始 lag 的阈值和交互，但它需要较深的树才能近似“13 个 offset 求和”“短期活动占长期活动比例”等稳定统计。在训练 case 很少时，显式给出这些统计通常更节省样本。

### 2.2 仓库中的方向性证据

`output/timewindow_offset_compare/step4_offset_comparison.csv` 中：

| offset 范围 | 模型 | MAPE | R² |
|---|---:|---:|---:|
| 0..12 | screen_K256 | 4.403% | 0.230 |
| 6..12 | screen_K256 | 5.692% | -0.290 |

这说明 `0..5` 的近邻历史很可能有用，适合做短/中/长多尺度特征。但由于当前 Step4 在测试集上比较 K，该结果只能用于生成假设，不能作为无偏结论。

## 3. 推荐特征

### 3.1 P0：每信号多尺度窗口统计

对入选信号 `s`，令 `x_k = TOG_s(t-k)`，建议在 offset `0..12` 上增加：

- 短窗和：`sum_0_2`、`sum_0_5`；
- 长窗和：`sum_0_12`；
- `mean`、`max`、非零 offset 数 `active_count`；
- 指数衰减和 `ewm = Σ exp(-k/τ) x_k`，先试 `τ=2, 4, 8`；
- 趋势：`x_0 - x_3`、短窗均值减长窗均值；
- burst：`max / (mean + ε)`；
- recent ratio：`sum_0_2 / (sum_0_12 + ε)`。

对于 level，优先使用：

- 短/长窗占空均值；
- `level_now`；
- 窗口内变化次数；
- 最长连续 0/1 长度（若只保留少量重要信号）。

建议先只加 sum、active-count、EWMA 和 trend。它们计算便宜、解释稳定，也容易保持稀疏；ratio、burst 会产生稠密浮点列，应在 signal shortlist 后生成。

Simmani 把每个信号在窗口中的翻转次数除以“位宽 × 窗口长度”，形成 toggle-density，并用 CSR 存储大规模稀疏 toggle-pattern matrix，说明窗口归一化活动特征与当前数据形态高度兼容：[Simmani 论文](https://simmani.github.io/assets/paper.pdf)。

### 3.2 P0：方向性翻转特征

当前 `TOG` 只表达“发生了变化”，不能区分上升和下降。利用已经对齐的 level，可构造：

```text
rise_s(t-k) = TOG_s(t-k) * LEVEL_s(t+1-k)
fall_s(t-k) = TOG_s(t-k) * (1 - LEVEL_s(t+1-k))
```

对于单 bit 信号，它们分别近似 0→1 和 1→0；对于总线/归约后的列，需要先确认 `TOG` 和 `level` 的数值语义再定义。建议同时生成 `rise/fall` 的短窗和、长窗和。早期 RTL 功耗模型也将切换状态进一步拆成上升/下降子变量，因此该增强有明确的物理动机：[ToggleFinder 相关论文检索页](https://ieeexplore.ieee.org/document/1594795/)。

注意：上述 IEEE 页面直接支持“RTL 周期功耗使用 switching/clock activity”，上升/下降拆分的具体依据来自 ToggleFinder 论文体系；实现前应以本项目 FSDB 导出语义做一次真值检查。

### 3.3 P1：模块/子系统活动特征

现有 signal name 含完整层次，例如 `boom_tile.core`、`frontend`、`lsu`、`rob`、`issue_unit`，而当前高 importance 信号确实分布在 LSU、Core、Frontend 等不同区域。可以按层次前缀聚合：

- `module_tog_sum_{short,long}`；
- `module_active_signal_count`；
- `module_level_occupancy_mean`；
- `module_tog_max`、`p90`；
- `module_share = module_tog_sum / global_tog_sum`；
- 全局 `total_tog`、活跃模块数、活跃信号数。

层次粒度先固定为少量稳定边界，如 `frontend/core/lsu/fpu/cache/uncore`，不要直接把任意路径层级 one-hot 化。MasterRTL 的树模型显式使用 toggle-rate 总和、平均值、`fanout × toggle-rate` 总和及节点数，且采用 module-level 功耗分解，这直接支持模块聚合方向：[MasterRTL, Section III-C](https://zhiyaoxie.com/files/TCAD25_MasterRTL.pdf)。

### 3.4 P1：结构加权活动

若能从综合网表、Liberty、SPEF 或已有 reduction 元数据取得结构信息，增加：

```text
weighted_toggle_s = TOG_s * fanout_s
cap_toggle_s      = TOG_s * Cload_s
module_cap_toggle = Σ_s TOG_s * Cload_s
```

优先级为 `Cload` > 扇出 > 无权重节点数。它比任意多项式交互更贴近动态功耗。MasterRTL 使用 `fanout × toggle-rate`；经典周期精确功耗工具也将 RTL switching/clock activity 映射到晶体管级宏模型：[MasterRTL](https://zhiyaoxie.com/files/TCAD25_MasterRTL.pdf)、[ASPDAC 2006 cycle-accurate power estimation](https://ieeexplore.ieee.org/document/1594795/)。

### 3.5 P1：冗余感知的信号/簇特征

当前 Step4 用 signal 的最大 gain 排序，会偏向偶然在某个 offset 被树选中的信号，也不会惩罚一组高度相关的重复信号。建议：

- 先直接比较仓库现成的 `step2_proxy_select.py --selector stability` 和 `--selector mrmr`；
- 按 train case/seed 重复选择，保留入选频率超过阈值的 signal；
- 对窗口 toggle pattern 相似的信号聚类，每簇保留代表 signal，同时增加簇内 sum、active-count、cluster-size；
- Step4 的 signal score 改为跨 fold 的 median/mean normalized gain，或在训练验证 fold 上做 grouped permutation importance，而不是单次训练的 max gain。

Simmani 的核心做法正是按 toggle-pattern 聚类并选代表信号；其论文也强调 workload 代表性对未见 workload 的泛化影响：[Simmani](https://simmani.github.io/assets/paper.pdf)。

### 3.6 P2：活动模式（power mode）特征

用少量模块统计构造 activity regime：idle、frontend-heavy、memory-heavy、integer-heavy、mixed。可采用：

- 规则阈值作为数值/布尔特征；或
- 仅用训练集拟合 MiniBatchKMeans/GMM，输出 cluster ID、到各中心距离。

该方法适合 BOOM/CoreMark 中明显的流水线阶段与模块激活差异，但必须在每个训练 fold 内拟合聚类器，不能先用全体数据聚类。早期 ToggleFinder 工作采用 power-mode classification 来区分不同功耗行为；近期 AtomPower 也把结构约束聚类用于缓解多重共线性：[AtomPower](https://www.jstage.jst.go.jp/article/elex/advpub/0/advpub_23.20260004/_article)。

## 4. 不建议优先做的项目

- **直接加入过去的 power label**：离线评估会大幅变好，但实际推理若拿不到上一周期真实功耗，就不可部署；除非明确是在线传感器场景。
- **把 case ID、绝对 row index 当特征**：容易记住 workload/时间位置，对新 case 泛化差。
- **全量二阶 signal×signal 交互**：列数平方增长，XGBoost 本身已能学习少量交互；优先只做有物理含义的 `TOG×level` 和模块聚合。
- **PCA 后直接替代稀疏原始特征**：可能破坏可解释性并变稠密。可作为小规模对照，不应作为首选。
- **先增强再在全数据上筛选**：会把验证信息泄漏到特征生成/选择中。

## 5. 必须配套修正的验证方案

### 5.1 严格保留最终测试 case

建议三层边界：

```text
最终 test cases：只在所有特征、K、HPO 参数确定后评估一次
train cases 内层：GroupKFold / Leave-One-Case-Out，用于特征选择和 HPO
同一 case 内：若做时间外推，再按时间前→后切并设置 gap >= max(offset)
```

普通随机 KFold 不适合当前数据：相邻 row 高度自相关，而且 Step5 当前对拼接后的 row 使用 KFold，会让同一 case 的相邻周期分散到 train/validation。scikit-learn 文档明确指出，普通 KFold/ShuffleSplit 在有时间自相关的数据上会给出不合理的泛化估计；有 group 结构时应按 group 切分：[时间序列 CV](https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation-of-time-series-data)、[GroupShuffleSplit 示例](https://scikit-learn.org/stable/modules/cross_validation.html)。

若目标是“泛化到未见 workload/case”，优先 `LeaveOneGroupOut(groups=stem)`；若目标是“同一 workload 的未来周期”，使用带 gap 的 `TimeSeriesSplit`。不要混用两个目标。

### 5.2 把所有选择放进训练折

- Step2 不得按 test MAPE 选最佳 batch；使用全训练 case、batch union、stability 或 mRMR。
- Step4 的 K 只由内层 validation case 选择。
- selector、聚类器、模块归一化、ratio 的分母统计都在 fold-train 上拟合或定义。
- final test 只产生一次最终报告。

### 5.3 指标

以 `MAE + NRMSE + R²` 为主，MAPE 为辅。当前代码用极小 epsilon 处理接近 0 的 label，低功耗点可能主导 MAPE；至少应同时报告按功耗分位区间的 MAE/MAPE，以及峰值区间（如 top 5% power）的误差。

## 6. 建议的流水线插入点

推荐增加 `Step2.5`，而不是在 Step1 对数万信号直接生成所有稠密统计：

```text
Step1   对齐并保存各 offset 的原始 TOG/level
Step2   仅用 train cases 做 signal shortlist（stability/mRMR/union）
Step2.5 对 shortlist 信号构造 raw lag + 多尺度 + rise/fall
        同时从全体信号流式计算少量 module/global aggregates
Step3   拼接增强特征，保留 feature family 元数据
Step4   train-grouped CV 的稳定性筛选；按 signal/family 聚合
Step5   GroupKFold/Leave-One-Case-Out HPO
Step6   train 全量重训，最终 test 一次
```

建议特征名采用：

```text
sig::{signal}::tog::lag0
sig::{signal}::tog::sum0_5
sig::{signal}::tog::ewm_tau4
sig::{signal}::rise::sum0_5
mod::{module}::tog::sum0_12
global::tog::active_count0_2
```

同时在 manifest 记录 `source_signal`、`module`、`family`、`offset_range`、`is_sparse`，避免 Step4 再依赖正则表达式猜测 signal 名。

## 7. 最小消融实验矩阵

先固定同一组 outer test cases 和 inner group folds，每组至少跑 3 个 seed：

| 实验 | 特征 | 目的 | 预计成本 |
|---|---|---|---:|
| E0 | raw lag 0..12 + 严格 group CV | 无泄漏新基线 | 低 |
| E1 | E0 + TOG sum/active-count/EWMA/trend | 检验多尺度活动 | 低 |
| E2 | E1 + rise/fall | 检验方向性切换 | 低-中 |
| E3 | E1 + module/global aggregates | 检验模块功耗模式 | 中 |
| E4 | E1 + stability/mRMR/cluster aggregates | 检验冗余控制 | 中 |
| E5 | E3 + fanout/Cload 加权 | 检验结构先验 | 中-高 |

晋级规则建议：inner-fold 平均 NRMSE 至少下降 2%，且最差 fold 不退化超过 3%；然后才允许在最终 test 上比较一次。还应记录峰值功耗 top-5% MAE、特征列数、CSR nnz、训练时间和模型 leaf 数，防止用大幅资源增长换来很小收益。

## 8. 推荐落地顺序

1. 先把 Step2/Step4/Step5 改为 case-grouped、test-once 协议；
2. 用 `offset=0..12` 建立严格 E0（仓库旧实验已经显示近邻 offset 有希望）；
3. 实现 Step2.5 的 TOG 多尺度统计；
4. 校验 FSDB 的 `TOG/level` 数值语义后加入 rise/fall；
5. 从层次信号名加入 `frontend/core/lsu/...` 模块聚合；
6. 若有网表/Liberty 元数据，再加入 fanout/Cload 加权；
7. 最后考虑聚类 activity regime 或更复杂模型。

