#!/usr/bin/env python3
"""Step 4（hybrid）：对 MCP/XGBoost 代理特征排序并生成下游 Top-K 数据。

脚本定位
========
上游 ``step3_hybrid_concat_features.py`` 会为 MCP、XGBoost 或两者生成独立的
train/test DMatrix buffer；本脚本读取这些 buffer，对两类代理特征分别建模和排序，
再把排序结果合并成 Step 5/6 使用的 ``union_K*`` 数据。这里的“hybrid”表示可以
融合两种代理选择器，并不表示把两个模型的预测值做加权平均。

总体流程
========
1. 读取 ``--hybrid-concat-meta``，由其中的 ``selected_selectors``/``selectors``
   决定启用 MCP、XGBoost，或同时启用二者。
2. ``load_selector`` 读取各 selector 的特征、标签和行级样本权重。buffer 没有权重
   时回退为全 1；所有 selector 使用同一随机顺序同步打乱训练行。
3. 按 selector 分支生成排名：

   * MCP 分支：用带 ``sample_weight`` 的 Ridge 拟合全部 MCP 代理列，按回归系数
     绝对值从大到小排序。
   * XGBoost 分支：用带权 DMatrix 训练筛选模型，按 feature gain 从大到小排序。
     ``--loss`` 可选择 XGBoost 内置损失或 ``training_policy`` 中的自定义损失；
     自定义损失会显式把样本权重乘到梯度和 Hessian。

4. 对 ``--final-top-ks`` 中的每个 K 调用 ``build_union``：每个已启用 selector
   最多贡献前 K 列，然后按特征名去重。去重顺序固定为 MCP 后 XGBoost，因此同名
   特征同时出现时保留 MCP 项。双 selector 的 union 列数通常介于 K 和 2K 之间，
   单 selector 时最多为 K。
5. 把 union 特征、标签和行权重保存为 train/test DMatrix buffer，并输出排名文件、
   特征名和 ``union_meta.json``，供 HPO 与最终重训练使用。

``--feature-input-mode`` 的两个特征构造分支
==============================================
``proxy_columns``（默认）
    直接从 Step 3 的 selector 矩阵中抽取排名命中的具体代理列并横向拼接。例如
    ``tog_o12_signal_a`` 入选后，只保留这个数据类型/offset 对应的列。横向拼接不
    改变样本行，所以标签和权重直接取第一个 selector 作为公共 reference；代码
    假定各 selector 来自同一批样本并具有相同的行权重。该模式不需要
    ``--tw-manifest`` 或 ``--namefile``，而且 ``mcp_linear_columns`` 会记录 union
    中可供后续 Ridge 使用的 MCP 列位置。

``signal_all_windows``
    先用 ``raw_signal_name`` 去掉 ``tog/level + offset`` 前缀，把代理列归并为原始
    信号；再通过 ``build_top_matrices_from_step1`` 从 Step 1 重新加载这些信号在
    所有可用数据类型和时间窗上的列。例如任一 ``signal_a`` 代理列入选后，最终
    可以展开为它的多个 TOG/LEVEL、多个 offset 列。因此 union 的实际列数可能
    大于入选代理列数。该模式必须同时提供 ``--tw-manifest`` 和 ``--namefile``；
    因为矩阵被重新构造，标签、权重、特征名和 source_blocks 也一起从 Step 1
    重新加载，以保证它们与新矩阵的行顺序严格对齐。此模式不设置
    ``mcp_linear_columns``。

样本权重说明
============
样本权重是“每一行一个”的训练权重，不是特征列权重。两个 selector 横向拼接时
不能把权重重复拼接；无论使用哪种 feature-input mode，最终都只有与样本行数相同
的一份 ``train_weight``/``test_weight``。本脚本不重新计算权重，只负责：

* 在 Ridge/XGBoost 排名模型训练时使用已有权重，使权重影响特征排名；
* 在构造 union 时保持特征、标签、权重的行对齐；
* 将权重写入输出 DMatrix，使 Step 5/6 继续进行带权训练。

脚本中的 ``metrics`` 是未加权 MAE/MAPE/RMSE/R2，仅用于按原始样本分布报告效果；
测试权重主要供 XGBoost 的 eval metric 使用，不会改变模型预测值。

主要选项与影响范围
==================
* ``--final-top-ks``：可同时生成多个 K 的 union，排序模型只训练一次。
* ``--feature-input-mode``：选择上述“具体代理列”或“原始信号全时间窗”分支。
* ``--tw-manifest``、``--namefile``：仅 ``signal_all_windows`` 分支必需。
* ``--ridge-*``：只影响 MCP/Ridge 排名分支。
* ``--xgb-rounds``、树结构和采样参数：只影响 XGBoost 排名分支。
* ``--loss``、``--relative-floor``、``--focus-power``：控制 XGBoost 损失分支；
  后两项只对相对误差/大误差聚焦类自定义损失有意义。
* ``--shuffle-seed``：控制所有 selector 共用的训练行打乱顺序；``--seed`` 控制
  XGBoost 自身随机性。
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import time
from collections import OrderedDict

import joblib
import numpy as np
import xgboost as xgb
from scipy.sparse import hstack
from sklearn.linear_model import Ridge
from utils.training_policy import XGB_LOSSES, configure_xgb_loss, shuffled_indices


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def load_names(path: str) -> list[str]:
    with open(path, "r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def load_selector(meta: dict, selector: str):
    item = meta["selectors"][selector]
    train = xgb.DMatrix(item["train_buffer"])
    test = xgb.DMatrix(item["test_buffer"])
    names = load_names(item["feature_names_file"])
    if train.num_col() != len(names) or test.num_col() != len(names):
        raise ValueError(f"{selector} feature-name count does not match buffers")
    train_weight = train.get_weight()
    test_weight = test.get_weight()
    if not train_weight.size:
        train_weight = np.ones(train.num_row())
    if not test_weight.size:
        test_weight = np.ones(test.num_row())
    return train.get_data().tocsr(), train.get_label(), train_weight, test.get_data().tocsr(), test.get_label(), test_weight, names


def metrics(y_true: np.ndarray, prediction: np.ndarray) -> OrderedDict:
    error = y_true - prediction
    denominator = np.maximum(np.abs(y_true), 1e-12)
    mse = float(np.mean(error * error))
    variance = float(np.var(y_true))
    return OrderedDict([
        ("n", int(y_true.size)), ("MAE", float(np.mean(np.abs(error)))),
        ("MAPE", float(np.mean(np.abs(error) / denominator) * 100.0)),
        ("RMSE", float(math.sqrt(mse))),
        ("R2", float(1.0 - mse / variance) if variance > 1e-12 else float("nan")),
    ])


def write_ranking(path: str, ranking: list[tuple[str, float, int]], score_name: str) -> None:
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["rank", "proxy", score_name, "source_column"])
        for rank, (name, score, column) in enumerate(ranking, 1):
            writer.writerow([rank, name, f"{score:.17g}", column])


def xgb_params(args) -> dict:
    # Defaults intentionally match step2_validation/xgb_power_v2_3.py.
    return {
        "objective": "reg:squarederror", "booster": "gbtree",
        "tree_method": args.tree_method, "max_bin": args.max_bin,
        "grow_policy": "lossguide", "max_depth": args.max_depth,
        "max_leaves": args.max_leaves, "learning_rate": args.learning_rate,
        "subsample": args.subsample, "colsample_bytree": args.colsample_bytree,
        "colsample_bynode": args.colsample_bynode, "nthread": args.nthread,
        "seed": args.seed, "eval_metric": "rmse",
    }


def raw_signal_name(feature_name: str) -> str:
    match = re.match(r"^(?:tog|level)_o-?\d+_(.+)$", feature_name)
    return match.group(1) if match else feature_name


def choose_ranked_proxies(
    top_k: int,
    rankings: OrderedDict[str, list[tuple[str, float, int]]],
) -> list[tuple[str, str, float, int]]:
    selected: list[tuple[str, str, float, int]] = []
    seen: set[str] = set()
    for selector in ("mcp", "xgb"):
        for name, score, column in rankings.get(selector, [])[:top_k]:
            if name not in seen:
                selected.append((selector, name, score, column))
                seen.add(name)
    return selected


def build_union(
    output_dir: str,
    top_k: int,
    rankings: OrderedDict[str, list[tuple[str, float, int]]],
    selector_data: OrderedDict,
    feature_input_mode: str,
    tw_manifest: dict | None,
    namefile: str | None,
) -> None:
    selected = choose_ranked_proxies(top_k, rankings)
    if not selected:
        raise ValueError(f"no proxy columns selected for K={top_k}")
    reference = next(iter(selector_data.values()))
    train_y, test_y = reference["train_y"], reference["test_y"]
    train_weight, test_weight = reference["train_weight"], reference["test_weight"]
    selected_names = [item[1] for item in selected]
    selected_signals = list(OrderedDict.fromkeys(raw_signal_name(name) for name in selected_names))

    if feature_input_mode == "proxy_columns":
        train_parts = [
            selector_data[selector]["train"][:, [column]]
            for selector, _name, _score, column in selected
        ]
        test_parts = [
            selector_data[selector]["test"][:, [column]]
            for selector, _name, _score, column in selected
        ]
        union_train = hstack(train_parts, format="csr")
        union_test = hstack(test_parts, format="csr")
        feature_names = selected_names
        source_blocks = []
    else:
        if tw_manifest is None or not namefile:
            raise ValueError(
                "signal_all_windows requires --tw-manifest and --namefile"
            )
        from step4_screen_features import build_top_matrices_from_step1
        (
            union_train, train_y, train_weight,
            union_test, test_y, test_weight,
            feature_names, source_blocks,
        ) = build_top_matrices_from_step1(
            tw_manifest, namefile, set(selected_signals)
        )

    directory = os.path.join(output_dir, f"union_K{top_k}")
    os.makedirs(directory, exist_ok=True)
    train_buffer = os.path.join(directory, f"train_union_top{top_k}.buffer")
    test_buffer = os.path.join(directory, f"test_union_top{top_k}.buffer")
    xgb.DMatrix(union_train, label=train_y, weight=train_weight, feature_names=feature_names).save_binary(train_buffer)
    xgb.DMatrix(union_test, label=test_y, weight=test_weight, feature_names=feature_names).save_binary(test_buffer)
    names_file = os.path.join(directory, "feature_names.txt")
    with open(names_file, "w", encoding="utf-8") as handle:
        handle.write("\n".join(feature_names) + "\n")
    for selector in rankings:
        with open(os.path.join(directory, f"top{top_k}_{selector}_proxies.txt"), "w", encoding="utf-8") as handle:
            for name, score, _ in rankings[selector][:top_k]:
                handle.write(f"{name}\t{score:.17g}\n")

    before_dedup = sum(min(top_k, len(ranking)) for ranking in rankings.values())
    mcp_selected = [item for item in selected if item[0] == "mcp"]
    xgb_selected = [item for item in selected if item[0] == "xgb"]
    union_meta = OrderedDict([
        ("schema_version", 1), ("final_top_k", top_k),
        ("feature_input_mode", feature_input_mode),
        ("selected_selectors", list(rankings)),
        ("k_mcp", min(top_k, len(rankings.get("mcp", [])))),
        ("k_xgb", min(top_k, len(rankings.get("xgb", [])))),
        ("selected_proxy_columns", len(selected)),
        ("selected_signals", selected_signals),
        ("union_columns", len(feature_names)),
        ("overlap_columns", before_dedup - len(selected)),
        ("mcp_linear_columns", list(range(len(mcp_selected))) if feature_input_mode == "proxy_columns" else []),
        ("mcp_ranked_proxies", [
            {"proxy": name, "abs_ridge_weight": score, "union_column": index}
            for index, (_selector, name, score, _column) in enumerate(mcp_selected)
        ]),
        ("xgb_ranked_proxies", [
            {"proxy": name, "gain": score}
            for _selector, name, score, _column in xgb_selected
        ]),
        ("source_blocks", source_blocks),
        ("train_buffer", os.path.abspath(train_buffer)),
        ("test_buffer", os.path.abspath(test_buffer)),
        ("feature_names_file", os.path.abspath(names_file)),
        ("train_shape", list(union_train.shape)), ("test_shape", list(union_test.shape)),
    ])
    with open(os.path.join(directory, "union_meta.json"), "w", encoding="utf-8") as handle:
        json.dump(union_meta, handle, indent=2, ensure_ascii=False)
    log(
        f"K={top_k}: selectors={list(rankings)}, proxies={len(selected)}, "
        f"signals={len(selected_signals)}, columns={len(feature_names)}, "
        f"mode={feature_input_mode} -> {directory}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Step4 Ridge/XGBoost proxy screening")
    parser.add_argument("--hybrid-concat-meta", required=True)
    parser.add_argument("--final-top-ks", default="256,512,1024,2048,4096")
    parser.add_argument(
        "--feature-input-mode",
        choices=("proxy_columns", "signal_all_windows"),
        default="proxy_columns",
    )
    parser.add_argument("--tw-manifest")
    parser.add_argument("--namefile")
    parser.add_argument("--ridge-alpha", type=float, default=8.0)
    parser.add_argument("--ridge-max-iter", type=int, default=2000)
    parser.add_argument("--xgb-rounds", type=int, default=200)
    parser.add_argument("--tree-method", choices=["hist", "approx"], default="hist")
    parser.add_argument("--max-bin", type=int, default=32)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--max-leaves", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--subsample", type=float, default=0.8)
    parser.add_argument("--colsample-bytree", type=float, default=0.01)
    parser.add_argument("--colsample-bynode", type=float, default=0.01)
    parser.add_argument("--nthread", type=int, default=32)
    parser.add_argument("--seed", type=int, default=108)
    parser.add_argument("--loss", choices=XGB_LOSSES, default="relative_error_focus")
    parser.add_argument("--relative-floor", type=float, default=1e-3)
    parser.add_argument("--focus-power", type=float, default=3.0)
    parser.add_argument("--shuffle-seed", type=int, default=108)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    if args.ridge_alpha < 0:
        parser.error("--ridge-alpha must be non-negative")
    top_ks = sorted(set(int(value) for value in args.final_top_ks.split(",") if value.strip()))
    if not top_ks or min(top_ks) < 1:
        parser.error("--final-top-ks must contain positive integers")
    if args.feature_input_mode == "signal_all_windows":
        if not args.tw_manifest or not args.namefile:
            parser.error(
                "signal_all_windows requires --tw-manifest and --namefile"
            )

    with open(args.hybrid_concat_meta, "r", encoding="utf-8") as handle:
        meta = json.load(handle, object_pairs_hook=OrderedDict)
    selectors = list(meta.get("selected_selectors") or meta["selectors"])
    if not selectors or any(selector not in {"mcp", "xgb"} for selector in selectors):
        raise ValueError(f"invalid selectors in hybrid meta: {selectors}")
    selector_data = OrderedDict()
    reference = None
    for selector in selectors:
        train, train_y, train_weight, test, test_y, test_weight, names = load_selector(meta, selector)
        selector_data[selector] = {
            "train": train, "train_y": train_y, "train_weight": train_weight,
            "test": test, "test_y": test_y, "test_weight": test_weight,
            "names": names,
        }
        if reference is None:
            reference = selector_data[selector]
        elif not np.array_equal(reference["train_y"], train_y) or not np.array_equal(reference["test_y"], test_y):
            raise ValueError(f"{selectors[0]} and {selector} labels differ")
    insufficient = {
        selector: len(data["names"])
        for selector, data in selector_data.items()
        if max(top_ks) > len(data["names"])
    }
    if insufficient:
        raise ValueError(f"requested K={max(top_ks)}, available columns={insufficient}")
    os.makedirs(args.output_dir, exist_ok=True)
    order = shuffled_indices(len(reference["train_y"]), args.shuffle_seed)
    for data in selector_data.values():
        data["train"] = data["train"][order]
        data["train_y"] = data["train_y"][order]
        data["train_weight"] = data["train_weight"][order]

    rankings = OrderedDict()
    if "mcp" in selector_data:
        data = selector_data["mcp"]
        log(f"fitting Ridge on all {len(data['names'])} merged MCP proxies")
        ridge = Ridge(
            alpha=args.ridge_alpha, fit_intercept=True, solver="lsqr",
            max_iter=args.ridge_max_iter, tol=1e-6,
        )
        ridge.fit(data["train"], data["train_y"], sample_weight=data["train_weight"])
        coefficients = np.asarray(ridge.coef_).reshape(-1)
        rankings["mcp"] = sorted(
            [(name, float(abs(coefficients[index])), index) for index, name in enumerate(data["names"])],
            key=lambda item: (-item[1], item[2]),
        )
        joblib.dump(ridge, os.path.join(args.output_dir, "mcp_ridge_model.joblib"))
        np.save(os.path.join(args.output_dir, "mcp_ridge_coefficients.npy"), coefficients)
        write_ranking(os.path.join(args.output_dir, "mcp_ridge_importance.csv"), rankings["mcp"], "abs_ridge_weight")
        ridge_eval = {
            "train": metrics(data["train_y"], ridge.predict(data["train"])),
            "test": metrics(data["test_y"], ridge.predict(data["test"])),
        }
        with open(os.path.join(args.output_dir, "mcp_ridge_metrics.json"), "w", encoding="utf-8") as handle:
            json.dump(ridge_eval, handle, indent=2, ensure_ascii=False)

    if "xgb" in selector_data:
        data = selector_data["xgb"]
        params, custom_objective = configure_xgb_loss(
            xgb_params(args), args.loss, args.relative_floor, args.focus_power
        )
        log(f"fitting XGBoost on all {len(data['names'])} merged XGB proxies; params={params}")
        dtrain = xgb.DMatrix(data["train"], label=data["train_y"], weight=data["train_weight"], feature_names=data["names"])
        dtest = xgb.DMatrix(data["test"], label=data["test_y"], weight=data["test_weight"], feature_names=data["names"])
        booster = xgb.train(
            params, dtrain, num_boost_round=args.xgb_rounds,
            evals=[(dtest, "test")], verbose_eval=10, obj=custom_objective,
        )
        booster.save_model(os.path.join(args.output_dir, "xgb_screen_model.json"))
        gain = booster.get_score(importance_type="gain")
        rankings["xgb"] = sorted(
            [(name, float(gain.get(name, gain.get(f"f{index}", 0.0))), index)
             for index, name in enumerate(data["names"])],
            key=lambda item: (-item[1], item[2]),
        )
        write_ranking(os.path.join(args.output_dir, "xgb_gain_importance.csv"), rankings["xgb"], "gain")
        xgb_eval = {
            "train": metrics(data["train_y"], booster.predict(dtrain)),
            "test": metrics(data["test_y"], booster.predict(dtest)),
            "params": params, "rounds": args.xgb_rounds,
        }
        with open(os.path.join(args.output_dir, "xgb_screen_metrics.json"), "w", encoding="utf-8") as handle:
            json.dump(xgb_eval, handle, indent=2, ensure_ascii=False)

    tw_manifest = None
    if args.tw_manifest:
        with open(args.tw_manifest, "r", encoding="utf-8") as handle:
            tw_manifest = json.load(handle, object_pairs_hook=OrderedDict)

    for top_k in top_ks:
        build_union(
            args.output_dir, top_k, rankings, selector_data,
            args.feature_input_mode, tw_manifest, args.namefile,
        )
    log("done")


if __name__ == "__main__":
    main()
