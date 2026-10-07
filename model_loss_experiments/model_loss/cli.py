"""实验命令行入口。"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from .data import load_dmatrix_buffer, resolve_step4_inputs
from .losses import available_losses
from .models import available_models
from .trainer import train_experiment


def _parser(default_config: Path) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Step4 后固定配置的模型 × 损失实验")
    source = parser.add_argument_group("Step4 输入")
    source.add_argument("--step4-dir", type=Path, help="screen_K*/union_K* 目录或其父目录")
    source.add_argument("--k", type=int, help="当 --step4-dir 是父目录时选择 K")
    source.add_argument("--train-buffer", type=Path)
    source.add_argument("--test-buffer", type=Path)
    source.add_argument("--feature-names", type=Path)
    parser.add_argument("--config", type=Path, default=default_config)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", choices=available_models(), default=None)
    parser.add_argument(
        "--losses", help=f"逗号分隔；可选 {','.join(available_losses())}，默认读配置文件"
    )
    parser.add_argument("--device", help="覆盖配置，如 auto/cpu/cuda")
    parser.add_argument("--epochs", type=int, help="覆盖配置")
    parser.add_argument("--seed", type=int, help="覆盖配置")
    return parser


def main() -> None:
    project_dir = Path(__file__).resolve().parents[1]
    args = _parser(project_dir / "configs" / "mlp.json").parse_args()
    with args.config.open(encoding="utf-8") as handle:
        config = json.load(handle)
    train_path, test_path, feature_path = resolve_step4_inputs(
        args.step4_dir, args.train_buffer, args.test_buffer, args.feature_names, args.k
    )
    print(f"加载训练数据: {train_path}", flush=True)
    train_data = load_dmatrix_buffer(train_path)
    print(f"加载测试数据: {test_path}", flush=True)
    test_data = load_dmatrix_buffer(test_path)
    if feature_path:
        feature_count = sum(
            1 for line in feature_path.read_text(encoding="utf-8").splitlines() if line.strip()
        )
        if feature_count != train_data.features.shape[1]:
            raise ValueError(
                f"feature_names 行数 {feature_count} != buffer 列数 {train_data.features.shape[1]}"
            )

    model_name = args.model or config.get("model", "mlp")
    losses = (
        [item.strip() for item in args.losses.split(",") if item.strip()]
        if args.losses else config.get("losses", ["mse"])
    )
    unknown = sorted(set(losses) - set(available_losses()))
    if unknown:
        raise ValueError(f"未知损失: {', '.join(unknown)}")
    training_config = dict(config.get("training", {}))
    for key, value in (("device", args.device), ("epochs", args.epochs), ("seed", args.seed)):
        if value is not None:
            training_config[key] = value
    model_config = dict(config.get("models", {}).get(model_name, {}))
    loss_config = dict(config.get("loss", {}))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_manifest = {
        "train_buffer": str(train_path),
        "test_buffer": str(test_path),
        "feature_names": str(feature_path) if feature_path else None,
        "model": model_name,
        "losses": losses,
        "config": config,
    }
    (args.output_dir / "run_manifest.json").write_text(
        json.dumps(run_manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    summaries = []
    for loss_name in losses:
        experiment_dir = args.output_dir / f"{model_name}__{loss_name}"
        print(f"开始实验: model={model_name}, loss={loss_name}", flush=True)
        summaries.append(
            train_experiment(
                train_data, test_data, model_name, model_config, loss_name,
                training_config, loss_config, experiment_dir,
            )
        )
    fields = sorted({key for row in summaries for key in row})
    with (args.output_dir / "comparison.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summaries)
    (args.output_dir / "comparison.json").write_text(
        json.dumps(summaries, indent=2, ensure_ascii=False, allow_nan=True), encoding="utf-8"
    )
    print(f"全部完成，汇总: {args.output_dir / 'comparison.csv'}", flush=True)
