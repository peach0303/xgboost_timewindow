"""可注册的逐样本回归损失。"""
from __future__ import annotations

import math
from typing import Callable

import torch
import torch.nn.functional as functional


LossCallable = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]


def _weighted_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    return torch.sum(values * weights) / torch.clamp(torch.sum(weights), min=1e-12)


def available_losses():
    return ("mse", "mae", "huber", "relative_mae", "relative_mse", "log_cosh")


def build_loss(name: str, *, relative_floor: float = 1e-3, huber_delta: float = 1.0) -> LossCallable:
    """构造损失；所有损失都使用 Step4 buffer 内的样本权重。"""
    if name not in available_losses():
        raise KeyError(f"未知损失 {name!r}；可选值: {', '.join(available_losses())}")
    if relative_floor <= 0:
        raise ValueError("relative_floor 必须大于 0")
    if huber_delta <= 0:
        raise ValueError("huber_delta 必须大于 0")

    def loss(prediction: torch.Tensor, target: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        error = prediction - target
        if name == "mse":
            values = error.square()
        elif name == "mae":
            values = error.abs()
        elif name == "huber":
            values = functional.huber_loss(prediction, target, reduction="none", delta=huber_delta)
        elif name == "relative_mae":
            values = error.abs() / torch.clamp(target.abs(), min=relative_floor)
        elif name == "relative_mse":
            values = (error / torch.clamp(target.abs(), min=relative_floor)).square()
        else:
            absolute = error.abs()
            values = absolute + functional.softplus(-2.0 * absolute) - math.log(2.0)
        return _weighted_mean(values, weight)

    return loss
