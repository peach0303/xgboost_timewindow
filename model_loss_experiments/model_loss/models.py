"""模型注册表；增加模型时只需注册新的构造函数。"""
from __future__ import annotations

from typing import Any, Dict

from torch import nn


ACTIVATIONS = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "tanh": nn.Tanh,
    "silu": nn.SiLU,
}


class MLPRegressor(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_sizes=(256, 128),
        activation: str = "relu",
        dropout: float = 0.0,
        batch_norm: bool = False,
    ) -> None:
        super().__init__()
        if not hidden_sizes or any(int(width) <= 0 for width in hidden_sizes):
            raise ValueError("hidden_sizes 必须是正整数列表")
        if activation not in ACTIVATIONS:
            raise ValueError(f"未知 activation={activation!r}")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout 必须位于 [0, 1)")
        layers = []
        previous = input_dim
        for width in hidden_sizes:
            width = int(width)
            layers.append(nn.Linear(previous, width))
            if batch_norm:
                layers.append(nn.BatchNorm1d(width))
            layers.append(ACTIVATIONS[activation]())
            if dropout:
                layers.append(nn.Dropout(dropout))
            previous = width
        layers.append(nn.Linear(previous, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, features):
        return self.network(features).squeeze(-1)


def available_models():
    return ("mlp",)


def build_model(name: str, input_dim: int, config: Dict[str, Any]) -> nn.Module:
    if name == "mlp":
        return MLPRegressor(input_dim=input_dim, **config)
    raise KeyError(f"未知模型 {name!r}；可选值: {', '.join(available_models())}")
