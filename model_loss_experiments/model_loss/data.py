"""Step4 DMatrix buffer 的发现与加载。"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from scipy.sparse import csr_matrix


@dataclass
class RegressionData:
    features: csr_matrix
    labels: np.ndarray
    weights: np.ndarray


def load_dmatrix_buffer(path: Path) -> RegressionData:
    """加载 Step4 生成的 XGBoost binary buffer，并保留样本权重。"""
    try:
        import xgboost as xgb
    except ImportError as exc:
        raise RuntimeError("读取 Step4 buffer 需要安装 xgboost") from exc

    matrix = xgb.DMatrix(str(path))
    features = matrix.get_data().tocsr().astype(np.float32)
    labels = matrix.get_label().astype(np.float32)
    weights = matrix.get_weight().astype(np.float32)
    if not weights.size:
        weights = np.ones(labels.size, dtype=np.float32)
    if features.shape[0] != labels.size or labels.size != weights.size:
        raise ValueError(
            f"{path} 行数不一致: X={features.shape[0]}, y={labels.size}, w={weights.size}"
        )
    if labels.size == 0:
        raise ValueError(f"{path} 不包含样本")
    if not np.isfinite(labels).all() or not np.isfinite(weights).all():
        raise ValueError(f"{path} 的标签或权重包含 NaN/Inf")
    if np.any(weights < 0) or float(weights.sum()) <= 0:
        raise ValueError(f"{path} 的样本权重必须非负且总和大于 0")
    return RegressionData(features, labels, weights)


def _unique_match(directory: Path, pattern: str, role: str) -> Path:
    matches = sorted(directory.glob(pattern))
    if len(matches) != 1:
        names = ", ".join(str(item) for item in matches) or "无"
        raise ValueError(f"{directory} 中{role}文件应恰好有 1 个（{pattern}），实际为: {names}")
    return matches[0]


def resolve_step4_inputs(
    step4_dir: Optional[Path],
    train_buffer: Optional[Path],
    test_buffer: Optional[Path],
    feature_names: Optional[Path],
    k: Optional[int] = None,
) -> Tuple[Path, Path, Optional[Path]]:
    """解析标准 screen_K* 或混合 union_K* 目录，也支持显式 buffer 路径。"""
    if train_buffer or test_buffer:
        if not train_buffer or not test_buffer:
            raise ValueError("--train-buffer 与 --test-buffer 必须同时给出")
        base = train_buffer.parent
    else:
        if step4_dir is None:
            raise ValueError("需要 --step4-dir，或同时给出 --train-buffer/--test-buffer")
        base = step4_dir
        if k is not None:
            candidates = [base / f"union_K{k}", base / f"screen_K{k}"]
            existing = [item for item in candidates if item.is_dir()]
            if len(existing) != 1:
                raise ValueError(
                    f"K={k} 时应唯一匹配 union_K{k} 或 screen_K{k}，实际: "
                    + (", ".join(str(item) for item in existing) or "无")
                )
            base = existing[0]
        train_buffer = _unique_match(base, "train*.buffer", "训练 buffer")
        test_buffer = _unique_match(base, "test*.buffer", "测试 buffer")

    train_buffer = train_buffer.resolve()
    test_buffer = test_buffer.resolve()
    for path in (train_buffer, test_buffer):
        if not path.is_file():
            raise FileNotFoundError(path)
    if feature_names is None:
        candidate = base / "feature_names.txt"
        feature_names = candidate if candidate.is_file() else None
    elif not feature_names.is_file():
        raise FileNotFoundError(feature_names)
    return train_buffer, test_buffer, feature_names.resolve() if feature_names else None
