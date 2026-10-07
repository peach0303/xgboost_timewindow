"""固定配置训练器：训练集划分验证集，测试集仅在训练完成后评估。"""
from __future__ import annotations

import copy
import csv
import json
import random
import time
from pathlib import Path
from typing import Any, Dict, Iterator, Tuple

import numpy as np
import torch
from scipy.sparse import csr_matrix
from sklearn.preprocessing import StandardScaler

from .data import RegressionData
from .losses import build_loss
from .metrics import regression_metrics
from .models import build_model


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("请求了 CUDA，但当前 PyTorch 无法使用 CUDA")
    return device


def _batches(
    features: csr_matrix,
    labels: np.ndarray,
    weights: np.ndarray,
    batch_size: int,
    device: torch.device,
    indices: np.ndarray,
) -> Iterator[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    for start in range(0, indices.size, batch_size):
        selected = indices[start:start + batch_size]
        dense = features[selected].toarray().astype(np.float32, copy=False)
        yield (
            torch.from_numpy(dense).to(device),
            torch.from_numpy(labels[selected]).to(device),
            torch.from_numpy(weights[selected]).to(device),
        )


def _predict(model, features, batch_size, device, label_mean, label_scale) -> np.ndarray:
    model.eval()
    outputs = []
    indices = np.arange(features.shape[0])
    dummy = np.zeros(features.shape[0], dtype=np.float32)
    with torch.no_grad():
        for batch_x, _, _ in _batches(features, dummy, np.ones_like(dummy), batch_size, device, indices):
            scaled = model(batch_x)
            outputs.append((scaled * label_scale + label_mean).cpu().numpy())
    return np.concatenate(outputs)


def _write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=True)


def _write_predictions(path: Path, labels, predictions, weights) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["label", "prediction", "error", "absolute_error", "sample_weight"])
        for target, prediction, weight in zip(labels, predictions, weights):
            error = float(prediction - target)
            writer.writerow([float(target), float(prediction), error, abs(error), float(weight)])


def train_experiment(
    train_data: RegressionData,
    test_data: RegressionData,
    model_name: str,
    model_config: Dict[str, Any],
    loss_name: str,
    training_config: Dict[str, Any],
    loss_config: Dict[str, Any],
    output_dir: Path,
) -> Dict[str, Any]:
    """训练单个“模型 × 损失”组合并保存可复现产物。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    if train_data.features.shape[1] != test_data.features.shape[1]:
        raise ValueError("训练与测试 buffer 的特征列数不一致")

    seed = int(training_config.get("seed", 108))
    set_seed(seed)
    device = select_device(str(training_config.get("device", "auto")))
    validation_ratio = float(training_config.get("validation_ratio", 0.15))
    if not 0.0 < validation_ratio < 1.0:
        raise ValueError("validation_ratio 必须位于 (0, 1)")
    row_count = train_data.labels.size
    if row_count < 3:
        raise ValueError("训练集至少需要 3 行")
    validation_size = min(row_count - 1, max(1, int(round(row_count * validation_ratio))))
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(row_count)
    validation_indices = permutation[:validation_size]
    fit_indices = permutation[validation_size:]

    scaler = StandardScaler(with_mean=False, copy=True)
    scaler.fit(train_data.features[fit_indices])
    # 全量矩阵保持 CSR，仅在 mini-batch 内转为稠密 Tensor。
    all_train_features = scaler.transform(train_data.features).tocsr().astype(np.float32)
    test_features = scaler.transform(test_data.features).tocsr().astype(np.float32)

    fit_labels = train_data.labels[fit_indices]
    label_mean = float(np.mean(fit_labels))
    label_scale = float(np.std(fit_labels))
    if label_scale < 1e-12:
        label_scale = 1.0

    model = build_model(model_name, all_train_features.shape[1], model_config).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_config.get("learning_rate", 1e-3)),
        weight_decay=float(training_config.get("weight_decay", 0.0)),
    )
    loss_function = build_loss(loss_name, **loss_config)
    batch_size = int(training_config.get("batch_size", 256))
    epochs = int(training_config.get("epochs", 100))
    patience = int(training_config.get("patience", 12))
    min_delta = float(training_config.get("min_delta", 0.0))
    gradient_clip = float(training_config.get("gradient_clip", 0.0))
    log_every = max(1, int(training_config.get("log_every", 1)))
    if batch_size < 1 or epochs < 1 or patience < 0:
        raise ValueError("batch_size/epochs 必须为正数，patience 必须非负")

    best_loss = float("inf")
    best_epoch = 0
    best_state = None
    stale_epochs = 0
    history = []
    started = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        shuffled = rng.permutation(fit_indices)
        train_total = 0.0
        train_weight = 0.0
        for batch_x, batch_y, batch_w in _batches(
            all_train_features, train_data.labels, train_data.weights,
            batch_size, device, shuffled,
        ):
            optimizer.zero_grad(set_to_none=True)
            raw_prediction = model(batch_x) * label_scale + label_mean
            loss = loss_function(raw_prediction, batch_y, batch_w)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"epoch {epoch} 出现非有限训练损失")
            loss.backward()
            if gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            optimizer.step()
            weight_sum = float(batch_w.sum().item())
            train_total += float(loss.item()) * weight_sum
            train_weight += weight_sum

        model.eval()
        validation_total = 0.0
        validation_weight = 0.0
        with torch.no_grad():
            for batch_x, batch_y, batch_w in _batches(
                all_train_features, train_data.labels, train_data.weights,
                batch_size, device, validation_indices,
            ):
                raw_prediction = model(batch_x) * label_scale + label_mean
                loss = loss_function(raw_prediction, batch_y, batch_w)
                weight_sum = float(batch_w.sum().item())
                validation_total += float(loss.item()) * weight_sum
                validation_weight += weight_sum
        train_loss = train_total / train_weight
        validation_loss = validation_total / validation_weight
        history.append({"epoch": epoch, "train_loss": train_loss, "validation_loss": validation_loss})
        if epoch == 1 or epoch % log_every == 0:
            print(
                f"[{model_name}/{loss_name}] epoch={epoch}/{epochs} "
                f"train={train_loss:.6g} val={validation_loss:.6g}",
                flush=True,
            )
        if validation_loss < best_loss - min_delta:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if patience > 0 and stale_epochs >= patience:
                print(f"[{model_name}/{loss_name}] early stop at epoch {epoch}", flush=True)
                break
    if best_state is None:
        raise RuntimeError("训练未产生有效 checkpoint")
    model.load_state_dict(best_state)

    train_prediction = _predict(
        model, all_train_features, batch_size, device, label_mean, label_scale
    )
    test_prediction = _predict(model, test_features, batch_size, device, label_mean, label_scale)
    relative_floor = float(loss_config.get("relative_floor", 1e-3))
    train_metrics = regression_metrics(
        train_data.labels, train_prediction, train_data.weights, relative_floor
    )
    test_metrics = regression_metrics(
        test_data.labels, test_prediction, test_data.weights, relative_floor
    )
    elapsed = time.time() - started
    metadata = {
        "model": model_name,
        "loss": loss_name,
        "input_features": int(all_train_features.shape[1]),
        "train_rows": int(row_count),
        "test_rows": int(test_data.labels.size),
        "fit_rows": int(fit_indices.size),
        "validation_rows": int(validation_indices.size),
        "parameter_count": int(parameter_count),
        "best_epoch": best_epoch,
        "best_validation_loss": best_loss,
        "elapsed_seconds": elapsed,
        "device": str(device),
        "seed": seed,
        "model_config": model_config,
        "training_config": training_config,
        "loss_config": loss_config,
    }
    torch.save(
        {
            "model_state_dict": {key: value.cpu() for key, value in model.state_dict().items()},
            "metadata": metadata,
            "feature_scale": scaler.scale_,
            "label_mean": label_mean,
            "label_scale": label_scale,
        },
        output_dir / "best_model.pt",
    )
    _write_json(output_dir / "metadata.json", metadata)
    _write_json(output_dir / "metrics_train.json", train_metrics)
    _write_json(output_dir / "metrics_test.json", test_metrics)
    with (output_dir / "history.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch", "train_loss", "validation_loss"])
        writer.writeheader()
        writer.writerows(history)
    _write_predictions(
        output_dir / "predictions_test.csv", test_data.labels, test_prediction, test_data.weights
    )
    return {**metadata, **{f"test_{key}": value for key, value in test_metrics.items()}}
