"""Optional PyTorch backends for time-window power regression.

PyTorch is imported lazily so the established XGBoost path has no new runtime
dependency.  LSTM/LTC inputs are reconstructed from feature names rather than
assuming that flattened buffer columns already form a sequence.
"""
from __future__ import annotations

import copy
import json
import re
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


FEATURE_RE = re.compile(r"^(tog|level)_o(\d+)_(.+)$")


@dataclass
class NeuralConfig:
    model_type: str
    epochs: int = 100
    batch_size: int = 128
    learning_rate: float = 1e-3
    hidden_size: int = 128
    num_layers: int = 1
    dropout: float = 0.1
    weight_decay: float = 1e-5
    patience: int = 12
    validation_ratio: float = 0.15
    gradient_clip: float = 1.0
    sequence_cache_mb: int = 512
    mixed_precision: str = "auto"
    seed: int = 108
    device: str = "auto"


def require_torch():
    try:
        import torch
    except ImportError as exc:
        raise SystemExit(
            "ERROR: neural model requested but PyTorch is unavailable. "
            "Install a site-compatible torch build, then rerun with "
            "--model-type bnn|bnn_fp32|lstm|ltc. The xgboost path remains dependency-free."
        ) from exc
    return torch


def load_neural_hpo_config(path: str, expected_model_type: str | None = None) -> dict[str, Any]:
    """Load and validate the JSON artifact emitted by neural HPO."""
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    model_type = payload.get("model_type")
    params = payload.get("params")
    if not isinstance(params, dict):
        raise ValueError(f"invalid neural HPO result (missing params object): {path}")
    if expected_model_type and model_type and model_type != expected_model_type:
        raise ValueError(
            f"HPO model_type {model_type!r} does not match requested model {expected_model_type!r}"
        )
    allowed = {
        "epochs", "batch_size", "learning_rate", "hidden_size", "num_layers",
        "dropout", "weight_decay", "patience", "validation_ratio",
        "gradient_clip", "sequence_cache_mb", "mixed_precision", "seed", "device",
    }
    return {key: value for key, value in params.items() if key in allowed}


def build_sequence_layout(feature_names: list[str]) -> dict[str, Any]:
    parsed = []
    for col, name in enumerate(feature_names):
        match = FEATURE_RE.match(name)
        if match:
            parsed.append((col, match.group(1), int(match.group(2)), match.group(3)))
    if not parsed:
        raise ValueError("no time-window feature names match TYPE_oOFFSET_signal")
    # Larger offset is older. Feed oldest -> newest to recurrent models.
    offsets = sorted({item[2] for item in parsed}, reverse=True)
    channels = sorted({(item[1], item[3]) for item in parsed})
    offset_pos = {value: idx for idx, value in enumerate(offsets)}
    channel_pos = {value: idx for idx, value in enumerate(channels)}
    column_map = np.full((len(offsets), len(channels)), -1, dtype=np.int64)
    for col, dtype, offset, signal in parsed:
        column_map[offset_pos[offset], channel_pos[(dtype, signal)]] = col
    return {
        "offsets": offsets,
        "channels": [f"{dtype}:{signal}" for dtype, signal in channels],
        "column_map": column_map,
        "matched_columns": len(parsed),
        "missing_slots": int(np.sum(column_map < 0)),
    }


def _batch_to_numpy(x, rows: np.ndarray, layout: dict[str, Any] | None) -> np.ndarray:
    xb = x[rows]
    if layout is None:
        return xb.toarray().astype(np.float32, copy=False)
    column_map = layout["column_map"]
    flat_map = column_map.reshape(-1)
    valid = flat_map >= 0
    out = np.zeros((rows.size, flat_map.size), dtype=np.float32)
    if np.any(valid):
        # One sparse column gather is substantially faster than one gather per
        # time step, especially inside the LSTM/LTC epoch loop.
        out[:, valid] = xb[:, flat_map[valid]].toarray()
    return out.reshape(rows.size, column_map.shape[0], column_map.shape[1])


def _make_models(torch):
    nn = torch.nn
    functional = torch.nn.functional

    class BinarySign(torch.autograd.Function):
        @staticmethod
        def forward(ctx, value):
            ctx.save_for_backward(value)
            return torch.where(value >= 0, torch.ones_like(value), -torch.ones_like(value))

        @staticmethod
        def backward(ctx, grad_output):
            (value,) = ctx.saved_tensors
            return grad_output * (value.abs() <= 1).to(grad_output.dtype)

    class BinaryLinear(nn.Linear):
        def forward(self, value):
            binary_weight = BinarySign.apply(self.weight)
            return functional.linear(value, binary_weight, self.bias)

    class BinaryActivation(nn.Module):
        def forward(self, value):
            return BinarySign.apply(torch.clamp(value, -1.0, 1.0))

    class BinaryMLP(nn.Module):
        def __init__(self, input_size, hidden_size, dropout):
            super().__init__()
            self.layers = nn.Sequential(
                BinaryLinear(input_size, hidden_size), nn.LayerNorm(hidden_size), BinaryActivation(),
                nn.Dropout(dropout), BinaryLinear(hidden_size, hidden_size),
                nn.LayerNorm(hidden_size), BinaryActivation(), nn.Linear(hidden_size, 1),
            )

        def forward(self, value):
            # Binary hidden activations and weights use STE; output stays real.
            return self.layers(value).squeeze(-1)

    class FullPrecisionBinaryTopology(nn.Module):
        """FP32 control model with the same topology as BinaryMLP."""
        def __init__(self, input_size, hidden_size, dropout):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Linear(input_size, hidden_size), nn.LayerNorm(hidden_size),
                nn.Hardtanh(-1.0, 1.0), nn.Dropout(dropout),
                nn.Linear(hidden_size, hidden_size), nn.LayerNorm(hidden_size),
                nn.Hardtanh(-1.0, 1.0), nn.Linear(hidden_size, 1),
            )

        def forward(self, value):
            return self.layers(value).squeeze(-1)

    class LSTMRegressor(nn.Module):
        def __init__(self, input_size, hidden_size, num_layers, dropout):
            super().__init__()
            self.lstm = nn.LSTM(
                input_size, hidden_size, num_layers=num_layers, batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
            self.head = nn.Sequential(nn.LayerNorm(hidden_size), nn.Linear(hidden_size, 1))

        def forward(self, value):
            output, _ = self.lstm(value)
            return self.head(output[:, -1]).squeeze(-1)

    class LTCCell(nn.Module):
        """Euler-discretized varying-time-constant recurrent cell."""
        def __init__(self, input_size, hidden_size):
            super().__init__()
            self.drive = nn.Linear(input_size + hidden_size, hidden_size)
            self.tau_gate = nn.Linear(input_size + hidden_size, hidden_size)
            self.hidden_size = hidden_size

        def forward(self, value, state):
            joined = torch.cat((value, state), dim=-1)
            drive = torch.tanh(self.drive(joined))
            tau = 0.1 + 1.9 * torch.sigmoid(self.tau_gate(joined))
            return torch.tanh(state + (-state + drive) / tau)

    class LTCRegressor(nn.Module):
        def __init__(self, input_size, hidden_size, dropout):
            super().__init__()
            self.cell = LTCCell(input_size, hidden_size)
            self.dropout = nn.Dropout(dropout)
            self.head = nn.Sequential(nn.LayerNorm(hidden_size), nn.Linear(hidden_size, 1))

        def forward(self, value):
            state = value.new_zeros((value.shape[0], self.cell.hidden_size))
            for time_idx in range(value.shape[1]):
                state = self.cell(value[:, time_idx], state)
            return self.head(self.dropout(state)).squeeze(-1)

    return BinaryMLP, FullPrecisionBinaryTopology, LSTMRegressor, LTCRegressor


def train_neural_regressor(
    x_train, y_train: np.ndarray, x_test, feature_names: list[str] | None,
    config: NeuralConfig, checkpoint_path: str | None, log,
    return_predictor: bool = False, predict_training: bool = True,
):
    torch = require_torch()
    BinaryMLP, FullPrecisionBinaryTopology, LSTMRegressor, LTCRegressor = _make_models(torch)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    device_name = config.device
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)

    layout = None
    if config.model_type in ("lstm", "ltc"):
        if not feature_names:
            raise ValueError(f"--feature-names is required for {config.model_type}")
        layout = build_sequence_layout(feature_names)
        input_size = len(layout["channels"])
        log(
            f"sequence layout: offsets={layout['offsets']}, channels={input_size}, "
            f"missing_slots={layout['missing_slots']}"
        )
    else:
        input_size = x_train.shape[1]

    if config.model_type == "bnn":
        model = BinaryMLP(input_size, config.hidden_size, config.dropout)
    elif config.model_type == "bnn_fp32":
        model = FullPrecisionBinaryTopology(input_size, config.hidden_size, config.dropout)
    elif config.model_type == "lstm":
        model = LSTMRegressor(input_size, config.hidden_size, config.num_layers, config.dropout)
    elif config.model_type == "ltc":
        model = LTCRegressor(input_size, config.hidden_size, config.dropout)
    else:
        raise ValueError(f"unsupported neural model: {config.model_type}")
    model.to(device)
    parameter_count = int(sum(parameter.numel() for parameter in model.parameters()))

    rng = np.random.default_rng(config.seed)
    order = rng.permutation(y_train.size)
    val_n = max(1, int(round(y_train.size * config.validation_ratio)))
    val_idx, fit_idx = order[:val_n], order[val_n:]
    if fit_idx.size == 0:
        raise ValueError("not enough training rows for a validation split")
    y_mean = float(np.mean(y_train[fit_idx]))
    y_std = max(float(np.std(y_train[fit_idx])), 1e-8)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay,
    )
    criterion = torch.nn.SmoothL1Loss()

    sequence_cache = {}
    if layout is not None and config.sequence_cache_mb > 0:
        row_bytes = int(layout["column_map"].size) * np.dtype(np.float32).itemsize
        required_bytes = int(x_train.shape[0] + x_test.shape[0]) * row_bytes
        limit_bytes = int(config.sequence_cache_mb) * 1024 * 1024
        if required_bytes <= limit_bytes:
            sequence_cache[id(x_train)] = _batch_to_numpy(
                x_train, np.arange(x_train.shape[0]), layout
            )
            sequence_cache[id(x_test)] = _batch_to_numpy(
                x_test, np.arange(x_test.shape[0]), layout
            )
            log(
                f"sequence cache enabled: {required_bytes / 1024**2:.1f} MiB "
                f"<= {config.sequence_cache_mb} MiB"
            )
        else:
            log(
                f"sequence cache skipped: {required_bytes / 1024**2:.1f} MiB "
                f"> {config.sequence_cache_mb} MiB"
            )

    use_amp = device.type == "cuda" and config.model_type in ("lstm", "ltc") and (
        config.mixed_precision == "on"
        or (config.mixed_precision == "auto" and torch.cuda.is_available())
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    def numpy_batch(x, rows):
        cached = sequence_cache.get(id(x))
        return cached[rows] if cached is not None else _batch_to_numpy(x, rows, layout)

    def tensor_batch(x, y, rows):
        values = torch.from_numpy(numpy_batch(x, rows)).to(device)
        labels = torch.from_numpy(((y[rows] - y_mean) / y_std).astype(np.float32)).to(device)
        return values, labels

    def validation_loss():
        model.eval()
        total, count = 0.0, 0
        with torch.no_grad():
            for start in range(0, val_idx.size, config.batch_size):
                rows = val_idx[start:start + config.batch_size]
                values, labels = tensor_batch(x_train, y_train, rows)
                with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                    batch_loss = criterion(model(values), labels)
                total += float(batch_loss) * rows.size
                count += rows.size
        return total / max(1, count)

    best_state, best_loss, stale, trained_epochs = None, float("inf"), 0, 0
    for epoch in range(1, config.epochs + 1):
        model.train()
        shuffled = rng.permutation(fit_idx)
        total = 0.0
        for start in range(0, shuffled.size, config.batch_size):
            rows = shuffled[start:start + config.batch_size]
            values, labels = tensor_batch(x_train, y_train, rows)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                loss = criterion(model(values), labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
            scaler.step(optimizer)
            scaler.update()
            total += float(loss.detach()) * rows.size
        val_loss = validation_loss()
        trained_epochs = epoch
        if epoch == 1 or epoch % 10 == 0:
            log(f"epoch={epoch}, train_loss={total/fit_idx.size:.6g}, val_loss={val_loss:.6g}")
        if val_loss < best_loss - 1e-7:
            best_loss, stale = val_loss, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
            if stale >= config.patience:
                log(f"early stop at epoch={epoch}; best_val_loss={best_loss:.6g}")
                break
    model.load_state_dict(best_state)

    def predict(x) -> np.ndarray:
        model.eval()
        parts = []
        all_rows = np.arange(x.shape[0])
        with torch.no_grad():
            for start in range(0, x.shape[0], config.batch_size):
                rows = all_rows[start:start + config.batch_size]
                values = torch.from_numpy(numpy_batch(x, rows)).to(device)
                with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                    batch_prediction = model(values)
                parts.append(batch_prediction.float().cpu().numpy())
        return np.concatenate(parts).astype(np.float64) * y_std + y_mean

    serializable_layout = None
    if layout is not None:
        serializable_layout = {key: value.tolist() if isinstance(value, np.ndarray) else value for key, value in layout.items()}
    if checkpoint_path:
        torch.save({
            "state_dict": model.state_dict(), "config": asdict(config),
            "input_size": input_size, "label_mean": y_mean, "label_std": y_std,
            "sequence_layout": serializable_layout, "best_validation_loss": best_loss,
            "trained_epochs": trained_epochs, "parameter_count": parameter_count,
        }, checkpoint_path)
    metadata = {
        "model_type": config.model_type, "device": str(device), "input_size": input_size,
        "best_validation_loss": best_loss, "trained_epochs": trained_epochs,
        "sequence_layout": serializable_layout, "parameter_count": parameter_count,
        "mixed_precision": use_amp, "sequence_cache_mb": config.sequence_cache_mb,
    }
    train_prediction = predict(x_train) if predict_training else None
    result = (train_prediction, predict(x_test), metadata)
    if return_predictor:
        return (*result, predict)
    return result
