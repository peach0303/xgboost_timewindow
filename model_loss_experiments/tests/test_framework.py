import numpy as np
import pytest
from scipy.sparse import csr_matrix

torch = pytest.importorskip("torch")

from model_loss.data import RegressionData, resolve_step4_inputs
from model_loss.losses import available_losses, build_loss
from model_loss.trainer import train_experiment


@pytest.mark.parametrize("name", available_losses())
def test_all_losses_are_finite_and_weighted(name):
    prediction = torch.tensor([2.0, 4.0])
    target = torch.tensor([1.0, 2.0])
    weights = torch.tensor([1.0, 3.0])
    value = build_loss(name, relative_floor=0.1, huber_delta=1.0)(prediction, target, weights)
    assert value.ndim == 0
    assert torch.isfinite(value)


def test_resolve_standard_step4_directory(tmp_path):
    screen = tmp_path / "screen_K32"
    screen.mkdir()
    (screen / "train_top32.buffer").touch()
    (screen / "test_top32.buffer").touch()
    (screen / "feature_names.txt").write_text("a\nb\n", encoding="utf-8")
    train, test, names = resolve_step4_inputs(tmp_path, None, None, None, 32)
    assert train.name == "train_top32.buffer"
    assert test.name == "test_top32.buffer"
    assert names.name == "feature_names.txt"


def test_mlp_training_smoke(tmp_path):
    rng = np.random.default_rng(7)
    dense = rng.normal(size=(96, 6)).astype(np.float32)
    dense[np.abs(dense) < 0.4] = 0.0
    labels = (3.0 + 1.8 * dense[:, 0] - 0.7 * dense[:, 2]).astype(np.float32)
    train = RegressionData(csr_matrix(dense[:72]), labels[:72], np.ones(72, dtype=np.float32))
    test = RegressionData(csr_matrix(dense[72:]), labels[72:], np.ones(24, dtype=np.float32))
    summary = train_experiment(
        train,
        test,
        "mlp",
        {"hidden_sizes": [16], "activation": "relu", "dropout": 0.0, "batch_norm": False},
        "mse",
        {
            "epochs": 20,
            "batch_size": 16,
            "learning_rate": 0.01,
            "validation_ratio": 0.2,
            "patience": 6,
            "device": "cpu",
            "seed": 7,
            "log_every": 20,
        },
        {"relative_floor": 0.001, "huber_delta": 1.0},
        tmp_path / "run",
    )
    assert np.isfinite(summary["test_RMSE"])
    assert (tmp_path / "run" / "best_model.pt").is_file()
    assert (tmp_path / "run" / "metrics_test.json").is_file()
    assert (tmp_path / "run" / "predictions_test.csv").is_file()
