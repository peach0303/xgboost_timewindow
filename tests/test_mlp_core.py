import sys
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "timewindow"))

from mlp_core import connection_importance, fit_model, predict, safe_mape


def test_small_sparse_regression_and_importance():
    rng = np.random.default_rng(7)
    dense = rng.normal(size=(120, 5)).astype(np.float32)
    dense[np.abs(dense) < 0.5] = 0
    labels = 2.5 * dense[:, 1] - 1.5 * dense[:, 3] + 0.2
    config = {
        "hidden_layer_sizes": [16], "activation": "relu", "alpha": 1e-5,
        "batch_size": 32, "learning_rate_init": 1e-2, "max_iter": 80,
        "n_iter_no_change": 10, "validation_fraction": 0.2, "random_state": 7,
    }
    model, x_scaler, y_scaler = fit_model(csr_matrix(dense), labels, config)
    predictions = predict(model, x_scaler, y_scaler, csr_matrix(dense))
    assert predictions.shape == labels.shape
    assert np.isfinite(predictions).all()
    assert safe_mape(labels + 10, predictions + 10) < 20
    importance = connection_importance(model, [f"x{i}" for i in range(5)])
    assert len(importance) == 5
    assert np.isclose(sum(score for _, score in importance), 1.0)
