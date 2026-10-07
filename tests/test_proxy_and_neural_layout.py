from argparse import Namespace
import json

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from src.timewindow.step2_proxy_select import select_features
from src.timewindow.utils.neural_models import (
    _batch_to_numpy, _make_models, build_sequence_layout, load_neural_hpo_config,
)


def selector_args(selector: str) -> Namespace:
    return Namespace(
        selector=selector, top_k=3, alpha=1e-4, l1_ratio=0.8,
        max_iter=200, tol=1e-3, seed=7, stability_fraction=0.6,
        stability_candidates=4, stability_runs=3, stability_threshold=0.6,
        prefilter_k=6, max_dense_elements=100_000, mi_neighbors=3,
        n_estimators=12, max_depth=4, min_samples_leaf=1,
        max_features=0.8, n_jobs=1, mrmr_pool_k=5,
        mrmr_redundancy=0.5,
    )


def test_all_selectors_return_valid_top_k():
    rng = np.random.default_rng(7)
    dense = rng.integers(0, 2, size=(80, 8)).astype(np.float64)
    y = 3.0 * dense[:, 1] - 2.0 * dense[:, 4] + rng.normal(0, 0.05, 80)
    x = csr_matrix(dense)
    for selector in (
        "f_regression", "mutual_info", "elastic_net", "stability",
        "extra_trees", "mrmr",
    ):
        selected, scores, _ = select_features(x, y, selector_args(selector))
        assert selected.shape == (3,)
        assert len(set(selected.tolist())) == 3
        assert np.all((selected >= 0) & (selected < x.shape[1]))
        assert scores.shape == (x.shape[1],)


def test_sequence_layout_uses_offset_order_and_zero_fills_missing_slots():
    names = [
        "tog_o2_sig_a", "level_o2_sig_a", "tog_o1_sig_a",
        "level_o1_sig_a", "tog_o2_sig_b",
    ]
    layout = build_sequence_layout(names)
    assert layout["offsets"] == [2, 1]
    assert layout["missing_slots"] == 1
    x = csr_matrix(np.array([[10, 20, 30, 40, 50]], dtype=np.float32))
    sequence = _batch_to_numpy(x, np.array([0]), layout)
    assert sequence.shape == (1, 2, 3)
    # level:sig_b is absent at both offsets; only tog:sig_b exists at offset 2.
    channel = layout["channels"].index("tog:sig_b")
    assert sequence[0, 0, channel] == 50
    assert sequence[0, 1, channel] == 0


def test_load_neural_hpo_config_filters_metadata_and_checks_model(tmp_path):
    result = tmp_path / "best_trial.json"
    result.write_text(json.dumps({
        "model_type": "bnn",
        "params": {
            "model_type": "bnn", "hidden_size": 64, "batch_size": 32,
            "learning_rate": 0.001, "unknown_metadata": "ignored",
        },
    }), encoding="utf-8")
    config = load_neural_hpo_config(str(result), "bnn")
    assert config == {
        "hidden_size": 64, "batch_size": 32, "learning_rate": 0.001,
    }
    with pytest.raises(ValueError, match="does not match"):
        load_neural_hpo_config(str(result), "lstm")


def test_optional_torch_model_forward_shapes():
    torch = pytest.importorskip("torch")
    BinaryMLP, FullPrecisionBinaryTopology, LSTMRegressor, LTCRegressor = _make_models(torch)

    bnn = BinaryMLP(input_size=6, hidden_size=8, dropout=0.0)
    assert bnn(torch.randn(4, 6)).shape == (4,)
    fp32 = FullPrecisionBinaryTopology(input_size=6, hidden_size=8, dropout=0.0)
    assert fp32(torch.randn(4, 6)).shape == (4,)
    assert all(parameter.dtype == torch.float32 for parameter in fp32.parameters())

    sequence = torch.randn(4, 3, 6)
    lstm = LSTMRegressor(
        input_size=6, hidden_size=8, num_layers=2, dropout=0.1
    )
    assert lstm(sequence).shape == (4,)

    ltc = LTCRegressor(input_size=6, hidden_size=8, dropout=0.0)
    assert ltc(sequence).shape == (4,)
