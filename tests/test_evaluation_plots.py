import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "timewindow"))

from utils.evaluation_plots import save_true_vs_pred_scatter


def test_true_vs_pred_scatter_is_written(tmp_path):
    true_values = np.linspace(1.0, 10.0, 100)
    predicted_values = true_values + np.sin(true_values) * 0.1
    output = tmp_path / "true_vs_pred_scatter.png"
    save_true_vs_pred_scatter(true_values, predicted_values, output, r2=0.99, max_points=25)
    assert output.is_file()
    assert output.stat().st_size > 0
