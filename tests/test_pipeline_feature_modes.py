from __future__ import annotations

import json
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix, save_npz

try:
    import xgboost as xgb
except ModuleNotFoundError:
    import pytest
    pytest.skip("xgboost is required for feature-mode tests", allow_module_level=True)


ROOT = Path(__file__).resolve().parents[1]
TIMEWINDOW = ROOT / "src" / "timewindow"
sys.path.insert(0, str(TIMEWINDOW))

from step4_hybrid_screen_features import build_union  # noqa: E402


def _write_combo(tmp_path: Path, combo_key: str, offset: int) -> dict:
    combo_dir = tmp_path / combo_key
    combo_dir.mkdir()
    paths: dict[str, Path] = {}
    for split, matrix, labels, weights in (
        ("train", [[1.0, 10.0], [2.0, 20.0]], [1.0, 2.0], [0.5, 1.5]),
        ("test", [[3.0, 30.0]], [3.0], [2.0]),
    ):
        feature = combo_dir / f"case.{split}.feature.npz"
        label = combo_dir / f"case.{split}.label.npy"
        weight = combo_dir / f"case.{split}.weight.npy"
        save_npz(feature, csr_matrix(np.asarray(matrix)))
        np.save(label, np.asarray(labels))
        np.save(weight, np.asarray(weights))
        (combo_dir / f"{split}_feature.f").write_text(f"{feature}\n")
        (combo_dir / f"{split}_label.f").write_text(f"{label}\n")
        paths[split] = feature
    return {
        "data_type": "tog",
        "offset": offset,
        "filelist_dir": str(combo_dir),
    }


def test_single_selector_can_expand_proxy_signal_to_all_windows(tmp_path: Path) -> None:
    namefile = tmp_path / "signals.txt"
    namefile.write_text("signal_0\nsignal_1\n")
    tw_manifest = {
        "train_stems": ["case.train"],
        "test_stems": ["case.test"],
        "combinations": OrderedDict([
            ("tog_offset2", _write_combo(tmp_path, "tog_offset2", 2)),
            ("tog_offset1", _write_combo(tmp_path, "tog_offset1", 1)),
        ]),
    }
    selector_data = OrderedDict([
        ("mcp", {
            "train": csr_matrix([[0.1, 0.2], [0.3, 0.4]]),
            "train_y": np.asarray([1.0, 2.0]),
            "train_weight": np.asarray([0.5, 1.5]),
            "test": csr_matrix([[0.5, 0.6]]),
            "test_y": np.asarray([3.0]),
            "test_weight": np.asarray([2.0]),
            "names": ["tog_o2_signal_0", "tog_o2_signal_1"],
        }),
    ])
    rankings = OrderedDict([
        ("mcp", [("tog_o2_signal_0", 2.0, 0), ("tog_o2_signal_1", 1.0, 1)]),
    ])

    output_dir = tmp_path / "screen"
    build_union(
        str(output_dir), 1, rankings, selector_data,
        "signal_all_windows", tw_manifest, str(namefile),
    )

    union_dir = output_dir / "union_K1"
    meta = json.loads((union_dir / "union_meta.json").read_text())
    assert meta["feature_input_mode"] == "signal_all_windows"
    assert meta["selected_selectors"] == ["mcp"]
    assert meta["selected_proxy_columns"] == 1
    assert meta["selected_signals"] == ["signal_0"]
    assert meta["union_columns"] == 2
    assert (union_dir / "feature_names.txt").read_text().splitlines() == [
        "tog_o2_signal_0",
        "tog_o1_signal_0",
    ]
    train = xgb.DMatrix(str(union_dir / "train_union_top1.buffer"))
    test = xgb.DMatrix(str(union_dir / "test_union_top1.buffer"))
    assert train.num_col() == 2
    assert test.num_col() == 2
    np.testing.assert_allclose(train.get_weight(), [0.5, 1.5])
    np.testing.assert_allclose(test.get_weight(), [2.0])
