import json
import sys
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix, save_npz


TIMEWINDOW_SRC = Path(__file__).resolve().parents[1] / "src" / "timewindow"
sys.path.insert(0, str(TIMEWINDOW_SRC))

from utils.posttrain_power_eval import (  # noqa: E402
    build_anchored_jump_detail,
    build_sampled_prediction_frame,
    evaluate_full_unsampled,
    restore_train_prediction_order,
)
from utils.training_policy import shuffled_indices  # noqa: E402

def test_restore_train_prediction_order_for_each_branch():
    original = np.arange(23, dtype=float) + 0.25
    predictions = original + np.arange(23) * 0.1
    permutation = shuffled_indices(len(original), 108)
    counts = {
        ("xgboost", "proxy_columns"): 3,
        ("xgboost", "signal_all_windows"): 1,
        ("mlp", "proxy_columns"): 2,
        ("mlp", "signal_all_windows"): 0,
    }
    for (model, mode), count in counts.items():
        row_order = np.arange(len(original))
        for _ in range(count):
            row_order = row_order[permutation]
        actual_true, actual_pred = restore_train_prediction_order(
            original[row_order], predictions[row_order],
            feature_input_mode=mode, model_type=model, shuffle_seed=108,
        )
        np.testing.assert_array_equal(actual_true, original)
        np.testing.assert_allclose(actual_pred, predictions)



def _write_sample_manifest(tmp_path, labels, boundary=1):
    label_dir = tmp_path / "labels"
    index_dir = tmp_path / "indices"
    label_dir.mkdir()
    index_dir.mkdir()
    stems = ["case_train", "case_test"]
    entries = {}
    for stem, values, indices, split in (
        (stems[0], labels[0], [0, 2], "train"),
        (stems[1], labels[1], [1, 3], "test"),
    ):
        label_path = label_dir / f"{stem}.npy"
        index_path = index_dir / f"{stem}.sample_idx.npy"
        np.save(label_path, values)
        np.save(index_path, np.asarray(indices, dtype=np.int64))
        entries[stem] = {
            "split": split,
            "label_file": str(label_path),
            "index_file": str(index_path),
            "available_rows": len(values) - boundary,
            "sampled_rows": len(indices),
        }
    sample_manifest = tmp_path / "sample_index_manifest.json"
    sample_manifest.write_text(json.dumps({"sampled": True, "stems": entries}))
    tw_manifest = tmp_path / "timewindow_manifest.json"
    tw_manifest.write_text(json.dumps({
        "stage": "after_one",
        "boundary_discard": boundary,
        "sample_index_manifest": str(sample_manifest),
        "train_stems": [stems[0]],
        "test_stems": [stems[1]],
    }))
    return tw_manifest, label_dir


def test_sampled_delta_uses_true_previous_cycle(tmp_path):
    labels = [
        np.asarray([10.0, 12.0, 20.0, 25.0, 40.0]),
        np.asarray([5.0, 8.0, 14.0, 15.0, 30.0]),
    ]
    tw_manifest, label_dir = _write_sample_manifest(tmp_path, labels)
    frame = build_sampled_prediction_frame(
        str(tw_manifest), str(label_dir),
        train_true=np.asarray([12.0, 25.0]),
        train_pred=np.asarray([13.0, 23.0]),
        test_true=np.asarray([14.0, 30.0]),
        test_pred=np.asarray([16.0, 27.0]),
    )
    assert frame["cycle_index"].tolist() == [1, 3, 2, 4]
    assert frame["true_prev_label"].tolist() == [10.0, 20.0, 8.0, 15.0]

    jumps = build_anchored_jump_detail(frame)
    assert jumps["pred_prev"].tolist() == jumps["real_prev"].tolist()
    np.testing.assert_allclose(
        jumps["abs_jump_error"].values,
        np.abs(frame["pred_label"].values - frame["true_label"].values),
    )


def test_full_evaluation_never_crosses_case_boundary(tmp_path):
    labels = [
        np.asarray([1.0, 2.0, 4.0, 7.0, 11.0]),
        np.asarray([100.0, 101.0, 103.0, 106.0, 110.0]),
    ]
    tw_manifest, label_dir = _write_sample_manifest(tmp_path, labels)
    level_dir = tmp_path / "level"
    tog_dir = tmp_path / "tog"
    level_dir.mkdir()
    tog_dir.mkdir()
    for stem, values in zip(("case_train", "case_test"), labels):
        rows = np.column_stack([values, values * 0.0])
        save_npz(level_dir / f"{stem}.after_one.tog.csr.npz", csr_matrix(rows))
        save_npz(tog_dir / f"{stem}.after_one.tog.csr.npz", csr_matrix(rows))
    namefile = tmp_path / "after_one.namefile.txt"
    namefile.write_text("sig0\nsig1\n")

    def predict_sparse(X):
        return np.asarray(X[:, 0].toarray()).reshape(-1)

    def power_metrics(y, pred):
        return {"n": int(len(y)), "mae": float(np.mean(np.abs(y - pred)))}

    summary = evaluate_full_unsampled(
        tw_manifest_path=str(tw_manifest),
        level_dir=str(level_dir),
        tog_dir=str(tog_dir),
        label_dir=str(label_dir),
        raw_namefile=str(namefile),
        feature_names=["level_o1_sig0"],
        predict_sparse=predict_sparse,
        out_dir=str(tmp_path / "full_eval"),
        power_evaluator=power_metrics,
    )
    assert summary["num_samples"] == 8
    assert summary["num_files"] == 2
    assert summary["num_jumps"] == 6
