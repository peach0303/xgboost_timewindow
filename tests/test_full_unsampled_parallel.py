import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.sparse import csr_matrix, save_npz


TIMEWINDOW_SRC = Path(__file__).resolve().parents[1] / "src" / "timewindow"
sys.path.insert(0, str(TIMEWINDOW_SRC))

from full_unsampled_eval import evaluate_case, prepare_tasks, reduce_cases  # noqa: E402
from mlp_core import fit_model, save_checkpoint  # noqa: E402


def _build_fixture(tmp_path: Path):
    labels = {
        "case_train": np.asarray([1.0, 2.0, 4.0, 7.0, 11.0]),
        "case_test": np.asarray([100.0, 101.0, 103.0, 106.0, 110.0]),
    }
    label_dir = tmp_path / "labels"
    level_dir = tmp_path / "level"
    tog_dir = tmp_path / "tog"
    result_dir = tmp_path / "result"
    for directory in (label_dir, level_dir, tog_dir, result_dir):
        directory.mkdir()
    entries = {}
    for stem, values in labels.items():
        label_path = label_dir / f"{stem}.npy"
        np.save(label_path, values)
        entries[stem] = {"label_file": str(label_path)}
        rows = np.column_stack([values, values * 0.0])
        save_npz(level_dir / f"{stem}.after_one.tog.csr.npz", csr_matrix(rows))
        save_npz(tog_dir / f"{stem}.after_one.tog.csr.npz", csr_matrix(rows))
    sample_manifest = tmp_path / "sample_index_manifest.json"
    sample_manifest.write_text(json.dumps({"stems": entries}))
    tw_manifest = tmp_path / "timewindow_manifest.json"
    tw_manifest.write_text(json.dumps({
        "stage": "after_one", "boundary_discard": 1,
        "sample_index_manifest": str(sample_manifest),
        "train_stems": ["case_train"], "test_stems": ["case_test"],
    }))
    namefile = tmp_path / "after_one.namefile.txt"
    namefile.write_text("sig0\nsig1\n")
    feature_names = tmp_path / "feature_names.txt"
    feature_names.write_text("level_o1_sig1\ntog_o1_sig0\n")
    values = np.arange(1.0, 12.0)
    train_x = np.column_stack([np.zeros(values.size), values])
    train_y = values
    matrix = xgb.DMatrix(
        train_x, label=train_y, feature_names=["level_o1_sig1", "tog_o1_sig0"],
    )
    booster = xgb.train({"objective": "reg:squarederror", "max_depth": 2}, matrix, 3)
    booster.save_model(result_dir / "xgbmodel.json")
    mlp_config = {
        "hidden_layer_sizes": [4], "activation": "relu", "alpha": 1e-5,
        "batch_size": 4, "learning_rate_init": 1e-2, "max_iter": 8,
        "random_state": 7,
    }
    mlp, x_scaler, y_scaler = fit_model(
        csr_matrix(train_x), train_y, mlp_config, early_stopping=False,
    )
    save_checkpoint(
        str(result_dir / "mlp_model.joblib"), mlp, x_scaler, y_scaler,
        {"model_type": "sklearn_mlp", "input_features": 2},
    )
    return tw_manifest, level_dir, tog_dir, label_dir, namefile, feature_names, result_dir


def test_parallel_case_evaluation_and_reduce(tmp_path, capsys):
    (
        tw_manifest, level_dir, tog_dir, label_dir, namefile, feature_names, result_dir,
    ) = _build_fixture(tmp_path)
    out_dir = tmp_path / "full_unsampled"
    task_manifest = out_dir / "case_tasks.json"
    prepared = prepare_tasks(str(tw_manifest), str(out_dir))
    assert prepared["num_tasks"] == 2

    for task_index in range(2):
        evaluate_case(
            task_manifest=str(task_manifest), task_index=task_index,
            level_dir=str(level_dir), tog_dir=str(tog_dir), label_dir=str(label_dir),
            raw_namefile=str(namefile), feature_names_path=str(feature_names),
            result_dir=str(result_dir), out_dir=str(out_dir), chunk_rows=2,
        )

    summary = reduce_cases(str(task_manifest), str(out_dir))
    assert summary["num_files"] == 2
    assert summary["num_samples"] == 8
    assert summary["num_jumps"] == 6
    predictions = pd.read_csv(out_dir / "prediction_samples.csv")
    assert predictions.groupby("case_name").size().to_dict() == {
        "case_test": 4, "case_train": 4,
    }
    jumps = pd.read_csv(out_dir / "power_and_delta_detail" / "jump_detail.csv")
    assert jumps.groupby("file_idx").size().to_dict() == {0: 3, 1: 3}
    output = capsys.readouterr().out
    assert "stage=load_tog_source" in output
    assert (out_dir / "evaluation_summary.json").is_file()
    assert (out_dir / "true_vs_pred_scatter.png").is_file()
    assert "stage=load_level_source" in output
    assert "stage=predict_chunk" in output
    assert "stage=build_and_write_jump_detail" in output
    assert "stage=reduce_true_vs_pred_scatter" in output
    assert "stage=reduce_total" in output


def test_mlp_parallel_case_evaluation_uses_saved_checkpoint(tmp_path):
    (
        tw_manifest, level_dir, tog_dir, label_dir, namefile, feature_names, result_dir,
    ) = _build_fixture(tmp_path)
    out_dir = tmp_path / "full_unsampled_mlp"
    prepared = prepare_tasks(str(tw_manifest), str(out_dir))
    for task_index in range(prepared["num_tasks"]):
        summary = evaluate_case(
            task_manifest=str(out_dir / "case_tasks.json"), task_index=task_index,
            level_dir=str(level_dir), tog_dir=str(tog_dir), label_dir=str(label_dir),
            raw_namefile=str(namefile), feature_names_path=str(feature_names),
            result_dir=str(result_dir), out_dir=str(out_dir), model_type="mlp", chunk_rows=2,
        )
        assert summary["model_type"] == "mlp"
    summary = reduce_cases(str(out_dir / "case_tasks.json"), str(out_dir))
    assert summary["num_samples"] == 8
    predictions = pd.read_csv(out_dir / "prediction_samples.csv")
    assert np.isfinite(predictions["pred_label"]).all()


def test_main_pipeline_embeds_case_dsub_and_reduce():
    pipeline = (Path(__file__).resolve().parents[1] / "cmd" / "run_timewindow_pipeline.csh").read_text()
    assert "--skip-full-eval" in pipeline
    assert "run_full_unsampled_eval_dsub.csh" not in pipeline
    assert "full_unsampled_eval.py case" in pipeline
    assert "full_unsampled_eval.py reduce" in pipeline
    assert "case_${case_id}.csh" in pipeline
    assert '--tw-manifest "${STEP3_TW_MANIFEST}"' in pipeline
    assert '--feature-names "${full_feature_names}"' in pipeline
    assert '--result-dir "${full_result_dir}"' in pipeline
    assert '--model-type mlp' in pipeline
    assert 'mlp_full_eval_case_submit' in pipeline
    assert 'MLP_FULL_EVAL_CHUNK_ROWS' in pipeline
