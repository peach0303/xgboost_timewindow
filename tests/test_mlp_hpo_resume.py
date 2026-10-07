import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "src" / "timewindow" / "step5_hpo_mlp.py"


def _run_hpo(train_buffer: Path, test_buffer: Path, names: Path, output: Path, trials: int):
    subprocess.run(
        [
            sys.executable, str(SCRIPT),
            "--train-buffer", str(train_buffer),
            "--test-buffer", str(test_buffer),
            "--feature-names", str(names),
            "--output-dir", str(output),
            "--epochs", "3", "--trial-num", str(trials),
            "--pop-size", "2", "--job-num", "1", "--nfold", "2",
            "--sampler", "Random", "--patience", "2",
            "--validation-ratio", "0.2", "--seed", "17",
        ],
        check=True,
    )


def test_hpo_persists_checkpoints_and_resumes_to_new_target(tmp_path):
    rng = np.random.default_rng(11)
    features = rng.normal(size=(48, 3)).astype(np.float32)
    labels = (2.0 * features[:, 0] - features[:, 1] + 0.5).astype(np.float32)
    train_buffer = tmp_path / "train.buffer"
    test_buffer = tmp_path / "test.buffer"
    matrix = xgb.DMatrix(features, label=labels)
    matrix.save_binary(train_buffer)
    matrix.save_binary(test_buffer)
    names = tmp_path / "feature_names.txt"
    names.write_text("f0\nf1\nf2\n")
    output = tmp_path / "hpo"

    _run_hpo(train_buffer, test_buffer, names, output, 2)
    first = pd.read_csv(output / "mlp_trials.csv")
    assert len(first) == 2
    assert (output / "optuna_study.db").is_file()
    assert (output / "best_trial.json").is_file()

    _run_hpo(train_buffer, test_buffer, names, output, 4)
    resumed = pd.read_csv(output / "mlp_trials.csv")
    assert len(resumed) == 4
    assert sorted(resumed["number"].tolist()) == [0, 1, 2, 3]
    progress = json.loads((output / "hpo_progress.json").read_text())
    assert progress["finished_trials"] == 4
    assert progress["target_trials"] == 4
    assert progress["checkpoint_reason"] == "complete"

    _run_hpo(train_buffer, test_buffer, names, output, 4)
    skipped = pd.read_csv(output / "mlp_trials.csv")
    assert len(skipped) == 4
    assert sorted(skipped["number"].tolist()) == [0, 1, 2, 3]
