from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "cmd"
    / "run_timewindow_neural_hpo.csh"
)
COMMON = SCRIPT.parent / "timewindow_common.csh"


def test_neural_runner_can_skip_hpo_and_use_direct_defaults():
    text = SCRIPT.read_text(encoding="utf-8")
    common = COMMON.read_text(encoding="utf-8")

    assert 'source "${SCRIPT_DIR}/timewindow_common.csh"' in text
    assert 'if ( ! $?NEURAL_PYTHON ) set NEURAL_PYTHON = "/home/data/chenyiheng/conda/envs/hw-model-train/bin/python"' in common
    assert '"$NEURAL_PYTHON" "${TIMEWINDOW_SRC}/step5_hpo_neural.py"' in text
    assert '"$NEURAL_PYTHON" "${TIMEWINDOW_SRC}/step6_retrain_eval.py"' in text
    assert 'if ( ! $?SKIP_NEURAL_HPO ) set SKIP_NEURAL_HPO = 0' in common
    assert 'SKIP_NEURAL_HPO must be 0 (run HPO) or 1 (direct retrain)' in text
    assert 'if ( "$SKIP_NEURAL_HPO" == "0" ) then' in text
    assert '--hpo-result "${HPO_OUT}/best_trial.json"' in text
    assert '_${MODEL_TYPE}_direct' in text

    expected_defaults = {
        "BNN": {
            "BATCH_SIZE": "256",
            "LEARNING_RATE": "0.001",
            "HIDDEN_SIZE": "128",
            "NUM_LAYERS": "2",
            "DROPOUT": "0.1",
        },
        "LSTM": {
            "BATCH_SIZE": "128",
            "LEARNING_RATE": "0.0005",
            "HIDDEN_SIZE": "128",
            "NUM_LAYERS": "2",
            "DROPOUT": "0.1",
        },
        "LTC": {
            "BATCH_SIZE": "128",
            "LEARNING_RATE": "0.001",
            "HIDDEN_SIZE": "128",
            "NUM_LAYERS": "1",
            "DROPOUT": "0.1",
        },
    }
    for model, defaults in expected_defaults.items():
        for name, value in defaults.items():
            assert (
                f"if ( ! $?{model}_{name} ) "
                f"set {model}_{name} = {value}"
            ) in common

    for option in (
        "--epochs",
        "--batch-size",
        "--learning-rate",
        "--hidden-size",
        "--num-layers",
        "--dropout",
        "--weight-decay",
        "--patience",
        "--validation-ratio",
        "--gradient-clip",
        "--sequence-cache-mb",
        "--mixed-precision",
    ):
        assert option in text
    assert '"bnn_fp32"' in text
