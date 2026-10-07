from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_main_pipeline_wires_optional_mlp_branch():
    pipeline = (ROOT / "cmd" / "run_timewindow_pipeline.csh").read_text(
        encoding="utf-8"
    )
    assert 'if ( ! $?skip_mlp_train ) set skip_mlp_train = 1' in pipeline
    assert 'if ( "$skip_mlp_train" == "0" ) then' in pipeline
    assert 'step5_hpo_mlp.py' in pipeline
    assert 'step6_retrain_eval_mlp.py' in pipeline
    assert 'step5_mlp_hpo' in pipeline
    assert 'step6_mlp_retrain' in pipeline
    assert 'best_trial.json' in pipeline
    assert 'hpo_progress.json' in pipeline
    assert '--skip-full-eval' in pipeline
    assert '--model-type mlp' in pipeline
    assert 'mlp_full_eval_case_submit' in pipeline
    assert 'eval/full_unsampled/evaluation_summary.json' in pipeline
