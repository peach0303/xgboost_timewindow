#!/usr/bin/env python3
"""Rebuild sampled posttrain evaluation from existing Step6 predictions.

This does not retrain a model or alter the Step3/4 buffers or HPO results.
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import xgboost as xgb

from utils.posttrain_power_eval import (
    build_sampled_prediction_frame,
    restore_train_prediction_order,
    save_anchored_delta_evaluation,
)
from utils.training_policy import shuffled_indices


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate existing Step6 sampled predictions only")
    parser.add_argument("--train-buffer", required=True)
    parser.add_argument("--test-buffer", required=True)
    parser.add_argument("--pred-dir", required=True, help="Existing Step6 pred directory")
    parser.add_argument("--tw-manifest", required=True)
    parser.add_argument("--label-dir", required=True)
    parser.add_argument("--output-dir", required=True, help="Separate directory for recovered sampled evaluation")
    parser.add_argument("--model-type", required=True, choices=["xgboost", "mlp"])
    parser.add_argument("--feature-input-mode", required=True, choices=["proxy_columns", "signal_all_windows"])
    parser.add_argument("--shuffle-seed", type=int, default=108)
    args = parser.parse_args()

    train_y = xgb.DMatrix(args.train_buffer).get_label()
    test_y = xgb.DMatrix(args.test_buffer).get_label()
    if args.model_type == "xgboost":
        # Step6 shuffles the loaded training DMatrix before saving train_predictions.txt.
        train_y = train_y[shuffled_indices(train_y.size, args.shuffle_seed)]
    train_pred = np.loadtxt(os.path.join(args.pred_dir, "train_predictions.txt"), ndmin=1)
    test_pred = np.loadtxt(os.path.join(args.pred_dir, "predictions.txt"), ndmin=1)
    train_y, train_pred = restore_train_prediction_order(
        train_y, train_pred, feature_input_mode=args.feature_input_mode,
        model_type=args.model_type, shuffle_seed=args.shuffle_seed,
    )
    frame = build_sampled_prediction_frame(
        args.tw_manifest, args.label_dir, train_y, train_pred, test_y, test_pred,
    )
    summary = save_anchored_delta_evaluation(frame, args.output_dir)
    print(f"Recovered {summary['num_samples']} sampled predictions in {args.output_dir}")


if __name__ == "__main__":
    main()
