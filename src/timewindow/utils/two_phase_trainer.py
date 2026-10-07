#!/usr/bin/env python3
"""
Two-Phase XGBoost Trainer.

Method A (Column Mask): Phase 1 zeroes out "near" offset columns, Phase 2
  continues training on full columns via xgb_model.
Method B (Residual / base_margin): Phase 1 trains on "far" features only,
  Phase 2 trains residual model on all features with base_margin.
"""
from __future__ import annotations

import re
from collections import defaultdict

import numpy as np
import xgboost as xgb


def extract_offset(feature_name: str) -> int | None:
    """Extract offset number from 'TYPE_oOFFSET_...' name."""
    m = re.match(r"^(?:tog|level)_o(\d+)_", feature_name)
    return int(m.group(1)) if m else None


def get_offset_column_groups(
    feature_names: list[str], cutoff_offset: int
) -> tuple[list[int], list[int]]:
    """
    Split column indices into 'far' (>cutoff) and 'near' (<=cutoff).
    Returns (far_cols, near_cols).
    """
    far_cols, near_cols = [], []
    for i, name in enumerate(feature_names):
        offset = extract_offset(name)
        if offset is not None and offset > cutoff_offset:
            far_cols.append(i)
        else:
            near_cols.append(i)
    return far_cols, near_cols


def zero_columns_csr(mat, zero_cols: set[int]):
    """Return CSR with specified columns zeroed (removed from sparse structure)."""
    col_arr = np.array(mat.indices)
    keep = ~np.isin(col_arr, list(zero_cols))
    new_data = mat.data[keep]
    new_indices = col_arr[keep]
    new_indptr = np.zeros_like(mat.indptr)
    n_rows = mat.shape[0]
    for r in range(n_rows):
        s0, s1 = mat.indptr[r], mat.indptr[r + 1]
        row_keep = keep[s0:s1]
        new_indptr[r + 1] = new_indptr[r] + row_keep.sum()
    from scipy.sparse import csr_matrix
    return csr_matrix((new_data, new_indices, new_indptr), shape=mat.shape)


# ---------------------------------------------------------------------------
# Method A: Column Mask
# ---------------------------------------------------------------------------
class TwoPhaseTrainerA:
    """Phase 1: near columns zeroed. Phase 2: full columns, continued training."""

    def __init__(
        self,
        feature_names: list[str],
        cutoff_offset: int,
        phase1_ratio: float,
        max_rounds: int,
    ):
        _, self.near_cols = get_offset_column_groups(feature_names, cutoff_offset)
        self.near_set = set(self.near_cols)
        self.phase1_rounds = int(max_rounds * phase1_ratio)
        self.phase2_rounds = max_rounds - self.phase1_rounds
        self.model_final: xgb.Booster | None = None

    def train(self, dtrain_full: xgb.DMatrix, dtest_full: xgb.DMatrix, params: dict, objective=None) -> xgb.Booster:
        # Phase 1: zero near columns
        X = dtrain_full.get_data().tocsr()
        Xe = dtest_full.get_data().tocsr()
        X1 = zero_columns_csr(X, self.near_set)
        Xe1 = zero_columns_csr(Xe, self.near_set)
        dt1 = xgb.DMatrix(X1, label=dtrain_full.get_label(), weight=dtrain_full.get_weight())
        de1 = xgb.DMatrix(Xe1, label=dtest_full.get_label(), weight=dtest_full.get_weight())

        model = xgb.train(
            params, dt1, num_boost_round=self.phase1_rounds,
            evals=[(de1, "eval_phase1")], verbose_eval=False, obj=objective,
        )

        # Phase 2: continue on full data
        model = xgb.train(
            params, dtrain_full, num_boost_round=self.phase2_rounds,
            xgb_model=model,
            evals=[(dtest_full, "eval_phase2")], verbose_eval=False, obj=objective,
        )
        self.model_final = model
        return model

    def predict(self, dtest: xgb.DMatrix) -> np.ndarray:
        return self.model_final.predict(dtest)


# ---------------------------------------------------------------------------
# Method B: Residual (base_margin)
# ---------------------------------------------------------------------------
class TwoPhaseTrainerB:
    """Phase 1: far features only. Phase 2: residual model on all features."""

    def __init__(
        self,
        feature_names: list[str],
        cutoff_offset: int,
        phase1_ratio: float,
        max_rounds: int,
    ):
        self.far_cols, _ = get_offset_column_groups(feature_names, cutoff_offset)
        self.phase1_rounds = int(max_rounds * phase1_ratio)
        self.phase2_rounds = max_rounds - self.phase1_rounds
        self.model_far: xgb.Booster | None = None
        self.model_residual: xgb.Booster | None = None

    def train(
        self, dtrain_all: xgb.DMatrix, dtest_all: xgb.DMatrix, params: dict, objective=None
    ) -> dict:
        # Phase 1: train on far columns only
        X_train_far = dtrain_all.get_data().tocsr()[:, self.far_cols]
        X_test_far = dtest_all.get_data().tocsr()[:, self.far_cols]
        dt_far = xgb.DMatrix(X_train_far, label=dtrain_all.get_label(), weight=dtrain_all.get_weight())
        de_far = xgb.DMatrix(X_test_far, label=dtest_all.get_label(), weight=dtest_all.get_weight())

        self.model_far = xgb.train(
            params, dt_far, num_boost_round=self.phase1_rounds,
            evals=[(de_far, "eval_phase1")], verbose_eval=False, obj=objective,
        )

        # Phase 2: residual model on all features with base_margin
        base_train = self.model_far.predict(dt_far)
        base_test = self.model_far.predict(de_far)
        dt_all = xgb.DMatrix(
            dtrain_all.get_data(), label=dtrain_all.get_label(), weight=dtrain_all.get_weight(), base_margin=base_train,
        )
        de_all = xgb.DMatrix(
            dtest_all.get_data(), label=dtest_all.get_label(), weight=dtest_all.get_weight(), base_margin=base_test,
        )

        self.model_residual = xgb.train(
            params, dt_all, num_boost_round=self.phase2_rounds,
            evals=[(de_all, "eval_phase2")], verbose_eval=False, obj=objective,
        )
        return {"far": self.model_far, "residual": self.model_residual}

    def predict(self, dtest_all: xgb.DMatrix) -> np.ndarray:
        X_test_far = dtest_all.get_data().tocsr()[:, self.far_cols]
        de_far = xgb.DMatrix(X_test_far)
        pred_far = self.model_far.predict(de_far)
        de_all = xgb.DMatrix(dtest_all.get_data(), base_margin=pred_far)
        # XGBoost predictions include base_margin, so this is already
        # phase1 prediction + residual trees. Do not add pred_far again.
        return self.model_residual.predict(de_all)
