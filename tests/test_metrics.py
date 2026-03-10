from __future__ import annotations

import numpy as np

from mlfd.metrics import mse, nrmse, primary_score, rmse


def test_basic_metrics() -> None:
    true = np.array([0.0, 1.0, 2.0], dtype=np.float32)
    pred = np.array([0.0, 2.0, 1.0], dtype=np.float32)
    assert np.isclose(mse(true, pred), 2.0 / 3.0)
    assert np.isclose(rmse(true, pred), np.sqrt(2.0 / 3.0))
    assert np.isclose(nrmse(true, pred), np.sqrt(2.0 / 3.0) / 2.0)


def test_primary_score_weighting() -> None:
    score = primary_score(0.1, 0.2, 0.3)
    assert np.isclose(score, 0.23)
