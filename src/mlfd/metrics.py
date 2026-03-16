from __future__ import annotations

import math

import numpy as np


def mse(true: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean(np.square(true - pred)))


def rmse(true: np.ndarray, pred: np.ndarray) -> float:
    return math.sqrt(mse(true, pred))


def nrmse(true: np.ndarray, pred: np.ndarray) -> float:
    denom = float(np.max(true) - np.min(true))
    if denom == 0.0:
        return 0.0
    return rmse(true, pred) / denom


def primary_score(recon_nrmse: float, rollout_nrmse_t100: float, rollout_nrmse_t150: float) -> float:
    return (0.2 * recon_nrmse) + (0.3 * rollout_nrmse_t100) + (0.5 * rollout_nrmse_t150)

