"""Metrics and ensemble-weight fitting."""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize


def rmsle(y_true, y_pred) -> float:
    """Root Mean Squared Logarithmic Error (predictions clipped at 0)."""
    y_pred = np.clip(np.asarray(y_pred, dtype=float), 0, None)
    y_true = np.asarray(y_true, dtype=float)
    return float(np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_true)) ** 2)))


def fit_blend_weights(pred_matrix: np.ndarray, y_true) -> np.ndarray:
    """Non-negative weights summing to 1 that minimise RMSLE of the blend.

    ``pred_matrix`` has shape (n_samples, n_models) and holds predictions in dollars.
    """
    n_models = pred_matrix.shape[1]

    def loss(w):
        return rmsle(y_true, pred_matrix @ w)

    result = minimize(
        loss,
        x0=np.full(n_models, 1.0 / n_models),
        method="SLSQP",
        bounds=[(0.0, 1.0)] * n_models,
        constraints=[{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}],
    )
    weights = np.clip(result.x, 0.0, None)
    return weights / weights.sum()
