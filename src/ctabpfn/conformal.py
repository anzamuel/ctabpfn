"""
Conformalized TabPFN prediction intervals.

Each variant splits the incoming train part 75/25 at the given seed, reads native quantiles for the calibration and test rows through `ctabpfn.inference`, and calibrates a symmetric split-conformal interval with one exact order statistic, no search and no refitting. `additive` shifts the native central interval outward by the CQR score's order statistic, `multiplicative` scales the two half widths around the native median, and `quantile_level` calibrates the quantile level itself, distributional conformal prediction on the native CDF. All three carry the finite-sample marginal coverage guarantee and return lower, upper, and the native median as center.
"""

import math

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.model_selection import train_test_split

from ctabpfn.inference import GRID, predict_quantiles

__all__ = ["additive", "multiplicative", "quantile_level"]


def _column(level: float) -> int:
    idx = int(np.argmin(np.abs(GRID - level)))
    if abs(GRID[idx] - level) > 1e-9:
        raise ValueError(f"quantile level {level} is not on the inference grid")
    return idx


def _prepare(
    X_train: ArrayLike, y_train: ArrayLike, X_test: ArrayLike, seed: int
) -> tuple[NDArray[np.float32], NDArray[np.float64], NDArray[np.float32]]:
    X_fit, X_cal, y_fit, y_cal = train_test_split(
        X_train, y_train, test_size=0.25, random_state=seed
    )
    q_cal = predict_quantiles(X_fit, y_fit, X_cal, seed)
    q_test = predict_quantiles(X_fit, y_fit, X_test, seed)
    return q_cal, np.asarray(y_cal, dtype=np.float64), q_test


def _order_statistic(scores: NDArray[np.float64], alpha: float) -> float:
    rank = math.ceil((1.0 - alpha) * (scores.size + 1))
    if rank > scores.size:
        return math.inf  # calibration set too small for the target coverage
    return float(np.sort(scores)[rank - 1])


def additive(
    X_train: ArrayLike,
    y_train: ArrayLike,
    X_test: ArrayLike,
    coverage: float,
    seed: int,
) -> tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
    """Shift the native central interval outward by the conformal CQR margin."""
    alpha = 1.0 - coverage
    lo, hi = _column(alpha / 2.0), _column(1.0 - alpha / 2.0)
    q_cal, y_cal, q_test = _prepare(X_train, y_train, X_test, seed)
    scores = np.maximum(q_cal[:, lo] - y_cal, y_cal - q_cal[:, hi])
    margin = _order_statistic(scores, alpha)
    return q_test[:, lo] - margin, q_test[:, hi] + margin, q_test[:, _column(0.5)]


def multiplicative(
    X_train: ArrayLike,
    y_train: ArrayLike,
    X_test: ArrayLike,
    coverage: float,
    seed: int,
) -> tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
    """Scale the native half widths around the median by the conformal ratio."""
    alpha = 1.0 - coverage
    lo, hi, mid = _column(alpha / 2.0), _column(1.0 - alpha / 2.0), _column(0.5)
    q_cal, y_cal, q_test = _prepare(X_train, y_train, X_test, seed)
    median = q_cal[:, mid].astype(np.float64)
    scores = np.maximum(
        (median - y_cal) / (median - q_cal[:, lo]),
        (y_cal - median) / (q_cal[:, hi] - median),
    )
    gamma = _order_statistic(scores, alpha)
    center = q_test[:, mid]
    return (
        center - gamma * (center - q_test[:, lo]),
        center + gamma * (q_test[:, hi] - center),
        center,
    )


def quantile_level(
    X_train: ArrayLike,
    y_train: ArrayLike,
    X_test: ArrayLike,
    coverage: float,
    seed: int,
) -> tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
    """Calibrate the native quantile level itself and read the interval off the grid."""
    alpha = 1.0 - coverage
    q_cal, y_cal, q_test = _prepare(X_train, y_train, X_test, seed)
    lower_cols = np.arange(GRID.size // 2)
    upper_cols = GRID.size - 1 - lower_cols
    covered = (q_cal[:, lower_cols] <= y_cal[:, None]) & (
        y_cal[:, None] <= q_cal[:, upper_cols]
    )
    covering = np.where(covered, lower_cols, -1)
    deepest = covering.max(axis=1)  # -1 when even the widest level misses
    rank = math.floor(alpha * (deepest.size + 1))
    chosen = int(np.sort(deepest)[rank - 1]) if rank >= 1 else -1
    chosen = max(chosen, 0)  # widest level when the guarantee wants wider
    center = q_test[:, _column(0.5)]
    return q_test[:, lower_cols[chosen]], q_test[:, upper_cols[chosen]], center
