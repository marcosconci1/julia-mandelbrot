"""
Combinatorial Purged Cross-Validation (CPCV) for regime/strategy evaluation.

Why CPCV instead of plain walk-forward
--------------------------------------
Arian, Norouzi & Seco (2024), "Backtest overfitting in the machine
learning era," *Knowledge-Based Systems*, ran a synthetic-controlled
head-to-head of K-fold, Purged K-fold, Walk-Forward, and CPCV across
hundreds of strategies. CPCV had the lowest Probability of Backtest
Overfitting (PBO) and the highest Deflated Sharpe Ratio test statistic.
Walk-forward, while popular, was **markedly worse than CPCV** for
backtest-overfitting control.

We keep ``walk_forward_adaptive.py`` for production-deployment realism
(it answers "would this have worked live?") but add CPCV here for
evaluation discipline (it answers "is this any better than chance?").

Mechanics
---------
- Partition T observations into N contiguous folds of equal size.
- Choose K folds to use as the *test set* per path; the remaining N-K
  folds are training.
- The number of distinct test-set selections is ``C(N, K)`` — for the
  typical N=6, K=2 setup that's 15 paths.
- **Purge**: drop training observations whose label horizon overlaps any
  test fold. Prevents leakage from "future" training labels touching
  test inputs through the label horizon.
- **Embargo**: skip a buffer of observations after each test fold before
  training resumes. Prevents serial-correlation leakage.

References
----------
- Arian, Norouzi & Seco (2024), KBS. *The* empirical validation that
  CPCV dominates walk-forward on backtest-overfitting control.
- López de Prado (2018) ch. 12 — original CPCV methodology.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Iterator, List, Tuple

import numpy as np


@dataclass(frozen=True)
class CPCVFold:
    """One CPCV fold: train indices, test indices, path identifier."""

    train: np.ndarray
    test: np.ndarray
    path_id: int


def _fold_boundaries(n_obs: int, n_folds: int) -> List[Tuple[int, int]]:
    """Equal-sized contiguous folds; last fold absorbs any remainder.

    Returns a list of (start, stop) tuples with right-open intervals.
    """
    if n_folds < 2:
        raise ValueError(f"n_folds must be >= 2, got {n_folds}")
    if n_obs < n_folds:
        raise ValueError(f"Need n_obs >= n_folds; got {n_obs} obs, {n_folds} folds")

    fold_size = n_obs // n_folds
    bounds = [(i * fold_size, (i + 1) * fold_size) for i in range(n_folds - 1)]
    bounds.append(((n_folds - 1) * fold_size, n_obs))
    return bounds


def _apply_purge_and_embargo(
    train_idx: np.ndarray,
    test_folds: List[Tuple[int, int]],
    label_horizon: int,
    embargo: int,
) -> np.ndarray:
    """Remove training indices that overlap test-fold label horizons or
    fall within the embargo window after any test fold.

    For each test fold [a, b):
    - Purge: drop train indices in [a - label_horizon, b)
    - Embargo: drop train indices in [b, b + embargo)
    """
    mask = np.ones_like(train_idx, dtype=bool)
    for (a, b) in test_folds:
        purge_start = a - label_horizon
        purge_stop = b
        embargo_stop = b + embargo
        mask &= ~((train_idx >= purge_start) & (train_idx < purge_stop))
        mask &= ~((train_idx >= b) & (train_idx < embargo_stop))
    return train_idx[mask]


def generate_cpcv_folds(
    n_obs: int,
    n_folds: int = 6,
    n_test_folds: int = 2,
    label_horizon: int = 0,
    embargo: int = 0,
) -> Iterator[CPCVFold]:
    """Yield CPCV train/test fold pairs.

    Parameters
    ----------
    n_obs
        Total number of time-ordered observations.
    n_folds
        Number of contiguous folds partitioning the data. Default 6.
    n_test_folds
        Number of folds used as the test set per path. Default 2.
        Number of paths returned is C(n_folds, n_test_folds).
    label_horizon
        Forward look-ahead of the labels in observations. E.g. if your
        label is "5-day forward return", pass 5. Training rows within
        ``label_horizon`` observations *before* a test fold are purged
        because their labels straddle into the test region.
    embargo
        Additional buffer (in observations) after each test fold before
        training resumes. Mitigates serial-correlation leakage.

    Yields
    ------
    CPCVFold(train_idx, test_idx, path_id)
    """
    if n_test_folds < 1 or n_test_folds >= n_folds:
        raise ValueError(
            f"Need 1 <= n_test_folds < n_folds; got {n_test_folds}/{n_folds}"
        )
    if label_horizon < 0 or embargo < 0:
        raise ValueError("label_horizon and embargo must be non-negative")

    bounds = _fold_boundaries(n_obs, n_folds)
    all_idx = np.arange(n_obs)

    for path_id, test_combo in enumerate(combinations(range(n_folds), n_test_folds)):
        test_ranges = [bounds[i] for i in test_combo]
        test_idx = np.concatenate(
            [np.arange(a, b) for (a, b) in test_ranges]
        )

        train_mask = np.ones(n_obs, dtype=bool)
        for (a, b) in test_ranges:
            train_mask[a:b] = False
        train_candidates = all_idx[train_mask]

        train_idx = _apply_purge_and_embargo(
            train_candidates, test_ranges, label_horizon, embargo
        )

        yield CPCVFold(train=train_idx, test=test_idx, path_id=path_id)


def cpcv_path_count(n_folds: int, n_test_folds: int) -> int:
    """Number of distinct CPCV paths for a given (n_folds, n_test_folds)."""
    from math import comb
    return comb(n_folds, n_test_folds)


def probability_of_backtest_overfitting(
    sharpe_in_sample: np.ndarray,
    sharpe_out_sample: np.ndarray,
) -> float:
    """Bailey, Borwein, López de Prado, Zhu (2017) PBO statistic.

    Computes the probability that the best in-sample-Sharpe strategy
    ranks below median out-of-sample. The standard interpretation is:
    PBO > 0.5 means selecting by in-sample performance is *worse than
    random* on the held-out data.

    Parameters
    ----------
    sharpe_in_sample
        Array shape (n_trials, n_paths) of in-sample Sharpe ratios.
    sharpe_out_sample
        Array shape (n_trials, n_paths), aligned with ``sharpe_in_sample``.

    Returns
    -------
    PBO in [0, 1]. Lower is better; > 0.5 indicates overfit selection.
    """
    sharpe_in_sample = np.asarray(sharpe_in_sample, dtype=float)
    sharpe_out_sample = np.asarray(sharpe_out_sample, dtype=float)
    if sharpe_in_sample.shape != sharpe_out_sample.shape:
        raise ValueError(
            f"Shape mismatch: {sharpe_in_sample.shape} vs {sharpe_out_sample.shape}"
        )
    if sharpe_in_sample.ndim != 2:
        raise ValueError(f"Expected 2D arrays, got {sharpe_in_sample.ndim}D")

    n_paths = sharpe_in_sample.shape[1]
    if n_paths < 2:
        raise ValueError("Need at least 2 paths to compute PBO")

    best_in = np.argmax(sharpe_in_sample, axis=0)
    ranks_out = np.argsort(np.argsort(sharpe_out_sample, axis=0), axis=0)
    median_rank = (sharpe_out_sample.shape[0] - 1) / 2.0

    chosen_ranks = ranks_out[best_in, np.arange(n_paths)]
    return float(np.mean(chosen_ranks < median_rank))
