"""Tests for juliams.regimes.cpcv — Combinatorial Purged CV.

Validates:
- Fold count matches C(N, K)
- Purge removes overlap between train labels and test inputs
- Embargo respected
- Test sets across paths cover every observation, with each obs appearing
  in the test set the same number of times (C(N-1, K-1) times)
- PBO statistic semantics
"""

from __future__ import annotations

import numpy as np
import pytest

from juliams.regimes.cpcv import (
    cpcv_path_count,
    generate_cpcv_folds,
    probability_of_backtest_overfitting,
)


def test_fold_count_matches_combination():
    """C(6, 2) = 15 paths."""
    folds = list(generate_cpcv_folds(n_obs=300, n_folds=6, n_test_folds=2))
    assert len(folds) == cpcv_path_count(6, 2) == 15


def test_each_observation_appears_in_test_C_n_minus_1_choose_k_minus_1_times():
    """Combinatorial identity: each obs is in a test fold in exactly
    C(N-1, K-1) of the C(N, K) paths."""
    n_obs, n_folds, n_test = 600, 6, 2
    folds = list(generate_cpcv_folds(n_obs, n_folds, n_test))
    test_count = np.zeros(n_obs, dtype=int)
    for f in folds:
        test_count[f.test] += 1
    # Expected: C(5, 1) = 5 for every obs.
    expected = cpcv_path_count(n_folds - 1, n_test - 1)
    # Allow off-by-one for the remainder in the last fold.
    fold_size = n_obs // n_folds
    interior = test_count[: (n_folds - 1) * fold_size]
    assert (interior == expected).all(), (
        f"Each obs should appear in {expected} test sets; got "
        f"min={interior.min()} max={interior.max()}"
    )


def test_train_and_test_are_disjoint():
    """A fold's train and test indices must never overlap."""
    folds = list(generate_cpcv_folds(n_obs=400, n_folds=5, n_test_folds=2))
    for f in folds:
        assert len(np.intersect1d(f.train, f.test)) == 0


def test_purge_removes_labels_overlapping_test():
    """With label_horizon=10, training observations within 10 of any
    test-fold start must be purged."""
    n_obs = 300
    fold_size = n_obs // 6
    folds = list(
        generate_cpcv_folds(
            n_obs, n_folds=6, n_test_folds=2, label_horizon=10, embargo=0
        )
    )
    for f in folds:
        for t_start in [f.test[0]] + [
            f.test[i] for i in range(1, len(f.test)) if f.test[i] != f.test[i - 1] + 1
        ]:
            # No training index should be within [t_start - 10, t_start).
            purged_zone = (f.train >= t_start - 10) & (f.train < t_start)
            assert not purged_zone.any(), (
                f"Path {f.path_id}: training indices found in purge zone "
                f"[{t_start - 10}, {t_start})"
            )


def test_embargo_skips_observations_after_test():
    """With embargo=5, no training index may fall within 5 obs after the
    end of any test fold."""
    n_obs = 300
    folds = list(
        generate_cpcv_folds(n_obs, n_folds=6, n_test_folds=2, label_horizon=0, embargo=5)
    )
    fold_size = n_obs // 6
    for f in folds:
        test = f.test
        # Find right edges of contiguous test runs.
        breaks = np.where(np.diff(test) > 1)[0]
        edges = np.concatenate([test[breaks], test[-1:]]) + 1
        for edge in edges:
            embargo_zone = (f.train >= edge) & (f.train < edge + 5)
            assert not embargo_zone.any(), (
                f"Path {f.path_id}: training indices found in embargo zone "
                f"[{edge}, {edge + 5})"
            )


def test_invalid_inputs_raise():
    with pytest.raises(ValueError):
        list(generate_cpcv_folds(n_obs=100, n_folds=1))
    with pytest.raises(ValueError):
        list(generate_cpcv_folds(n_obs=100, n_folds=6, n_test_folds=6))
    with pytest.raises(ValueError):
        list(generate_cpcv_folds(n_obs=100, n_folds=6, n_test_folds=0))
    with pytest.raises(ValueError):
        list(generate_cpcv_folds(n_obs=3, n_folds=6))
    with pytest.raises(ValueError):
        list(generate_cpcv_folds(n_obs=100, label_horizon=-1))


def test_pbo_perfect_overfit_is_one():
    """If in-sample ranking is reversed out-of-sample, PBO = 1."""
    n_trials, n_paths = 5, 10
    sharpe_is = np.tile(np.arange(n_trials, dtype=float)[:, None], (1, n_paths))
    # Out-of-sample: reversed — best in-sample is worst out-of-sample.
    sharpe_oos = -sharpe_is
    pbo = probability_of_backtest_overfitting(sharpe_is, sharpe_oos)
    assert pbo == pytest.approx(1.0)


def test_pbo_perfect_carry_is_zero():
    """If in-sample ranking matches out-of-sample perfectly, PBO = 0."""
    n_trials, n_paths = 5, 10
    sharpe_is = np.tile(np.arange(n_trials, dtype=float)[:, None], (1, n_paths))
    sharpe_oos = sharpe_is.copy()
    pbo = probability_of_backtest_overfitting(sharpe_is, sharpe_oos)
    assert pbo == pytest.approx(0.0)


def test_pbo_random_is_near_half():
    """Random shuffling should produce PBO near 0.5."""
    rng = np.random.default_rng(0)
    n_trials, n_paths = 10, 50
    sharpe_is = rng.standard_normal((n_trials, n_paths))
    sharpe_oos = rng.standard_normal((n_trials, n_paths))
    pbo = probability_of_backtest_overfitting(sharpe_is, sharpe_oos)
    assert 0.3 < pbo < 0.7


def test_pbo_shape_mismatch_raises():
    with pytest.raises(ValueError):
        probability_of_backtest_overfitting(
            np.ones((5, 3)), np.ones((4, 3))
        )


def test_pbo_single_path_raises():
    with pytest.raises(ValueError):
        probability_of_backtest_overfitting(
            np.ones((5, 1)), np.ones((5, 1))
        )
