"""Tests for juliams.regimes.bocpd — Bayesian Online CPD.

Falsifiable claims:
- On synthetic mean-shift data, MAP run length resets within a few
  observations of the true change point.
- On stationary data, MAP run length grows monotonically (no false CPs).
- The detector is causal: future poisoning leaves past output unchanged.
- Run-length posterior is a valid distribution (rows sum to ~1).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from juliams.regimes.bocpd import detect_change_points_bocpd


def test_detects_mean_shift_within_few_steps():
    rng = np.random.default_rng(0)
    s = pd.Series(np.concatenate([rng.normal(0, 1, 100), rng.normal(2, 1, 100)]))
    res = detect_change_points_bocpd(s, expected_run_length=50)
    rl = res.map_run_length.values

    # Run length grows up to t=100 (approx 100 at index 99).
    assert rl[99] >= 90, f"Pre-CP MAP run length too small: {rl[99]}"
    # Within 10 steps after the CP, MAP run length must drop dramatically.
    post_cp_min = rl[100:110].min()
    assert post_cp_min < 20, (
        f"BOCPD failed to reset run length after CP; min in [100,110] = {post_cp_min}"
    )


def test_detects_variance_shift():
    rng = np.random.default_rng(1)
    # Same mean, very different variance.
    s = pd.Series(np.concatenate([rng.normal(0, 0.5, 150), rng.normal(0, 3.0, 150)]))
    res = detect_change_points_bocpd(s, expected_run_length=80)
    rl = res.map_run_length.values
    assert rl[149] >= 100  # well-grown before CP
    assert rl[150:175].min() < 25  # reset after CP


def test_no_change_on_stationary_series_lets_run_length_grow():
    rng = np.random.default_rng(2)
    s = pd.Series(rng.normal(0, 1, 300))
    res = detect_change_points_bocpd(s, expected_run_length=200)
    rl = res.map_run_length.values
    # Last 50 observations should mostly have large MAP run lengths.
    assert (rl[-50:] > 100).mean() > 0.8, (
        f"On stationary data the MAP run length should grow; "
        f"got mean={rl[-50:].mean():.1f}"
    )


def test_posterior_rows_sum_to_one():
    rng = np.random.default_rng(3)
    s = pd.Series(rng.normal(0, 1, 100))
    res = detect_change_points_bocpd(s, expected_run_length=50)
    row_sums = res.run_length_posterior.sum(axis=1)
    np.testing.assert_allclose(row_sums, 1.0, atol=1e-10)


def test_posterior_is_lower_triangular_in_run_length():
    """At time t the run length cannot exceed t. Entries beyond t must
    be zero."""
    rng = np.random.default_rng(4)
    n = 50
    s = pd.Series(rng.normal(0, 1, n))
    res = detect_change_points_bocpd(s)
    post = res.run_length_posterior
    for t in range(n):
        # Entries at columns > t+1 must be zero.
        assert (post[t, t + 2 :] == 0).all(), f"Row {t} has support beyond run length t+1"


def test_causal_no_future_leakage():
    """Output at time t must depend only on data through index t."""
    rng = np.random.default_rng(5)
    s = pd.Series(rng.normal(0, 1, 200))
    res_orig = detect_change_points_bocpd(s, expected_run_length=100)

    s_pert = s.copy()
    s_pert.iloc[150:] = 999.0
    res_pert = detect_change_points_bocpd(s_pert, expected_run_length=100)

    # MAP run length and change probability up to index 149 must match.
    np.testing.assert_array_equal(
        res_orig.map_run_length.iloc[:150].values,
        res_pert.map_run_length.iloc[:150].values,
    )
    np.testing.assert_allclose(
        res_orig.change_probability.iloc[:150].values,
        res_pert.change_probability.iloc[:150].values,
    )


def test_invalid_inputs_raise():
    s = pd.Series(np.zeros(50))
    with pytest.raises(ValueError):
        detect_change_points_bocpd(s, expected_run_length=0)
    with pytest.raises(ValueError):
        detect_change_points_bocpd(s, prior_kappa=-1)
    with pytest.raises(ValueError):
        detect_change_points_bocpd(s, prior_alpha=0)
    with pytest.raises(ValueError):
        detect_change_points_bocpd(s, prior_beta=0)


def test_handles_nan_input():
    s = pd.Series([np.nan, np.nan, 0.1, 0.2, 0.3, 5.0, 5.1, 5.2])
    res = detect_change_points_bocpd(s, expected_run_length=10)
    # First two outputs (where input was NaN) must be NaN.
    assert res.map_run_length.iloc[:2].isna().all()
    # Remaining outputs must be valid.
    assert res.map_run_length.iloc[2:].notna().all()


def test_empty_input_returns_empty():
    res = detect_change_points_bocpd(pd.Series([], dtype=float))
    assert len(res.map_run_length) == 0
    assert res.run_length_posterior.shape == (0, 1)


def test_higher_expected_run_length_means_fewer_change_signals():
    """A larger λ biases toward longer regimes — on identical data the
    MAP run length should be larger (or equal) for higher λ."""
    rng = np.random.default_rng(6)
    s = pd.Series(rng.normal(0, 1, 250))
    res_short = detect_change_points_bocpd(s, expected_run_length=50)
    res_long = detect_change_points_bocpd(s, expected_run_length=500)
    # Compare mean MAP run length on the second half.
    mean_short = res_short.map_run_length.iloc[125:].mean()
    mean_long = res_long.map_run_length.iloc[125:].mean()
    assert mean_long >= mean_short
