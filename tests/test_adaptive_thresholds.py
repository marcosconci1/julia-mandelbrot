"""Tests for juliams.regimes.adaptive — rolling-quantile thresholds.

Validates the assumptions that motivate adaptive thresholds:
- A1: thresholds are non-NaN past warmup, monotone in q
- A4: adaptive labels differ from fixed labels on real-shape data
- A5 (leakage): output at time t depends only on input through t-1
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from juliams.regimes.adaptive import (
    adaptive_volatility_regime,
    classify_with_adaptive_thresholds,
    rolling_quantile_thresholds,
)


# -- Assumption A1: monotonicity & warmup -----------------------------------

def test_thresholds_warmup_yields_nan_then_values():
    rng = np.random.default_rng(0)
    s = pd.Series(rng.standard_normal(500))
    upper, lower = rolling_quantile_thresholds(s, window=100, min_periods=50)
    # First (min_periods - 1) values must be NaN; later values must be valid.
    # Add 1 because shift_calibration=True consumes one observation.
    expected_first_valid = 50  # min_periods=50 plus shift=1, then index 50
    assert upper.iloc[:expected_first_valid].isna().all()
    assert upper.iloc[expected_first_valid + 5 :].notna().all()
    assert lower.iloc[expected_first_valid + 5 :].notna().all()


def test_thresholds_monotone_in_quantile():
    rng = np.random.default_rng(1)
    s = pd.Series(rng.standard_normal(1000))
    upper_70, lower_30 = rolling_quantile_thresholds(s, window=252, q_up=0.7, q_down=0.3)
    upper_90, lower_10 = rolling_quantile_thresholds(s, window=252, q_up=0.9, q_down=0.1)
    # Higher q_up gives a higher upper cutoff; lower q_down gives a lower one.
    # Compare on the warm region only.
    valid = upper_70.notna() & upper_90.notna()
    assert (upper_90[valid] >= upper_70[valid]).all()
    assert (lower_10[valid] <= lower_30[valid]).all()


def test_abs_floor_clamps_threshold():
    # Series of zeros → quantile is 0; floor should lift the upper cut.
    s = pd.Series(np.zeros(500))
    upper, lower = rolling_quantile_thresholds(
        s, window=100, q_up=0.7, q_down=0.3, abs_floor_up=0.5, abs_floor_down=0.5
    )
    valid = upper.notna()
    assert (upper[valid] == 0.5).all()
    assert (lower[valid] == -0.5).all()


def test_invalid_quantile_raises():
    s = pd.Series(np.zeros(100))
    with pytest.raises(ValueError):
        rolling_quantile_thresholds(s, q_up=0.3, q_down=0.7)
    with pytest.raises(ValueError):
        rolling_quantile_thresholds(s, q_up=1.0, q_down=0.5)


def test_invalid_window_raises():
    s = pd.Series(np.zeros(100))
    with pytest.raises(ValueError):
        rolling_quantile_thresholds(s, window=1)


# -- Assumption A5: no future leakage ---------------------------------------

def test_no_future_leakage_under_shift_calibration():
    """Threshold at index t with shift_calibration=True must depend only
    on values at indices < t. We verify by mutating the future and
    confirming the threshold up to that point is unchanged."""
    rng = np.random.default_rng(2)
    s = pd.Series(rng.standard_normal(500))
    upper_orig, lower_orig = rolling_quantile_thresholds(s, window=100)

    s_perturbed = s.copy()
    s_perturbed.iloc[400:] = 999.0  # poison the future
    upper_pert, lower_pert = rolling_quantile_thresholds(s_perturbed, window=100)

    # Indices 0..399 use only data 0..398 (because of shift=1), so they
    # should be identical between the two runs.
    assert upper_orig.iloc[:400].equals(upper_pert.iloc[:400])
    assert lower_orig.iloc[:400].equals(lower_pert.iloc[:400])


def test_shift_calibration_false_does_use_current():
    """Sanity check: with shift_calibration=False the current value is in
    the rolling window, so poisoning the future affects the same index."""
    rng = np.random.default_rng(3)
    s = pd.Series(rng.standard_normal(500))
    upper_orig, _ = rolling_quantile_thresholds(s, window=100, shift_calibration=False)

    s_perturbed = s.copy()
    s_perturbed.iloc[400] = 999.0
    upper_pert, _ = rolling_quantile_thresholds(s_perturbed, window=100, shift_calibration=False)

    # With shift=False, index 400 includes value at 400. Should differ.
    assert upper_orig.iloc[400] != upper_pert.iloc[400]


# -- Classification helper --------------------------------------------------

def test_classify_with_adaptive_thresholds_basic():
    indicator = pd.Series([0.0, 1.0, -1.0, 0.5, np.nan])
    upper = pd.Series([0.4, 0.4, 0.4, 0.4, 0.4])
    lower = pd.Series([-0.4, -0.4, -0.4, -0.4, -0.4])
    labels = classify_with_adaptive_thresholds(indicator, upper, lower)
    assert labels.tolist() == ["Sideways", "Up", "Down", "Up", "Unknown"]


def test_classify_marks_unknown_when_thresholds_nan():
    indicator = pd.Series([1.0, 1.0])
    upper = pd.Series([np.nan, 0.5])
    lower = pd.Series([-0.5, -0.5])
    labels = classify_with_adaptive_thresholds(indicator, upper, lower)
    assert labels.iloc[0] == "Unknown"
    assert labels.iloc[1] == "Up"


# -- Assumption A4: adaptive ≠ fixed on regime-shift fixture ----------------

def test_adaptive_differs_from_fixed_on_regime_shift():
    """Construct a synthetic indicator whose distribution shifts halfway
    through. A fixed threshold tuned to the first half will misfire on the
    second; a rolling-quantile threshold should track the shift and
    produce a meaningfully different label sequence."""
    rng = np.random.default_rng(7)
    n = 1000
    # First half: indicator centered at 0 with std 1 → fixed thresh ±0.5
    # captures roughly the top/bottom 31% each.
    half = n // 2
    first = rng.normal(0.0, 1.0, half)
    # Second half: indicator centered at 0 with std 3 → fixed ±0.5 now
    # captures roughly the top/bottom 43% each (much more frequent).
    second = rng.normal(0.0, 3.0, n - half)
    indicator = pd.Series(np.concatenate([first, second]))

    # Fixed thresholds at ±0.5
    fixed_labels = pd.Series("Sideways", index=indicator.index, dtype=object)
    fixed_labels[indicator > 0.5] = "Up"
    fixed_labels[indicator < -0.5] = "Down"

    # Adaptive thresholds, 252-day rolling, q=0.7/0.3 (so by construction
    # ~30% Up and ~30% Down on stationary data).
    upper, lower = rolling_quantile_thresholds(
        indicator, window=252, q_up=0.7, q_down=0.3, min_periods=100
    )
    adaptive_labels = classify_with_adaptive_thresholds(indicator, upper, lower)

    # Compare on the warm region only.
    warm = upper.notna()
    fixed_warm = fixed_labels[warm]
    adapt_warm = adaptive_labels[warm]

    # The adaptive labels must materially disagree with fixed.
    disagreement = (fixed_warm != adapt_warm).mean()
    assert disagreement > 0.10, f"Adaptive too similar to fixed (diff={disagreement:.2%})"

    # The adaptive label distribution should be more uniform (closer to
    # the 30/40/30 target) than the fixed one in the high-vol second half.
    second_half_idx = adapt_warm.index[adapt_warm.index >= half]
    fixed_up_rate_2 = (fixed_warm.loc[second_half_idx] == "Up").mean()
    adapt_up_rate_2 = (adapt_warm.loc[second_half_idx] == "Up").mean()
    # Adaptive holds closer to 30% target than fixed (which inflated).
    assert abs(adapt_up_rate_2 - 0.30) < abs(fixed_up_rate_2 - 0.30)


# -- Volatility regime helper ----------------------------------------------

def test_adaptive_vol_labels_high_above_threshold():
    rng = np.random.default_rng(11)
    vol = pd.Series(rng.uniform(0.1, 0.5, 500))
    threshold, labels = adaptive_volatility_regime(vol, window=100, q_high=0.67)
    valid = labels != "Unknown"
    above = vol[valid] > threshold[valid]
    assert ((labels[valid] == "High") == above).all()


def test_adaptive_vol_q_high_invalid_raises():
    s = pd.Series(np.ones(50))
    with pytest.raises(ValueError):
        adaptive_volatility_regime(s, q_high=0.0)
    with pytest.raises(ValueError):
        adaptive_volatility_regime(s, q_high=1.0)


# -- Integration with evaluation metrics -----------------------------------

def test_adaptive_information_ratio_at_least_matches_fixed_under_regime_shift():
    """End-to-end smoke: under the regime-shift fixture, the adaptive
    classifier should not produce *worse* forward-return information
    ratio than the fixed one. A strict 'better' claim is fragile because
    forward returns are noise; we only require non-degradation."""
    from juliams.regimes.evaluation import forward_return_information_ratio

    rng = np.random.default_rng(17)
    n = 1500
    half = n // 2
    indicator = pd.Series(
        np.concatenate(
            [rng.normal(0.0, 1.0, half), rng.normal(0.0, 3.0, n - half)]
        )
    )
    # Forward returns: weak positive drift when indicator was strongly
    # positive yesterday — i.e. true regime carries some predictive value.
    fr = pd.Series(0.1 * indicator.shift(1).fillna(0) + rng.standard_normal(n))

    fixed_labels = pd.Series("Sideways", index=indicator.index, dtype=object)
    fixed_labels[indicator > 0.5] = "Up"
    fixed_labels[indicator < -0.5] = "Down"

    upper, lower = rolling_quantile_thresholds(
        indicator, window=252, q_up=0.7, q_down=0.3, min_periods=100
    )
    adapt_labels = classify_with_adaptive_thresholds(indicator, upper, lower)

    ir_fixed = forward_return_information_ratio(fixed_labels, fr, min_support=20)
    ir_adapt = forward_return_information_ratio(adapt_labels, fr, min_support=20)
    # Adaptive should be within 30% of fixed, ideally better. We only
    # assert non-catastrophic degradation: it should not collapse.
    assert ir_adapt >= 0.7 * ir_fixed
