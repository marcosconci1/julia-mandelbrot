"""Tests for juliams.regimes.walk_forward_adaptive.

The headline test is :func:`test_no_future_leakage_at_any_index` — it
poisons the future and asserts every past output is byte-identical.
This is the formal version of assumption A5.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from juliams.regimes.evaluation import (
    forward_return_information_ratio,
    regime_distribution_entropy,
    regime_stability,
)
from juliams.regimes.walk_forward_adaptive import (
    AdaptiveRegimeConfig,
    fit_adaptive_regime_walkforward,
)


def _synthetic_prices(n: int = 1500, seed: int = 0) -> pd.DataFrame:
    """Realistic-shape price series with a vol regime shift midway."""
    rng = np.random.default_rng(seed)
    half = n // 2
    rets = np.concatenate(
        [
            rng.normal(0.0005, 0.01, half),
            rng.normal(-0.0002, 0.025, n - half),
        ]
    )
    prices = 100.0 * np.exp(np.cumsum(rets))
    return pd.DataFrame({"Close": prices})


# -- Causality / leakage ----------------------------------------------------

def test_no_future_leakage_at_any_index():
    """Headline causality test: poison the future, assert the past
    threshold and regime outputs are byte-identical."""
    df = _synthetic_prices(seed=1)
    out_orig = fit_adaptive_regime_walkforward(df)

    df_poisoned = df.copy()
    df_poisoned.iloc[1000:, df_poisoned.columns.get_loc("Close")] *= 5.0
    out_pert = fit_adaptive_regime_walkforward(df_poisoned)

    # Threshold at index 1000 must use only data from indices 0..999.
    # Trend strength at index t may legitimately depend on prices through
    # index t, but the threshold is what governs leakage. The regime
    # decision combines current trend_strength with past-only thresholds,
    # so its leakage characteristic equals the threshold's.
    assert out_orig.upper_threshold.iloc[:1000].equals(
        out_pert.upper_threshold.iloc[:1000]
    )
    assert out_orig.lower_threshold.iloc[:1000].equals(
        out_pert.lower_threshold.iloc[:1000]
    )

    # Regime label at index t depends on trend_strength_t (causal up to t)
    # AND threshold_t (strictly past). For t < 1000, both inputs are
    # untouched, so regime labels must match.
    assert out_orig.regime.iloc[:1000].equals(out_pert.regime.iloc[:1000])


def test_thresholds_are_nan_in_warmup():
    df = _synthetic_prices(n=400)
    cfg = AdaptiveRegimeConfig(threshold_window=252, min_threshold_periods=200)
    out = fit_adaptive_regime_walkforward(df, cfg)
    # Need (trend_window=20) for trend_strength to be valid, then
    # (min_threshold_periods=200) more rows for the threshold.
    # First valid threshold is at index ≈ 20 + 200 = 220.
    assert out.upper_threshold.iloc[:200].isna().all()
    assert out.upper_threshold.iloc[300:].notna().any()


def test_regime_labels_are_in_expected_set():
    df = _synthetic_prices()
    out = fit_adaptive_regime_walkforward(df)
    assert set(out.regime.unique()).issubset({"Up", "Down", "Sideways", "Unknown"})


def test_warmup_region_is_unknown():
    df = _synthetic_prices(n=500)
    out = fit_adaptive_regime_walkforward(df)
    # First 20 rows have NaN trend_strength so regime is Unknown.
    assert (out.regime.iloc[:20] == "Unknown").all()


# -- Behavioural / metric tests --------------------------------------------

def test_regime_distribution_is_non_degenerate():
    """The adaptive classifier must not collapse all days into one regime."""
    df = _synthetic_prices(n=2000)
    out = fit_adaptive_regime_walkforward(df)
    valid = out.regime[out.regime != "Unknown"]
    counts = valid.value_counts(normalize=True)
    # No single regime should dominate beyond 80% (degenerate signal).
    assert counts.max() < 0.80
    # At least 2 of the 3 active regimes must each get > 5% of days.
    active = (counts > 0.05).sum()
    assert active >= 2


def test_entropy_above_floor():
    """Distribution entropy must exceed 0.5 bits — i.e. the classifier is
    actually discriminating, not always outputting the same label."""
    df = _synthetic_prices(n=2000)
    out = fit_adaptive_regime_walkforward(df)
    valid = out.regime[out.regime != "Unknown"]
    h = regime_distribution_entropy(valid)
    assert h > 0.5


def test_some_regime_persistence():
    """Regime stability should be > 0.5 — labels persist across days
    rather than flipping every observation."""
    df = _synthetic_prices(n=2000)
    out = fit_adaptive_regime_walkforward(df)
    valid = out.regime[out.regime != "Unknown"]
    s = regime_stability(valid)
    assert s > 0.5


def test_information_ratio_beats_random_labels():
    """The adaptive regime label must carry more forward-return
    information than a random shuffle of the same labels."""
    df = _synthetic_prices(n=2000, seed=5)
    out = fit_adaptive_regime_walkforward(df)
    fwd_ret = np.log(df["Close"]).diff(5).shift(-5)

    valid_mask = out.regime != "Unknown"
    ir_real = forward_return_information_ratio(
        out.regime[valid_mask], fwd_ret[valid_mask], min_support=20
    )

    rng = np.random.default_rng(99)
    shuffled = pd.Series(
        rng.permutation(out.regime[valid_mask].values),
        index=out.regime[valid_mask].index,
    )
    ir_shuffled = forward_return_information_ratio(
        shuffled, fwd_ret[valid_mask], min_support=20
    )

    # Real labels should beat shuffled. Allow a small tolerance for noise.
    assert ir_real >= ir_shuffled, (
        f"Adaptive regime no better than random: real={ir_real:.3f} "
        f"shuffled={ir_shuffled:.3f}"
    )


# -- Config validation ------------------------------------------------------

def test_config_overrides_propagate():
    df = _synthetic_prices(n=500)
    cfg_a = AdaptiveRegimeConfig(threshold_window=100, q_up=0.7, q_down=0.3)
    cfg_b = AdaptiveRegimeConfig(threshold_window=100, q_up=0.9, q_down=0.1)
    out_a = fit_adaptive_regime_walkforward(df, cfg_a)
    out_b = fit_adaptive_regime_walkforward(df, cfg_b)
    # Wider quantile range → fewer Up/Down labels.
    valid_a = (out_a.regime != "Unknown").sum()
    valid_b = (out_b.regime != "Unknown").sum()
    assert valid_a > 0 and valid_b > 0
    up_rate_a = (out_a.regime == "Up").sum() / valid_a
    up_rate_b = (out_b.regime == "Up").sum() / valid_b
    assert up_rate_b < up_rate_a, (
        f"q_up=0.9 should produce fewer Up labels than q_up=0.7: "
        f"{up_rate_b:.3f} vs {up_rate_a:.3f}"
    )
