"""Tests for EWMA-normalised trend strength.

Falsifiable claim (A2): EWMA-z reacts faster than rolling-z to a vol
regime break, but does not exhibit higher variance in stable periods.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from juliams.features.trend import (
    compute_ewma_trend_strength,
    compute_trend_strength,
)


def _synthetic_price(returns: np.ndarray, start: float = 100.0) -> pd.DataFrame:
    prices = start * np.exp(np.cumsum(returns))
    return pd.DataFrame({"Close": prices})


def test_ewma_trend_strength_matches_rolling_shape_on_constant_vol():
    """In a stationary high-vol regime EWMA and rolling-window denominators
    converge in expectation, so the trend-strength series should be
    similar (correlation > 0.9 over the warm region)."""
    rng = np.random.default_rng(0)
    rets = rng.normal(0.0, 0.01, 1000)
    df = _synthetic_price(rets)

    rolling_ts = compute_trend_strength(df, window=20)
    ewma_ts = compute_ewma_trend_strength(df, window=20, halflife=20.0)

    # Compare on rows where both are valid.
    paired = pd.concat([rolling_ts, ewma_ts], axis=1).dropna()
    paired.columns = ["rolling", "ewma"]
    corr = paired["rolling"].corr(paired["ewma"])
    assert corr > 0.9, f"Rolling and EWMA should agree closely on stationary data (got corr={corr:.3f})"


def test_ewma_reacts_faster_to_vol_step():
    """Inject a vol step at index 500 (3x increase). EWMA std should
    track the new level faster than a 20-day rolling std.

    Operationalised: at index 530 (30 days after the shock), the EWMA
    std must be closer to the post-shock true std (0.03) than the
    rolling std is."""
    rng = np.random.default_rng(1)
    n = 1000
    pre = rng.normal(0.0, 0.01, 500)
    post = rng.normal(0.0, 0.03, n - 500)
    rets = np.concatenate([pre, post])

    log_returns = pd.Series(rets)
    rolling_std = log_returns.rolling(window=20, min_periods=10).std()
    ewma_std = log_returns.ewm(halflife=20.0, adjust=False).std()

    truth_post = 0.03
    rolling_err = abs(rolling_std.iloc[530] - truth_post)
    ewma_err = abs(ewma_std.iloc[530] - truth_post)
    # EWMA should be closer (lower error) to the new true std at +30 days.
    assert ewma_err < rolling_err, (
        f"EWMA failed to react faster: rolling_err={rolling_err:.4f} "
        f"ewma_err={ewma_err:.4f}"
    )


def test_ewma_trend_strength_handles_zero_vol_safely():
    """If returns are all zero, vol is zero — function must not blow up
    or emit inf, thanks to min_vol clamp + inf→nan replacement."""
    df = pd.DataFrame({"Close": [100.0] * 100})
    ts = compute_ewma_trend_strength(df, window=20, halflife=20.0)
    assert not np.isinf(ts).any()


def test_ewma_trend_strength_no_future_leakage():
    """compute_ewma_trend_strength at index t must depend only on
    data through index t (the underlying ewm and rolling are causal).
    Verify by poisoning the future and re-computing."""
    rng = np.random.default_rng(2)
    rets = rng.normal(0.0, 0.01, 500)
    df = _synthetic_price(rets)

    ts_orig = compute_ewma_trend_strength(df, window=20, halflife=20.0)

    df_perturbed = df.copy()
    df_perturbed.iloc[400:] = df_perturbed.iloc[400:] * 10.0
    ts_pert = compute_ewma_trend_strength(df_perturbed, window=20, halflife=20.0)

    # Indices 0..399 must be identical.
    assert ts_orig.iloc[:400].equals(ts_pert.iloc[:400])


def test_ewma_halflife_smaller_means_more_reactive():
    """Smaller half-life → vol estimate moves more in response to a shock.
    Verify by measuring jump in vol from t=499 to t=510 across two
    halflives."""
    rng = np.random.default_rng(3)
    pre = rng.normal(0.0, 0.01, 500)
    post = rng.normal(0.0, 0.05, 100)  # 5x shock
    rets = pd.Series(np.concatenate([pre, post]))

    fast = rets.ewm(halflife=5.0, adjust=False).std()
    slow = rets.ewm(halflife=50.0, adjust=False).std()

    fast_jump = fast.iloc[510] - fast.iloc[499]
    slow_jump = slow.iloc[510] - slow.iloc[499]
    assert fast_jump > slow_jump > 0
