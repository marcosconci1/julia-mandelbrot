"""Offline synthetic tests for unconditional left-tail columns (spec section 6.5)."""

from __future__ import annotations

import math
import numpy as np
import pandas as pd
import pytest

import julia_mandelbrot.tail as tail_mod
from julia_mandelbrot.config import IndicatorConfig
from julia_mandelbrot.data_adapters import PriceSeries
from julia_mandelbrot.tail import (
    cvar,
    drawdown,
    tail_frame,
)


def _series(values, name: str = "returns") -> pd.Series:
    index = pd.date_range("2020-01-01", periods=len(values), freq="B", tz="UTC")
    return pd.Series(values, index=index, name=name, dtype=float)


def test_known_answer_cvar_on_fixed_return_array():
    """Known-answer CVaR on a fixed return array with hand-computed worst-q mean."""
    # Window = 10, quantile = 0.90 -> alpha = 0.10 -> k_worst = ceil(0.10 * 10) = 1
    # Window = 10, quantile = 0.95 -> alpha = 0.05 -> k_worst = ceil(0.05 * 10) = 1
    # Window = 10, quantile = 0.80 -> alpha = 0.20 -> k_worst = ceil(0.20 * 10) = 2
    # Window = 5, quantile = 0.95 -> alpha = 0.05 -> k_worst = ceil(0.05 * 5) = ceil(0.25) = 1
    # Window = 5, quantile = 0.60 -> alpha = 0.40 -> k_worst = ceil(0.40 * 5) = 2

    # Fixed returns of 10 observations:
    # Sorted: -0.10, -0.05, -0.02, 0.00, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06
    rets = _series([0.01, -0.05, 0.03, 0.02, -0.10, 0.00, 0.04, -0.02, 0.05, 0.06])
    
    # 1. quantile=0.95, window=10:
    # alpha = 0.05. k_worst = ceil(0.05 * 10) = 1. Worst return is -0.10.
    res_95 = cvar(rets, window=10, quantile=0.95)
    assert np.isnan(res_95.iloc[:9]).all()
    assert res_95.iloc[9] == pytest.approx(-0.10, abs=1e-12)

    # 2. quantile=0.80, window=10:
    # alpha = 0.20. k_worst = ceil(0.20 * 10) = 2. Worst returns are -0.10 and -0.05.
    # Mean of worst 2 = (-0.10 + -0.05) / 2 = -0.075.
    res_80 = cvar(rets, window=10, quantile=0.80)
    assert res_80.iloc[9] == pytest.approx(-0.075, abs=1e-12)

    # 3. Window=5, quantile=0.60:
    # alpha = 0.40. k_worst = ceil(0.40 * 5) = 2.
    # Row 4 (first 5 obs): [0.01, -0.05, 0.03, 0.02, -0.10]
    # Sorted worst 2: -0.10, -0.05. Mean = -0.075.
    # Row 5 (obs 1..5): [-0.05, 0.03, 0.02, -0.10, 0.00]
    # Sorted worst 2: -0.10, -0.05. Mean = -0.075.
    # Row 6 (obs 2..6): [0.03, 0.02, -0.10, 0.00, 0.04]
    # Sorted worst 2: -0.10, 0.00. Mean = -0.05.
    res_win5 = cvar(rets, window=5, quantile=0.60)
    assert np.isnan(res_win5.iloc[:4]).all()
    assert res_win5.iloc[4] == pytest.approx(-0.075, abs=1e-12)
    assert res_win5.iloc[5] == pytest.approx(-0.075, abs=1e-12)
    assert res_win5.iloc[6] == pytest.approx(-0.050, abs=1e-12)


def test_cvar_monotonicity_making_tail_worse_increases_loss_magnitude():
    """CVaR monotonicity: making the worst tail worse increases |cvar|."""
    # Base series
    rng = np.random.default_rng(42)
    base_returns = _series(rng.normal(0.0005, 0.02, size=300))
    cvar_base = cvar(base_returns, window=252, quantile=0.95)

    # Degraded series where negative returns are deepened (made more negative)
    worse_returns = base_returns.copy()
    # Find the worst 10% returns and double their loss magnitude
    neg_mask = worse_returns < 0.0
    worse_returns[neg_mask] = worse_returns[neg_mask] * 1.5

    cvar_worse = cvar(worse_returns, window=252, quantile=0.95)

    valid_mask = cvar_base.notna()
    # With negative sign convention, worse tail means lower (more negative) cvar,
    # and strictly greater absolute magnitude |cvar|.
    assert (cvar_worse[valid_mask] <= cvar_base[valid_mask]).all()
    assert (cvar_worse[valid_mask].abs() >= cvar_base[valid_mask].abs()).all()
    assert (cvar_worse[valid_mask] < cvar_base[valid_mask]).any()


def test_known_drawdown_on_constructed_price_path():
    """Known drawdown on a constructed price path (e.g. monotonic rise then 30% crash -> -0.30)."""
    # Monotonic rise from 100 to 200 (over 6 steps: 0..5), then crash at t=6 to 140 (30% drop from peak 200)
    # Then recovery to 160 at t=7, 180 at t=8, and new peak 210 at t=9
    prices_list = [100.0, 120.0, 140.0, 160.0, 180.0, 200.0, 140.0, 160.0, 180.0, 210.0]
    prices = _series(prices_list, name="Close")

    # With window=5:
    # t=0..3: < window observations -> NaN
    # t=4: [100, 120, 140, 160, 180] -> peak 180, current 180 -> dd = 0.0
    # t=5: [120, 140, 160, 180, 200] -> peak 200, current 200 -> dd = 0.0
    # t=6: [140, 160, 180, 200, 140] -> peak 200, current 140 -> dd = (140 - 200) / 200 = -0.30
    # t=7: [160, 180, 200, 140, 160] -> peak 200, current 160 -> dd = (160 - 200) / 200 = -0.20
    # t=8: [180, 200, 140, 160, 180] -> peak 200, current 180 -> dd = (180 - 200) / 200 = -0.10
    # t=9: [200, 140, 160, 180, 210] -> peak 210, current 210 -> dd = 0.0
    dd = drawdown(prices, window=5)

    assert np.isnan(dd.iloc[:4]).all()
    assert dd.iloc[4] == pytest.approx(0.0, abs=1e-12)
    assert dd.iloc[5] == pytest.approx(0.0, abs=1e-12)
    assert dd.iloc[6] == pytest.approx(-0.30, abs=1e-12)
    assert dd.iloc[7] == pytest.approx(-0.20, abs=1e-12)
    assert dd.iloc[8] == pytest.approx(-0.10, abs=1e-12)
    assert dd.iloc[9] == pytest.approx(0.0, abs=1e-12)

    # Test equivalent reconstruction from returns
    returns = prices.pct_change().dropna()
    # returns has 9 items, index 1..9 corresponding to prices index 1..9
    # Window of 4 returns ending at t=6:
    # prices: t=2(140), t=3(160), t=4(180), t=5(200), t=6(140)
    # returns: r_3, r_4, r_5, r_6
    # wealth from cumprod(1+r): [1+r3, ..., (1+r3)*...*(1+r6)]
    # peak wealth is at t=5 (proportional to 200), end wealth is at t=6 (proportional to 140)
    # Drawdown is (140 - 200) / 200 = -0.30
    dd_from_ret = drawdown(returns, window=4, is_returns=True)
    assert dd_from_ret.loc[prices.index[6]] == pytest.approx(-0.30, abs=1e-12)


def test_no_lookahead_perturb_last_return_earlier_rows_unchanged():
    """No-lookahead: perturb the last return or price; earlier rows remain strictly unchanged."""
    rng = np.random.default_rng(123)
    p_vals = 100.0 * np.exp(np.cumsum(rng.normal(0.0002, 0.015, size=350)))
    prices = _series(p_vals, name="Close")
    returns = prices.pct_change()

    cvar_orig = cvar(returns, window=252, quantile=0.95)
    dd_orig = drawdown(prices, window=252)
    tf_orig = tail_frame(prices, window=252, quantile=0.95)

    # Perturb the final bar drastically
    prices_perturbed = prices.copy()
    prices_perturbed.iloc[-1] = prices_perturbed.iloc[-1] * 0.50
    returns_perturbed = prices_perturbed.pct_change()

    cvar_pert = cvar(returns_perturbed, window=252, quantile=0.95)
    dd_pert = drawdown(prices_perturbed, window=252)
    tf_pert = tail_frame(prices_perturbed, window=252, quantile=0.95)

    # Verify earlier rows (0..N-2) are bitwise identical
    pd.testing.assert_series_equal(cvar_orig.iloc[:-1], cvar_pert.iloc[:-1])
    pd.testing.assert_series_equal(dd_orig.iloc[:-1], dd_pert.iloc[:-1])
    pd.testing.assert_frame_equal(tf_orig.iloc[:-1], tf_pert.iloc[:-1])

    # Final row should differ due to the perturbation
    assert cvar_orig.iloc[-1] != cvar_pert.iloc[-1]
    assert dd_orig.iloc[-1] != dd_pert.iloc[-1]


def test_insufficient_window_gives_nan_and_alignment_of_tail_frame():
    """Insufficient window yields NaN; alignment of tail_frame with input prices."""
    # Under-length series < 252
    short_prices = _series(np.linspace(100.0, 110.0, 100), name="Close")
    tf_short = tail_frame(short_prices, window=252)

    assert tf_short.shape == (100, 2)
    assert list(tf_short.columns) == ["cvar", "drawdown"]
    assert tf_short.index.equals(short_prices.index)
    assert tf_short["cvar"].isna().all()
    assert tf_short["drawdown"].isna().all()

    # Valid length series
    full_prices = _series(np.linspace(100.0, 150.0, 300), name="Close")
    tf_full = tail_frame(full_prices, window=252)

    assert tf_full.index.equals(full_prices.index)
    assert tf_full["cvar"].iloc[:252].isna().all()  # returns have NaN at index 0, so valid at index 252
    assert tf_full["cvar"].iloc[252:].notna().all()
    assert tf_full["drawdown"].iloc[:251].isna().all()
    assert tf_full["drawdown"].iloc[251:].notna().all()


def test_priceseries_container_support_and_config_defaults():
    """Verify PriceSeries container support and IndicatorConfig defaults."""
    df = pd.DataFrame(
        {
            "Open": np.linspace(100, 110, 260),
            "High": np.linspace(101, 111, 260),
            "Low": np.linspace(99, 109, 260),
            "Close": np.linspace(100, 110, 260),
            "Volume": np.full(260, 1000.0),
        },
        index=pd.date_range("2021-01-01", periods=260, freq="B", tz="UTC"),
    )
    ps = PriceSeries(data=df, source="test")
    tf = tail_frame(ps)

    cfg = IndicatorConfig()
    assert cfg.tail_window == 252
    assert cfg.tail_quantile == 0.95
    assert tf.index.equals(ps.data.index)
    assert list(tf.columns) == ["cvar", "drawdown"]
    assert not tf.iloc[-1].isna().any()


def test_docstrings_cite_spec_and_orthogonality():
    """Docstrings state causality and unconditional orthogonality."""
    for fn in (cvar, drawdown, tail_frame, tail_mod):
        doc = fn.__doc__ or ""
        assert "lookahead" in doc.lower() or "causal" in doc.lower()
        assert "orthogonal" in doc.lower() or "unconditional" in doc.lower()
