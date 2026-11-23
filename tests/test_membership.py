"""Offline synthetic tests for soft membership scores and entropy (spec section 6.2)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from julia_mandelbrot.cells import cell_frame
from julia_mandelbrot.config import IndicatorConfig
from julia_mandelbrot.data_adapters import PriceSeries
from julia_mandelbrot.membership import (
    logistic_score,
    membership_entropy,
    membership_frame,
    trend_membership,
    volatility_membership,
)


def _series(values, name: str = "value") -> pd.Series:
    index = pd.date_range("2024-01-01", periods=len(values), freq="B", tz="UTC")
    return pd.Series(values, index=index, name=name, dtype=float)


def _prices(length: int, seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    log_returns = rng.normal(0.0005, 0.01, size=length)
    close = 100.0 * np.exp(np.cumsum(log_returns))
    index = pd.date_range("2023-01-02", periods=length, freq="B", tz="UTC")
    return pd.Series(close, index=index, name="Close", dtype=float)


def test_logistic_score_known_values_and_extreme_stability():
    # Center maps exactly to 0.5.
    assert logistic_score(0.0, center=0.0, scale=1.0) == pytest.approx(0.5)
    assert logistic_score(2.5, center=2.5, scale=0.4) == pytest.approx(0.5)

    # Known analytical points: z = +/- 1 -> 1 / (1 + e^{-/+ 1}).
    assert logistic_score(1.0, center=0.0, scale=1.0) == pytest.approx(
        1.0 / (1.0 + np.exp(-1.0))
    )
    assert logistic_score(-1.0, center=0.0, scale=1.0) == pytest.approx(
        1.0 / (1.0 + np.exp(1.0))
    )

    # Large |x| values remain numerically stable (no overflow warning, finite)
    # and approach 0 and 1 without reaching exact 0.0 or 1.0.
    extremes = _series([-1e6, -100.0, 0.0, 100.0, 1e6], name="x")
    scores = logistic_score(extremes, center=0.0, scale=1.0)
    assert isinstance(scores, pd.Series)
    assert scores.index.equals(extremes.index)
    assert np.isfinite(scores).all()
    assert 0.0 < scores.iloc[0] < 1e-12
    assert 0.0 < scores.iloc[1] < 1e-12
    assert scores.iloc[2] == pytest.approx(0.5)
    assert 1.0 - 1e-12 < scores.iloc[3] < 1.0
    assert 1.0 - 1e-12 < scores.iloc[4] < 1.0


def test_logistic_score_rejects_non_positive_scale():
    with pytest.raises(ValueError, match="scale"):
        logistic_score(1.0, center=0.0, scale=0.0)
    with pytest.raises(ValueError, match="scale"):
        logistic_score(_series([1.0, 2.0]), center=0.0, scale=-0.5)


def test_trend_and_volatility_membership_known_centers_and_monotonicity():
    grid = _series(np.linspace(-3.0, 3.0, 61), name="trend")
    lo = _series(np.full(len(grid), -1.0), name="lo")
    hi = _series(np.full(len(grid), 1.0), name="hi")

    trend_m = trend_membership(grid, lo, hi)
    assert trend_m.name == "trend_m"
    assert trend_m.index.equals(grid.index)
    # Midpoint of (-1, 1) is 0.0 -> score is 0.5; thresholds map to z = -1 and +1.
    midpoint_idx = len(grid) // 2
    assert trend_m.iloc[midpoint_idx] == pytest.approx(0.5)
    assert np.all(np.diff(trend_m.to_numpy()) > 0.0)
    assert (trend_m > 0.0).all() and (trend_m < 1.0).all()

    vol_grid = _series(np.linspace(0.0, 1.0, 51), name="vol_percentile")
    vol_thresh = _series(np.full(len(vol_grid), 0.60), name="q67_threshold")
    vol_m = volatility_membership(vol_grid, vol_thresh)
    assert vol_m.name == "vol_m"
    assert vol_m.index.equals(vol_grid.index)
    assert vol_m.iloc[30] == pytest.approx(0.5)
    assert np.all(np.diff(vol_m.to_numpy()) > 0.0)
    assert (vol_m > 0.0).all() and (vol_m < 1.0).all()


def test_membership_entropy_identical_scores_extremes_and_clipping():
    # Identical scores normalize to (0.5, 0.5) and achieve maximum entropy log(2).
    identical_trend = _series([0.5, 0.25, 0.85], name="trend_m")
    identical_vol = _series([0.5, 0.25, 0.85], name="vol_m")
    max_ent = membership_entropy(identical_trend, identical_vol)
    assert max_ent.name == "entropy"
    np.testing.assert_allclose(max_ent.to_numpy(), np.log(2.0), rtol=1e-12)

    # One score pinned at an extreme (0.0 or negative before clipping) and the
    # other high produces near-minimum entropy, strictly positive due to clipping.
    extreme_trend = _series([0.0, 1.0, 0.0], name="trend_m")
    extreme_vol = _series([1.0, 0.0, 0.8], name="vol_m")
    near_min = membership_entropy(extreme_trend, extreme_vol, clip=1e-6)
    assert np.isfinite(near_min).all()
    assert (near_min > 0.0).all()
    assert (near_min < 1e-4).all()
    assert (near_min < max_ent).all()

    # Larger clip raises the floor when one score is pinned to 0.
    coarser_clip = membership_entropy(0.0, 1.0, clip=1e-2)
    finer_clip = membership_entropy(0.0, 1.0, clip=1e-6)
    assert 0.0 < finer_clip < coarser_clip < np.log(2.0)

    with pytest.raises(ValueError, match="clip"):
        membership_entropy(0.5, 0.5, clip=0.0)
    with pytest.raises(ValueError, match="clip"):
        membership_entropy(0.5, 0.5, clip=0.5)


def test_no_lookahead_perturbing_last_bar_leaves_prior_membership_unchanged():
    prices = _prices(60, seed=7)
    baseline = membership_frame(prices, window=5, percentile_window=10)

    perturbed = prices.copy()
    perturbed.iloc[-1] = prices.iloc[-1] * 1.8
    updated = membership_frame(perturbed, window=5, percentile_window=10)

    pd.testing.assert_frame_equal(baseline.iloc[:-1], updated.iloc[:-1])
    assert baseline["trend_m"].iloc[-1] != updated["trend_m"].iloc[-1]


def test_membership_frame_alignment_columns_and_price_series_input():
    prices = _prices(60, seed=11)
    frame = membership_frame(prices, window=5, percentile_window=10)
    expected_cells = cell_frame(prices, window=5, percentile_window=10)

    assert frame.index.equals(prices.index)
    for col in (*expected_cells.columns, "trend_m", "vol_m", "entropy"):
        assert col in frame.columns
    pd.testing.assert_frame_equal(frame.loc[:, expected_cells.columns], expected_cells)

    valid_rows = frame.loc[frame["valid"]]
    assert not valid_rows.empty
    assert valid_rows[["trend_m", "vol_m", "entropy"]].notna().all().all()
    assert ((valid_rows["trend_m"] > 0.0) & (valid_rows["trend_m"] < 1.0)).all()
    assert ((valid_rows["vol_m"] > 0.0) & (valid_rows["vol_m"] < 1.0)).all()
    assert ((valid_rows["entropy"] > 0.0) & (valid_rows["entropy"] <= np.log(2.0) + 1e-12)).all()

    # Also accepts a PriceSeries container and uses IndicatorConfig defaults.
    ohlcv = pd.DataFrame(
        {
            "Open": prices,
            "High": prices * 1.01,
            "Low": prices * 0.99,
            "Close": prices,
            "Volume": np.full(len(prices), 1000.0),
        },
        index=prices.index,
    )
    ps = PriceSeries(ohlcv, "synthetic")
    ps_frame = membership_frame(ps, window=5, percentile_window=10)
    assert ps_frame.index.equals(ps.data.index)
    pd.testing.assert_frame_equal(
        membership_frame(ps.data["Close"], window=5, percentile_window=10),
        ps_frame,
    )

    default_prices = _prices(325, seed=3)
    default_frame = membership_frame(default_prices)
    first_valid = max(
        IndicatorConfig().volatility_window + IndicatorConfig().percentile_window,
        IndicatorConfig().trend_window - 1,
    )
    assert default_frame.index.equals(default_prices.index)
    assert not default_frame["valid"].iloc[:first_valid].any()
    assert default_frame["valid"].iloc[first_valid:].all()
    assert default_frame.loc[default_frame["valid"], ["trend_m", "vol_m", "entropy"]].notna().all().all()


def test_public_docstrings_cite_section_6_2_and_score_not_probability_caveat():
    for fn in (
        logistic_score,
        trend_membership,
        volatility_membership,
        membership_entropy,
        membership_frame,
    ):
        doc = fn.__doc__ or ""
        assert "caveat" in doc.lower()
        assert "score" in doc.lower()
        assert "not" in doc.lower() and "probabilit" in doc.lower()
        assert "Arrieta-Ibarra" in doc
