"""Offline synthetic tests for the six-cell classification (spec section 6.1)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from julia_mandelbrot.cells import cell_frame, classify_cells
from julia_mandelbrot.config import IndicatorConfig


def _series(values, name="value") -> pd.Series:
    index = pd.date_range("2024-01-01", periods=len(values), freq="B", tz="UTC")
    return pd.Series(values, index=index, name=name, dtype=float)


def _classify(trend, vol_percentile, trend_lo, trend_hi, vol_threshold):
    return classify_cells(
        _series(trend, "trend"),
        _series(vol_percentile, "vol_percentile"),
        _series(trend_lo, "trend_lo"),
        _series(trend_hi, "trend_hi"),
        _series(vol_threshold, "vol_threshold"),
    )


def test_exact_classification_one_row_per_cell_plus_nan_row():
    trend = [0.0, 0.5, 1.0, 0.1, 0.6, 0.9, np.nan]
    vol_percentile = [0.30, 0.70, 0.90, 0.90, 0.67, 0.50, 0.40]
    trend_lo = [0.20] * 7
    trend_hi = [0.80] * 7
    vol_threshold = [0.67] * 7

    result = _classify(trend, vol_percentile, trend_lo, trend_hi, vol_threshold)

    assert result["trend_state"].tolist()[:6] == [
        "low",
        "mid",
        "high",
        "low",
        "mid",
        "high",
    ]
    assert result["vol_state"].tolist()[:6] == [
        "low",
        "high",
        "high",
        "high",
        "low",
        "low",
    ]
    assert result["cell"].tolist()[:6] == [
        "low|low",
        "mid|high",
        "high|high",
        "low|high",
        "mid|low",
        "high|low",
    ]
    assert set(result["cell"].tolist()[:6]) == {
        "low|low",
        "mid|low",
        "high|low",
        "low|high",
        "mid|high",
        "high|high",
    }
    # Equality with a threshold stays mid / low (strict < and >, <= on vol).
    assert result["cell"].iloc[4] == "mid|low"
    # NaN in any input: invalid, no cell.
    assert result["valid"].iloc[6] == False  # noqa: E712
    assert result["cell"].iloc[6] is None
    assert result["trend_state"].iloc[6] is None
    assert result["vol_state"].iloc[6] is None


def test_trend_state_boundary_equality_is_mid():
    result = _classify(
        trend=[0.20, 0.80],
        vol_percentile=[0.67, 0.67],
        trend_lo=[0.20, 0.20],
        trend_hi=[0.80, 0.80],
        vol_threshold=[0.67, 0.67],
    )
    assert result["cell"].tolist() == ["mid|low", "mid|low"]
    assert result["valid"].tolist() == [True, True]


@pytest.mark.parametrize(
    "column",
    ["trend", "vol_percentile", "trend_lo", "trend_hi", "vol_threshold"],
)
def test_nan_in_any_input_invalidates_the_row(column):
    values = {
        "trend": [0.5, 0.5],
        "vol_percentile": [0.5, 0.5],
        "trend_lo": [0.2, 0.2],
        "trend_hi": [0.8, 0.8],
        "vol_threshold": [0.67, 0.67],
    }
    values[column][1] = np.nan
    result = _classify(
        values["trend"],
        values["vol_percentile"],
        values["trend_lo"],
        values["trend_hi"],
        values["vol_threshold"],
    )
    assert result["valid"].tolist() == [True, False]
    assert result["cell"].iloc[1] is None


def _prices(length: int, seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    log_returns = rng.normal(0.0005, 0.01, size=length)
    close = 100.0 * np.exp(np.cumsum(log_returns))
    index = pd.date_range("2023-01-02", periods=length, freq="B", tz="UTC")
    return pd.Series(close, index=index, name="Close", dtype=float)


def test_perturbing_last_bar_does_not_change_prior_cells():
    prices = _prices(60)
    full = cell_frame(prices, window=5, percentile_window=10)

    perturbed = prices.copy()
    perturbed.iloc[-1] = prices.iloc[-1] * 1.5
    changed = cell_frame(perturbed, window=5, percentile_window=10)

    # The last bar moves the last row's windows but nothing before it:
    # thresholds are expanding estimates through t-1, rolling windows end at t.
    pd.testing.assert_frame_equal(full.iloc[:-1], changed.iloc[:-1])


def test_cell_frame_alignment_index_and_validity():
    prices = _prices(60, seed=1)
    frame = cell_frame(prices, window=5, percentile_window=10)

    assert frame.index.equals(prices.index)
    for column in (
        "trend_strength",
        "realized_volatility",
        "realized_volatility_percentile",
        "q30_threshold",
        "q70_threshold",
        "q67_threshold",
        "trend_state",
        "vol_state",
        "cell",
        "valid",
    ):
        assert column in frame.columns

    # First rows have unfilled rolling windows: invalid until every window
    # feeding the row is full. realized_volatility needs 5 returns (first
    # valid at index 5), the 10-bar percentile of it fills at 5 + 10 - 1,
    # and the t-1 threshold needs one more observation.
    assert not frame["valid"].iloc[:15].any()
    assert frame["valid"].iloc[15:].all()
    assert frame.loc[frame["valid"], "cell"].str.contains(r"^(?:low|mid|high)\|(?:low|high)$").all()

    # Defaults come from IndicatorConfig: the percentile window stacks on
    # the realized-volatility window, and the t-1 threshold lags one more.
    default_frame = cell_frame(_prices(320, seed=2))
    first_valid = max(
        IndicatorConfig().volatility_window + IndicatorConfig().percentile_window,
        IndicatorConfig().trend_window - 1,
    )
    assert not default_frame["valid"].iloc[:first_valid].any()
    assert default_frame["valid"].iloc[first_valid:].all()
