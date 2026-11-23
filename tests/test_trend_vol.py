import numpy as np
import pandas as pd
import pytest

from julia_mandelbrot.trend_vol import (
    realized_volatility,
    rolling_percentile,
    trend_strength,
    trend_thresholds,
    volatility_threshold,
)


def test_constant_log_slope_has_spec_normalized_trend_strength():
    n = 5
    close = pd.Series(np.exp(np.arange(12, dtype=float) * 0.02))
    result = trend_strength(close, n)
    np.testing.assert_allclose(result.dropna(), np.sqrt(n), rtol=1e-12)


def test_flat_series_has_zero_trend():
    result = trend_strength(pd.Series(np.full(8, 12.0)), 4)
    np.testing.assert_allclose(result.dropna(), 0.0)


def test_realized_volatility_matches_hand_calculation():
    returns = np.array([0.01, -0.02, 0.03, 0.015, -0.005])
    close = pd.Series(np.exp(np.r_[0.0, np.cumsum(returns)]))
    result = realized_volatility(close, 3)
    expected = [np.std(returns[i : i + 3], ddof=1) for i in range(3)]
    np.testing.assert_allclose(result.dropna(), expected, rtol=1e-12)


def test_last_bar_does_not_change_prior_thresholds_or_percentile():
    values = pd.Series([1.0, 4.0, 2.0, 8.0, 3.0, 7.0, 6.0])
    changed = values.copy()
    changed.iloc[-1] = 1000.0
    for original, perturbed in zip(trend_thresholds(values), trend_thresholds(changed)):
        pd.testing.assert_series_equal(original.iloc[:-1], perturbed.iloc[:-1])
    pd.testing.assert_series_equal(
        volatility_threshold(values).iloc[:-1], volatility_threshold(changed).iloc[:-1]
    )
    pd.testing.assert_series_equal(
        rolling_percentile(values, 4).iloc[:-1], rolling_percentile(changed, 4).iloc[:-1]
    )


@pytest.mark.parametrize("function", [trend_strength, realized_volatility, rolling_percentile])
def test_insufficient_input_length_raises(function):
    with pytest.raises(ValueError, match="insufficient"):
        function(pd.Series([1.0, 2.0, 3.0]), 4)
