"""Trend and volatility features for regime classification.

All rolling features use the current and preceding observations only. Quantile
thresholds are expanding estimates through t-1, so a row never sets its own
classification threshold.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data_adapters import PriceSeries


def _close(prices: PriceSeries | pd.Series) -> pd.Series:
    values = prices.data["Close"] if isinstance(prices, PriceSeries) else prices
    if not isinstance(values, pd.Series):
        raise TypeError("prices must be a PriceSeries or pandas Series")
    values = pd.to_numeric(values, errors="coerce").astype(float)
    if values.isna().any() or (values <= 0).any():
        raise ValueError("prices must contain finite positive close values")
    return values


def _window(window: int) -> int:
    if isinstance(window, bool) or not isinstance(window, (int, np.integer)) or window < 2:
        raise ValueError("window must be an integer of at least 2")
    return int(window)


def realized_volatility(prices: PriceSeries | pd.Series, window: int) -> pd.Series:
    """Rolling sample standard deviation of log returns.

    Each value uses only returns available through its timestamp, with no
    lookahead. The first return is undefined, so a full price window is needed.
    """
    n = _window(window)
    close = _close(prices)
    if len(close) < n:
        raise ValueError("insufficient observations for window")
    returns = np.log(close).diff()
    return returns.rolling(n, min_periods=n).std(ddof=1).rename("realized_volatility")


def trend_strength(prices: PriceSeries | pd.Series, window: int) -> pd.Series:
    """OLS log-price slope normalized by volatility.

    The closed-form slope is divided by realized volatility / sqrt(window),
    and each rolling value uses data only through its own timestamp.
    """
    n = _window(window)
    close = _close(prices)
    if len(close) < n:
        raise ValueError("insufficient observations for window")
    log_price = np.log(close)
    x = np.arange(n, dtype=float)
    centered_x = x - x.mean()
    denominator = float(np.dot(centered_x, centered_x))
    slope = log_price.rolling(n, min_periods=n).apply(
        lambda values: float(np.dot(centered_x, values - values.mean()) / denominator), raw=True
    )
    # Sigma is defined as the root mean square of the n-1 returns;
    # realized_volatility is also exposed as the separately requested sample SD.
    returns = log_price.diff()
    sigma = np.sqrt(returns.pow(2).rolling(n, min_periods=n).mean())
    normalized = slope / (sigma / np.sqrt(n))
    normalized = normalized.where(sigma != 0, 0.0)
    return normalized.rename("trend_strength")


def rolling_percentile(series: pd.Series, window: int) -> pd.Series:
    """Percentile rank of each value in its trailing window.

    The trailing window includes the current value and never includes future
    observations; ties receive their average rank.
    """
    n = _window(window)
    if not isinstance(series, pd.Series):
        raise TypeError("series must be a pandas Series")
    if len(series) < n:
        raise ValueError("insufficient observations for window")
    return series.rolling(n, min_periods=n).apply(
        lambda values: float(pd.Series(values).rank(pct=True).iloc[-1]), raw=True
    ).rename(f"{series.name or 'value'}_percentile")


def expanding_quantile_threshold(series: pd.Series, quantile: float) -> pd.Series:
    """Expanding quantile threshold through t-1 (no lookahead)."""
    if not isinstance(series, pd.Series):
        raise TypeError("series must be a pandas Series")
    if not 0 <= quantile <= 1:
        raise ValueError("quantile must be between 0 and 1")
    return series.shift(1).expanding(min_periods=1).quantile(quantile).rename(
        f"q{int(round(quantile * 100)):02d}_threshold"
    )


def trend_thresholds(trend: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Return 30th and 70th trend quantiles through t-1."""
    return expanding_quantile_threshold(trend, 0.30), expanding_quantile_threshold(trend, 0.70)


def volatility_threshold(volatility_percentile: pd.Series) -> pd.Series:
    """Return the 67th percentile threshold through t-1."""
    return expanding_quantile_threshold(volatility_percentile, 0.67)
