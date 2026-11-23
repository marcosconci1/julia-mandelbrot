"""Configuration defaults transcribed from the research specification.

These values are research defaults to be calibrated on data. This module
defines configuration only; it does not implement indicator calculations.
"""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class IndicatorConfig:
    """Tunable windows and thresholds for the indicator columns.

    Defaults: 63-session trend and volatility
    windows; 252-session percentile, roughness, and tail windows; 30th/70th
    trend quantiles; 67th volatility percentile; a 756-session transition
    window with a 30-transition count floor; and variogram lags 1 through 10.
    """

    trend_window: int = 63
    volatility_window: int = 63
    percentile_window: int = 252
    trend_lower_quantile: float = 0.30
    trend_upper_quantile: float = 0.70
    volatility_quantile: float = 0.67
    membership_trend_lower_quantile: float = 0.30
    membership_trend_upper_quantile: float = 0.70
    membership_volatility_quantile: float = 0.67
    transition_window: int = 756
    transition_count_floor: int = 30
    variogram_min_lag: int = 1
    variogram_max_lag: int = 10
    roughvol_window: int = 252
    roughvol_realized_volatility_window: int = 20
    tail_window: int = 252
    tail_quantile: float = 0.95
