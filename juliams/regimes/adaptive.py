"""
Adaptive (data-driven) thresholds for regime classification.

Replaces fixed thresholds like ``trend_strength > 0.2`` with rolling
empirical quantiles of the indicator itself, so the cut-points track
the recent distribution rather than being calibrated once.

References
----------
- Hurst, Ooi, Pedersen (2017), "A Century of Evidence on Trend-Following
  Investing" — empirical use of distribution-driven trend cutoffs.
- Moskowitz, Ooi, Pedersen (2012), "Time Series Momentum" — sign-of-z
  as the canonical signal, motivating distribution-aware refinements.
- Wang & Lin (2020), "Regime-Switching Factor Investing with HMMs",
  *J. Risk and Financial Management* 13(12):311 — recent confirmation
  that distribution-aware regime cutoffs remain the default in
  peer-reviewed factor research post-2020.

Design notes
------------
- All rolling windows use ``min_periods`` so we never silently fabricate
  a threshold from a too-short window. Result is NaN until warm-up is
  satisfied; callers must handle this.
- ``shift_calibration=True`` (default) is strictly causal: the threshold
  at time t uses data through t-1 only. This matches how a live system
  must operate and keeps walk-forward backtests honest.
- An ``abs_floor`` parameter prevents the failure mode where rolling
  quantiles force a constant regime frequency in genuinely flat periods.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd


def rolling_quantile_thresholds(
    indicator: pd.Series,
    window: int = 252,
    q_up: float = 0.70,
    q_down: float = 0.30,
    abs_floor_up: float = 0.0,
    abs_floor_down: float = 0.0,
    min_periods: Optional[int] = None,
    shift_calibration: bool = True,
) -> Tuple[pd.Series, pd.Series]:
    """Compute rolling upper and lower quantile thresholds for an indicator.

    Parameters
    ----------
    indicator
        Series of indicator values (e.g. trend_strength z-scores).
    window
        Rolling window length in observations. Default 252 ≈ one trading year.
    q_up, q_down
        Empirical quantiles for the upper and lower cutoffs. Must satisfy
        ``0 < q_down < q_up < 1``.
    abs_floor_up, abs_floor_down
        Hard floors. The returned upper threshold is at least
        ``abs_floor_up``; the lower threshold is at most ``-abs_floor_down``
        (i.e. ``abs_floor_down`` is the magnitude on the negative side).
        Both default to 0 for backwards compatibility but should be set in
        production to avoid forcing a constant trigger frequency in flat
        markets.
    min_periods
        Minimum observations required to emit a non-NaN threshold.
        Defaults to ``window // 2``.
    shift_calibration
        When True (default), the threshold at index t uses data only
        through t-1 — a strict causality guarantee for live or walk-forward
        use. Set False only when you specifically want the in-sample fit.

    Returns
    -------
    (upper, lower)
        Two Series aligned with the input index.
    """
    if not 0.0 < q_down < q_up < 1.0:
        raise ValueError(f"Need 0 < q_down < q_up < 1; got q_down={q_down}, q_up={q_up}")
    if window < 2:
        raise ValueError(f"window must be >= 2, got {window}")

    if min_periods is None:
        min_periods = max(2, window // 2)

    source = indicator.shift(1) if shift_calibration else indicator
    roll = source.rolling(window=window, min_periods=min_periods)

    upper_q = roll.quantile(q_up)
    lower_q = roll.quantile(q_down)

    upper = upper_q.where(upper_q >= abs_floor_up, abs_floor_up)
    lower = lower_q.where(lower_q <= -abs_floor_down, -abs_floor_down)

    # NaN warm-up region is preserved from the rolling output.
    upper = upper.where(upper_q.notna(), np.nan)
    lower = lower.where(lower_q.notna(), np.nan)

    return upper, lower


def classify_with_adaptive_thresholds(
    indicator: pd.Series,
    upper: pd.Series,
    lower: pd.Series,
    up_label: str = "Up",
    down_label: str = "Down",
    sideways_label: str = "Sideways",
    unknown_label: str = "Unknown",
) -> pd.Series:
    """Vectorised regime classification using per-row thresholds.

    Returns ``unknown_label`` where the indicator or either threshold is
    NaN (the warm-up region or external gaps).
    """
    out = pd.Series(unknown_label, index=indicator.index, dtype=object)
    valid = indicator.notna() & upper.notna() & lower.notna()
    out.loc[valid & (indicator > upper)] = up_label
    out.loc[valid & (indicator < lower)] = down_label
    out.loc[valid & (indicator >= lower) & (indicator <= upper)] = sideways_label
    return out


def adaptive_volatility_regime(
    volatility: pd.Series,
    window: int = 252,
    q_high: float = 0.67,
    min_periods: Optional[int] = None,
    shift_calibration: bool = True,
) -> Tuple[pd.Series, pd.Series]:
    """Adaptive high/low volatility threshold via rolling quantile.

    Returns
    -------
    (threshold, regime_labels)
        ``threshold`` is the rolling quantile cutoff; ``regime_labels`` is
        a Series of {"High", "Low", "Unknown"}.
    """
    if not 0.0 < q_high < 1.0:
        raise ValueError(f"q_high must be in (0, 1), got {q_high}")
    if window < 2:
        raise ValueError(f"window must be >= 2, got {window}")

    if min_periods is None:
        min_periods = max(2, window // 2)

    source = volatility.shift(1) if shift_calibration else volatility
    threshold = source.rolling(window=window, min_periods=min_periods).quantile(q_high)

    labels = pd.Series("Unknown", index=volatility.index, dtype=object)
    valid = volatility.notna() & threshold.notna()
    labels.loc[valid & (volatility > threshold)] = "High"
    labels.loc[valid & (volatility <= threshold)] = "Low"

    return threshold, labels
