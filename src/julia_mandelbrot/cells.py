"""Six-cell regime classification from causal trend and volatility thresholds.

The cells discretize the (trend_strength, realized-volatility percentile)
plane: trend strength crossed with the volatility percentile around expanding
causal quantile thresholds. Soft membership is computed separately in :mod:`membership`.

No lookahead: every threshold series passed to :func:`classify_cells` is an
expanding estimate through t-1 (see ``trend_vol.expanding_quantile_threshold``),
so a row never sets its own classification boundary. A row is ``valid`` only
when every rolling window feeding it was full — that is, when none of the
inputs is NaN at that row.
"""

from __future__ import annotations

import pandas as pd

from .config import IndicatorConfig
from .data_adapters import PriceSeries
from .trend_vol import (
    realized_volatility,
    rolling_percentile,
    trend_strength,
    trend_thresholds,
    volatility_threshold,
)

TREND_STATES: tuple[str, str, str] = ("low", "mid", "high")
VOL_STATES: tuple[str, str] = ("low", "high")


def _as_series(values: pd.Series, name: str) -> pd.Series:
    if not isinstance(values, pd.Series):
        raise TypeError(f"{name} must be a pandas Series")
    return pd.to_numeric(values, errors="coerce").astype(float)


def classify_cells(
    trend: pd.Series,
    vol_percentile: pd.Series,
    trend_lo: pd.Series,
    trend_hi: pd.Series,
    vol_threshold: pd.Series,
) -> pd.DataFrame:
    """Classify each row into one of the six regime cells.

    ``trend_state`` is "low" when ``trend < trend_lo``, "high" when
    ``trend > trend_hi``, and "mid" otherwise; equality with a threshold
    stays "mid". ``vol_state`` is "low" when
    ``vol_percentile <= vol_threshold`` and "high" otherwise. ``cell`` is
    the combination label ``"<trend_state>|<vol_state>"``.

    All threshold series must already be causal (through t-1) — this
    function only compares against them, so a row never contributes to its
    own thresholds. A row with NaN in any input is not classifiable: it
    gets ``cell=None`` and ``valid=False``.

    Returns a DataFrame indexed like the (aligned) inputs with columns
    ``trend_state``, ``vol_state``, ``cell``, and ``valid``.
    """
    inputs = {
        "trend": _as_series(trend, "trend"),
        "vol_percentile": _as_series(vol_percentile, "vol_percentile"),
        "trend_lo": _as_series(trend_lo, "trend_lo"),
        "trend_hi": _as_series(trend_hi, "trend_hi"),
        "vol_threshold": _as_series(vol_threshold, "vol_threshold"),
    }
    index = inputs["trend"].index
    for name, series in inputs.items():
        if not series.index.equals(index):
            raise ValueError(f"{name} must share the trend index")

    trend_state = pd.Series("mid", index=index, dtype=object)
    trend_state = trend_state.mask(inputs["trend"] < inputs["trend_lo"], "low")
    trend_state = trend_state.mask(inputs["trend"] > inputs["trend_hi"], "high")

    vol_state = pd.Series("high", index=index, dtype=object)
    vol_state = vol_state.mask(
        inputs["vol_percentile"] <= inputs["vol_threshold"], "low"
    )

    valid = pd.concat(inputs, axis=1).notna().all(axis=1)
    cell = (trend_state + "|" + vol_state).where(valid, None)

    return pd.DataFrame(
        {
            "trend_state": trend_state.where(valid, None),
            "vol_state": vol_state.where(valid, None),
            "cell": cell,
            "valid": valid,
        }
    )


def cell_frame(
    prices: PriceSeries | pd.Series,
    window: int | None = None,
    percentile_window: int | None = None,
) -> pd.DataFrame:
    """Chain the trend and volatility features into one aligned regime frame.

    Columns: the raw ``trend_strength``, ``realized_volatility``, its
    trailing ``realized_volatility_percentile``, the t-1 threshold series
    (``q30_threshold``, ``q70_threshold``, ``q67_threshold``), and the
    classification columns from :func:`classify_cells` (``trend_state``,
    ``vol_state``, ``cell``, ``valid``).

    Windows default to :class:`config.IndicatorConfig`: 63 sessions for
    trend and realized volatility, a 252-session percentile window. Every
    threshold is computed through t-1, so no row uses its own or any later
    observation; early rows with unfilled windows are ``valid=False``.
    """
    config = IndicatorConfig()
    trend_n = config.trend_window if window is None else window
    vol_n = config.volatility_window if window is None else window
    percentile_n = (
        config.percentile_window if percentile_window is None else percentile_window
    )

    trend = trend_strength(prices, trend_n)
    volatility = realized_volatility(prices, vol_n)
    vol_percentile = rolling_percentile(volatility, percentile_n)
    trend_lo, trend_hi = trend_thresholds(trend)
    vol_threshold = volatility_threshold(vol_percentile)

    classified = classify_cells(trend, vol_percentile, trend_lo, trend_hi, vol_threshold)

    frame = pd.concat(
        [trend, volatility, vol_percentile, trend_lo, trend_hi, vol_threshold, classified],
        axis=1,
    )
    return frame
