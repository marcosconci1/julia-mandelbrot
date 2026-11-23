"""Unconditional left-tail risk metrics: CVaR and max drawdown.

This module implements unconditional left-tail loss descriptors on a trailing window:
- :func:`cvar`: Trailing-window Expected Shortfall (CVaR) at a confidence quantile
  (default ``0.95``, corresponding to the worst ``1 - q = 5%`` return fraction).
- :func:`drawdown`: Trailing-window maximum drawdown from the window peak, expressed
  as a negative fractional return relative to peak (e.g. ``-0.23`` = 23% below peak).
- :func:`tail_frame`: Integrates both descriptors into an aligned DataFrame with
  columns ``cvar`` and ``drawdown``, indexed identically to input prices.

ORTHOGONALITY TO REGIME CELLS:
Per Hou, Kong, Wang, and Wu (2026), tail severity
is separated from drift and central dispersion factors. Common loss severity is
orthogonal to the center of the distribution. The tail descriptors implemented
here are UNCONDITIONAL loss descriptors:
1. They enter as columns strictly independent of the six regime cells :math:`(\\mu, \\sigma)`.
2. They are NEVER gated by, conditioned on, or mixed with regime cell assignments.
3. ``survival`` does not become a seventh cell; it is a side gate.
   A market state row can be "Bull Quiet" and simultaneously fragile in the left tail.

EMPIRICAL ESTIMATION ONLY (no Hill, no POT):
Per Candia and Herrera (2024), without a declared and
stable out-of-sample threshold parameter :math:`k`, a Hill estimator on a 252-day window
is noise. Therefore, no Hill estimator and no Peaks-Over-Threshold (POT) / GPD model
are used. CVaR is computed strictly via empirical worst-:math:`q` expectation.

STRICT CAUSALITY (NO LOOKAHEAD):
Every window at row :math:`t` is strictly causal, ending at :math:`t` inclusive and
accessing observations :math:`u \\le t` only. Perturbing or appending rows after :math:`t`
leaves rows through :math:`t` strictly invariant.
"""

from __future__ import annotations

import math
import operator
from typing import Any

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import pandas as pd

from .config import IndicatorConfig
from .data_adapters import PriceSeries
from .trend_vol import _close


def _validate_series(series: Any, name: str) -> pd.Series:
    """Validate that input is a pandas Series."""
    if not isinstance(series, pd.Series):
        raise TypeError(f"{name} must be a pandas Series")
    return series


def _validate_window(window: Any, name: str = "window") -> int:
    """Validate that window is an integer >= 2."""
    if isinstance(window, bool):
        raise ValueError(f"{name} must be an integer of at least 2, not bool")
    try:
        int_val = operator.index(window)
    except TypeError as exc:
        raise ValueError(f"{name} must be an integer of at least 2") from exc
    if int_val < 2:
        raise ValueError(f"{name} must be an integer of at least 2")
    return int(int_val)


def _validate_tail_quantile(quantile: Any) -> float:
    """Validate confidence quantile q in (0, 1) and return alpha = 1 - q."""
    if isinstance(quantile, bool) or not isinstance(quantile, (int, float)):
        raise TypeError("quantile must be a float")
    q = float(quantile)
    if not math.isfinite(q) or not (0.0 < q < 1.0):
        raise ValueError("quantile must be finite and strictly between 0 and 1")
    # Alpha is the fraction of worst observations in the left tail.
    # If user provides 0.95, alpha = 0.05. If user provides 0.05, alpha = 0.05.
    alpha = 1.0 - q if q > 0.5 else q
    return round(alpha, 10)


def cvar(
    returns: pd.Series,
    window: int | None = None,
    quantile: float | None = None,
) -> pd.Series:
    """Compute trailing-window empirical Expected Shortfall (CVaR) of returns.

    Orthogonality rule (Hou et al., 2026):
    CVaR is an unconditional left-tail loss descriptor estimated across all returns
    in the trailing window. It is strictly orthogonal to the (mu, sigma) regime cells
    and is never gated by or conditioned on cell membership.

    Estimator and tie/rounding rule:
    This function uses the empirical worst-fraction
    mean rather than parametric or Hill/POT estimators. In each window of size ``W``,
    the count of worst observations is determined by the ceiling rule:
        ``k_worst = max(1, math.ceil(round(alpha * W, 9)))``
    where ``alpha = min(quantile, 1 - quantile)`` (for quantile 0.95, ``alpha = 0.05``).
    The empirical CVaR is the arithmetic mean of the ``k_worst`` smallest (most negative)
    returns within the trailing window of length ``window``.

    Sign convention:
    Reported as a negative return mean (e.g. ``-0.035`` for a 3.5% expected shortfall loss),
    so that it reads natively in return space alongside returns and drawdowns.

    No-lookahead rule:
    At row ``t``, the rolling window ends at ``t`` inclusive (using returns up to and
    including ``t``). Earlier rows are strictly unaffected by subsequent data.
    Rows with fewer than ``window`` observations or windows containing any NaN
    yield ``NaN``.

    Parameters
    ----------
    returns : pd.Series
        Series of asset returns (arithmetic or log returns), indexed by timestamp.
    window : int, optional
        Rolling trailing window length. Defaults to ``IndicatorConfig().tail_window`` (252).
    quantile : float, optional
        Confidence quantile level. Defaults to ``IndicatorConfig().tail_quantile`` (0.95).

    Returns
    -------
    pd.Series
        Trailing empirical CVaR series indexed like ``returns``, named ``cvar``.
    """
    _validate_series(returns, "returns")
    cfg = IndicatorConfig()
    win = cfg.tail_window if window is None else _validate_window(window, "window")
    q = cfg.tail_quantile if quantile is None else quantile
    alpha = _validate_tail_quantile(q)

    n = len(returns)
    out = pd.Series(float("nan"), index=returns.index, name="cvar", dtype="float64")
    if n < win:
        return out

    values = pd.to_numeric(returns, errors="coerce").to_numpy(dtype=np.float64)
    k_worst = max(1, math.ceil(round(alpha * win, 9)))

    # Vectorized sliding window view over the 1D array
    sub_windows = sliding_window_view(values, window_shape=win)
    # Check for NaNs within each window
    valid_mask = np.isfinite(sub_windows).all(axis=1)

    cvar_vals = np.full(len(sub_windows), np.nan, dtype=np.float64)
    if np.any(valid_mask):
        valid_sub = sub_windows[valid_mask]
        # Partition to find the k_worst smallest values per row efficiently
        if k_worst >= win:
            cvar_vals[valid_mask] = np.mean(valid_sub, axis=1)
        else:
            # np.partition puts the k_worst smallest elements into indices 0..k_worst-1
            part = np.partition(valid_sub, kth=k_worst - 1, axis=1)
            worst_k = part[:, :k_worst]
            cvar_vals[valid_mask] = np.mean(worst_k, axis=1)

    out.iloc[win - 1 :] = cvar_vals
    return out


def drawdown(
    prices_or_returns: PriceSeries | pd.Series,
    window: int | None = None,
    *,
    is_returns: bool | None = None,
) -> pd.Series:
    """Compute trailing-window maximum drawdown from the window peak.

    Orthogonality rule:
    Drawdown describes the left-tail path loss unconditionally, measuring depth below
    the trailing window's high-water mark. It is completely independent of the
    regime classification state.

    Input convention:
    Accepts either:
    1. A :class:`PriceSeries` or price :class:`pd.Series` (e.g. Close prices), or
    2. A return :class:`pd.Series`.
    If ``is_returns`` is None, the function auto-detects: if the object is a
    :class:`PriceSeries`, it is treated as prices; if it is a :class:`pd.Series` whose
    name is ``"returns"`` or ``"return"``, or if ``is_returns=True`` is explicitly passed,
    it is treated as returns and reconstructed into a wealth index via ``(1 + r).cumprod()``
    internally. Otherwise, if values are strictly positive prices or a standard price
    series, it computes drawdown directly from price levels:
        ``drawdown_t = (P_t - max_{u in [t - window + 1, t]} P_u) / max_{u in [t - window + 1, t]} P_u``

    Output sign convention:
    Reported as a non-positive fraction (e.g. ``-0.23`` means current value is 23%
    below the trailing window peak; ``0.0`` indicates a new peak within the window).

    No-lookahead rule:
    At row ``t``, the peak is evaluated strictly over ``[t - window + 1, t]`` inclusive.
    Future rows ``> t`` never affect row ``t``.
    Rows with fewer than ``window`` observations yield ``NaN``.

    Parameters
    ----------
    prices_or_returns : PriceSeries or pd.Series
        Price series, PriceSeries container, or return series.
    window : int, optional
        Rolling trailing window length. Defaults to ``IndicatorConfig().tail_window`` (252).
    is_returns : bool, optional
        Explicit indicator whether the input represents returns. If None, auto-detected.

    Returns
    -------
    pd.Series
        Trailing drawdown series indexed like input, named ``drawdown``.
    """
    cfg = IndicatorConfig()
    win = cfg.tail_window if window is None else _validate_window(window, "window")

    # Detect PriceSeries or extract Series
    if isinstance(prices_or_returns, PriceSeries):
        s = _close(prices_or_returns)
        treat_as_returns = False
    elif isinstance(prices_or_returns, pd.Series):
        s = prices_or_returns
        if is_returns is True:
            treat_as_returns = True
        elif is_returns is False:
            treat_as_returns = False
        else:
            # Auto-detect by series name
            treat_as_returns = str(s.name).lower() in ("returns", "return", "log_returns", "log_return")
    else:
        raise TypeError("prices_or_returns must be a pd.Series or PriceSeries")

    n = len(s)
    out = pd.Series(float("nan"), index=s.index, name="drawdown", dtype="float64")
    if n < win:
        return out

    numeric_vals = pd.to_numeric(s, errors="coerce").to_numpy(dtype=np.float64)

    if treat_as_returns:
        # Reconstruct wealth index via cumulative product on (1 + r)
        wealth = np.empty_like(numeric_vals)
        wealth[:] = np.nan
        valid_ret_mask = np.isfinite(numeric_vals)
        if np.all(valid_ret_mask):
            wealth = np.cumprod(1.0 + numeric_vals)
        else:
            wealth_accum = 1.0
            for i, r in enumerate(numeric_vals):
                if np.isnan(r):
                    wealth[i] = np.nan
                else:
                    wealth_accum *= (1.0 + r)
                    wealth[i] = wealth_accum
        p_vals = wealth
    else:
        p_vals = numeric_vals

    sub_windows = sliding_window_view(p_vals, window_shape=win)
    valid_mask = np.isfinite(sub_windows).all(axis=1)

    dd_vals = np.full(len(sub_windows), np.nan, dtype=np.float64)
    if np.any(valid_mask):
        valid_sub = sub_windows[valid_mask]
        current_val = valid_sub[:, -1]
        peaks = np.max(valid_sub, axis=1)
        # Avoid division by zero or negative peaks if pathological
        safe_peaks = np.where(peaks > 0.0, peaks, np.nan)
        dd = (current_val - safe_peaks) / safe_peaks
        # Numerical guard: clipping max to 0.0
        dd_vals[valid_mask] = np.minimum(dd, 0.0)

    out.iloc[win - 1 :] = dd_vals
    return out


def tail_frame(
    prices: PriceSeries | pd.Series,
    window: int | None = None,
    quantile: float | None = None,
    *,
    config: IndicatorConfig | None = None,
) -> pd.DataFrame:
    """Construct aligned DataFrame of unconditional left-tail columns.

    Computes trailing empirical :func:`cvar` and :func:`drawdown` over ``window``
    (default ``IndicatorConfig().tail_window = 252``) and ``quantile`` (default
    ``IndicatorConfig().tail_quantile = 0.95``).

    Orthogonality to regime cells (Hou et al., 2026):
    The returned DataFrame columns ``cvar`` and ``drawdown`` are unconditional on
    the regime state :math:`(\\mu, \\sigma)`. They describe observed left-tail risk
    independently of cell boundaries and are never gated by state membership.

    Causality and return convention:
    Returns are computed causally from prices as arithmetic simple returns:
        ``r_t = (P_t - P_{t-1}) / P_{t-1}``.
    Every window at row ``t`` ends at ``t`` inclusive, strictly preventing lookahead.

    Parameters
    ----------
    prices : PriceSeries or pd.Series
        Price series or :class:`PriceSeries` container.
    window : int, optional
        Trailing window size. Defaults to ``config.tail_window`` (252).
    quantile : float, optional
        Confidence quantile level. Defaults to ``config.tail_quantile`` (0.95).
    config : IndicatorConfig, optional
        Configuration instance for defaults.

    Returns
    -------
    pd.DataFrame
        DataFrame indexed identically to ``prices`` with columns ``cvar`` and ``drawdown``.
    """
    cfg = IndicatorConfig() if config is None else config
    win = cfg.tail_window if window is None else _validate_window(window, "window")
    q = cfg.tail_quantile if quantile is None else quantile

    s = _close(prices) if isinstance(prices, PriceSeries) else _validate_series(prices, "prices")
    # Returns for CVaR: percentage change (causal, r_t = (P_t - P_{t-1}) / P_{t-1})
    returns = s.pct_change()

    cvar_col = cvar(returns, window=win, quantile=q)
    drawdown_col = drawdown(s, window=win, is_returns=False)

    return pd.DataFrame(
        {
            "cvar": cvar_col,
            "drawdown": drawdown_col,
        },
        index=s.index,
    )
