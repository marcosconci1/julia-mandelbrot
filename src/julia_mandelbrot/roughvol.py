"""Realized log-volatility roughness by variogram.

This module estimates the roughness of realized log-volatility (``hurst_logvol``)
and its log-log variogram regression fit quality (``hurst_r2``) on a trailing
window using fixed short lags ``1..10``.

CRITICAL READING LIMIT AND DAILY-BAR CAVEAT:
1. **Observed-series descriptor, not latent-process identification**:
   Per Cont and Das (2024), realized volatility exhibits an apparent Hurst
   exponent ``H < 1/2`` even when the underlying instantaneous volatility
   process is diffusive (Brownian). The bias arises from estimation error in
   the realized measure itself. Therefore, ``hurst_logvol`` describes the
   **observed** realized log-volatility series only; it does **not** identify
   fractional latent volatility, is not called the Hurst exponent of latent
   volatility, and does not pick a directional side (Bull or Bear). Its
   authorized use is as an input to volatility forecasting alongside long-run
   persistence (Bennedsen, Lunde, and Pakkanen, 2022).
2. **Fixed short lags 1..10**:
   Per Mouti (2026), the variogram OLS regression
   uses fixed lags ``tau = 1, ..., 10`` only, because fitting beyond short lags
   is contaminated by mean reversion of the fractional Ornstein-Uhlenbeck (fOU)
   process.
3. **Daily-bar smoothing caveat**:
   This package computes realized volatility from daily close-to-close log
   returns over a short trailing window (default ``rv_window = 20``), a coarser
   object than intraday high-frequency realized variance. Time averaging smooths
   the observed path and raises ``H`` above intraday reference medians.
4. **No lookahead (strict causality)**:
   Every rolling window at row ``t`` uses observations through ``t`` only
   (windows end at ``t`` inclusive and never access ``t + 1`` or later).
"""

from __future__ import annotations

from collections.abc import Sequence
import operator

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import pandas as pd

from .config import IndicatorConfig
from .data_adapters import PriceSeries
from .trend_vol import _close


def _validate_int_at_least(value: int, minimum: int, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer of at least {minimum}, not bool")
    try:
        int_val = operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be an integer of at least {minimum}") from exc
    if int_val < minimum:
        raise ValueError(f"{name} must be an integer of at least {minimum}")
    return int(int_val)


def _validate_lags(min_lag: int | None, max_lag: int | None, cfg: IndicatorConfig) -> tuple[int, int]:
    lo = (
        cfg.variogram_min_lag
        if min_lag is None
        else _validate_int_at_least(min_lag, 1, "min_lag")
    )
    hi = (
        cfg.variogram_max_lag
        if max_lag is None
        else _validate_int_at_least(max_lag, 1, "max_lag")
    )
    if hi <= lo:
        raise ValueError("max_lag must be strictly greater than min_lag")
    return lo, hi


def realized_log_vol(
    prices: PriceSeries | pd.Series,
    rv_window: int | None = None,
) -> pd.Series:
    """Compute the natural logarithm of short-window realized volatility.

    Realized volatility is the trailing sample standard deviation (``ddof=1``)
    of one-period log returns over ``rv_window`` sessions (defaulting to
    ``IndicatorConfig().roughvol_realized_volatility_window = 20``), followed
    by the natural logarithm ``log(sigma_t)``.

    No-lookahead rule:
    At row ``t``, the trailing return window uses prices through ``t`` only
    (ending at ``t`` inclusive) and never accesses ``t + 1`` or later. Rows
    with fewer than ``rv_window`` returns or non-positive realized volatility
    (e.g. a constant price window) are ``NaN``.

    Daily-bar and identification caveat (Cont and Das, 2024; Bennedsen, Lunde,
    and Pakkanen, 2022; Mouti, 2026):
    This series is constructed from daily bars over fixed short windows for
    subsequent variogram estimation on lags ``1..10``. Downstream roughness
    estimates describe the observed daily series only and do not identify
    fractional latent volatility.

    Parameters
    ----------
    prices : PriceSeries or pd.Series
        Positive daily close price series or :class:`PriceSeries` container.
    rv_window : int, optional
        Rolling window length in sessions for realized volatility of log
        returns. Defaults to
        ``IndicatorConfig().roughvol_realized_volatility_window`` (20).

    Returns
    -------
    pd.Series
        Series named ``"realized_log_vol"`` aligned to ``prices``.
    """
    cfg = IndicatorConfig()
    n = (
        cfg.roughvol_realized_volatility_window
        if rv_window is None
        else _validate_int_at_least(rv_window, 2, "rv_window")
    )
    close = _close(prices)
    returns = np.log(close).diff()
    rv = returns.rolling(n, min_periods=n).std(ddof=1)
    log_rv = np.log(rv.where(rv > 0.0))
    return log_rv.rename("realized_log_vol")


def variogram_hurst(
    series: pd.Series | np.ndarray | Sequence[float],
    min_lag: int | None = None,
    max_lag: int | None = None,
) -> tuple[float, float]:
    """Estimate the Hurst exponent ``H`` and regression ``R^2`` by semi-variogram OLS.

    On a 1D window ``x`` of observations, the empirical semi-variogram at lag
    ``k`` for ``k in [min_lag, ..., max_lag]`` (fixed defaults ``1..10``) is:

    ``gamma(k) = 0.5 * mean[(x_{i + k} - x_i)^2]``

    Fitting ``log(gamma(k)) = 2 * H * log(k) + c`` by ordinary least squares
    across the lags yields ``H = slope / 2`` and the coefficient of
    determination ``R^2``. If the input window has insufficient observations
    (``len(series) <= max_lag``), contains any non-finite value, produces any
    non-positive ``gamma(k) <= 0``, or has zero variance in ``log(gamma(k))``
    across lags, ``(nan, nan)`` is returned.

    No-lookahead rule:
    This estimator operates strictly on the supplied trailing window ending at
    ``t`` inclusive and never inspects observations outside that window.

    Daily-bar and identification caveat (Cont and Das, 2024; Mouti, 2026;
    Bennedsen, Lunde, and Pakkanen, 2022):
    Lags are fixed by default to ``1..10`` to avoid mean-reversion
    contamination at longer lags (Mouti, 2026). When applied to daily realized
    log-volatility, ``H`` describes the observed series only and does not
    identify fractional latent volatility (Cont and Das, 2024).

    Parameters
    ----------
    series : pd.Series, np.ndarray, or Sequence[float]
        1D sequence of observations (typically a trailing window of
        ``realized_log_vol``).
    min_lag : int, optional
        Minimum variogram lag (inclusive). Defaults to
        ``IndicatorConfig().variogram_min_lag`` (1).
    max_lag : int, optional
        Maximum variogram lag (inclusive). Defaults to
        ``IndicatorConfig().variogram_max_lag`` (10).

    Returns
    -------
    tuple[float, float]
        ``(H, r2)`` where ``H`` is half the log-log OLS slope and ``r2`` is the
        regression ``R^2`` in ``[0, 1]`` (or ``(nan, nan)`` on a failed fit).
    """
    if isinstance(series, bool) or not isinstance(series, (pd.Series, np.ndarray, Sequence)):
        raise TypeError("series must be a 1D pandas Series, numpy ndarray, or sequence")

    cfg = IndicatorConfig()
    lo_lag, hi_lag = _validate_lags(min_lag, max_lag, cfg)

    arr = np.asarray(series, dtype=float)
    if arr.ndim != 1:
        raise ValueError("series must be one-dimensional")

    nan_pair = (float("nan"), float("nan"))
    if len(arr) <= hi_lag or not np.isfinite(arr).all():
        return nan_pair

    lags = np.arange(lo_lag, hi_lag + 1, dtype=float)
    gammas = np.empty(len(lags), dtype=float)
    for idx, k in enumerate(range(lo_lag, hi_lag + 1)):
        diff = arr[k:] - arr[:-k]
        gamma_k = 0.5 * float(np.mean(diff * diff))
        if not np.isfinite(gamma_k) or gamma_k <= 0.0:
            return nan_pair
        gammas[idx] = gamma_k

    log_lags = np.log(lags)
    log_gammas = np.log(gammas)

    du = log_lags - float(np.mean(log_lags))
    dy = log_gammas - float(np.mean(log_gammas))

    ss_uu = float(np.dot(du, du))
    ss_yy = float(np.dot(dy, dy))
    ss_uy = float(np.dot(du, dy))

    if ss_uu <= 0.0 or ss_yy <= 0.0 or not np.isfinite(ss_yy) or not np.isfinite(ss_uy):
        return nan_pair

    slope = ss_uy / ss_uu
    hurst = 0.5 * slope
    r2 = float(np.clip((ss_uy * ss_uy) / (ss_uu * ss_yy), 0.0, 1.0))
    return float(hurst), r2


def rolling_roughness(
    logvol: pd.Series,
    window: int | None = None,
    min_lag: int | None = None,
    max_lag: int | None = None,
) -> pd.DataFrame:
    """Compute rolling variogram Hurst exponent and ``R^2`` over ``logvol``.

    Rolls a trailing window of ``window`` sessions (defaulting to
    ``IndicatorConfig().roughvol_window = 252``) over ``logvol``, estimating
    ``hurst_logvol`` and ``hurst_r2`` on each window using semi-variogram OLS
    on lags ``min_lag..max_lag`` (default fixed lags ``1..10``).

    No-lookahead rule:
    Each row ``t`` uses only the trailing window ``logvol[t - window + 1 : t + 1]``
    ending at ``t`` inclusive and never inspects future observations ``> t``.
    Rows with insufficient data (the initial ``window - 1`` warm-up rows, or
    any window containing ``NaN``) or a failed regression fit (e.g. non-positive
    semi-variogram ``gamma(k) <= 0`` on a constant window) receive ``NaN`` in
    both ``hurst_logvol`` and ``hurst_r2``.

    Daily-bar and identification caveat (Cont and Das, 2024; Mouti, 2026;
    Bennedsen, Lunde, and Pakkanen, 2022):
    ``hurst_logvol`` describes the observed daily realized log-volatility
    series on fixed short lags ``1..10`` only; it does not identify fractional
    latent volatility (Cont and Das, 2024) and does not pick a directional
    market side.

    Parameters
    ----------
    logvol : pd.Series
        Realized log-volatility series (typically from :func:`realized_log_vol`).
    window : int, optional
        Rolling window length in sessions. Defaults to
        ``IndicatorConfig().roughvol_window`` (252).
    min_lag : int, optional
        Minimum variogram lag (inclusive). Defaults to
        ``IndicatorConfig().variogram_min_lag`` (1).
    max_lag : int, optional
        Maximum variogram lag (inclusive). Defaults to
        ``IndicatorConfig().variogram_max_lag`` (10).

    Returns
    -------
    pd.DataFrame
        DataFrame indexed like ``logvol`` with columns ``hurst_logvol`` and
        ``hurst_r2``.
    """
    if not isinstance(logvol, pd.Series):
        raise TypeError("logvol must be a pandas Series")

    cfg = IndicatorConfig()
    win = (
        cfg.roughvol_window
        if window is None
        else _validate_int_at_least(window, 2, "window")
    )
    lo_lag, hi_lag = _validate_lags(min_lag, max_lag, cfg)

    n_obs = len(logvol)
    hurst_out = np.full(n_obs, np.nan, dtype=float)
    r2_out = np.full(n_obs, np.nan, dtype=float)

    if n_obs >= win and win > hi_lag:
        values = pd.to_numeric(logvol, errors="coerce").to_numpy(dtype=float, copy=True)
        windows = sliding_window_view(values, win)  # shape: (n_windows, win)
        n_windows = windows.shape[0]
        lags = np.arange(lo_lag, hi_lag + 1, dtype=float)
        n_lags = len(lags)

        valid_window = np.isfinite(windows).all(axis=1)
        gammas = np.empty((n_windows, n_lags), dtype=float)
        for idx, k in enumerate(range(lo_lag, hi_lag + 1)):
            diff = windows[:, k:] - windows[:, :-k]
            gammas[:, idx] = 0.5 * np.mean(diff * diff, axis=1)

        valid_gamma = (
            valid_window
            & np.isfinite(gammas).all(axis=1)
            & (gammas > 0.0).all(axis=1)
        )

        safe_gammas = np.where(valid_gamma[:, None], gammas, 1.0)
        log_gammas = np.log(safe_gammas)
        log_lags = np.log(lags)

        du = log_lags - np.mean(log_lags)
        ss_uu = float(np.dot(du, du))
        dy = log_gammas - np.mean(log_gammas, axis=1, keepdims=True)
        ss_uy = np.sum(dy * du, axis=1)
        ss_yy = np.sum(dy * dy, axis=1)

        valid_fit = (
            valid_gamma
            & (ss_uu > 0.0)
            & (ss_yy > 0.0)
            & np.isfinite(ss_uy)
            & np.isfinite(ss_yy)
        )

        safe_ss_yy = np.where(valid_fit, ss_yy, 1.0)
        slopes = np.where(valid_fit, ss_uy / ss_uu, np.nan)
        r2_vals = np.where(
            valid_fit,
            np.clip((ss_uy * ss_uy) / (ss_uu * safe_ss_yy), 0.0, 1.0),
            np.nan,
        )

        hurst_out[win - 1 :] = 0.5 * slopes
        r2_out[win - 1 :] = r2_vals

    return pd.DataFrame(
        {
            "hurst_logvol": hurst_out,
            "hurst_r2": r2_out,
        },
        index=logvol.index,
    )


def roughness_frame(
    prices: PriceSeries | pd.Series,
    rv_window: int | None = None,
    window: int | None = None,
    min_lag: int | None = None,
    max_lag: int | None = None,
    *,
    config: IndicatorConfig | None = None,
) -> pd.DataFrame:
    """Chain :func:`realized_log_vol` and :func:`rolling_roughness` into an aligned frame.

    Computes ``realized_log_vol`` on ``prices`` using ``rv_window`` (default
    ``20``) and feeds it into :func:`rolling_roughness` with ``window``
    (default ``252``) and variogram lags ``min_lag..max_lag`` (default fixed
    lags ``1..10``), returning a DataFrame aligned to the input price index
    with columns ``realized_log_vol``, ``hurst_logvol``, and ``hurst_r2``.

    No-lookahead rule:
    Every row ``t`` depends only on prices at or before ``t`` (windows end at
    ``t`` inclusive; no future price ``> t`` ever enters row ``t``).

    Daily-bar and identification caveat (Cont and Das, 2024; Mouti, 2026;
    Bennedsen, Lunde, and Pakkanen, 2022):
    ``hurst_logvol`` is estimated on fixed lags ``1..10`` from daily-bar
    realized log-volatility. It describes the observed series only and does
    not identify fractional latent volatility (Cont and Das, 2024).

    Parameters
    ----------
    prices : PriceSeries or pd.Series
        Positive daily close price series or :class:`PriceSeries` container.
    rv_window : int, optional
        Short realized-volatility window length. Defaults to
        ``config.roughvol_realized_volatility_window`` (20).
    window : int, optional
        Long variogram rolling window length. Defaults to
        ``config.roughvol_window`` (252).
    min_lag : int, optional
        Minimum variogram lag (inclusive). Defaults to
        ``config.variogram_min_lag`` (1).
    max_lag : int, optional
        Maximum variogram lag (inclusive). Defaults to
        ``config.variogram_max_lag`` (10).
    config : IndicatorConfig, optional
        Configuration instance providing default parameters.

    Returns
    -------
    pd.DataFrame
        DataFrame indexed like ``prices`` with columns ``realized_log_vol``,
        ``hurst_logvol``, and ``hurst_r2``.
    """
    cfg = IndicatorConfig() if config is None else config
    rv_n = cfg.roughvol_realized_volatility_window if rv_window is None else rv_window
    win = cfg.roughvol_window if window is None else window
    lo_lag = cfg.variogram_min_lag if min_lag is None else min_lag
    hi_lag = cfg.variogram_max_lag if max_lag is None else max_lag

    logvol = realized_log_vol(prices, rv_window=rv_n)
    roughness = rolling_roughness(
        logvol,
        window=win,
        min_lag=lo_lag,
        max_lag=hi_lag,
    )
    return pd.concat([logvol, roughness], axis=1)
