"""Soft regime membership scores and Shannon entropy.

CRITICAL CAVEAT:
Membership values produced by this module are soft **scores**, NOT
probabilities. Per Arrieta-Ibarra, Gujral, Tannen, Tygert, and Xu (2022),
a normalized score is not a probability without a calibration check via a
reliability diagram comparing predicted values against realized outcome
frequencies. Until calibrated, ``trend_m`` and
``vol_m`` must be read strictly as dimensionless logistic scores and
``entropy`` as a low-conviction / ambiguity diagnostic on those scores.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .cells import cell_frame
from .config import IndicatorConfig
from .data_adapters import PriceSeries
from .trend_vol import expanding_quantile_threshold

_LOGISTIC_Z_LIMIT: float = 36.0
_MIN_SCALE: float = 1e-12
_DEFAULT_VOL_SCALE: float = 0.10
_DEFAULT_ENTROPY_CLIP: float = 1e-6


def _validate_positive_scale(scale_arr: np.ndarray) -> None:
    finite_mask = ~np.isnan(scale_arr)
    if np.any(~np.isfinite(scale_arr[finite_mask])) or np.any(scale_arr[finite_mask] <= 0.0):
        raise ValueError("scale must be finite and strictly positive")


def _extract_index_and_arrays(
    *values: tuple[str, pd.Series | np.ndarray | float],
) -> tuple[pd.Index | None, bool, list[np.ndarray]]:
    index: pd.Index | None = None
    any_ndarray = False
    arrays: list[np.ndarray] = []

    for name, value in values:
        if isinstance(value, bool):
            raise TypeError(f"{name} must be numeric, not bool")
        if isinstance(value, pd.Series):
            if index is None:
                index = value.index
            elif not value.index.equals(index):
                raise ValueError(f"{name} must share the same index")
            numeric = pd.to_numeric(value, errors="coerce").astype(float)
            arrays.append(numeric.to_numpy(dtype=float, copy=True))
        elif isinstance(value, (np.ndarray, list, tuple)):
            any_ndarray = True
            arrays.append(np.asarray(value, dtype=float))
        elif isinstance(value, (int, float, np.number)):
            arrays.append(np.asarray(float(value), dtype=float))
        else:
            raise TypeError(f"{name} must be a pandas Series, numpy ndarray, or float")

    return index, any_ndarray, arrays


def _wrap_output(
    result: np.ndarray,
    index: pd.Index | None,
    any_ndarray: bool,
    name: str,
) -> pd.Series | np.ndarray | float:
    if index is not None:
        return pd.Series(result, index=index, name=name, dtype=float)
    if any_ndarray or result.ndim > 0:
        return np.asarray(result, dtype=float)
    return float(result)


def logistic_score(
    x: pd.Series | np.ndarray | float,
    center: pd.Series | np.ndarray | float = 0.0,
    scale: pd.Series | np.ndarray | float = 1.0,
) -> pd.Series | np.ndarray | float:
    """Compute the standard logistic score ``1 / (1 + exp(-(x - center) / scale))``.

    Implements a vectorized sigmoid mapping with
    standardized arguments clamped to ``[-36, 36]`` so large ``|x|`` values
    never overflow ``exp`` and asymptotically approach ``0`` and ``1`` without
    underflowing to exact ``0.0`` or ``1.0`` in IEEE-754 float64.

    Important calibration caveat (Arrieta-Ibarra et al., 2022):
    the returned value is a soft **score**, NOT a
    probability. Without a reliability diagram against realized frequencies,
    there is no probability interpretation.
    """
    index, any_ndarray, (x_arr, center_arr, scale_arr) = _extract_index_and_arrays(
        ("x", x),
        ("center", center),
        ("scale", scale),
    )
    _validate_positive_scale(scale_arr)

    z = (x_arr - center_arr) / scale_arr
    z_clamped = np.clip(z, -_LOGISTIC_Z_LIMIT, _LOGISTIC_Z_LIMIT)
    score = 1.0 / (1.0 + np.exp(-z_clamped))
    return _wrap_output(score, index, any_ndarray, "logistic_score")


def trend_membership(
    trend: pd.Series | np.ndarray | float,
    lo_threshold: pd.Series | np.ndarray | float,
    hi_threshold: pd.Series | np.ndarray | float,
) -> pd.Series | np.ndarray | float:
    """Compute the trend-dimension logistic score from causal t-1 thresholds.

    Parameterization choice:
    - ``center = 0.5 * (lo_threshold + hi_threshold)``, the midpoint between the
      causal ``t-1`` lower and upper trend quantile thresholds (so a trend at
      the midpoint maps to ``0.5``).
    - ``scale = max(0.5 * (hi_threshold - lo_threshold), 1e-12)``, the half-spread
      of the ``t-1`` inter-quantile band (so ``lo_threshold`` and ``hi_threshold``
      sit at ``z = -1`` and ``z = +1``, and degenerate zero-spread bars on a
      single-observation expanding start remain well-defined).

    Important calibration caveat (Arrieta-Ibarra et al., 2022):
    ``trend_m`` is a soft **score**, NOT a
    probability. Without a reliability diagram there is no probability.
    """
    index, any_ndarray, (trend_arr, lo_arr, hi_arr) = _extract_index_and_arrays(
        ("trend", trend),
        ("lo_threshold", lo_threshold),
        ("hi_threshold", hi_threshold),
    )
    spread = hi_arr - lo_arr
    valid_spread = ~np.isnan(spread)
    if np.any(spread[valid_spread] < 0.0):
        raise ValueError("hi_threshold must be greater than or equal to lo_threshold")

    center = 0.5 * (lo_arr + hi_arr)
    scale = np.where(valid_spread, np.maximum(0.5 * spread, _MIN_SCALE), np.nan)
    score = logistic_score(trend_arr, center=center, scale=scale)
    return _wrap_output(np.asarray(score, dtype=float), index, any_ndarray, "trend_m")


def volatility_membership(
    vol_percentile: pd.Series | np.ndarray | float,
    threshold: pd.Series | np.ndarray | float,
    scale: pd.Series | np.ndarray | float = _DEFAULT_VOL_SCALE,
) -> pd.Series | np.ndarray | float:
    """Compute the volatility-dimension logistic score around the t-1 threshold.

    The logistic is centered on the causal ``t-1`` volatility percentile
    threshold (``center = threshold``), with ``scale`` defaulting to ``0.10`` on
    the unit ``[0, 1]`` percentile domain so that values at the threshold map to
    ``0.5`` and extreme percentiles approach ``0`` and ``1``.

    Important calibration caveat (Arrieta-Ibarra et al., 2022):
    ``vol_m`` is a soft **score**, NOT a
    probability. Without a reliability diagram there is no probability.
    """
    index, any_ndarray, (vol_arr, thresh_arr, scale_arr) = _extract_index_and_arrays(
        ("vol_percentile", vol_percentile),
        ("threshold", threshold),
        ("scale", scale),
    )
    score = logistic_score(vol_arr, center=thresh_arr, scale=scale_arr)
    return _wrap_output(np.asarray(score, dtype=float), index, any_ndarray, "vol_m")


def membership_entropy(
    trend_m: pd.Series | np.ndarray | float,
    vol_m: pd.Series | np.ndarray | float,
    clip: float = _DEFAULT_ENTROPY_CLIP,
) -> pd.Series | np.ndarray | float:
    """Compute the Shannon entropy of the two normalized membership scores.

    Both input scores ``trend_m`` and ``vol_m`` are first clipped to
    ``[clip, 1 - clip]`` (default ``clip = 1e-6``) so extreme values approach
    ``0`` or ``1`` without reaching them, preventing ``log(0)`` singularities
    and degenerate zero weights. The two clipped scores are then normalized to
    sum to ``1`` (``p_trend = t / (t + v)``, ``p_vol = v / (t + v)``) and
    aggregated via Shannon entropy:

    ``H = -(p_trend * log(p_trend) + p_vol * log(p_vol))``

    Higher entropy indicates a more ambiguous / lower-conviction state
    (reaching its maximum ``log(2)`` when the two scores are identical, and
    approaching ``0`` when one score is pinned near ``0`` and the other is not).

    Important calibration caveat (Arrieta-Ibarra et al., 2022):
    the normalized inputs and resulting entropy
    summarize uncalibrated **scores**, NOT probabilities. Without a reliability
    diagram there is no probability.
    """
    if isinstance(clip, bool) or not isinstance(clip, (int, float, np.number)):
        raise TypeError("clip must be a float")
    clip_val = float(clip)
    if not np.isfinite(clip_val) or not (0.0 < clip_val < 0.5):
        raise ValueError("clip must be finite and strictly between 0 and 0.5")

    index, any_ndarray, (trend_arr, vol_arr) = _extract_index_and_arrays(
        ("trend_m", trend_m),
        ("vol_m", vol_m),
    )

    t_clipped = np.clip(trend_arr, clip_val, 1.0 - clip_val)
    v_clipped = np.clip(vol_arr, clip_val, 1.0 - clip_val)
    total = t_clipped + v_clipped
    p_trend = t_clipped / total
    p_vol = v_clipped / total
    entropy = -(p_trend * np.log(p_trend) + p_vol * np.log(p_vol))
    return _wrap_output(entropy, index, any_ndarray, "entropy")


def membership_frame(
    prices: PriceSeries | pd.Series,
    window: int | None = None,
    percentile_window: int | None = None,
    *,
    config: IndicatorConfig | None = None,
    clip: float = _DEFAULT_ENTROPY_CLIP,
) -> pd.DataFrame:
    """Combine the regime cell frame with soft membership columns.

    Chains :func:`cells.cell_frame` with ``trend_m``, ``vol_m``, and ``entropy``
    aligned to the input price index. Window and quantile defaults come from
    :class:`config.IndicatorConfig`. All thresholds are expanding estimates
    through ``t-1``, so no row uses future observations.

    Important calibration caveat (Arrieta-Ibarra et al., 2022):
    ``trend_m`` and ``vol_m`` are soft **scores**,
    NOT probabilities, and ``entropy`` is an ambiguity diagnostic on those
    scores. Without a reliability diagram there is no probability.
    """
    cfg = IndicatorConfig() if config is None else config
    base_window = cfg.trend_window if window is None else window
    base_percentile_window = (
        cfg.percentile_window if percentile_window is None else percentile_window
    )

    frame = cell_frame(
        prices,
        window=base_window,
        percentile_window=base_percentile_window,
    )

    if cfg.membership_trend_lower_quantile == cfg.trend_lower_quantile:
        trend_lo = frame["q30_threshold"]
    else:
        trend_lo = expanding_quantile_threshold(
            frame["trend_strength"], cfg.membership_trend_lower_quantile
        )

    if cfg.membership_trend_upper_quantile == cfg.trend_upper_quantile:
        trend_hi = frame["q70_threshold"]
    else:
        trend_hi = expanding_quantile_threshold(
            frame["trend_strength"], cfg.membership_trend_upper_quantile
        )

    if cfg.membership_volatility_quantile == cfg.volatility_quantile:
        vol_thresh = frame["q67_threshold"]
    else:
        vol_thresh = expanding_quantile_threshold(
            frame["realized_volatility_percentile"], cfg.membership_volatility_quantile
        )

    trend_m = trend_membership(frame["trend_strength"], trend_lo, trend_hi)
    vol_m = volatility_membership(frame["realized_volatility_percentile"], vol_thresh)
    entropy = membership_entropy(trend_m, vol_m, clip=clip)

    return pd.concat([frame, trend_m, vol_m, entropy], axis=1)
