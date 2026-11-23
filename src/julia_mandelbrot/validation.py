"""Validation harness for the market-state indicator columns.

This module implements four out-of-sample validation checks
and aggregates their verdicts:

1. :func:`reliability_diagram`: Per-column calibration via quantile-binned
   reliability diagrams comparing mean predicted scores against realized binary
   outcome frequencies, plus the sample-size-weighted miscalibration summary
   (Arrieta-Ibarra, Gujral, Tannen, Tygert, and Xu, 2022).
2. :func:`transition_log_loss`: Out-of-sample categorical log-loss of causal
   :func:`transitions.next_cell_probabilities` predictions on transitions
   arriving in the holdout slice after the estimation window, benchmarked
   against (a) the persistence baseline (predicting the current cell persists)
   and (b) the causal unconditional baseline (global cell frequencies through
   ``t - 1``), with ``1e-12`` epsilon clamping for zero-probability cells.
3. :func:`roughness_sanity`: Diagnostic sanity verification on rolling variogram
   roughness (:func:`roughvol.rolling_roughness`), checking that ``hurst_logvol``
   is finite on enough eligible windows, median variogram regression ``R^2`` sits
   at or above threshold, and ``H`` lies within a sane ``[0, 1]`` band. Per
   Cont and Das (2024), passing this check describes the observed daily series
   only and never identifies fractional latent volatility.
4. :func:`walk_forward`: Expanding-window walk-forward evaluation with a purged
   horizon between training and evaluation slices (Falck, Rej, and Thesmar,
   2022), scoring one-step-ahead cell classification
   hit rate against the persistence baseline without lookahead leakage.
5. :func:`run_validation`: Aggregates all validation verdicts into a structured
   dictionary of ``{value..., "pass": bool}`` entries.

CRITICAL MEMBERSHIP SEMANTICS:
A failed validation test demotes the corresponding column from downstream use.
For soft membership columns, **even on pass, membership remains a soft SCORE**:
passing the reliability diagram check only licenses further study and never
promotes a logistic membership score to probability status (Arrieta-Ibarra
et al., 2022).
"""

from __future__ import annotations

import math
import operator
from typing import Any

import numpy as np
import pandas as pd

from .cells import TREND_STATES, VOL_STATES, cell_frame
from .config import IndicatorConfig
from .data_adapters import PriceSeries
from .membership import membership_frame
from .roughvol import realized_log_vol, rolling_roughness
from .transitions import CANONICAL_CELLS, next_cell_probabilities
from .trend_vol import _close, realized_volatility, rolling_percentile, trend_strength

# Validation thresholds and protocol constants.
DEFAULT_RELIABILITY_BINS: int = 10
MAX_MISCALIBRATION: float = 0.25
LOG_LOSS_EPS: float = 1e-12
ROUGHNESS_MIN_FINITE_RATIO: float = 0.80
ROUGHNESS_MIN_MEDIAN_R2: float = 0.80
ROUGHNESS_MIN_HURST: float = 0.0
ROUGHNESS_MAX_HURST: float = 1.0
WALK_FORWARD_TRAIN_BARS: int = 504
WALK_FORWARD_HORIZON: int = 1
WALK_FORWARD_STEP: int = 63

_CELL_TO_CODE: dict[str, int] = {cell: idx for idx, cell in enumerate(CANONICAL_CELLS)}


def _validate_series(series: Any, name: str) -> pd.Series:
    if not isinstance(series, pd.Series):
        raise TypeError(f"{name} must be a pandas Series")
    return series


def _validate_positive_int(value: Any, name: str, minimum: int = 1) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer of at least {minimum}, not bool")
    try:
        int_val = operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be an integer of at least {minimum}") from exc
    if int_val < minimum:
        raise ValueError(f"{name} must be an integer of at least {minimum}")
    return int(int_val)


def reliability_diagram(
    score: pd.Series,
    outcome: pd.Series,
    bins: int = DEFAULT_RELIABILITY_BINS,
) -> pd.DataFrame:
    """Compute a quantile-binned reliability diagram and weighted miscalibration summary.

    Bins ``score`` into ``bins`` quantile bins (default deciles, ``bins = 10``)
    and reports, for each bin, the sample count, the mean predicted score
    (``mean_score``), the realized frequency of the binary ``outcome``
    (``frequency``), the per-bin absolute calibration gap
    ``abs_error = |mean_score - frequency|``, and the sample-size-weighted
    miscalibration summary across all bins:

    ``miscalibration = sum(count * |mean_score - frequency|) / sum(count)``

    Basis and interpretation limit (Arrieta-Ibarra, Gujral, Tannen, Tygert, and Xu, 2022):
    A normalized or logistic score is not a probability. Even when a membership
    column exhibits low miscalibration and monotonic realized frequencies across
    quantile bins, it remains a soft **score**; calibration only licenses
    further study, not probability status.

    Parameters
    ----------
    score : pd.Series
        Continuous predicted score series.
    outcome : pd.Series
        Realized binary outcome series (0/1 or boolean) aligned to ``score.index``.
    bins : int, default 10
        Number of quantile bins.

    Returns
    -------
    pd.DataFrame
        DataFrame with one row per non-empty quantile bin and columns
        ``bin``, ``count``, ``mean_score``, ``frequency``, ``abs_error``, and
        ``miscalibration``. The overall weighted miscalibration summary is also
        stored in ``df.attrs["miscalibration"]``.
    """
    _validate_series(score, "score")
    _validate_series(outcome, "outcome")
    if not score.index.equals(outcome.index):
        raise ValueError("score and outcome must share the same index")
    n_bins = _validate_positive_int(bins, "bins", minimum=1)

    score_num = pd.to_numeric(score, errors="coerce").astype("float64")
    if outcome.dtype == bool:
        outcome_num = outcome.astype("float64")
    else:
        outcome_num = pd.to_numeric(outcome, errors="coerce").astype("float64")

    valid_mask = np.isfinite(score_num.to_numpy()) & np.isfinite(outcome_num.to_numpy())
    valid_score = score_num.loc[valid_mask]
    valid_outcome = outcome_num.loc[valid_mask]

    columns = ["bin", "count", "mean_score", "frequency", "abs_error", "miscalibration"]
    if valid_score.empty:
        empty = pd.DataFrame(columns=columns)
        empty.attrs["miscalibration"] = float("nan")
        return empty

    eff_bins = min(n_bins, len(valid_score))
    if eff_bins <= 1 or valid_score.nunique() <= 1:
        bin_ids = pd.Series(0, index=valid_score.index, dtype="int64")
    else:
        try:
            qcut_bins = pd.qcut(valid_score, q=eff_bins, labels=False, duplicates="drop")
            if qcut_bins.isna().all():
                bin_ids = pd.Series(0, index=valid_score.index, dtype="int64")
            else:
                bin_ids = qcut_bins.astype("int64")
        except ValueError:
            bin_ids = pd.Series(0, index=valid_score.index, dtype="int64")

    grouped_Frame = pd.DataFrame(
        {"score": valid_score, "outcome": valid_outcome, "bin": bin_ids}
    )
    agg = (
        grouped_Frame.groupby("bin", as_index=False, sort=True)
        .agg(
            count=("score", "count"),
            mean_score=("score", "mean"),
            frequency=("outcome", "mean"),
        )
        .reset_index(drop=True)
    )
    agg["bin"] = np.arange(len(agg), dtype="int64")
    agg["count"] = agg["count"].astype("int64")
    agg["mean_score"] = agg["mean_score"].astype("float64")
    agg["frequency"] = agg["frequency"].astype("float64")
    agg["abs_error"] = (agg["mean_score"] - agg["frequency"]).abs().astype("float64")

    total_count = float(agg["count"].sum())
    miscalibration = (
        float(np.dot(agg["count"].to_numpy(dtype=float), agg["abs_error"].to_numpy(dtype=float)) / total_count)
        if total_count > 0.0
        else float("nan")
    )
    agg["miscalibration"] = float(miscalibration)
    agg.attrs["miscalibration"] = float(miscalibration)
    return agg.loc[:, columns]


def transition_log_loss(
    cells: pd.Series,
    window: int | None = None,
    floor: int | None = None,
    *,
    eps: float = LOG_LOSS_EPS,
) -> dict[str, Any]:
    """Compare causal transition probability predictions against persistence and unconditional baselines by log-loss.

    Evaluates :func:`transitions.next_cell_probabilities` on one-step transitions
    ``(cells[t] -> cells[t + 1])`` arriving in the holdout slice AFTER the initial
    estimation window ``window`` (i.e. prediction indices ``t >= first_valid + window``).
    At each holdout row ``t`` where ``cells[t]``, ``cells[t + 1]``, and the
    predicted next-cell distribution are valid, categorical log-loss
    ``-log(p_predicted(cells[t + 1]))`` is scored for:

    1. **Transition model** (``transition_log_loss``): causal transition matrix
       probabilities from :func:`transitions.next_cell_probabilities` using only
       transitions arriving at or before ``t - 1``.
    2. **Persistence baseline** (``persistence_log_loss``): predicts that the
       current cell ``cells[t]`` persists at ``t + 1`` with probability ``1.0``
       (and ``0.0`` for all other cells).
    3. **Unconditional baseline** (``unconditional_log_loss``): predicts global
       empirical cell frequencies observed strictly through ``t - 1``
       (``cells[0..t-1]``).

    Zero-probability predictions are clamped to ``[eps, 1.0]`` with
    ``eps = 1e-12`` (:data:`LOG_LOSS_EPS`) so log-loss remains finite on unseen
    or non-persistent transitions. The transition
    matrix passes validation only if it beats both the persistence baseline and
    the unconditional baseline out of sample.

    Parameters
    ----------
    cells : pd.Series
        Sequence of discrete regime cell labels.
    window : int, optional
        Trailing estimation window length. Defaults to
        ``IndicatorConfig().transition_window`` (756).
    floor : int, optional
        Minimum required outgoing transition count per cell. Defaults to
        ``IndicatorConfig().transition_count_floor`` (30).
    eps : float, default 1e-12
        Clamping floor for zero-probability cells.

    Returns
    -------
    dict[str, Any]
        Dictionary containing ``transition_log_loss``, ``persistence_log_loss``,
        ``unconditional_log_loss`` (plus aliases ``transition``, ``persistence``,
        ``unconditional``), ``n_evaluated``, ``beats_persistence``,
        ``beats_unconditional``, and ``pass``.
    """
    _validate_series(cells, "cells")
    cfg = IndicatorConfig()
    win = (
        cfg.transition_window
        if window is None
        else _validate_positive_int(window, "window", minimum=1)
    )
    flr = (
        cfg.transition_count_floor
        if floor is None
        else _validate_positive_int(floor, "floor", minimum=1)
    )
    if isinstance(eps, bool) or not isinstance(eps, (int, float)) or not (0.0 < float(eps) < 0.5):
        raise ValueError("eps must be a float strictly between 0 and 0.5")
    eps_val = float(eps)

    probs = next_cell_probabilities(cells, window=win, floor=flr)
    prob_cols = [col for col in probs.columns if col.startswith("p_next_")]
    states = [col.removeprefix("p_next_") for col in prob_cols]
    state_to_idx = {state: idx for idx, state in enumerate(states)}

    non_null_positions = np.flatnonzero(cells.notna().to_numpy())
    if len(non_null_positions) == 0 or not states:
        return {
            "transition_log_loss": float("nan"),
            "persistence_log_loss": float("nan"),
            "unconditional_log_loss": float("nan"),
            "transition": float("nan"),
            "persistence": float("nan"),
            "unconditional": float("nan"),
            "n_evaluated": 0,
            "beats_persistence": False,
            "beats_unconditional": False,
            "pass": False,
        }

    first_valid_pos = int(non_null_positions[0])
    holdout_start = first_valid_pos + win

    cell_values = cells.to_numpy(dtype=object)
    prob_matrix = probs.loc[:, prob_cols].to_numpy(dtype=float)

    # Precompute causal global cell counts through t - 1 (indices 0..t-1).
    n_obs = len(cells)
    n_states = len(states)
    state_indicators = np.zeros((n_obs, n_states), dtype=float)
    valid_cell_mask = np.zeros(n_obs, dtype=float)
    for pos in non_null_positions:
        label = cell_values[pos]
        if label in state_to_idx:
            state_indicators[pos, state_to_idx[label]] = 1.0
            valid_cell_mask[pos] = 1.0

    # Cumulative counts shifted by 1 so row t holds counts over 0..t-1.
    cum_state_counts = np.vstack(
        [np.zeros((1, n_states), dtype=float), np.cumsum(state_indicators[:-1], axis=0)]
    )
    cum_totals = np.concatenate([[0.0], np.cumsum(valid_cell_mask[:-1])])

    model_losses: list[float] = []
    pers_losses: list[float] = []
    uncond_losses: list[float] = []

    for t in range(holdout_start, n_obs - 1):
        cur_cell = cell_values[t]
        next_cell = cell_values[t + 1]
        if pd.isna(cur_cell) or pd.isna(next_cell):
            continue
        if cur_cell not in state_to_idx or next_cell not in state_to_idx:
            continue

        row_probs = prob_matrix[t]
        if not np.isfinite(row_probs).all():
            continue
        if cum_totals[t] <= 0.0:
            continue

        target_idx = state_to_idx[next_cell]

        p_model = float(np.clip(row_probs[target_idx], eps_val, 1.0))
        p_pers = 1.0 if next_cell == cur_cell else eps_val
        p_uncond_raw = cum_state_counts[t, target_idx] / cum_totals[t]
        p_uncond = float(np.clip(p_uncond_raw, eps_val, 1.0))

        model_losses.append(-math.log(p_model))
        pers_losses.append(-math.log(p_pers))
        uncond_losses.append(-math.log(p_uncond))

    n_eval = len(model_losses)
    if n_eval == 0:
        return {
            "transition_log_loss": float("nan"),
            "persistence_log_loss": float("nan"),
            "unconditional_log_loss": float("nan"),
            "transition": float("nan"),
            "persistence": float("nan"),
            "unconditional": float("nan"),
            "n_evaluated": 0,
            "beats_persistence": False,
            "beats_unconditional": False,
            "pass": False,
        }

    transition_ll = float(np.mean(model_losses))
    persistence_ll = float(np.mean(pers_losses))
    unconditional_ll = float(np.mean(uncond_losses))
    beats_persistence = bool(transition_ll < persistence_ll)
    beats_unconditional = bool(transition_ll < unconditional_ll)
    passed = bool(beats_persistence and beats_unconditional)

    return {
        "transition_log_loss": transition_ll,
        "persistence_log_loss": persistence_ll,
        "unconditional_log_loss": unconditional_ll,
        "transition": transition_ll,
        "persistence": persistence_ll,
        "unconditional": unconditional_ll,
        "n_evaluated": int(n_eval),
        "beats_persistence": beats_persistence,
        "beats_unconditional": beats_unconditional,
        "pass": passed,
    }


def roughness_sanity(
    logvol: pd.Series,
    window: int | None = None,
    *,
    min_finite_ratio: float = ROUGHNESS_MIN_FINITE_RATIO,
    min_median_r2: float = ROUGHNESS_MIN_MEDIAN_R2,
    min_hurst: float = ROUGHNESS_MIN_HURST,
    max_hurst: float = ROUGHNESS_MAX_HURST,
) -> dict[str, Any]:
    """Run sanity checks on rolling variogram roughness.

    Verifies three conditions on :func:`roughvol.rolling_roughness`:
    1. ``finite_pass``: ``hurst_logvol`` is finite on at least
       ``min_finite_ratio`` (default ``0.80``) of eligible post-warmup windows.
    2. ``r2_pass``: the median variogram regression ``R^2`` across finite
       windows is at least ``min_median_r2`` (default ``0.80``).
    3. ``hurst_band_pass``: the median ``hurst_logvol`` sits within the sane
       band ``[min_hurst, max_hurst]`` (default ``[0.0, 1.0]``) and at least
       ``min_finite_ratio`` of finite windows lie within that band.

    Reading limit (Cont and Das, 2024):
    Passing this sanity check validates the observed daily realized
    log-volatility column as a well-behaved descriptor for volatility
    forecasting (Bennedsen, Lunde, and Pakkanen, 2022); it does NOT authorize
    the claim that latent instantaneous volatility is fractional.

    Parameters
    ----------
    logvol : pd.Series
        Realized log-volatility series (e.g. from :func:`roughvol.realized_log_vol`).
    window : int, optional
        Rolling variogram window length. Defaults to
        ``IndicatorConfig().roughvol_window`` (252).
    min_finite_ratio : float, default 0.80
        Minimum required fraction of finite post-warmup estimates.
    min_median_r2 : float, default 0.80
        Minimum required median variogram ``R^2``.
    min_hurst : float, default 0.0
        Lower bound of the sane Hurst exponent band.
    max_hurst : float, default 1.0
        Upper bound of the sane Hurst exponent band.

    Returns
    -------
    dict[str, Any]
        Dictionary of summary metrics and boolean verdicts ``finite_pass``,
        ``r2_pass``, ``hurst_band_pass``, and ``pass``.
    """
    _validate_series(logvol, "logvol")
    cfg = IndicatorConfig()
    win = (
        cfg.roughvol_window
        if window is None
        else _validate_positive_int(window, "window", minimum=2)
    )

    rough = rolling_roughness(logvol, window=win)
    h_arr = rough["hurst_logvol"].to_numpy(dtype=float)
    r2_arr = rough["hurst_r2"].to_numpy(dtype=float)

    logvol_num = pd.to_numeric(logvol, errors="coerce").to_numpy(dtype=float)
    n_total = len(logvol_num)
    n_valid_input = int(np.isfinite(logvol_num).sum())
    n_post_warmup = max(0, n_total - win + 1)
    n_eligible = max(0, n_valid_input - win + 1)

    finite_mask = np.isfinite(h_arr) & np.isfinite(r2_arr)
    n_finite = int(np.sum(finite_mask))

    denom = n_eligible if n_eligible > 0 else n_post_warmup
    finite_ratio = float(n_finite / denom) if denom > 0 else 0.0

    # If the input has interior NaNs that eliminate most post-warmup windows,
    # also check coverage against post-warmup slots when leading warmup is accounted for.
    finite_pass = bool(
        n_post_warmup > 0
        and n_finite > 0
        and finite_ratio >= float(min_finite_ratio)
    )

    if n_finite > 0:
        h_finite = h_arr[finite_mask]
        r2_finite = r2_arr[finite_mask]
        median_hurst = float(np.median(h_finite))
        mean_hurst = float(np.mean(h_finite))
        median_r2 = float(np.median(r2_finite))
        mean_r2 = float(np.mean(r2_finite))
        in_band_ratio = float(
            np.mean((h_finite >= float(min_hurst)) & (h_finite <= float(max_hurst)))
        )
    else:
        median_hurst = float("nan")
        mean_hurst = float("nan")
        median_r2 = float("nan")
        mean_r2 = float("nan")
        in_band_ratio = 0.0

    r2_pass = bool(np.isfinite(median_r2) and median_r2 >= float(min_median_r2))
    hurst_band_pass = bool(
        np.isfinite(median_hurst)
        and (float(min_hurst) <= median_hurst <= float(max_hurst))
        and in_band_ratio >= float(min_finite_ratio)
    )
    passed = bool(finite_pass and r2_pass and hurst_band_pass)

    return {
        "n_total": int(n_total),
        "n_eligible": int(denom),
        "n_finite": int(n_finite),
        "finite_ratio": finite_ratio,
        "median_hurst": median_hurst,
        "mean_hurst": mean_hurst,
        "median_r2": median_r2,
        "mean_r2": mean_r2,
        "in_band_ratio": in_band_ratio,
        "finite_pass": finite_pass,
        "r2_pass": r2_pass,
        "hurst_band_pass": hurst_band_pass,
        "pass": passed,
    }


def walk_forward(
    prices: PriceSeries | pd.Series,
    train_bars: int = WALK_FORWARD_TRAIN_BARS,
    horizon: int = WALK_FORWARD_HORIZON,
    step: int = WALK_FORWARD_STEP,
    *,
    window: int | None = None,
    percentile_window: int | None = None,
) -> pd.DataFrame:
    """Run an expanding-window walk-forward validation with a purged horizon.

    Protocol (Falck, Rej, and Thesmar, 2022):
    For each expanding fold ``k`` with training end ``train_end`` advancing from
    ``train_bars`` in increments of ``step``:

    1. **Causal training fit through t - 1**:
       Using ONLY the training slice ``0 .. train_end - 1``, fit the causal
       threshold chain (``train_q30``, ``train_q70`` on ``trend_strength`` and
       ``train_q67`` on ``realized_volatility_percentile``) and estimate the
       6x6 empirical transition count matrix strictly from transitions completed
       within ``0 .. train_end - 1``.
    2. **Purged horizon**:
       Purge ``horizon`` bars ``[train_end, train_end + horizon)`` between the
       training slice and the evaluation slice so that multi-period or
       one-step horizon targets cannot leak across the split boundary.
    3. **Out-of-sample evaluation**:
       On the evaluation slice ``[eval_start, eval_end)`` where
       ``eval_start = train_end + horizon`` and
       ``eval_end = min(eval_start + step, len(prices) - 1)``, classify cells
       against the frozen training thresholds and score one-step-ahead cell
       classification hit rate (modal trained transition vs realized next cell)
       against the persistence baseline (``cell[t + 1] == cell[t]``).

    Parameters
    ----------
    prices : PriceSeries or pd.Series
        Positive close price series or :class:`PriceSeries` container.
    train_bars : int, default 504
        Initial expanding training window size in bars.
    horizon : int, default 1
        Number of bars purged between each training slice and evaluation slice.
    step : int, default 63
        Number of bars per evaluation slice / step between expanding folds.
    window : int, optional
        Trend and realized volatility rolling window. Defaults to
        ``IndicatorConfig().trend_window`` (63), scaled down automatically only
        when ``train_bars`` is shorter than the default warmup length.
    percentile_window : int, optional
        Realized volatility percentile rolling window. Defaults to
        ``IndicatorConfig().percentile_window`` (252), scaled down automatically
        only when ``train_bars`` is shorter than the default warmup length.

    Returns
    -------
    pd.DataFrame
        DataFrame with one row per walk-forward fold and columns:
        ``fold``, ``train_start``, ``train_end``, ``train_bars``,
        ``purge_start``, ``purge_end``, ``purged_bars``, ``eval_start``,
        ``eval_end``, ``n_eval``, ``train_q30``, ``train_q70``, ``train_q67``,
        ``hit_rate``, ``persistence_hit_rate``, and ``hit_rate_diff``.
    """
    close = _close(prices)
    n_train_init = _validate_positive_int(train_bars, "train_bars", minimum=4)
    n_horizon = _validate_positive_int(horizon, "horizon", minimum=1)
    n_step = _validate_positive_int(step, "step", minimum=1)

    cfg = IndicatorConfig()
    default_warmup = cfg.volatility_window + cfg.percentile_window
    if window is not None:
        trend_n = _validate_positive_int(window, "window", minimum=2)
        vol_n = trend_n
    elif n_train_init > default_warmup + 2:
        trend_n = cfg.trend_window
        vol_n = cfg.volatility_window
    else:
        trend_n = max(2, min(cfg.trend_window, n_train_init // 4))
        vol_n = trend_n

    if percentile_window is not None:
        pct_n = _validate_positive_int(percentile_window, "percentile_window", minimum=2)
    elif n_train_init > default_warmup + 2:
        pct_n = cfg.percentile_window
    else:
        pct_n = max(2, min(cfg.percentile_window, n_train_init // 2))

    columns = [
        "fold",
        "train_start",
        "train_end",
        "train_bars",
        "purge_start",
        "purge_end",
        "purged_bars",
        "eval_start",
        "eval_end",
        "n_eval",
        "train_q30",
        "train_q70",
        "train_q67",
        "hit_rate",
        "persistence_hit_rate",
        "hit_rate_diff",
    ]

    n_obs = len(close)
    if n_obs <= n_train_init + n_horizon or n_obs < max(trend_n, vol_n + pct_n):
        return pd.DataFrame(columns=columns)

    # Causal rolling features: value at index t depends strictly on close[0..t].
    trend_ser = trend_strength(close, trend_n)
    vol_ser = realized_volatility(close, vol_n)
    vol_pct_ser = rolling_percentile(vol_ser, pct_n)

    trend_arr = trend_ser.to_numpy(dtype=float)
    vol_pct_arr = vol_pct_ser.to_numpy(dtype=float)
    valid_feat = np.isfinite(trend_arr) & np.isfinite(vol_pct_arr)

    rows: list[dict[str, Any]] = []
    fold_idx = 0

    for train_end in range(n_train_init, n_obs - n_horizon, n_step):
        eval_start = train_end + n_horizon
        if eval_start >= n_obs - 1:
            break
        eval_end = min(eval_start + n_step, n_obs - 1)

        # Fit thresholds strictly on the training slice 0 .. train_end - 1.
        train_mask = valid_feat[:train_end]
        if int(np.sum(train_mask)) < 2:
            continue

        train_trend_valid = trend_arr[:train_end][train_mask]
        train_vol_valid = vol_pct_arr[:train_end][train_mask]

        train_q30 = float(np.quantile(train_trend_valid, cfg.trend_lower_quantile))
        train_q70 = float(np.quantile(train_trend_valid, cfg.trend_upper_quantile))
        train_q67 = float(np.quantile(train_vol_valid, cfg.volatility_quantile))

        # Classify training slice cells using only training parameters to fit transition dynamics.
        t_state_train = np.ones(train_end, dtype=int)
        t_state_train[trend_arr[:train_end] < train_q30] = 0
        t_state_train[trend_arr[:train_end] > train_q70] = 2

        v_state_train = np.ones(train_end, dtype=int)
        v_state_train[vol_pct_arr[:train_end] <= train_q67] = 0

        train_codes = t_state_train * 2 + v_state_train
        valid_train_trans = train_mask[:-1] & train_mask[1:]
        if np.any(valid_train_trans):
            from_codes = train_codes[:-1][valid_train_trans]
            to_codes = train_codes[1:][valid_train_trans]
            trans_matrix = np.bincount(
                from_codes * 6 + to_codes, minlength=36
            ).reshape(6, 6)
        else:
            trans_matrix = np.zeros((6, 6), dtype=int)

        # Modal next cell per origin state; ties or zero counts default to persistence.
        modal_next = np.arange(6, dtype=int)
        for state_code in range(6):
            row_counts = trans_matrix[state_code]
            row_total = int(row_counts.sum())
            if row_total > 0:
                max_count = int(row_counts.max())
                if row_counts[state_code] == max_count:
                    modal_next[state_code] = state_code
                else:
                    modal_next[state_code] = int(np.argmax(row_counts))

        # Classify evaluation slice [eval_start .. eval_end] using frozen training thresholds.
        eval_slice_trend = trend_arr[eval_start : eval_end + 1]
        eval_slice_vol = vol_pct_arr[eval_start : eval_end + 1]
        eval_slice_valid = valid_feat[eval_start : eval_end + 1]

        t_state_eval = np.ones(len(eval_slice_trend), dtype=int)
        t_state_eval[eval_slice_trend < train_q30] = 0
        t_state_eval[eval_slice_trend > train_q70] = 2

        v_state_eval = np.ones(len(eval_slice_vol), dtype=int)
        v_state_eval[eval_slice_vol <= train_q67] = 0

        eval_codes = t_state_eval * 2 + v_state_eval
        valid_eval_steps = eval_slice_valid[:-1] & eval_slice_valid[1:]
        n_eval = int(np.sum(valid_eval_steps))
        if n_eval == 0:
            continue

        cur_eval = eval_codes[:-1][valid_eval_steps]
        next_eval = eval_codes[1:][valid_eval_steps]
        pred_eval = modal_next[cur_eval]

        hit_rate = float(np.mean(pred_eval == next_eval))
        persistence_hit_rate = float(np.mean(cur_eval == next_eval))

        rows.append(
            {
                "fold": int(fold_idx),
                "train_start": 0,
                "train_end": int(train_end),
                "train_bars": int(train_end),
                "purge_start": int(train_end),
                "purge_end": int(eval_start),
                "purged_bars": int(n_horizon),
                "eval_start": int(eval_start),
                "eval_end": int(eval_end),
                "n_eval": int(n_eval),
                "train_q30": train_q30,
                "train_q70": train_q70,
                "train_q67": train_q67,
                "hit_rate": hit_rate,
                "persistence_hit_rate": persistence_hit_rate,
                "hit_rate_diff": float(hit_rate - persistence_hit_rate),
            }
        )
        fold_idx += 1

    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows, columns=columns)


def run_validation(
    prices: PriceSeries | pd.Series,
    *,
    config: IndicatorConfig | None = None,
    bins: int = DEFAULT_RELIABILITY_BINS,
    train_bars: int | None = None,
    horizon: int = WALK_FORWARD_HORIZON,
    step: int | None = None,
) -> dict[str, dict[str, Any]]:
    """Execute the full validation suite on ``prices``.

    Aggregates four validation verdicts, where every entry is a dictionary of
    diagnostic values plus ``"pass": bool``:

    - ``"membership"``: Reliability-diagram calibration check on ``trend_m`` and
      ``vol_m`` against realized next-bar high-trend and high-volatility state
      indicators on valid rows.
      **CRITICAL SEMANTIC RULE (Arrieta-Ibarra et al., 2022)**: Even when ``"pass"`` is ``True``,
      membership remains a soft **SCORE** (``"status": "score"``). Calibration
      only licenses further study, NEVER probability status.
    - ``"transition"``: Out-of-sample log-loss comparison of
      :func:`transitions.next_cell_probabilities` against the persistence and
      unconditional baselines (:func:`transition_log_loss`).
    - ``"roughness"``: Variogram roughness sanity check on
      :func:`roughvol.realized_log_vol` (:func:`roughness_sanity`).
    - ``"walk_forward"``: Expanding-window purged walk-forward classification
      protocol summary (:func:`walk_forward`).

    Parameters
    ----------
    prices : PriceSeries or pd.Series
        Positive daily close prices or :class:`PriceSeries` container.
    config : IndicatorConfig, optional
        Configuration instance providing default windows and floors.
    bins : int, default 10
        Number of quantile bins for the membership reliability diagrams.
    train_bars : int, optional
        Initial training bars for walk-forward evaluation. Defaults to
        :data:`WALK_FORWARD_TRAIN_BARS` (504), scaled automatically when
        ``prices`` is shorter than the default multi-year history.
    horizon : int, default 1
        Purged horizon in bars between walk-forward train and evaluation slices.
    step : int, optional
        Walk-forward step size in bars. Defaults to :data:`WALK_FORWARD_STEP` (63).

    Returns
    -------
    dict[str, dict[str, Any]]
        Dictionary mapping ``"membership"``, ``"transition"``, ``"roughness"``,
        and ``"walk_forward"`` to their verdict dictionaries.
    """
    cfg = IndicatorConfig() if config is None else config
    close = _close(prices)
    n_obs = len(close)

    # Adapt window sizes gracefully only when a short series (< default warmups) is supplied.
    default_cell_warmup = cfg.volatility_window + cfg.percentile_window
    if n_obs > default_cell_warmup + 30:
        cell_win = cfg.trend_window
        pct_win = cfg.percentile_window
    else:
        cell_win = max(2, min(cfg.trend_window, n_obs // 8))
        pct_win = max(2, min(cfg.percentile_window, n_obs // 4))

    m_frame = membership_frame(
        close,
        window=cell_win,
        percentile_window=pct_win,
        config=cfg,
    )

    # 1. Membership reliability diagram check.
    valid_curr = m_frame["valid"].to_numpy(dtype=bool)
    valid_pair = pd.Series(False, index=m_frame.index, dtype=bool)
    if len(m_frame) > 1:
        valid_pair.iloc[:-1] = valid_curr[:-1] & valid_curr[1:]

    next_trend_high = (
        (m_frame["trend_state"].shift(-1) == "high")
        .loc[valid_pair]
        .astype("float64")
    )
    next_vol_high = (
        (m_frame["vol_state"].shift(-1) == "high")
        .loc[valid_pair]
        .astype("float64")
    )
    trend_scores = m_frame.loc[valid_pair, "trend_m"]
    vol_scores = m_frame.loc[valid_pair, "vol_m"]

    trend_diag = reliability_diagram(trend_scores, next_trend_high, bins=bins)
    vol_diag = reliability_diagram(vol_scores, next_vol_high, bins=bins)

    trend_miscal = float(trend_diag.attrs.get("miscalibration", float("nan")))
    vol_miscal = float(vol_diag.attrs.get("miscalibration", float("nan")))
    mean_miscal = (
        0.5 * (trend_miscal + vol_miscal)
        if np.isfinite(trend_miscal) and np.isfinite(vol_miscal)
        else float("nan")
    )

    trend_rising = bool(
        len(trend_diag) >= 2
        and float(trend_diag["frequency"].iloc[-1]) > float(trend_diag["frequency"].iloc[0])
    )
    vol_rising = bool(
        len(vol_diag) >= 2
        and float(vol_diag["frequency"].iloc[-1]) > float(vol_diag["frequency"].iloc[0])
    )
    membership_pass = bool(
        np.isfinite(mean_miscal)
        and mean_miscal <= MAX_MISCALIBRATION
        and trend_rising
        and vol_rising
    )

    membership_verdict: dict[str, Any] = {
        "trend_miscalibration": trend_miscal,
        "vol_miscalibration": vol_miscal,
        "mean_miscalibration": mean_miscal,
        "max_miscalibration_threshold": MAX_MISCALIBRATION,
        "trend_frequency_rising": trend_rising,
        "vol_frequency_rising": vol_rising,
        "n_bins_trend": int(len(trend_diag)),
        "n_bins_vol": int(len(vol_diag)),
        "status": "score",
        "note": (
            "Even on pass, membership remains a soft score; calibration only "
            "licenses further study, not probability status (Arrieta-Ibarra et al., 2022)."
        ),
        "pass": membership_pass,
    }

    # 2. Transition log-loss check.
    n_valid_cells = int(m_frame["cell"].notna().sum())
    if n_valid_cells > cfg.transition_window + 20:
        trans_win = cfg.transition_window
        trans_floor = cfg.transition_count_floor
    else:
        trans_win = max(5, n_valid_cells // 2)
        trans_floor = max(1, min(cfg.transition_count_floor, n_valid_cells // 12))

    transition_verdict = transition_log_loss(
        m_frame["cell"],
        window=trans_win,
        floor=trans_floor,
    )

    # 3. Roughness sanity check.
    if n_obs > cfg.roughvol_realized_volatility_window + cfg.roughvol_window + 10:
        rv_win = cfg.roughvol_realized_volatility_window
        rough_win = cfg.roughvol_window
    else:
        rv_win = max(2, min(cfg.roughvol_realized_volatility_window, n_obs // 10))
        rough_win = max(12, min(cfg.roughvol_window, n_obs // 2))

    logvol = realized_log_vol(close, rv_window=rv_win)
    roughness_verdict = roughness_sanity(logvol, window=rough_win)

    # 4. Walk-forward check.
    if train_bars is not None:
        wf_train = train_bars
    elif n_obs > WALK_FORWARD_TRAIN_BARS + horizon + 20:
        wf_train = WALK_FORWARD_TRAIN_BARS
    else:
        wf_train = max(20, int(n_obs * 0.6))

    if step is not None:
        wf_step = step
    elif n_obs > WALK_FORWARD_TRAIN_BARS + WALK_FORWARD_STEP:
        wf_step = WALK_FORWARD_STEP
    else:
        wf_step = max(5, (n_obs - wf_train - horizon) // 3)

    wf_frame = walk_forward(
        close,
        train_bars=wf_train,
        horizon=horizon,
        step=wf_step,
        window=cell_win,
        percentile_window=pct_win,
    )

    if wf_frame.empty:
        wf_verdict: dict[str, Any] = {
            "n_folds": 0,
            "total_evaluated": 0,
            "purged_horizon": int(horizon),
            "mean_hit_rate": float("nan"),
            "mean_persistence_hit_rate": float("nan"),
            "pass": False,
        }
    else:
        weights = wf_frame["n_eval"].to_numpy(dtype=float)
        total_eval = float(weights.sum())
        mean_hit = float(np.dot(weights, wf_frame["hit_rate"].to_numpy(dtype=float)) / total_eval)
        mean_pers = float(
            np.dot(weights, wf_frame["persistence_hit_rate"].to_numpy(dtype=float)) / total_eval
        )
        purged_ok = bool((wf_frame["eval_start"] - wf_frame["train_end"] == horizon).all())
        wf_pass = bool(
            len(wf_frame) > 0
            and total_eval > 0
            and purged_ok
            and np.isfinite(mean_hit)
            and mean_hit >= mean_pers
        )
        wf_verdict = {
            "n_folds": int(len(wf_frame)),
            "total_evaluated": int(total_eval),
            "purged_horizon": int(horizon),
            "mean_hit_rate": mean_hit,
            "mean_persistence_hit_rate": mean_pers,
            "pass": wf_pass,
        }

    return {
        "membership": membership_verdict,
        "transition": transition_verdict,
        "roughness": roughness_verdict,
        "walk_forward": wf_verdict,
    }
