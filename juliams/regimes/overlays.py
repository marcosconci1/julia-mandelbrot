"""
Adaptive regime overlays applied as additive columns to the main DataFrame.

These functions are pure: they take a DataFrame plus parameters and return
the DataFrame with new columns appended. The legacy ``regime`` column is
never modified — overlays are observation, not replacement.

All overlays are causal (no future leakage) by construction:
- ``apply_adaptive_threshold_overlay`` calls
  :func:`fit_adaptive_regime_walkforward` which uses ``shift_calibration=True``.
- ``apply_ewma_overlay`` uses ``compute_ewma_trend_strength`` which is causal
  by construction (rolling slope + EWMA of past returns).
- ``apply_markov_overlay`` returns *filtered* (causal) probabilities, not
  smoothed. Smoothed probabilities use the whole sample and would be
  inappropriate for a column meant to support live decisions.
- ``apply_bocpd_overlay`` is online by definition.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def apply_adaptive_threshold_overlay(
    df: pd.DataFrame,
    window: int = 252,
    q_up: float = 0.70,
    q_down: float = 0.30,
    floor: float = 0.10,
    price_col: str = "Close",
    halflife: float = 20.0,
) -> pd.DataFrame:
    """Add a ``regime_adaptive`` column using rolling-quantile thresholds.

    Always uses EWMA-z internally (the rolling-quantile thresholds are
    most meaningful on a vol-scaled signal). The halflife is independent
    of any caller-supplied EWMA overlay.
    """
    from juliams.regimes.walk_forward_adaptive import (
        AdaptiveRegimeConfig,
        fit_adaptive_regime_walkforward,
    )

    cfg = AdaptiveRegimeConfig(
        trend_halflife=halflife,
        threshold_window=window,
        q_up=q_up,
        q_down=q_down,
        abs_floor=floor,
    )
    out = fit_adaptive_regime_walkforward(df, cfg, price_col=price_col)
    df = df.copy()
    df["regime_adaptive"] = out.regime
    return df


def apply_ewma_overlay(
    df: pd.DataFrame,
    halflife: float,
    window: int = 20,
    price_col: str = "Close",
) -> pd.DataFrame:
    """Add a ``trend_strength_ewma`` column."""
    from juliams.features.trend import compute_ewma_trend_strength

    df = df.copy()
    df["trend_strength_ewma"] = compute_ewma_trend_strength(
        df, window=window, halflife=halflife, price_col=price_col
    )
    return df


def _fetch_vol_series(
    ticker: str,
    start: object,
    end: object,
) -> Optional[pd.Series]:
    """Fetch a vol-index ticker's daily close. Returns None on any
    failure so callers can fall back to the univariate path."""
    try:
        import yfinance as yf
    except ImportError:
        logger.warning("yfinance not available; cannot auto-fetch vol channel")
        return None
    try:
        raw = yf.download(
            ticker,
            start=start,
            end=end,
            progress=False,
            auto_adjust=False,
        )
    except Exception as exc:  # network, rate-limit, etc.
        logger.warning("Failed to fetch vol-index %s: %s", ticker, exc)
        return None
    if raw is None or len(raw) == 0:
        logger.warning("No data returned for vol-index %s", ticker)
        return None
    close = raw["Close"] if "Close" in raw.columns else raw.iloc[:, 0]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close.name = "implied_vol"
    # Vol indices are quoted in percent; convert to fraction for
    # numerical comparability with returns.
    return close.astype(float) / 100.0


def apply_markov_overlay(
    df: pd.DataFrame,
    return_col: str = "log_return",
    min_dwell: int = 1,
    threshold: float = 0.5,
    vol_channel: object = None,
) -> pd.DataFrame:
    """Add ``markov_prob_high`` and ``markov_state`` columns.

    Uses *filtered* probabilities so the column is causal (suitable for
    columns that downstream code may treat as live signals). Apply
    ``min_dwell`` post-processing if requested to suppress spurious
    flips during structural breaks.

    Parameters
    ----------
    vol_channel
        - ``None`` (default): use the univariate Markov fit on returns
          only.
        - ``str``: a yfinance-compatible ticker (e.g. ``"^GVZ"``); the
          implied-vol series is auto-fetched and used as the second
          channel in a multivariate HMM fit.
        - ``pd.Series``: a pre-aligned implied-vol series, used
          directly without any fetch.

        If a fetch fails (no network, rate-limit, ticker unavailable),
        the function emits a warning and falls back to the univariate
        path so downstream callers still get the standard columns.
    """
    from juliams.regimes.markov import (
        enforce_min_dwell,
        fit_markov_variance_regime,
        label_markov_regimes,
    )

    if return_col not in df.columns:
        if "Close" in df.columns:
            returns = np.log(df["Close"]).diff()
        else:
            raise ValueError(
                f"DataFrame missing both '{return_col}' and 'Close' columns; "
                "cannot derive returns for Markov overlay."
            )
    else:
        returns = df[return_col]

    vol_series: Optional[pd.Series] = None
    if isinstance(vol_channel, str):
        vol_series = _fetch_vol_series(
            vol_channel, start=df.index.min(), end=df.index.max()
        )
    elif isinstance(vol_channel, pd.Series):
        vol_series = vol_channel
    elif vol_channel is not None:
        raise TypeError(
            f"vol_channel must be None, str (ticker), or pd.Series; "
            f"got {type(vol_channel).__name__}"
        )

    df = df.copy()

    if vol_series is not None:
        # Try the multivariate fit. On any failure (alignment, fit
        # convergence, missing optional dep), fall back to univariate.
        try:
            from juliams.regimes.markov import fit_multivariate_markov_regime
            # Align by calendar date to absorb tz differences between
            # the price fetch (typically tz-aware) and the IV fetch
            # (typically tz-naive). We normalise both indices to plain
            # dates, reindex, then restore the original returns index
            # for the output columns.
            ret_dates = pd.DatetimeIndex(returns.index.normalize().tz_localize(None))
            vol_dates = pd.DatetimeIndex(vol_series.index)
            if vol_dates.tz is not None:
                vol_dates = vol_dates.tz_convert(None)
            vol_dates = vol_dates.normalize()
            vol_by_date = pd.Series(
                vol_series.values, index=vol_dates, name="implied_vol"
            )
            aligned_iv = vol_by_date.reindex(ret_dates).values
            features = pd.DataFrame(
                {
                    "log_return": returns.values,
                    "implied_vol": aligned_iv,
                },
                index=returns.index,
            )
            multi = fit_multivariate_markov_regime(features)
            df["markov_prob_high"] = multi.filtered_prob_high
            from juliams.regimes.markov import enforce_min_dwell
            labels = pd.Series("Unknown", index=multi.filtered_prob_high.index, dtype=object)
            valid = multi.filtered_prob_high.notna()
            labels.loc[valid & (multi.filtered_prob_high > threshold)] = "High"
            labels.loc[valid & (multi.filtered_prob_high <= threshold)] = "Low"
            if min_dwell > 1:
                labels = enforce_min_dwell(labels, min_days=min_dwell)
            df["markov_state"] = labels
            return df
        except Exception as exc:
            logger.warning(
                "Multivariate Markov fit failed (%s); falling back to univariate.",
                exc,
            )

    # Univariate fallback (original behaviour).
    fit = fit_markov_variance_regime(returns)
    labels = label_markov_regimes(fit, threshold=threshold, use_filtered=True)
    if min_dwell > 1:
        labels = enforce_min_dwell(labels, min_days=min_dwell)

    df["markov_prob_high"] = fit.filtered_prob_high
    df["markov_state"] = labels
    return df


def apply_consensus_overlay(
    df: pd.DataFrame,
    window_days: int = 5,
    rl_drop_threshold: int = 30,
    prob_high_jump_threshold: float = 0.3,
    cooldown_days: int = 10,
) -> pd.DataFrame:
    """Add a ``consensus_event`` boolean column.

    Requires that ``bocpd_run_length`` and ``markov_prob_high`` already
    exist on ``df`` (i.e. the BOCPD and Markov overlays were applied
    earlier). Raises ValueError otherwise.
    """
    from juliams.regimes.consensus import detect_consensus_change_points

    missing = {"bocpd_run_length", "markov_prob_high"} - set(df.columns)
    if missing:
        raise ValueError(
            f"apply_consensus_overlay requires columns {sorted(missing)} "
            "(produced by apply_bocpd_overlay and apply_markov_overlay); "
            "enable the bocpd and markov overlays first."
        )

    events = detect_consensus_change_points(
        df["bocpd_run_length"],
        df["markov_prob_high"],
        window_days=window_days,
        rl_drop_threshold=rl_drop_threshold,
        prob_high_jump_threshold=prob_high_jump_threshold,
        cooldown_days=cooldown_days,
    )
    df = df.copy()
    df["consensus_event"] = events
    return df


def apply_bocpd_overlay(
    df: pd.DataFrame,
    expected_run_length: float = 100.0,
    return_col: str = "log_return",
    method: str = "standard",
    df_cap: Optional[float] = None,
    varx: Optional[float] = None,
    omega: float = 1.0,
    robustness_bandwidth: float = 3.0,
) -> pd.DataFrame:
    """Add ``bocpd_run_length`` and ``bocpd_change_prob`` columns.

    Parameters
    ----------
    method
        ``"standard"`` (default) uses the Adams-MacKay (2007) BOCPD with
        Gaussian / NIG conjugate. Pass ``df_cap`` to cap predictive
        Student-t tails (Sellier & Dellaportas 2023). ``"dsm"`` uses
        the Diffusion Score Matching BOCPD of Altamirano-Briol-Knoblauch
        (ICML 2023), which is more robust to heavy-tailed returns.
    varx
        Observation noise variance for DSM. If None, defaults to the
        sample variance of the returns series.
    omega, robustness_bandwidth
        DSM hyperparameters; see ``juliams.regimes.dsm_bocpd``.
    df_cap
        Standard-BOCPD-only: Student-t degrees-of-freedom cap.
    """
    if method not in {"standard", "dsm"}:
        raise ValueError(
            f"Unknown bocpd method {method!r}; choose 'standard' or 'dsm'."
        )

    if return_col not in df.columns:
        if "Close" in df.columns:
            returns = np.log(df["Close"]).diff()
        else:
            raise ValueError(
                f"DataFrame missing both '{return_col}' and 'Close' columns; "
                "cannot derive returns for BOCPD overlay."
            )
    else:
        returns = df[return_col]

    if method == "standard":
        from juliams.regimes.bocpd import detect_change_points_bocpd
        res = detect_change_points_bocpd(
            returns,
            expected_run_length=expected_run_length,
            df_cap=df_cap,
        )
    else:
        from juliams.regimes.dsm_bocpd import detect_change_points_dsm_bocpd
        # Default varx to the sample variance of the clean returns.
        if varx is None:
            varx = float(returns.dropna().var())
            if varx <= 0:
                varx = 1e-8
        res = detect_change_points_dsm_bocpd(
            returns,
            expected_run_length=expected_run_length,
            varx=varx,
            omega=omega,
            robustness_bandwidth=robustness_bandwidth,
        )

    df = df.copy()
    df["bocpd_run_length"] = res.map_run_length
    df["bocpd_change_prob"] = res.change_probability
    return df
