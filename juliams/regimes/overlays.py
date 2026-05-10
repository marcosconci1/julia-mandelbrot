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

from typing import Optional

import numpy as np
import pandas as pd


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


def apply_markov_overlay(
    df: pd.DataFrame,
    return_col: str = "log_return",
    min_dwell: int = 1,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """Add ``markov_prob_high`` and ``markov_state`` columns.

    Uses *filtered* probabilities so the column is causal (suitable for
    columns that downstream code may treat as live signals). Apply
    ``min_dwell`` post-processing if requested to suppress spurious
    flips during structural breaks.
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

    fit = fit_markov_variance_regime(returns)
    labels = label_markov_regimes(fit, threshold=threshold, use_filtered=True)
    if min_dwell > 1:
        labels = enforce_min_dwell(labels, min_days=min_dwell)

    df = df.copy()
    df["markov_prob_high"] = fit.filtered_prob_high
    df["markov_state"] = labels
    return df


def apply_bocpd_overlay(
    df: pd.DataFrame,
    expected_run_length: float = 100.0,
    return_col: str = "log_return",
) -> pd.DataFrame:
    """Add ``bocpd_run_length`` and ``bocpd_change_prob`` columns."""
    from juliams.regimes.bocpd import detect_change_points_bocpd

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

    res = detect_change_points_bocpd(
        returns, expected_run_length=expected_run_length
    )
    df = df.copy()
    df["bocpd_run_length"] = res.map_run_length
    df["bocpd_change_prob"] = res.change_probability
    return df
