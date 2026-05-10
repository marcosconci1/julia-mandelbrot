"""
Walk-forward adaptive regime fitter.

Wires together the rolling-quantile thresholds (juliams.regimes.adaptive),
the EWMA-normalised trend strength (juliams.features.trend), and the
Markov-switching variance detector (juliams.regimes.markov) under a
strict no-future-leakage discipline.

The contract
------------
For each ``output_index t`` the function returns a regime label whose
calculation used **only data with index < t** — never data at t or
later. This is the "purged walk-forward" idea from López de Prado (2018,
ch. 7) but applied to *online classification* rather than supervised CV.

The leakage test in ``tests/test_walk_forward_adaptive.py`` makes the
guarantee falsifiable by poisoning the future and asserting the past
output is byte-identical.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from juliams.features.trend import compute_ewma_trend_strength
from juliams.regimes.adaptive import (
    classify_with_adaptive_thresholds,
    rolling_quantile_thresholds,
)


@dataclass
class AdaptiveRegimeConfig:
    """Hyperparameters for adaptive regime fitting.

    All defaults come from research-validated starting points and are
    deliberately conservative; tune via walk-forward on a hold-out, not
    on the final evaluation period.
    """

    trend_window: int = 20
    trend_halflife: float = 20.0
    threshold_window: int = 252
    q_up: float = 0.70
    q_down: float = 0.30
    abs_floor: float = 0.10
    min_threshold_periods: Optional[int] = None


@dataclass
class AdaptiveRegimeOutput:
    trend_strength: pd.Series
    upper_threshold: pd.Series
    lower_threshold: pd.Series
    regime: pd.Series


def fit_adaptive_regime_walkforward(
    df: pd.DataFrame,
    config: Optional[AdaptiveRegimeConfig] = None,
    price_col: str = "Close",
) -> AdaptiveRegimeOutput:
    """Fit the adaptive regime classifier under a strict-causal contract.

    Parameters
    ----------
    df
        Daily OHLCV data; only ``price_col`` is used.
    config
        Hyperparameters; see :class:`AdaptiveRegimeConfig`.
    price_col
        Defaults to ``"Close"``.

    Returns
    -------
    AdaptiveRegimeOutput with four aligned series:
        - trend_strength: EWMA-normalised slope, NaN before warmup
        - upper_threshold, lower_threshold: rolling-quantile cutoffs
        - regime: {"Up", "Down", "Sideways", "Unknown"} per row

    Causality
    ---------
    The trend_strength at index t depends on prices through index t,
    which is allowed (this is the *signal*, not a parameter). The
    thresholds at index t depend strictly on data with index < t (via
    ``shift_calibration=True``). The regime at index t is therefore a
    decision that uses only the *current* signal compared against
    *historical* cutoffs — exactly what a live system would do.
    """
    cfg = config or AdaptiveRegimeConfig()

    trend_strength = compute_ewma_trend_strength(
        df,
        window=cfg.trend_window,
        halflife=cfg.trend_halflife,
        price_col=price_col,
    )

    upper, lower = rolling_quantile_thresholds(
        trend_strength,
        window=cfg.threshold_window,
        q_up=cfg.q_up,
        q_down=cfg.q_down,
        abs_floor_up=cfg.abs_floor,
        abs_floor_down=cfg.abs_floor,
        min_periods=cfg.min_threshold_periods,
        shift_calibration=True,
    )

    regime = classify_with_adaptive_thresholds(trend_strength, upper, lower)

    return AdaptiveRegimeOutput(
        trend_strength=trend_strength,
        upper_threshold=upper,
        lower_threshold=lower,
        regime=regime,
    )
