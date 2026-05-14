"""
Signal readiness flags for regime indicator output.
"""

from __future__ import annotations

from typing import Mapping, Optional, Sequence

import numpy as np
import pandas as pd


DEFAULT_REQUIRED_COLUMNS = (
    "trend_strength",
    "volatility",
    "volatility_percentile",
    "regime",
)


def add_indicator_signal_flags(
    df: pd.DataFrame,
    config: Optional[Mapping[str, object]] = None,
    required_columns: Sequence[str] = DEFAULT_REQUIRED_COLUMNS,
    causal: bool = True,
) -> pd.DataFrame:
    result = df.copy()
    cfg = dict(config or {})
    warmup_bars = _warmup_bars(cfg)
    observations = pd.Series(np.arange(1, len(result) + 1), index=result.index)

    result["calibration_observations"] = observations
    result["warmup_bars"] = warmup_bars
    result["warmup_complete"] = observations >= warmup_bars
    result["causal_mode"] = bool(causal)

    valid = result["warmup_complete"].copy()
    for column in required_columns:
        if column not in result.columns:
            valid &= False
            continue
        values = result[column]
        valid &= values.notna()
        if column == "regime":
            valid &= values.astype(str).ne("Unknown")
        elif pd.api.types.is_numeric_dtype(values):
            valid &= np.isfinite(values.astype(float))

    result["valid_signal"] = valid.astype(bool)
    return result


def _warmup_bars(config: Mapping[str, object]) -> int:
    keys = (
        "trend_window",
        "volatility_window",
        "volatility_percentile_lookback",
        "hurst_window",
        "tail_risk_window",
    )
    windows = []
    for key in keys:
        value = config.get(key)
        if value is None:
            continue
        try:
            windows.append(max(1, int(value)))
        except (TypeError, ValueError):
            continue
    return max(windows) if windows else 1
