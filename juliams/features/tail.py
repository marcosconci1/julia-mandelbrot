"""
Tail-risk and survival diagnostics.

These features add a non-Gaussian risk layer to the regime model. The goal is
to flag fragile states where historical moments, Gaussian variance, or a plain
rebound signal may be misleading.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def _clean_returns(values: np.ndarray | pd.Series) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    return array[np.isfinite(array)]


def value_at_risk(
    returns: np.ndarray | pd.Series,
    alpha: float = 0.95,
) -> float:
    """
    Return positive loss VaR for a return sample.

    A return of -4% contributes a positive loss of 0.04. The result is a loss
    magnitude, not a signed return.
    """
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1")

    clean = _clean_returns(returns)
    if clean.size == 0:
        return np.nan
    losses = -clean
    return float(np.quantile(losses, alpha))


def conditional_value_at_risk(
    returns: np.ndarray | pd.Series,
    alpha: float = 0.95,
) -> float:
    """
    Return positive expected shortfall beyond VaR.
    """
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1")

    clean = _clean_returns(returns)
    if clean.size == 0:
        return np.nan

    losses = -clean
    threshold = np.quantile(losses, alpha)
    tail_losses = losses[losses >= threshold]
    if tail_losses.size == 0:
        return np.nan
    return float(np.mean(tail_losses))


def hill_tail_index(
    returns: np.ndarray | pd.Series,
    tail_fraction: float = 0.10,
    min_tail_observations: int = 5,
) -> float:
    """
    Estimate the left-tail Pareto exponent using the Hill estimator.

    Lower values indicate fatter tails. The estimator is applied to positive
    losses, so gains do not enter the left-tail estimate.
    """
    if not 0 < tail_fraction < 1:
        raise ValueError("tail_fraction must be between 0 and 1")
    if min_tail_observations <= 1:
        raise ValueError("min_tail_observations must be greater than 1")

    clean = _clean_returns(returns)
    losses = -clean
    losses = losses[losses > 0]
    if losses.size < min_tail_observations:
        return np.nan

    k = max(min_tail_observations, int(np.floor(losses.size * tail_fraction)))
    k = min(k, losses.size)
    top_losses = np.sort(losses)[-k:]
    threshold = top_losses[0]
    if threshold <= 0:
        return np.nan

    logs = np.log(top_losses / threshold)
    mean_log = np.mean(logs)
    if mean_log <= 0:
        return np.nan
    return float(1.0 / mean_log)


def rolling_max_drawdown(
    returns: pd.Series,
    window: int,
) -> pd.Series:
    """
    Compute rolling max drawdown from simple returns.
    """
    if window <= 0:
        raise ValueError("window must be positive")

    def calculate(values: np.ndarray) -> float:
        clean = np.nan_to_num(values.astype(float), nan=0.0, posinf=0.0, neginf=0.0)
        equity = np.cumprod(1.0 + clean)
        if equity.size == 0:
            return np.nan
        peak = np.maximum.accumulate(equity)
        drawdown = equity / peak - 1.0
        return float(np.min(drawdown))

    min_periods = min(window, max(5, window // 2))
    return returns.rolling(window=window, min_periods=min_periods).apply(calculate, raw=True)


def classify_survival_regime(
    loss_cvar: pd.Series,
    tail_index: pd.Series,
    rolling_drawdown: pd.Series,
    fragile_cvar: float = 0.04,
    stressed_cvar: float = 0.02,
    fragile_tail_index: float = 3.0,
    stressed_tail_index: float = 4.0,
    fragile_drawdown: float = -0.15,
    stressed_drawdown: float = -0.07,
) -> pd.Series:
    """
    Classify a market state into Resilient, Stressed, Fragile, or Unknown.
    """
    if fragile_cvar <= 0 or stressed_cvar <= 0:
        raise ValueError("CVaR thresholds must be positive")
    if fragile_tail_index <= 0 or stressed_tail_index <= 0:
        raise ValueError("tail-index thresholds must be positive")
    if fragile_drawdown >= 0 or stressed_drawdown >= 0:
        raise ValueError("drawdown thresholds must be negative")

    def classify(cvar_value: float, alpha_value: float, dd_value: float) -> str:
        has_signal = any(pd.notna(value) for value in (cvar_value, alpha_value, dd_value))
        if not has_signal:
            return "Unknown"

        fragile = (
            (pd.notna(cvar_value) and cvar_value >= fragile_cvar)
            or (pd.notna(alpha_value) and alpha_value <= fragile_tail_index)
            or (pd.notna(dd_value) and dd_value <= fragile_drawdown)
        )
        if fragile:
            return "Fragile"

        stressed = (
            (pd.notna(cvar_value) and cvar_value >= stressed_cvar)
            or (pd.notna(alpha_value) and alpha_value <= stressed_tail_index)
            or (pd.notna(dd_value) and dd_value <= stressed_drawdown)
        )
        if stressed:
            return "Stressed"

        return "Resilient"

    return pd.Series(
        [
            classify(cvar_value, alpha_value, dd_value)
            for cvar_value, alpha_value, dd_value in zip(loss_cvar, tail_index, rolling_drawdown)
        ],
        index=loss_cvar.index,
    )


def survival_score(
    loss_cvar: pd.Series,
    tail_index: pd.Series,
    rolling_drawdown: pd.Series,
    fragile_cvar: float = 0.04,
    fragile_tail_index: float = 3.0,
    fragile_drawdown: float = -0.15,
) -> pd.Series:
    """
    Return a 0-1 fragility score where 1 is most fragile.
    """
    cvar_component = (loss_cvar / fragile_cvar).clip(lower=0.0, upper=1.0)
    drawdown_component = (rolling_drawdown.abs() / abs(fragile_drawdown)).clip(lower=0.0, upper=1.0)
    tail_component = ((fragile_tail_index / tail_index).replace([np.inf, -np.inf], np.nan)).clip(
        lower=0.0,
        upper=1.0,
    )
    return pd.concat([cvar_component, drawdown_component, tail_component], axis=1).mean(axis=1, skipna=True)


def compute_tail_risk_features(
    df: pd.DataFrame,
    config: Optional[dict] = None,
    price_col: str = "Close",
    return_col: str = "log_return",
) -> pd.DataFrame:
    """
    Add tail-risk and survival columns to a price DataFrame.
    """
    if config is None:
        config = {}
    if price_col not in df.columns and return_col not in df.columns:
        raise ValueError(f"DataFrame must contain '{price_col}' or '{return_col}'")

    result = df.copy()
    if return_col not in result.columns:
        result[return_col] = np.log(result[price_col]).diff()

    returns = result[return_col].astype(float)
    window = int(config.get("tail_risk_window", 252))
    if window <= 0:
        raise ValueError("tail_risk_window must be positive")

    min_periods = min(window, max(20, window // 2))
    var_alpha = float(config.get("tail_var_alpha", 0.95))
    cvar_alpha = float(config.get("tail_cvar_alpha", 0.95))
    tail_fraction = float(config.get("hill_tail_fraction", 0.10))
    min_tail_observations = int(config.get("hill_min_tail_observations", 5))

    result["loss_var"] = returns.rolling(window=window, min_periods=min_periods).apply(
        lambda values: value_at_risk(values, alpha=var_alpha),
        raw=True,
    )
    result["loss_cvar"] = returns.rolling(window=window, min_periods=min_periods).apply(
        lambda values: conditional_value_at_risk(values, alpha=cvar_alpha),
        raw=True,
    )
    result["tail_index"] = returns.rolling(window=window, min_periods=min_periods).apply(
        lambda values: hill_tail_index(
            values,
            tail_fraction=tail_fraction,
            min_tail_observations=min_tail_observations,
        ),
        raw=True,
    )
    result["excess_kurtosis"] = returns.rolling(window=window, min_periods=min_periods).kurt()
    result["rolling_max_drawdown"] = rolling_max_drawdown(returns, window=window)
    if price_col in result.columns:
        close = result[price_col].astype(float)
        result["drawdown"] = close / close.cummax() - 1.0

    result["survival_regime"] = classify_survival_regime(
        result["loss_cvar"],
        result["tail_index"],
        result["rolling_max_drawdown"],
        fragile_cvar=float(config.get("fragile_cvar", 0.04)),
        stressed_cvar=float(config.get("stressed_cvar", 0.02)),
        fragile_tail_index=float(config.get("fragile_tail_index", 3.0)),
        stressed_tail_index=float(config.get("stressed_tail_index", 4.0)),
        fragile_drawdown=float(config.get("fragile_drawdown", -0.15)),
        stressed_drawdown=float(config.get("stressed_drawdown", -0.07)),
    )
    result["survival_score"] = survival_score(
        result["loss_cvar"],
        result["tail_index"],
        result["rolling_max_drawdown"],
        fragile_cvar=float(config.get("fragile_cvar", 0.04)),
        fragile_tail_index=float(config.get("fragile_tail_index", 3.0)),
        fragile_drawdown=float(config.get("fragile_drawdown", -0.15)),
    )

    return result
