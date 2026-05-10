"""
Pre-adjustment diagnostics for regime models.

These helpers are meant to answer why a regime system behaved the way it did
before thresholds or exposure rules are changed.
"""

from __future__ import annotations

from collections import Counter
from typing import Dict, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from .backtest import DEFAULT_REGIME_EXPOSURE, RegimeBacktestResult, backtest_regime_strategy, regime_exposure


DEFAULT_DIAGNOSTIC_EXPOSURE_MAPS: Dict[str, Dict[str, float]] = {
    "current": {
        "Bull Quiet": 1.0,
        "Bull Volatile": 0.75,
        "Sideways Quiet": 0.25,
        "Sideways Volatile": 0.0,
        "Bear Quiet": 0.0,
        "Bear Volatile": 0.0,
    },
    "aggressive": {
        "Bull Quiet": 1.0,
        "Bull Volatile": 1.0,
        "Sideways Quiet": 0.5,
        "Sideways Volatile": 0.25,
        "Bear Quiet": 0.0,
        "Bear Volatile": 0.0,
    },
    "defensive": {
        "Bull Quiet": 1.0,
        "Bull Volatile": 0.5,
        "Sideways Quiet": 0.0,
        "Sideways Volatile": 0.0,
        "Bear Quiet": 0.0,
        "Bear Volatile": 0.0,
    },
    "binary_bull": {
        "Bull Quiet": 1.0,
        "Bull Volatile": 1.0,
        "Sideways Quiet": 0.0,
        "Sideways Volatile": 0.0,
        "Bear Quiet": 0.0,
        "Bear Volatile": 0.0,
    },
}


def _regime_group(regime: object) -> str:
    label = str(regime)
    if label.startswith(("Bull", "Up")):
        return "bull"
    if label.startswith(("Bear", "Down")):
        return "bear"
    if label.startswith("Sideways"):
        return "sideways"
    return "unknown"


def _slice_window(
    df: pd.DataFrame,
    evaluation_start: Optional[object] = None,
    evaluation_end: Optional[object] = None,
) -> pd.DataFrame:
    result = df
    if evaluation_start is not None:
        result = result.loc[pd.Timestamp(evaluation_start):]
    if evaluation_end is not None:
        result = result.loc[:pd.Timestamp(evaluation_end)]
    return result


def _compound(returns: pd.Series) -> float:
    clean = returns.replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        return np.nan
    return float((1.0 + clean).prod() - 1.0)


def compute_regime_forward_diagnostics(
    df: pd.DataFrame,
    regime_col: str = "regime_name",
    price_col: str = "Close",
    horizons: Sequence[int] = (1, 5, 10, 20),
    evaluation_start: Optional[object] = None,
    evaluation_end: Optional[object] = None,
    min_observations: int = 1,
) -> pd.DataFrame:
    """
    Measure future returns conditioned on the current regime.

    This tests whether regimes have forward return separation before any
    exposure map or threshold is changed.
    """
    if regime_col not in df.columns:
        raise ValueError(f"Column '{regime_col}' not found in DataFrame")
    if price_col not in df.columns:
        raise ValueError(f"Column '{price_col}' not found in DataFrame")
    if min_observations <= 0:
        raise ValueError("min_observations must be positive")

    evaluation = _slice_window(df.copy(), evaluation_start, evaluation_end)
    rows = []

    for horizon in horizons:
        if horizon <= 0:
            raise ValueError("horizons must contain positive integers")

        fwd_return = evaluation[price_col].shift(-horizon) / evaluation[price_col] - 1.0
        overall = fwd_return.dropna()
        overall_mean = float(overall.mean()) if len(overall) else np.nan

        for regime, regime_data in evaluation.assign(_fwd_return=fwd_return).groupby(regime_col):
            returns = regime_data["_fwd_return"].replace([np.inf, -np.inf], np.nan).dropna()
            if len(returns) < min_observations:
                continue

            rows.append(
                {
                    "regime": regime,
                    "regime_group": _regime_group(regime),
                    "horizon": int(horizon),
                    "count": int(len(returns)),
                    "mean_return": float(returns.mean()),
                    "median_return": float(returns.median()),
                    "compound_return": _compound(returns),
                    "positive_pct": float((returns > 0).mean() * 100.0),
                    "edge_vs_overall": float(returns.mean() - overall_mean) if pd.notna(overall_mean) else np.nan,
                }
            )

    return pd.DataFrame(rows).sort_values(["horizon", "regime_group", "regime"]).reset_index(drop=True)


def measure_bullish_turn_lag(
    df: pd.DataFrame,
    regime_col: str = "regime_name",
    price_col: str = "Close",
    evaluation_start: Optional[object] = None,
    evaluation_end: Optional[object] = None,
) -> Dict[str, object]:
    """
    Measure how long the model took to turn bullish after the local price low.
    """
    if regime_col not in df.columns:
        raise ValueError(f"Column '{regime_col}' not found in DataFrame")
    if price_col not in df.columns:
        raise ValueError(f"Column '{price_col}' not found in DataFrame")

    evaluation = _slice_window(df.copy(), evaluation_start, evaluation_end)
    if evaluation.empty:
        return {
            "trough_date": None,
            "trough_price": np.nan,
            "first_bull_date": None,
            "first_bull_price": np.nan,
            "lag_periods": None,
            "missed_return": np.nan,
            "latest_regime": "Unknown",
        }

    trough_date = evaluation[price_col].idxmin()
    trough_pos = evaluation.index.get_loc(trough_date)
    after_trough = evaluation.iloc[trough_pos:]
    bull_mask = after_trough[regime_col].astype(str).map(lambda value: value.startswith(("Bull", "Up")))
    bull_rows = after_trough.loc[bull_mask]

    if bull_rows.empty:
        first_bull_date = None
        first_bull_price = np.nan
        lag_periods = None
        missed_return = np.nan
    else:
        first_bull_date = bull_rows.index[0]
        first_bull_price = float(bull_rows[price_col].iloc[0])
        lag_periods = int(after_trough.index.get_loc(first_bull_date))
        missed_return = first_bull_price / float(evaluation.loc[trough_date, price_col]) - 1.0

    return {
        "trough_date": trough_date,
        "trough_price": float(evaluation.loc[trough_date, price_col]),
        "first_bull_date": first_bull_date,
        "first_bull_price": first_bull_price,
        "lag_periods": lag_periods,
        "missed_return": missed_return,
        "latest_regime": str(evaluation[regime_col].iloc[-1]),
    }


def compare_exposure_maps(
    df: pd.DataFrame,
    exposure_maps: Optional[Mapping[str, Mapping[str, float]]] = None,
    regime_col: str = "regime_name",
    price_col: str = "Close",
    evaluation_start: Optional[object] = None,
    evaluation_end: Optional[object] = None,
    asset_name: Optional[str] = None,
    transaction_cost_bps: float = 0.0,
) -> pd.DataFrame:
    """
    Backtest several exposure maps against the same unchanged regimes.
    """
    maps = DEFAULT_DIAGNOSTIC_EXPOSURE_MAPS if exposure_maps is None else exposure_maps
    rows = []

    for name, exposure_map in maps.items():
        result = backtest_regime_strategy(
            df,
            regime_col=regime_col,
            price_col=price_col,
            exposure_map=exposure_map,
            transaction_cost_bps=transaction_cost_bps,
            evaluation_start=evaluation_start,
            evaluation_end=evaluation_end,
            asset_name=asset_name,
        )
        metrics = result.metrics
        rows.append(
            {
                "map": name,
                "asset": asset_name,
                "strategy_return": metrics["strategy_return"],
                "gross_strategy_return": metrics["gross_strategy_return"],
                "buy_hold_return": metrics["buy_hold_return"],
                "excess_return": metrics["excess_return"],
                "transaction_cost_bps": metrics["transaction_cost_bps"],
                "total_transaction_cost": metrics["total_transaction_cost"],
                "turnover": metrics["turnover"],
                "mean_exposure": metrics["mean_exposure"],
                "max_drawdown_strategy": metrics["max_drawdown_strategy"],
                "market_story": metrics["market_story"],
                "strategy_posture": metrics["strategy_posture"],
                "relative_result": metrics["relative_result"],
            }
        )

    return pd.DataFrame(rows).sort_values("excess_return", ascending=False).reset_index(drop=True)


def compute_rebound_signal(
    df: pd.DataFrame,
    price_col: str = "Close",
    fast_window: int = 5,
    drawdown_lookback: int = 20,
    min_rebound: float = 0.03,
    min_fast_return: float = 0.02,
    require_above_fast_ma: bool = True,
) -> pd.Series:
    """
    Detect a fast rebound using only current and historical prices.

    A signal requires price to recover from a recent rolling low and to show
    positive short-window momentum. It is intended as a candidate overlay,
    not a replacement for the slower regime classifier.
    """
    if price_col not in df.columns:
        raise ValueError(f"Column '{price_col}' not found in DataFrame")
    if fast_window <= 0:
        raise ValueError("fast_window must be positive")
    if drawdown_lookback <= 0:
        raise ValueError("drawdown_lookback must be positive")
    if min_rebound < 0:
        raise ValueError("min_rebound must be non-negative")

    close = df[price_col].astype(float)
    rolling_low = close.rolling(drawdown_lookback, min_periods=fast_window).min()
    rebound_from_low = close / rolling_low - 1.0
    fast_return = close / close.shift(fast_window) - 1.0

    signal = (rebound_from_low >= min_rebound) & (fast_return >= min_fast_return)

    if require_above_fast_ma:
        fast_ma = close.rolling(fast_window, min_periods=fast_window).mean()
        signal = signal & (close >= fast_ma)

    return signal.fillna(False)


def backtest_rebound_overlay(
    df: pd.DataFrame,
    regime_col: str = "regime_name",
    price_col: str = "Close",
    evaluation_start: Optional[object] = None,
    evaluation_end: Optional[object] = None,
    asset_name: Optional[str] = None,
    rebound_exposure: float = 0.75,
    fast_window: int = 5,
    drawdown_lookback: int = 20,
    min_rebound: float = 0.03,
    min_fast_return: float = 0.02,
    require_above_fast_ma: bool = True,
    transaction_cost_bps: float = 0.0,
) -> RegimeBacktestResult:
    """
    Test a faster rebound overlay without changing the base regime labels.

    When the rebound signal is active and the base regime exposure is below
    `rebound_exposure`, the temporary backtest regime is set to `Rebound Overlay`.
    The underlying backtest still applies a one-period signal lag.
    """
    if regime_col not in df.columns:
        raise ValueError(f"Column '{regime_col}' not found in DataFrame")
    if not 0 <= rebound_exposure <= 1:
        raise ValueError("rebound_exposure must be between 0 and 1")

    result = df.copy()
    signal = compute_rebound_signal(
        result,
        price_col=price_col,
        fast_window=fast_window,
        drawdown_lookback=drawdown_lookback,
        min_rebound=min_rebound,
        min_fast_return=min_fast_return,
        require_above_fast_ma=require_above_fast_ma,
    )
    base_exposure = regime_exposure(result[regime_col])
    overlay_mask = signal & (base_exposure < rebound_exposure)

    result["_rebound_signal"] = signal
    result["_base_regime_name"] = result[regime_col]
    result["_overlay_regime_name"] = result[regime_col].where(~overlay_mask, "Rebound Overlay")

    exposure_map = dict(DEFAULT_REGIME_EXPOSURE)
    exposure_map["Rebound Overlay"] = rebound_exposure

    backtest = backtest_regime_strategy(
        result,
        regime_col="_overlay_regime_name",
        price_col=price_col,
        exposure_map=exposure_map,
        transaction_cost_bps=transaction_cost_bps,
        evaluation_start=evaluation_start,
        evaluation_end=evaluation_end,
        asset_name=asset_name,
    )
    backtest.metrics["rebound_signal_count"] = int(_slice_window(result, evaluation_start, evaluation_end)["_rebound_signal"].sum())
    backtest.metrics["rebound_signal_share"] = float(
        _slice_window(result, evaluation_start, evaluation_end)["_rebound_signal"].mean()
        if len(_slice_window(result, evaluation_start, evaluation_end))
        else 0.0
    )
    backtest.metrics["latest_base_regime"] = (
        str(_slice_window(result, evaluation_start, evaluation_end)["_base_regime_name"].iloc[-1])
        if len(_slice_window(result, evaluation_start, evaluation_end))
        else "Unknown"
    )
    return backtest


def compare_rebound_overlay(
    df: pd.DataFrame,
    regime_col: str = "regime_name",
    price_col: str = "Close",
    evaluation_start: Optional[object] = None,
    evaluation_end: Optional[object] = None,
    asset_name: Optional[str] = None,
    rebound_exposure: float = 0.75,
    fast_window: int = 5,
    drawdown_lookback: int = 20,
    min_rebound: float = 0.03,
    min_fast_return: float = 0.02,
    require_above_fast_ma: bool = True,
    transaction_cost_bps: float = 0.0,
) -> pd.DataFrame:
    """
    Compare the current regime strategy with the rebound overlay.
    """
    baseline = backtest_regime_strategy(
        df,
        regime_col=regime_col,
        price_col=price_col,
        transaction_cost_bps=transaction_cost_bps,
        evaluation_start=evaluation_start,
        evaluation_end=evaluation_end,
        asset_name=asset_name,
    )
    overlay = backtest_rebound_overlay(
        df,
        regime_col=regime_col,
        price_col=price_col,
        evaluation_start=evaluation_start,
        evaluation_end=evaluation_end,
        asset_name=asset_name,
        rebound_exposure=rebound_exposure,
        fast_window=fast_window,
        drawdown_lookback=drawdown_lookback,
        min_rebound=min_rebound,
        min_fast_return=min_fast_return,
        require_above_fast_ma=require_above_fast_ma,
        transaction_cost_bps=transaction_cost_bps,
    )

    rows = []
    for name, result in (("current", baseline), ("rebound_overlay", overlay)):
        metrics = result.metrics
        rows.append(
            {
                "variant": name,
                "asset": asset_name,
                "strategy_return": metrics["strategy_return"],
                "buy_hold_return": metrics["buy_hold_return"],
                "excess_return": metrics["excess_return"],
                "transaction_cost_bps": metrics["transaction_cost_bps"],
                "total_transaction_cost": metrics["total_transaction_cost"],
                "turnover": metrics["turnover"],
                "mean_exposure": metrics["mean_exposure"],
                "max_drawdown_strategy": metrics["max_drawdown_strategy"],
                "market_story": metrics["market_story"],
                "strategy_posture": metrics["strategy_posture"],
                "relative_result": metrics["relative_result"],
                "rebound_signal_share": metrics.get("rebound_signal_share", 0.0),
            }
        )

    return pd.DataFrame(rows)


def summarize_variant_comparison_by_group(
    comparison: pd.DataFrame,
    asset_groups: Mapping[str, Sequence[str]],
    baseline_variant: str = "current",
    candidate_variant: str = "rebound_overlay",
) -> pd.DataFrame:
    """
    Summarize whether a candidate variant helps by asset group.
    """
    required = {"asset", "variant", "strategy_return", "excess_return", "mean_exposure"}
    missing = required.difference(comparison.columns)
    if missing:
        raise ValueError(f"comparison is missing required columns: {sorted(missing)}")

    rows = []

    for group_name, assets in asset_groups.items():
        group = comparison[comparison["asset"].isin(assets)]
        baseline = group[group["variant"] == baseline_variant].set_index("asset")
        candidate = group[group["variant"] == candidate_variant].set_index("asset")
        common_assets = baseline.index.intersection(candidate.index)
        if common_assets.empty:
            continue

        baseline_returns = baseline.loc[common_assets, "strategy_return"].astype(float)
        candidate_returns = candidate.loc[common_assets, "strategy_return"].astype(float)
        improvement = candidate_returns - baseline_returns

        rows.append(
            {
                "group": group_name,
                "asset_count": int(len(common_assets)),
                "baseline_mean_strategy_return": float(baseline_returns.mean()),
                "candidate_mean_strategy_return": float(candidate_returns.mean()),
                "mean_improvement": float(improvement.mean()),
                "improved_count": int((improvement > 0).sum()),
                "degraded_count": int((improvement < 0).sum()),
                "candidate_mean_excess_return": float(candidate.loc[common_assets, "excess_return"].astype(float).mean()),
                "candidate_median_exposure": float(candidate.loc[common_assets, "mean_exposure"].astype(float).median()),
            }
        )

    return pd.DataFrame(rows)


def summarize_group_narratives(
    metrics_by_asset: Mapping[str, Mapping[str, object] | RegimeBacktestResult],
    asset_groups: Mapping[str, Sequence[str]],
) -> pd.DataFrame:
    """
    Aggregate narrative backtest metrics by asset group.
    """
    rows = []

    for group_name, assets in asset_groups.items():
        group_metrics = []
        for asset in assets:
            item = metrics_by_asset.get(asset)
            if item is None:
                continue
            if isinstance(item, RegimeBacktestResult):
                group_metrics.append(item.metrics)
            else:
                group_metrics.append(item)

        if not group_metrics:
            continue

        stories = Counter(str(metric.get("market_story", "unknown")) for metric in group_metrics)
        postures = Counter(str(metric.get("strategy_posture", "unknown")) for metric in group_metrics)
        relatives = Counter(str(metric.get("relative_result", "unknown")) for metric in group_metrics)

        rows.append(
            {
                "group": group_name,
                "asset_count": len(group_metrics),
                "mean_strategy_return": float(np.mean([metric["strategy_return"] for metric in group_metrics])),
                "mean_buy_hold_return": float(np.mean([metric["buy_hold_return"] for metric in group_metrics])),
                "mean_excess_return": float(np.mean([metric["excess_return"] for metric in group_metrics])),
                "median_exposure": float(np.median([metric["mean_exposure"] for metric in group_metrics])),
                "top_story": stories.most_common(1)[0][0],
                "top_posture": postures.most_common(1)[0][0],
                "top_relative_result": relatives.most_common(1)[0][0],
            }
        )

    return pd.DataFrame(rows)
