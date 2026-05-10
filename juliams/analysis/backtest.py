"""
Regime-aware strategy backtesting and narrative diagnostics.

The functions in this module are intentionally simple: they test whether a
regime classification would have produced coherent exposure decisions without
using same-day information. They are not a complete portfolio engine.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional

import numpy as np
import pandas as pd


DEFAULT_REGIME_EXPOSURE: Dict[str, float] = {
    "Bull Quiet": 1.0,
    "Bull Volatile": 0.75,
    "Sideways Quiet": 0.25,
    "Sideways Volatile": 0.0,
    "Bear Quiet": 0.0,
    "Bear Volatile": 0.0,
    "Up-LowVol": 1.0,
    "Up-HighVol": 0.75,
    "Sideways-LowVol": 0.25,
    "Sideways-HighVol": 0.0,
    "Down-LowVol": 0.0,
    "Down-HighVol": 0.0,
}


@dataclass(frozen=True)
class RegimeBacktestResult:
    """Container for a regime strategy backtest."""

    data: pd.DataFrame
    metrics: Dict[str, object]
    narrative: Dict[str, object]


def _compound_return(returns: pd.Series) -> float:
    clean = returns.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return float((1.0 + clean).prod() - 1.0)


def _max_drawdown(returns: pd.Series) -> float:
    clean = returns.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    equity = (1.0 + clean).cumprod()
    if equity.empty:
        return 0.0
    drawdown = equity / equity.cummax() - 1.0
    return float(drawdown.min())


def _regime_share(regime_mix: Mapping[str, float], prefixes: tuple[str, ...]) -> float:
    return float(sum(value for regime, value in regime_mix.items() if str(regime).startswith(prefixes)))


def _volatility_share(regime_mix: Mapping[str, float], suffixes: tuple[str, ...]) -> float:
    return float(sum(value for regime, value in regime_mix.items() if str(regime).endswith(suffixes)))


def _format_pct(value: object) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{float(value) * 100:.2f}%"


def _market_story(buy_hold_return: float) -> str:
    if buy_hold_return >= 0.05:
        return "risk_on_rebound"
    if buy_hold_return <= -0.05:
        return "risk_off_selloff"
    return "choppy_or_rangebound"


def _posture(exposure: float) -> str:
    if exposure < 0.33:
        return "defensive"
    if exposure > 0.66:
        return "fully_invested"
    return "selective"


def _relative_result(excess_return: float) -> str:
    if excess_return > 0.01:
        return "outperformed"
    if excess_return < -0.01:
        return "lagged"
    return "matched"


def regime_exposure(
    regimes: pd.Series,
    exposure_map: Optional[Mapping[str, float]] = None,
) -> pd.Series:
    """
    Convert regime labels to target exposure.

    Unknown or unmapped regimes receive zero exposure. Exposure is clipped to
    the [0, 1] range because this helper models a long-only sanity check.
    """
    mapping = dict(DEFAULT_REGIME_EXPOSURE if exposure_map is None else exposure_map)
    exposure = regimes.map(mapping).fillna(0.0).astype(float)
    return exposure.clip(lower=0.0, upper=1.0)


def backtest_regime_strategy(
    df: pd.DataFrame,
    regime_col: str = "regime_name",
    price_col: str = "Close",
    return_col: Optional[str] = None,
    exposure_map: Optional[Mapping[str, float]] = None,
    signal_lag: int = 1,
    transaction_cost_bps: float = 0.0,
    evaluation_start: Optional[object] = None,
    evaluation_end: Optional[object] = None,
    asset_name: Optional[str] = None,
) -> RegimeBacktestResult:
    """
    Backtest a long-only regime exposure rule.

    The default `signal_lag=1` means today's return is traded using yesterday's
    regime signal, which avoids lookahead bias. If an evaluation window is
    supplied, positions are still computed on the full input first so the first
    evaluation row can use pre-window signal history.
    """
    if signal_lag < 0:
        raise ValueError("signal_lag must be non-negative")
    if transaction_cost_bps < 0:
        raise ValueError("transaction_cost_bps must be non-negative")
    if regime_col not in df.columns:
        raise ValueError(f"Column '{regime_col}' not found in DataFrame")
    if return_col is None and price_col not in df.columns and "return" not in df.columns:
        raise ValueError(f"Column '{price_col}' or 'return' must be present")

    result = df.copy()

    if return_col:
        if return_col not in result.columns:
            raise ValueError(f"Column '{return_col}' not found in DataFrame")
        asset_returns = result[return_col].astype(float)
    elif "return" in result.columns:
        asset_returns = result["return"].astype(float)
    else:
        asset_returns = result[price_col].pct_change()

    result["asset_return"] = asset_returns.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    result["target_exposure"] = regime_exposure(result[regime_col], exposure_map)
    result["position"] = result["target_exposure"].shift(signal_lag).fillna(0.0)
    result["gross_strategy_return"] = result["position"] * result["asset_return"]
    result["turnover"] = result["position"].diff().abs().fillna(result["position"].abs())
    result["transaction_cost"] = result["turnover"] * (transaction_cost_bps / 10_000.0)
    result["strategy_return"] = result["gross_strategy_return"] - result["transaction_cost"]

    evaluation = result
    if evaluation_start is not None:
        evaluation = evaluation.loc[pd.Timestamp(evaluation_start):]
    if evaluation_end is not None:
        evaluation = evaluation.loc[:pd.Timestamp(evaluation_end)]

    regime_mix = (
        evaluation[regime_col]
        .dropna()
        .astype(str)
        .value_counts(normalize=True)
        .mul(100.0)
        .round(1)
        .to_dict()
    )

    strategy_return = _compound_return(evaluation["strategy_return"])
    buy_hold_return = _compound_return(evaluation["asset_return"])
    excess_return = strategy_return - buy_hold_return

    metrics: Dict[str, object] = {
        "asset": asset_name,
        "observations": int(len(evaluation)),
        "start_date": evaluation.index.min() if len(evaluation.index) else None,
        "end_date": evaluation.index.max() if len(evaluation.index) else None,
        "strategy_return": strategy_return,
        "gross_strategy_return": _compound_return(evaluation["gross_strategy_return"]),
        "buy_hold_return": buy_hold_return,
        "excess_return": excess_return,
        "transaction_cost_bps": float(transaction_cost_bps),
        "total_transaction_cost": float(evaluation["transaction_cost"].sum()) if len(evaluation) else 0.0,
        "turnover": float(evaluation["turnover"].sum()) if len(evaluation) else 0.0,
        "mean_exposure": float(evaluation["position"].mean()) if len(evaluation) else 0.0,
        "latest_regime": str(evaluation[regime_col].iloc[-1]) if len(evaluation) else "Unknown",
        "latest_position": float(evaluation["position"].iloc[-1]) if len(evaluation) else 0.0,
        "max_drawdown_strategy": _max_drawdown(evaluation["strategy_return"]),
        "max_drawdown_buy_hold": _max_drawdown(evaluation["asset_return"]),
        "regime_mix": regime_mix,
        "bull_share": _regime_share(regime_mix, ("Bull", "Up")),
        "bear_share": _regime_share(regime_mix, ("Bear", "Down")),
        "sideways_share": _regime_share(regime_mix, ("Sideways",)),
        "high_vol_share": _volatility_share(regime_mix, ("Volatile", "HighVol")),
        "low_vol_share": _volatility_share(regime_mix, ("Quiet", "LowVol")),
    }
    metrics["market_story"] = _market_story(buy_hold_return)
    metrics["strategy_posture"] = _posture(float(metrics["mean_exposure"]))
    metrics["relative_result"] = _relative_result(excess_return)

    narrative = build_backtest_narrative(metrics)
    return RegimeBacktestResult(data=result, metrics=metrics, narrative=narrative)


def build_backtest_narrative(metrics: Mapping[str, object]) -> Dict[str, object]:
    """
    Convert backtest metrics into a compact qualitative market story.
    """
    asset = metrics.get("asset") or "Asset"
    story = str(metrics.get("market_story", "choppy_or_rangebound"))
    posture = str(metrics.get("strategy_posture", "selective"))
    relative = str(metrics.get("relative_result", "matched"))
    latest_regime = str(metrics.get("latest_regime", "Unknown"))

    story_labels = {
        "risk_on_rebound": "risk-on rebound",
        "risk_off_selloff": "risk-off selloff",
        "choppy_or_rangebound": "choppy or range-bound tape",
    }
    posture_labels = {
        "defensive": "defensive",
        "selective": "selective",
        "fully_invested": "fully invested",
    }
    relative_labels = {
        "outperformed": "outperformed buy-and-hold",
        "lagged": "lagged buy-and-hold",
        "matched": "roughly matched buy-and-hold",
    }

    headline = (
        f"{asset}: {posture_labels.get(posture, posture)} regime posture "
        f"{relative_labels.get(relative, relative)} during a {story_labels.get(story, story)}."
    )

    if story == "risk_on_rebound" and posture == "defensive":
        interpretation = (
            "The classifier appears to have recognized the bullish turn late or required too much "
            "confirmation, so it captured less upside than a passive position."
        )
    elif story == "risk_off_selloff" and posture == "defensive":
        interpretation = (
            "The classifier narrative is capital preservation: low exposure is consistent with "
            "avoiding a falling or unstable market."
        )
    elif story == "risk_on_rebound" and posture == "fully_invested":
        interpretation = (
            "The classifier narrative is trend participation: high exposure is consistent with "
            "a broad advance."
        )
    elif story == "risk_on_rebound" and posture == "selective":
        interpretation = (
            "The classifier captured part of the rebound but still treated the move as unconfirmed "
            "or rotational, so it stayed below passive exposure."
        )
    elif story == "risk_off_selloff" and posture == "fully_invested":
        interpretation = (
            "The classifier may be too slow to de-risk because high exposure persisted into a selloff."
        )
    elif story == "choppy_or_rangebound" and posture == "defensive":
        interpretation = (
            "The classifier narrative is risk control in a mixed tape; low exposure can reduce noise "
            "but may miss mild gains."
        )
    else:
        interpretation = (
            "The classifier narrative is selective participation, which is reasonable when the "
            "price action is mixed or regimes are rotating."
        )

    evidence = [
        f"latest regime: {latest_regime}",
        f"buy-and-hold return: {_format_pct(metrics.get('buy_hold_return'))}",
        f"strategy return: {_format_pct(metrics.get('strategy_return'))}",
        f"mean exposure: {_format_pct(metrics.get('mean_exposure'))}",
        f"strategy drawdown: {_format_pct(metrics.get('max_drawdown_strategy'))}",
        f"buy-and-hold drawdown: {_format_pct(metrics.get('max_drawdown_buy_hold'))}",
        f"bull/bear/sideways mix: {metrics.get('bull_share', 0):.1f}%/"
        f"{metrics.get('bear_share', 0):.1f}%/{metrics.get('sideways_share', 0):.1f}%",
        f"high-volatility share: {metrics.get('high_vol_share', 0):.1f}%",
    ]

    return {
        "headline": headline,
        "market_story": story,
        "posture": posture,
        "relative_result": relative,
        "interpretation": interpretation,
        "evidence": evidence,
    }
