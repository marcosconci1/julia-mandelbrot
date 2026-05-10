"""
Group-aware rebound overlay diagnostics.

This module keeps the rebound overlay as a diagnostic layer. It does not fetch
data and expects callers to pass already prepared DataFrames with price and
regime columns.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from .backtest import DEFAULT_REGIME_EXPOSURE, RegimeBacktestResult, backtest_regime_strategy, regime_exposure
from .diagnostics import compute_rebound_signal


DEFAULT_ASSET_GROUPS: Dict[str, tuple[str, ...]] = {
    "indices": ("SPY", "QQQ", "IWM", "DIA", "^GSPC", "^IXIC", "^RUT", "^DJI"),
    "mega_cap_tech": ("AAPL", "MSFT", "NVDA", "GOOGL", "GOOG", "AMZN", "META", "TSLA"),
    "diversifiers": ("TLT", "IEF", "GLD", "SLV"),
    "crypto": ("BTC-USD", "ETH-USD", "SOL-USD", "BTCUSDT", "ETHUSDT", "SOLUSDT"),
}


@dataclass(frozen=True)
class ReboundOverlayProfile:
    """
    Parameters for a group-specific rebound overlay.

    Disabled profiles still carry parameters so callers can enable and test
    them explicitly without changing the public shape of the config.
    """

    enabled: bool
    rebound_exposure: float = 0.75
    fast_window: int = 5
    drawdown_lookback: int = 20
    min_rebound: float = 0.03
    min_fast_return: float = 0.02
    require_above_fast_ma: bool = True
    require_prior_drawdown: bool = False
    drawdown_gate_lookback: int = 20
    min_prior_drawdown: float = -0.05
    max_overlay_holding_period: Optional[int] = 5
    overlay_stop_loss: Optional[float] = -0.03
    require_survival_guard: bool = True
    survival_regime_col: str = "survival_regime"
    fragile_survival_regimes: tuple[str, ...] = ("Fragile",)

    def overlay_kwargs(self) -> Dict[str, object]:
        return {
            "rebound_exposure": self.rebound_exposure,
            "fast_window": self.fast_window,
            "drawdown_lookback": self.drawdown_lookback,
            "min_rebound": self.min_rebound,
            "min_fast_return": self.min_fast_return,
            "require_above_fast_ma": self.require_above_fast_ma,
        }


DEFAULT_GROUP_OVERLAY_PROFILES: Dict[str, ReboundOverlayProfile] = {
    "indices": ReboundOverlayProfile(
        enabled=True,
        rebound_exposure=0.75,
        fast_window=5,
        drawdown_lookback=20,
        min_rebound=0.03,
        min_fast_return=0.02,
        require_prior_drawdown=True,
        drawdown_gate_lookback=20,
        min_prior_drawdown=-0.05,
        max_overlay_holding_period=5,
        overlay_stop_loss=-0.03,
    ),
    "mega_cap_tech": ReboundOverlayProfile(
        enabled=True,
        rebound_exposure=0.75,
        fast_window=5,
        drawdown_lookback=20,
        min_rebound=0.035,
        min_fast_return=0.02,
        require_prior_drawdown=True,
        drawdown_gate_lookback=20,
        min_prior_drawdown=-0.06,
        max_overlay_holding_period=5,
        overlay_stop_loss=-0.04,
    ),
    "diversifiers": ReboundOverlayProfile(
        enabled=False,
        rebound_exposure=0.50,
        fast_window=7,
        drawdown_lookback=30,
        min_rebound=0.04,
        min_fast_return=0.015,
    ),
    "crypto": ReboundOverlayProfile(
        enabled=False,
        rebound_exposure=0.50,
        fast_window=7,
        drawdown_lookback=30,
        min_rebound=0.08,
        min_fast_return=0.04,
    ),
    "other": ReboundOverlayProfile(enabled=False),
}


def _normalize_symbol(symbol: object) -> str:
    return str(symbol).strip().upper()


def _coerce_profile(profile: ReboundOverlayProfile | Mapping[str, object]) -> ReboundOverlayProfile:
    if isinstance(profile, ReboundOverlayProfile):
        return profile
    return ReboundOverlayProfile(**dict(profile))


def _profile_for_group(
    group: str,
    profiles: Optional[Mapping[str, ReboundOverlayProfile | Mapping[str, object]]] = None,
) -> ReboundOverlayProfile:
    configured = DEFAULT_GROUP_OVERLAY_PROFILES if profiles is None else profiles
    profile = configured.get(group) or configured.get("other") or ReboundOverlayProfile(enabled=False)
    return _coerce_profile(profile)


def _as_float(value: object, default: float = np.nan) -> float:
    if value is None or pd.isna(value):
        return default
    return float(value)


def _prior_drawdown_gate(
    df: pd.DataFrame,
    price_col: str,
    lookback: int,
    min_drawdown: float,
) -> pd.Series:
    if price_col not in df.columns:
        raise ValueError(f"Column '{price_col}' not found in DataFrame")
    if lookback <= 0:
        raise ValueError("drawdown_gate_lookback must be positive")
    if min_drawdown > 0:
        raise ValueError("min_prior_drawdown must be zero or negative")

    close = df[price_col].astype(float)
    rolling_peak = close.rolling(lookback, min_periods=2).max()
    drawdown = close / rolling_peak - 1.0
    recent_worst_drawdown = drawdown.rolling(lookback, min_periods=2).min()
    return (recent_worst_drawdown <= min_drawdown).fillna(False)


def _survival_gate(df: pd.DataFrame, profile: ReboundOverlayProfile) -> pd.Series:
    if not profile.require_survival_guard:
        return pd.Series(True, index=df.index)
    if profile.survival_regime_col not in df.columns:
        return pd.Series(True, index=df.index)

    fragile = set(profile.fragile_survival_regimes)
    return ~df[profile.survival_regime_col].astype(str).isin(fragile)


def _apply_overlay_risk_guard(
    raw_overlay_mask: pd.Series,
    prices: pd.Series,
    max_holding_period: Optional[int],
    stop_loss: Optional[float],
) -> tuple[pd.Series, pd.Series, pd.Series]:
    if max_holding_period is not None and max_holding_period <= 0:
        raise ValueError("max_overlay_holding_period must be positive when provided")
    if stop_loss is not None and stop_loss > 0:
        raise ValueError("overlay_stop_loss must be zero or negative when provided")

    active_values = []
    stop_values = []
    max_hold_values = []
    active = False
    entry_price = np.nan
    holding_period = 0

    for signal, price in zip(raw_overlay_mask.fillna(False), prices.astype(float)):
        stop_triggered = False
        max_hold_triggered = False

        if active:
            if stop_loss is not None and pd.notna(entry_price) and pd.notna(price):
                stop_triggered = (price / entry_price - 1.0) <= stop_loss
            if max_holding_period is not None:
                max_hold_triggered = holding_period >= max_holding_period
            if stop_triggered or max_hold_triggered or not signal:
                active = False
                entry_price = np.nan
                holding_period = 0

        if not active and signal and not (stop_triggered or max_hold_triggered):
            active = True
            entry_price = price
            holding_period = 0

        if active:
            holding_period += 1

        active_values.append(active)
        stop_values.append(stop_triggered)
        max_hold_values.append(max_hold_triggered)

    index = raw_overlay_mask.index
    return (
        pd.Series(active_values, index=index, dtype=bool),
        pd.Series(stop_values, index=index, dtype=bool),
        pd.Series(max_hold_values, index=index, dtype=bool),
    )


def backtest_drawdown_gated_rebound_overlay(
    df: pd.DataFrame,
    profile: ReboundOverlayProfile,
    regime_col: str = "regime_name",
    price_col: str = "Close",
    evaluation_start: Optional[object] = None,
    evaluation_end: Optional[object] = None,
    asset_name: Optional[str] = None,
    transaction_cost_bps: float = 0.0,
) -> RegimeBacktestResult:
    """
    Backtest rebound overlay with an optional prior-drawdown prerequisite.
    """
    if regime_col not in df.columns:
        raise ValueError(f"Column '{regime_col}' not found in DataFrame")

    result = df.copy()
    rebound_signal = compute_rebound_signal(
        result,
        price_col=price_col,
        fast_window=profile.fast_window,
        drawdown_lookback=profile.drawdown_lookback,
        min_rebound=profile.min_rebound,
        min_fast_return=profile.min_fast_return,
        require_above_fast_ma=profile.require_above_fast_ma,
    )

    if profile.require_prior_drawdown:
        drawdown_gate = _prior_drawdown_gate(
            result,
            price_col=price_col,
            lookback=profile.drawdown_gate_lookback,
            min_drawdown=profile.min_prior_drawdown,
        )
    else:
        drawdown_gate = pd.Series(True, index=result.index)

    survival_gate = _survival_gate(result, profile)
    base_exposure = regime_exposure(result[regime_col])
    raw_overlay_mask = rebound_signal & drawdown_gate & survival_gate & (base_exposure < profile.rebound_exposure)
    overlay_mask, stop_triggered, max_hold_triggered = _apply_overlay_risk_guard(
        raw_overlay_mask,
        result[price_col],
        max_holding_period=profile.max_overlay_holding_period,
        stop_loss=profile.overlay_stop_loss,
    )

    result["_rebound_signal"] = rebound_signal
    result["_drawdown_gate"] = drawdown_gate
    result["_survival_gate"] = survival_gate
    result["_raw_overlay_signal"] = raw_overlay_mask
    result["_overlay_stop_triggered"] = stop_triggered
    result["_overlay_max_hold_triggered"] = max_hold_triggered
    result["_base_regime_name"] = result[regime_col]
    result["_overlay_regime_name"] = result[regime_col].where(~overlay_mask, "Rebound Overlay")

    exposure_map = dict(DEFAULT_REGIME_EXPOSURE)
    exposure_map["Rebound Overlay"] = profile.rebound_exposure

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

    evaluation = result
    if evaluation_start is not None:
        evaluation = evaluation.loc[pd.Timestamp(evaluation_start):]
    if evaluation_end is not None:
        evaluation = evaluation.loc[:pd.Timestamp(evaluation_end)]

    backtest.metrics["rebound_signal_count"] = int(evaluation["_rebound_signal"].sum())
    backtest.metrics["rebound_signal_share"] = float(evaluation["_rebound_signal"].mean()) if len(evaluation) else 0.0
    backtest.metrics["drawdown_gate_count"] = int(evaluation["_drawdown_gate"].sum())
    backtest.metrics["drawdown_gate_share"] = float(evaluation["_drawdown_gate"].mean()) if len(evaluation) else 0.0
    backtest.metrics["survival_gate_count"] = int(evaluation["_survival_gate"].sum())
    backtest.metrics["survival_gate_share"] = float(evaluation["_survival_gate"].mean()) if len(evaluation) else 0.0
    backtest.metrics["raw_overlay_signal_count"] = int(evaluation["_raw_overlay_signal"].sum())
    backtest.metrics["raw_overlay_signal_share"] = (
        float(evaluation["_raw_overlay_signal"].mean()) if len(evaluation) else 0.0
    )
    backtest.metrics["overlay_active_count"] = int(evaluation["_overlay_regime_name"].eq("Rebound Overlay").sum())
    backtest.metrics["overlay_active_share"] = (
        float(evaluation["_overlay_regime_name"].eq("Rebound Overlay").mean()) if len(evaluation) else 0.0
    )
    backtest.metrics["overlay_stop_count"] = int(evaluation["_overlay_stop_triggered"].sum())
    backtest.metrics["overlay_max_hold_count"] = int(evaluation["_overlay_max_hold_triggered"].sum())
    backtest.metrics["latest_base_regime"] = (
        str(evaluation["_base_regime_name"].iloc[-1]) if len(evaluation) else "Unknown"
    )
    return backtest


def asset_group_for_symbol(
    symbol: str,
    asset_groups: Optional[Mapping[str, Sequence[str]]] = None,
    default_group: str = "other",
) -> str:
    """
    Return the configured group for a ticker symbol.

    Matching is case-insensitive and deterministic. Symbols not present in the
    configured groups are assigned to `default_group`.
    """
    groups = DEFAULT_ASSET_GROUPS if asset_groups is None else asset_groups
    target = _normalize_symbol(symbol)

    for group, symbols in groups.items():
        if target in {_normalize_symbol(item) for item in symbols}:
            return str(group)
    return default_group


def _result_row(
    *,
    asset: str,
    group: str,
    variant: str,
    result: RegimeBacktestResult,
    profile: ReboundOverlayProfile,
    profile_group: str,
    current_metrics: Optional[Mapping[str, object]] = None,
) -> Dict[str, object]:
    metrics = result.metrics
    baseline = metrics if current_metrics is None else current_metrics

    strategy_return = _as_float(metrics.get("strategy_return"))
    excess_return = _as_float(metrics.get("excess_return"))
    mean_exposure = _as_float(metrics.get("mean_exposure"))
    drawdown = _as_float(metrics.get("max_drawdown_strategy"))

    baseline_strategy_return = _as_float(baseline.get("strategy_return"))
    baseline_excess_return = _as_float(baseline.get("excess_return"))
    baseline_mean_exposure = _as_float(baseline.get("mean_exposure"))
    baseline_drawdown = _as_float(baseline.get("max_drawdown_strategy"))

    strategy_delta = strategy_return - baseline_strategy_return
    excess_delta = excess_return - baseline_excess_return
    exposure_delta = mean_exposure - baseline_mean_exposure
    drawdown_delta = drawdown - baseline_drawdown

    return {
        "asset": asset,
        "group": group,
        "variant": variant,
        "profile_group": profile_group,
        "overlay_enabled": bool(profile.enabled and variant == "group_rebound"),
        "observations": int(metrics.get("observations", 0)),
        "start_date": metrics.get("start_date"),
        "end_date": metrics.get("end_date"),
        "strategy_return": strategy_return,
        "gross_strategy_return": _as_float(metrics.get("gross_strategy_return")),
        "buy_hold_return": _as_float(metrics.get("buy_hold_return")),
        "excess_return": excess_return,
        "transaction_cost_bps": _as_float(metrics.get("transaction_cost_bps", 0.0), default=0.0),
        "total_transaction_cost": _as_float(metrics.get("total_transaction_cost", 0.0), default=0.0),
        "turnover": _as_float(metrics.get("turnover", 0.0), default=0.0),
        "mean_exposure": mean_exposure,
        "max_drawdown_strategy": drawdown,
        "max_drawdown_buy_hold": _as_float(metrics.get("max_drawdown_buy_hold")),
        "latest_regime": metrics.get("latest_regime", "Unknown"),
        "market_story": metrics.get("market_story", "unknown"),
        "strategy_posture": metrics.get("strategy_posture", "unknown"),
        "relative_result": metrics.get("relative_result", "unknown"),
        "rebound_signal_count": int(metrics.get("rebound_signal_count", 0) or 0),
        "rebound_signal_share": _as_float(metrics.get("rebound_signal_share", 0.0), default=0.0),
        "drawdown_gate_count": int(metrics.get("drawdown_gate_count", 0) or 0),
        "drawdown_gate_share": _as_float(metrics.get("drawdown_gate_share", 0.0), default=0.0),
        "survival_gate_count": int(metrics.get("survival_gate_count", 0) or 0),
        "survival_gate_share": _as_float(metrics.get("survival_gate_share", 0.0), default=0.0),
        "raw_overlay_signal_count": int(metrics.get("raw_overlay_signal_count", 0) or 0),
        "raw_overlay_signal_share": _as_float(metrics.get("raw_overlay_signal_share", 0.0), default=0.0),
        "overlay_active_count": int(metrics.get("overlay_active_count", 0) or 0),
        "overlay_active_share": _as_float(metrics.get("overlay_active_share", 0.0), default=0.0),
        "overlay_stop_count": int(metrics.get("overlay_stop_count", 0) or 0),
        "overlay_max_hold_count": int(metrics.get("overlay_max_hold_count", 0) or 0),
        "strategy_return_delta_vs_current": strategy_delta,
        "excess_return_delta_vs_current": excess_delta,
        "mean_exposure_delta_vs_current": exposure_delta,
        "drawdown_delta_vs_current": drawdown_delta,
        "improved_vs_current": bool(strategy_delta > 0.0) if variant == "group_rebound" else False,
        "rebound_exposure": profile.rebound_exposure,
        "fast_window": profile.fast_window,
        "drawdown_lookback": profile.drawdown_lookback,
        "min_rebound": profile.min_rebound,
        "min_fast_return": profile.min_fast_return,
        "require_above_fast_ma": profile.require_above_fast_ma,
        "require_prior_drawdown": profile.require_prior_drawdown,
        "drawdown_gate_lookback": profile.drawdown_gate_lookback,
        "min_prior_drawdown": profile.min_prior_drawdown,
        "max_overlay_holding_period": profile.max_overlay_holding_period,
        "overlay_stop_loss": profile.overlay_stop_loss,
        "require_survival_guard": profile.require_survival_guard,
        "survival_regime_col": profile.survival_regime_col,
        "fragile_survival_regimes": ",".join(profile.fragile_survival_regimes),
    }


def compare_group_aware_rebound(
    asset_frames: Mapping[str, pd.DataFrame],
    asset_groups: Optional[Mapping[str, Sequence[str]]] = None,
    profiles: Optional[Mapping[str, ReboundOverlayProfile | Mapping[str, object]]] = None,
    regime_col: str = "regime_name",
    price_col: str = "Close",
    evaluation_start: Optional[object] = None,
    evaluation_end: Optional[object] = None,
    transaction_cost_bps: float = 0.0,
) -> pd.DataFrame:
    """
    Compare current regime exposure with a group-aware rebound overlay.

    `asset_frames` must contain prepared DataFrames keyed by ticker. The
    candidate variant applies the rebound overlay only when the asset group's
    profile is enabled; otherwise it repeats the current regime strategy.
    """
    rows = []

    for asset in sorted(asset_frames):
        frame = asset_frames[asset]
        group = asset_group_for_symbol(asset, asset_groups)
        profile = _profile_for_group(group, profiles)

        current = backtest_regime_strategy(
            frame,
            regime_col=regime_col,
            price_col=price_col,
            transaction_cost_bps=transaction_cost_bps,
            evaluation_start=evaluation_start,
            evaluation_end=evaluation_end,
            asset_name=asset,
        )

        if profile.enabled:
            candidate = backtest_drawdown_gated_rebound_overlay(
                frame,
                profile=profile,
                regime_col=regime_col,
                price_col=price_col,
                evaluation_start=evaluation_start,
                evaluation_end=evaluation_end,
                asset_name=asset,
                transaction_cost_bps=transaction_cost_bps,
            )
        else:
            candidate = backtest_regime_strategy(
                frame,
                regime_col=regime_col,
                price_col=price_col,
                transaction_cost_bps=transaction_cost_bps,
                evaluation_start=evaluation_start,
                evaluation_end=evaluation_end,
                asset_name=asset,
            )

        rows.append(
            _result_row(
                asset=asset,
                group=group,
                variant="current",
                result=current,
                profile=profile,
                profile_group=group,
            )
        )
        rows.append(
            _result_row(
                asset=asset,
                group=group,
                variant="group_rebound",
                result=candidate,
                profile=profile,
                profile_group=group,
                current_metrics=current.metrics,
            )
        )

    return pd.DataFrame(rows)


def _window_value(window: Mapping[str, object], *names: str) -> Optional[object]:
    for name in names:
        if name in window:
            return window[name]
    return None


def _coerce_window(window: object, position: int) -> tuple[str, Optional[object], Optional[object]]:
    if isinstance(window, Mapping):
        label = _window_value(window, "label", "name", "window")
        start = _window_value(window, "start", "evaluation_start", "from")
        end = _window_value(window, "end", "evaluation_end", "to")
        return str(label or f"window_{position + 1}"), start, end

    if isinstance(window, Sequence) and not isinstance(window, (str, bytes)):
        if len(window) == 2:
            start, end = window
            return f"window_{position + 1}", start, end
        if len(window) == 3:
            first, second, third = window
            if isinstance(first, str) and not _looks_like_date(first):
                return first, second, third
            return str(third), first, second

    raise ValueError("windows must contain mappings or (start, end[, label]) tuples")


def _looks_like_date(value: str) -> bool:
    try:
        pd.Timestamp(value)
    except (TypeError, ValueError):
        return False
    return True


def run_walk_forward_group_rebound(
    asset_frames: Mapping[str, pd.DataFrame],
    windows: Sequence[Mapping[str, object] | Sequence[object]],
    asset_groups: Optional[Mapping[str, Sequence[str]]] = None,
    profiles: Optional[Mapping[str, ReboundOverlayProfile | Mapping[str, object]]] = None,
    regime_col: str = "regime_name",
    price_col: str = "Close",
    transaction_cost_bps: float = 0.0,
) -> pd.DataFrame:
    """
    Run group-aware rebound comparisons over multiple evaluation windows.

    Each window can be a mapping with `label`, `start`, and `end` keys, a
    `(start, end)` tuple, a `(start, end, label)` tuple, or a
    `(label, start, end)` tuple where the first item is not date-like.
    """
    frames = []

    for position, window in enumerate(windows):
        label, start, end = _coerce_window(window, position)
        comparison = compare_group_aware_rebound(
            asset_frames,
            asset_groups=asset_groups,
            profiles=profiles,
            regime_col=regime_col,
            price_col=price_col,
            transaction_cost_bps=transaction_cost_bps,
            evaluation_start=start,
            evaluation_end=end,
        )
        comparison.insert(0, "window", label)
        comparison.insert(1, "window_start", pd.Timestamp(start) if start is not None else None)
        comparison.insert(2, "window_end", pd.Timestamp(end) if end is not None else None)
        frames.append(comparison)

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


def summarize_group_rebound_results(
    comparison: pd.DataFrame,
    baseline_variant: str = "current",
    candidate_variant: str = "group_rebound",
) -> pd.DataFrame:
    """
    Aggregate a comparison or walk-forward output by window, group, and variant.
    """
    required = {"group", "variant", "strategy_return", "buy_hold_return", "excess_return", "mean_exposure"}
    missing = required.difference(comparison.columns)
    if missing:
        raise ValueError(f"comparison is missing required columns: {sorted(missing)}")

    group_cols = ["group", "variant"]
    if "window" in comparison.columns:
        group_cols.insert(0, "window")

    output_columns = group_cols + [
        "asset_count",
        "mean_strategy_return",
        "mean_buy_hold_return",
        "mean_excess_return",
        "median_exposure",
        "mean_strategy_delta_vs_current",
        "improved_count",
        "overlay_enabled_count",
        "top_market_story",
        "top_strategy_posture",
        "is_candidate",
        "is_baseline",
    ]

    working = comparison.copy()
    if "observations" in working.columns:
        observations = pd.to_numeric(working["observations"], errors="coerce").fillna(0)
        working = working[observations > 0]

    if working.empty:
        return pd.DataFrame(columns=output_columns)

    grouped = (
        working.groupby(group_cols, dropna=False)
        .agg(
            asset_count=("asset", "nunique"),
            mean_strategy_return=("strategy_return", "mean"),
            mean_buy_hold_return=("buy_hold_return", "mean"),
            mean_excess_return=("excess_return", "mean"),
            median_exposure=("mean_exposure", "median"),
            mean_strategy_delta_vs_current=("strategy_return_delta_vs_current", "mean"),
            improved_count=("improved_vs_current", "sum"),
            overlay_enabled_count=("overlay_enabled", "sum"),
            top_market_story=("market_story", lambda values: values.value_counts().index[0] if len(values) else "unknown"),
            top_strategy_posture=("strategy_posture", lambda values: values.value_counts().index[0] if len(values) else "unknown"),
        )
        .reset_index()
    )

    grouped["is_candidate"] = grouped["variant"].eq(candidate_variant)
    grouped["is_baseline"] = grouped["variant"].eq(baseline_variant)
    return grouped.reindex(columns=output_columns)
