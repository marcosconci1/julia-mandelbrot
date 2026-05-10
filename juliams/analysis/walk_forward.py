"""
Walk-forward diagnostics for group-aware rebound overlays.

This module exposes a small compatibility layer around the group-aware rebound
diagnostics so callers can test current-vs-overlay behavior across multiple
asset groups and date windows.
"""

from __future__ import annotations

from typing import Mapping, Optional, Sequence

import pandas as pd

from .group_rebound import (
    DEFAULT_ASSET_GROUPS,
    DEFAULT_GROUP_OVERLAY_PROFILES,
    ReboundOverlayProfile,
    compare_group_aware_rebound,
    run_walk_forward_group_rebound,
    summarize_group_rebound_results,
)


DEFAULT_WALK_FORWARD_WINDOW_SPECS: tuple[tuple[str, int, int], ...] = (
    ("recent_1m", 31, 0),
    ("prior_3m", 123, 92),
    ("prior_6m", 214, 183),
    ("prior_12m", 396, 365),
)


def narrative_alignment_score(market_story: object, strategy_posture: object) -> float:
    """
    Score whether the strategy posture fits the observed market narrative.

    The score is intentionally coarse. It is a research diagnostic, not a
    learned sentiment model: risk-on rebounds should not stay defensive,
    risk-off selloffs should not stay fully invested, and choppy tapes should
    avoid all-or-nothing posture.
    """
    story = str(market_story)
    posture = str(strategy_posture)
    scores = {
        "risk_on_rebound": {
            "fully_invested": 1.0,
            "selective": 0.7,
            "defensive": 0.0,
        },
        "risk_off_selloff": {
            "defensive": 1.0,
            "selective": 0.6,
            "fully_invested": 0.0,
        },
        "choppy_or_rangebound": {
            "selective": 1.0,
            "defensive": 0.7,
            "fully_invested": 0.2,
        },
    }
    return scores.get(story, {}).get(posture, 0.0)


def add_narrative_alignment(
    comparison: pd.DataFrame,
    aligned_threshold: float = 0.5,
) -> pd.DataFrame:
    """
    Add narrative alignment diagnostics to a comparison DataFrame.
    """
    if not 0 <= aligned_threshold <= 1:
        raise ValueError("aligned_threshold must be between 0 and 1")

    result = comparison.copy()
    if {"market_story", "strategy_posture"}.issubset(result.columns):
        result["narrative_alignment_score"] = [
            narrative_alignment_score(story, posture)
            for story, posture in zip(result["market_story"], result["strategy_posture"])
        ]
        result["narrative_aligned"] = result["narrative_alignment_score"] >= aligned_threshold
    return result


def _profiles_from_enabled_groups(
    overlay_enabled_groups: Optional[set[str] | Sequence[str]] = None,
    require_prior_drawdown: Optional[bool] = None,
    drawdown_gate_lookback: Optional[int] = None,
    min_prior_drawdown: Optional[float] = None,
    max_overlay_holding_period: Optional[int] = None,
    overlay_stop_loss: Optional[float] = None,
    **overlay_kwargs: object,
) -> dict[str, ReboundOverlayProfile]:
    enabled_groups = set(overlay_enabled_groups or {"indices", "mega_cap_tech"})
    profiles: dict[str, ReboundOverlayProfile] = {}

    for group, profile in DEFAULT_GROUP_OVERLAY_PROFILES.items():
        profile_kwargs = profile.overlay_kwargs()
        profile_kwargs.update(overlay_kwargs)
        profile_kwargs["require_prior_drawdown"] = (
            profile.require_prior_drawdown if require_prior_drawdown is None else require_prior_drawdown
        )
        profile_kwargs["drawdown_gate_lookback"] = (
            profile.drawdown_gate_lookback if drawdown_gate_lookback is None else drawdown_gate_lookback
        )
        profile_kwargs["min_prior_drawdown"] = (
            profile.min_prior_drawdown if min_prior_drawdown is None else min_prior_drawdown
        )
        profile_kwargs["max_overlay_holding_period"] = (
            profile.max_overlay_holding_period
            if max_overlay_holding_period is None
            else max_overlay_holding_period
        )
        profile_kwargs["overlay_stop_loss"] = (
            profile.overlay_stop_loss if overlay_stop_loss is None else overlay_stop_loss
        )
        profiles[group] = ReboundOverlayProfile(
            enabled=group in enabled_groups,
            **profile_kwargs,
        )

    return profiles


def build_recent_walk_forward_windows(
    end_date: object,
    window_specs: Sequence[tuple[str, int, int]] = DEFAULT_WALK_FORWARD_WINDOW_SPECS,
) -> list[dict[str, object]]:
    """
    Build named calendar windows ending at offsets from a reference date.

    Each spec is `(name, start_offset_days, end_offset_days)`, where offsets
    are measured backwards from `end_date`.
    """
    end = pd.Timestamp(end_date).normalize()
    windows = []
    for name, start_offset, end_offset in window_specs:
        if start_offset < end_offset:
            raise ValueError("window start offset must be greater than or equal to end offset")
        windows.append(
            {
                "name": name,
                "start": end - pd.Timedelta(days=int(start_offset)),
                "end": end - pd.Timedelta(days=int(end_offset)),
            }
        )
    return windows


def compare_group_rebound_overlay(
    asset_frames: Mapping[str, pd.DataFrame],
    asset_groups: Optional[Mapping[str, Sequence[str]]] = None,
    overlay_enabled_groups: Optional[set[str] | Sequence[str]] = None,
    regime_col: str = "regime_name",
    price_col: str = "Close",
    evaluation_start: Optional[object] = None,
    evaluation_end: Optional[object] = None,
    rebound_exposure: float = 0.75,
    fast_window: int = 5,
    drawdown_lookback: int = 20,
    min_rebound: float = 0.03,
    min_fast_return: float = 0.02,
    require_above_fast_ma: bool = True,
    require_prior_drawdown: Optional[bool] = None,
    drawdown_gate_lookback: Optional[int] = None,
    min_prior_drawdown: Optional[float] = None,
    max_overlay_holding_period: Optional[int] = None,
    overlay_stop_loss: Optional[float] = None,
    transaction_cost_bps: float = 0.0,
) -> pd.DataFrame:
    """
    Compare current regime strategy with group-aware rebound overlay.

    `overlay_enabled_groups` controls which asset groups are allowed to use the
    overlay. Disabled groups return the same metrics as the current strategy for
    their `group_rebound` row.
    """
    profiles = _profiles_from_enabled_groups(
        overlay_enabled_groups,
        require_prior_drawdown=require_prior_drawdown,
        drawdown_gate_lookback=drawdown_gate_lookback,
        min_prior_drawdown=min_prior_drawdown,
        max_overlay_holding_period=max_overlay_holding_period,
        overlay_stop_loss=overlay_stop_loss,
        rebound_exposure=rebound_exposure,
        fast_window=fast_window,
        drawdown_lookback=drawdown_lookback,
        min_rebound=min_rebound,
        min_fast_return=min_fast_return,
        require_above_fast_ma=require_above_fast_ma,
    )
    return compare_group_aware_rebound(
        asset_frames,
        asset_groups=asset_groups,
        profiles=profiles,
        regime_col=regime_col,
        price_col=price_col,
        transaction_cost_bps=transaction_cost_bps,
        evaluation_start=evaluation_start,
        evaluation_end=evaluation_end,
    )


def run_walk_forward_diagnostics(
    asset_frames: Mapping[str, pd.DataFrame],
    windows: Sequence[Mapping[str, object] | Sequence[object]],
    asset_groups: Optional[Mapping[str, Sequence[str]]] = None,
    overlay_enabled_groups: Optional[set[str] | Sequence[str]] = None,
    regime_col: str = "regime_name",
    price_col: str = "Close",
    rebound_exposure: float = 0.75,
    fast_window: int = 5,
    drawdown_lookback: int = 20,
    min_rebound: float = 0.03,
    min_fast_return: float = 0.02,
    require_above_fast_ma: bool = True,
    require_prior_drawdown: Optional[bool] = None,
    drawdown_gate_lookback: Optional[int] = None,
    min_prior_drawdown: Optional[float] = None,
    max_overlay_holding_period: Optional[int] = None,
    overlay_stop_loss: Optional[float] = None,
    transaction_cost_bps: float = 0.0,
) -> pd.DataFrame:
    """
    Run group-aware rebound comparisons across multiple evaluation windows.
    """
    profiles = _profiles_from_enabled_groups(
        overlay_enabled_groups,
        require_prior_drawdown=require_prior_drawdown,
        drawdown_gate_lookback=drawdown_gate_lookback,
        min_prior_drawdown=min_prior_drawdown,
        max_overlay_holding_period=max_overlay_holding_period,
        overlay_stop_loss=overlay_stop_loss,
        rebound_exposure=rebound_exposure,
        fast_window=fast_window,
        drawdown_lookback=drawdown_lookback,
        min_rebound=min_rebound,
        min_fast_return=min_fast_return,
        require_above_fast_ma=require_above_fast_ma,
    )
    return run_walk_forward_group_rebound(
        asset_frames,
        windows=windows,
        asset_groups=asset_groups,
        profiles=profiles,
        regime_col=regime_col,
        price_col=price_col,
        transaction_cost_bps=transaction_cost_bps,
    )


def evaluate_group_rebound_promotion(
    comparison: pd.DataFrame,
    eligible_groups: Sequence[str] = ("indices", "mega_cap_tech"),
    protected_groups: Sequence[str] = ("diversifiers", "crypto"),
    candidate_variant: str = "group_rebound",
    min_eligible_improved_share: float = 0.60,
    max_candidate_drawdown_delta: float = -0.02,
    max_protected_degraded_share: float = 0.0,
) -> dict[str, object]:
    """
    Evaluate whether group rebound is ready to promote from diagnostic mode.

    Drawdown deltas are measured as candidate drawdown minus current drawdown;
    more negative values mean deeper drawdowns. Promotion is conservative.
    """
    required = {
        "group",
        "variant",
        "improved_vs_current",
        "drawdown_delta_vs_current",
        "strategy_return_delta_vs_current",
    }
    missing = required.difference(comparison.columns)
    if missing:
        raise ValueError(f"comparison is missing required columns: {sorted(missing)}")
    if not 0 <= min_eligible_improved_share <= 1:
        raise ValueError("min_eligible_improved_share must be between 0 and 1")
    if not 0 <= max_protected_degraded_share <= 1:
        raise ValueError("max_protected_degraded_share must be between 0 and 1")

    candidate = comparison[comparison["variant"] == candidate_variant].copy()
    if "observations" in candidate.columns:
        observations = pd.to_numeric(candidate["observations"], errors="coerce").fillna(0)
        candidate = candidate[observations > 0]
    eligible = candidate[candidate["group"].isin(eligible_groups)]
    protected = candidate[candidate["group"].isin(protected_groups)]

    eligible_count = int(len(eligible))
    protected_count = int(len(protected))
    eligible_improved_share = float(eligible["improved_vs_current"].mean()) if eligible_count else 0.0
    protected_degraded_share = (
        float((protected["strategy_return_delta_vs_current"] < 0).mean()) if protected_count else 0.0
    )
    worst_drawdown_delta = (
        float(candidate["drawdown_delta_vs_current"].min()) if len(candidate) else 0.0
    )

    passes = (
        eligible_count > 0
        and eligible_improved_share >= min_eligible_improved_share
        and protected_degraded_share <= max_protected_degraded_share
        and worst_drawdown_delta >= max_candidate_drawdown_delta
    )

    reasons = []
    if eligible_count == 0:
        reasons.append("no eligible group observations")
    if eligible_improved_share < min_eligible_improved_share:
        reasons.append("eligible improvement share below threshold")
    if protected_degraded_share > max_protected_degraded_share:
        reasons.append("protected groups degraded")
    if worst_drawdown_delta < max_candidate_drawdown_delta:
        reasons.append("candidate drawdown worsened beyond threshold")

    return {
        "promote": bool(passes),
        "eligible_count": eligible_count,
        "eligible_improved_share": eligible_improved_share,
        "protected_count": protected_count,
        "protected_degraded_share": protected_degraded_share,
        "worst_drawdown_delta": worst_drawdown_delta,
        "reasons": reasons,
    }


def _nonempty_candidate(
    comparison: pd.DataFrame,
    candidate_variant: str,
    eligible_groups: Sequence[str],
) -> pd.DataFrame:
    candidate = comparison[
        (comparison["variant"] == candidate_variant)
        & (comparison["group"].isin(eligible_groups))
    ].copy()
    if "observations" in candidate.columns:
        observations = pd.to_numeric(candidate["observations"], errors="coerce").fillna(0)
        candidate = candidate[observations > 0]
    return candidate


def _window_count(candidate: pd.DataFrame) -> int:
    if candidate.empty:
        return 0
    if "window" not in candidate.columns:
        return 1
    return int(candidate["window"].dropna().nunique())


def _recent_window_slice(
    comparison: pd.DataFrame,
    variant: str,
    groups: Sequence[str],
    recent_window: str,
) -> pd.DataFrame:
    rows = comparison[
        (comparison["variant"] == variant)
        & (comparison["group"].isin(groups))
    ].copy()
    if "observations" in rows.columns:
        observations = pd.to_numeric(rows["observations"], errors="coerce").fillna(0)
        rows = rows[observations > 0]
    if "window" in rows.columns and recent_window in set(rows["window"].dropna()):
        rows = rows[rows["window"] == recent_window]
    return rows


def _promotion_without_parameter_stability(
    comparison: pd.DataFrame,
    eligible_groups: Sequence[str],
    protected_groups: Sequence[str],
    candidate_variant: str,
    baseline_variant: str,
    min_eligible_improved_share: float,
    max_candidate_drawdown_delta: float,
    max_protected_degraded_share: float,
    min_windows: int,
    recent_window: str,
    min_recent_narrative_score: float,
    min_candidate_narrative_advantage: float,
    required_transaction_cost_bps: float,
) -> dict[str, object]:
    comparison = add_narrative_alignment(comparison)
    basic = evaluate_group_rebound_promotion(
        comparison,
        eligible_groups=eligible_groups,
        protected_groups=protected_groups,
        candidate_variant=candidate_variant,
        min_eligible_improved_share=min_eligible_improved_share,
        max_candidate_drawdown_delta=max_candidate_drawdown_delta,
        max_protected_degraded_share=max_protected_degraded_share,
    )
    reasons = list(basic["reasons"])
    candidate = _nonempty_candidate(comparison, candidate_variant, eligible_groups)
    windows_tested = _window_count(candidate)

    if windows_tested < min_windows:
        reasons.append("insufficient non-empty walk-forward windows")

    transaction_cost_checked = False
    min_transaction_cost_bps = 0.0
    if required_transaction_cost_bps > 0:
        if "transaction_cost_bps" not in candidate.columns or candidate.empty:
            reasons.append("transaction cost check missing")
        else:
            min_transaction_cost_bps = float(
                pd.to_numeric(candidate["transaction_cost_bps"], errors="coerce").fillna(0.0).min()
            )
            transaction_cost_checked = min_transaction_cost_bps >= required_transaction_cost_bps
            if not transaction_cost_checked:
                reasons.append("transaction cost check below required bps")
    else:
        transaction_cost_checked = True
        if "transaction_cost_bps" in candidate.columns and not candidate.empty:
            min_transaction_cost_bps = float(
                pd.to_numeric(candidate["transaction_cost_bps"], errors="coerce").fillna(0.0).min()
            )

    candidate_recent = _recent_window_slice(comparison, candidate_variant, eligible_groups, recent_window)
    baseline_recent = _recent_window_slice(comparison, baseline_variant, eligible_groups, recent_window)
    narrative_count = int(len(candidate_recent))
    candidate_narrative_score = (
        float(candidate_recent["narrative_alignment_score"].mean())
        if "narrative_alignment_score" in candidate_recent.columns and narrative_count
        else 0.0
    )
    baseline_narrative_score = (
        float(baseline_recent["narrative_alignment_score"].mean())
        if "narrative_alignment_score" in baseline_recent.columns and len(baseline_recent)
        else 0.0
    )

    if "narrative_alignment_score" not in comparison.columns:
        reasons.append("narrative alignment check missing")
    elif narrative_count == 0:
        reasons.append("no recent narrative observations")
    elif candidate_narrative_score < min_recent_narrative_score:
        reasons.append("recent narrative alignment below threshold")
    elif candidate_narrative_score + 1e-12 < baseline_narrative_score + min_candidate_narrative_advantage:
        reasons.append("candidate narrative alignment worse than current")

    passes = (
        bool(basic["promote"])
        and windows_tested >= min_windows
        and transaction_cost_checked
        and narrative_count > 0
        and candidate_narrative_score >= min_recent_narrative_score
        and candidate_narrative_score + 1e-12 >= baseline_narrative_score + min_candidate_narrative_advantage
    )

    return {
        **basic,
        "promote": bool(passes),
        "reasons": reasons,
        "windows_tested": windows_tested,
        "min_windows": int(min_windows),
        "transaction_cost_checked": bool(transaction_cost_checked),
        "required_transaction_cost_bps": float(required_transaction_cost_bps),
        "min_transaction_cost_bps": float(min_transaction_cost_bps),
        "recent_window": recent_window,
        "recent_narrative_count": narrative_count,
        "recent_narrative_alignment_score": candidate_narrative_score,
        "current_recent_narrative_alignment_score": baseline_narrative_score,
        "min_recent_narrative_score": float(min_recent_narrative_score),
    }


def evaluate_research_grade_rebound_promotion(
    comparison: pd.DataFrame,
    eligible_groups: Sequence[str] = ("indices", "mega_cap_tech"),
    protected_groups: Sequence[str] = ("diversifiers", "crypto"),
    candidate_variant: str = "group_rebound",
    baseline_variant: str = "current",
    min_eligible_improved_share: float = 0.60,
    max_candidate_drawdown_delta: float = -0.02,
    max_protected_degraded_share: float = 0.0,
    min_windows: int = 3,
    recent_window: str = "recent_1m",
    min_recent_narrative_score: float = 0.70,
    min_candidate_narrative_advantage: float = 0.0,
    required_transaction_cost_bps: float = 5.0,
    parameter_results: Optional[Mapping[str, pd.DataFrame]] = None,
    require_parameter_stability: bool = True,
    min_parameter_pass_share: float = 0.67,
) -> dict[str, object]:
    """
    Conservative promotion gate for turning a rebound overlay into default logic.

    The gate encodes the research checks implied by momentum, rebound-crash, and
    backtest-overfitting literature: out-of-sample windows, costs, drawdown
    safety, narrative coherence, and parameter stability.
    """
    if min_windows <= 0:
        raise ValueError("min_windows must be positive")
    if required_transaction_cost_bps < 0:
        raise ValueError("required_transaction_cost_bps must be non-negative")
    if not 0 <= min_recent_narrative_score <= 1:
        raise ValueError("min_recent_narrative_score must be between 0 and 1")
    if not 0 <= min_parameter_pass_share <= 1:
        raise ValueError("min_parameter_pass_share must be between 0 and 1")

    result = _promotion_without_parameter_stability(
        comparison=comparison,
        eligible_groups=eligible_groups,
        protected_groups=protected_groups,
        candidate_variant=candidate_variant,
        baseline_variant=baseline_variant,
        min_eligible_improved_share=min_eligible_improved_share,
        max_candidate_drawdown_delta=max_candidate_drawdown_delta,
        max_protected_degraded_share=max_protected_degraded_share,
        min_windows=min_windows,
        recent_window=recent_window,
        min_recent_narrative_score=min_recent_narrative_score,
        min_candidate_narrative_advantage=min_candidate_narrative_advantage,
        required_transaction_cost_bps=required_transaction_cost_bps,
    )

    reasons = list(result["reasons"])
    parameter_checks: dict[str, bool] = {}
    if parameter_results:
        for name, frame in parameter_results.items():
            parameter_eval = _promotion_without_parameter_stability(
                comparison=frame,
                eligible_groups=eligible_groups,
                protected_groups=protected_groups,
                candidate_variant=candidate_variant,
                baseline_variant=baseline_variant,
                min_eligible_improved_share=min_eligible_improved_share,
                max_candidate_drawdown_delta=max_candidate_drawdown_delta,
                max_protected_degraded_share=max_protected_degraded_share,
                min_windows=min_windows,
                recent_window=recent_window,
                min_recent_narrative_score=min_recent_narrative_score,
                min_candidate_narrative_advantage=min_candidate_narrative_advantage,
                required_transaction_cost_bps=required_transaction_cost_bps,
            )
            parameter_checks[str(name)] = bool(parameter_eval["promote"])

    if parameter_checks:
        parameter_pass_share = sum(parameter_checks.values()) / len(parameter_checks)
    else:
        parameter_pass_share = 0.0

    parameter_stability_checked = bool(parameter_checks) or not require_parameter_stability
    if require_parameter_stability and not parameter_checks:
        reasons.append("parameter stability checks missing")
    elif require_parameter_stability and parameter_pass_share < min_parameter_pass_share:
        reasons.append("parameter stability pass share below threshold")

    promote = (
        bool(result["promote"])
        and parameter_stability_checked
        and (
            not require_parameter_stability
            or parameter_pass_share >= min_parameter_pass_share
        )
    )

    return {
        **result,
        "promote": bool(promote),
        "reasons": reasons,
        "parameter_stability_checked": bool(parameter_stability_checked),
        "parameter_pass_share": float(parameter_pass_share),
        "min_parameter_pass_share": float(min_parameter_pass_share),
        "parameter_checks": parameter_checks,
    }


__all__ = [
    "DEFAULT_ASSET_GROUPS",
    "DEFAULT_GROUP_OVERLAY_PROFILES",
    "DEFAULT_WALK_FORWARD_WINDOW_SPECS",
    "ReboundOverlayProfile",
    "build_recent_walk_forward_windows",
    "compare_group_aware_rebound",
    "compare_group_rebound_overlay",
    "add_narrative_alignment",
    "evaluate_group_rebound_promotion",
    "evaluate_research_grade_rebound_promotion",
    "narrative_alignment_score",
    "run_walk_forward_diagnostics",
    "run_walk_forward_group_rebound",
    "summarize_group_rebound_results",
]
