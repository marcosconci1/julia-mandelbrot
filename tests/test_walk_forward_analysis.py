import pandas as pd
import pytest


walk_forward = pytest.importorskip("juliams.analysis.walk_forward")

# Public API contract:
# - compare_group_rebound_overlay(asset_frames, asset_groups, overlay_enabled_groups, ...)
#   returns a long DataFrame with one row per asset/variant. Expected variants are
#   "current" and "group_rebound".
# - run_walk_forward_diagnostics(asset_frames, windows, asset_groups,
#   overlay_enabled_groups, ...) returns the same long metrics with window metadata.
# These tests use synthetic prices only and require prior-day signal use.
compare_group_rebound_overlay = walk_forward.compare_group_rebound_overlay
run_walk_forward_diagnostics = walk_forward.run_walk_forward_diagnostics
build_recent_walk_forward_windows = walk_forward.build_recent_walk_forward_windows
evaluate_group_rebound_promotion = walk_forward.evaluate_group_rebound_promotion
evaluate_research_grade_rebound_promotion = walk_forward.evaluate_research_grade_rebound_promotion
narrative_alignment_score = walk_forward.narrative_alignment_score
summarize_group_rebound_results = walk_forward.summarize_group_rebound_results


def _frame(close, regimes=None, start="2024-01-01"):
    index = pd.date_range(start, periods=len(close), freq="D")
    if regimes is None:
        regimes = ["Bear Quiet"] * len(close)
    return pd.DataFrame({"Close": close, "regime_name": regimes}, index=index)


def _row(result, asset, variant):
    rows = result[(result["asset"] == asset) & (result["variant"] == variant)]
    assert len(rows) == 1
    return rows.iloc[0]


def _run_group_overlay(asset_frames, enabled_groups):
    return compare_group_rebound_overlay(
        asset_frames,
        asset_groups={
            "indices": ["SPY"],
            "mega_cap_tech": ["MSFT"],
            "diversifiers": ["GLD"],
            "crypto": ["BTC-USD"],
        },
        overlay_enabled_groups=enabled_groups,
        fast_window=1,
        drawdown_lookback=3,
        min_rebound=0.05,
        min_fast_return=0.05,
        require_above_fast_ma=False,
        rebound_exposure=0.75,
    )


def test_group_rebound_overlay_is_enabled_only_for_index_and_tech_rebounds():
    result = _run_group_overlay(
        {
            "SPY": _frame([100.0, 90.0, 99.0, 108.9]),
            "MSFT": _frame([50.0, 45.0, 49.5, 54.45]),
            "GLD": _frame([100.0, 90.0, 99.0, 89.1]),
            "BTC-USD": _frame([100.0, 90.0, 99.0, 108.9]),
        },
        enabled_groups={"indices", "mega_cap_tech"},
    )

    assert {
        "asset",
        "group",
        "variant",
        "strategy_return",
        "mean_exposure",
        "overlay_enabled",
    }.issubset(result.columns)

    spy_current = _row(result, "SPY", "current")
    spy_overlay = _row(result, "SPY", "group_rebound")
    msft_current = _row(result, "MSFT", "current")
    msft_overlay = _row(result, "MSFT", "group_rebound")
    gld_current = _row(result, "GLD", "current")
    gld_overlay = _row(result, "GLD", "group_rebound")
    btc_current = _row(result, "BTC-USD", "current")
    btc_overlay = _row(result, "BTC-USD", "group_rebound")

    assert spy_current["strategy_return"] == 0.0
    assert spy_overlay["strategy_return"] == pytest.approx(0.075)
    assert bool(spy_overlay["overlay_enabled"]) is True
    assert spy_overlay["strategy_return"] > spy_current["strategy_return"]

    assert msft_current["strategy_return"] == 0.0
    assert msft_overlay["strategy_return"] == pytest.approx(0.075)
    assert bool(msft_overlay["overlay_enabled"]) is True
    assert msft_overlay["strategy_return"] > msft_current["strategy_return"]

    assert gld_overlay["group"] == "diversifiers"
    assert bool(gld_overlay["overlay_enabled"]) is False
    assert gld_overlay["strategy_return"] == pytest.approx(gld_current["strategy_return"])

    assert btc_overlay["group"] == "crypto"
    assert bool(btc_overlay["overlay_enabled"]) is False
    assert btc_overlay["strategy_return"] == pytest.approx(btc_current["strategy_return"])


def test_disabled_groups_fall_back_to_current_strategy_even_when_overlay_would_fire():
    result = _run_group_overlay(
        {
            "SPY": _frame([100.0, 90.0, 99.0, 108.9]),
            "MSFT": _frame([50.0, 45.0, 49.5, 54.45]),
            "GLD": _frame([100.0, 90.0, 99.0, 89.1]),
            "BTC-USD": _frame([100.0, 90.0, 99.0, 108.9]),
        },
        enabled_groups={"indices", "mega_cap_tech"},
    )

    for asset in ("GLD", "BTC-USD"):
        current = _row(result, asset, "current")
        group_rebound = _row(result, asset, "group_rebound")

        assert bool(group_rebound["overlay_enabled"]) is False
        assert group_rebound["strategy_return"] == pytest.approx(current["strategy_return"])
        assert group_rebound["mean_exposure"] == pytest.approx(current["mean_exposure"])


def test_walk_forward_diagnostics_include_multiple_windows_and_variants():
    diagnostics = run_walk_forward_diagnostics(
        {
            "SPY": _frame([100.0, 90.0, 99.0, 108.9, 98.01, 107.811, 118.5921, 130.45131]),
            "GLD": _frame([100.0, 90.0, 99.0, 89.1, 98.01, 88.209, 97.0299, 87.32691]),
        },
        windows=[
            {"name": "first_rebound", "start": "2024-01-01", "end": "2024-01-04"},
            {"name": "second_rebound", "start": "2024-01-05", "end": "2024-01-08"},
        ],
        asset_groups={
            "indices": ["SPY"],
            "diversifiers": ["GLD"],
        },
        overlay_enabled_groups={"indices"},
        fast_window=1,
        drawdown_lookback=3,
        min_rebound=0.05,
        min_fast_return=0.05,
        require_above_fast_ma=False,
        rebound_exposure=0.75,
    )

    assert {
        "window",
        "window_start",
        "window_end",
        "asset",
        "group",
        "variant",
        "strategy_return",
        "buy_hold_return",
        "mean_exposure",
    }.issubset(diagnostics.columns)
    assert set(diagnostics["window"]) == {"first_rebound", "second_rebound"}
    assert set(diagnostics["variant"]) == {"current", "group_rebound"}

    expected_rows = 2 * 2 * 2
    assert len(diagnostics) == expected_rows


def test_group_rebound_overlay_uses_prior_day_signal_for_expected_returns():
    result = _run_group_overlay(
        {
            "SPY": _frame([100.0, 90.0, 99.0, 108.9]),
            "MSFT": _frame([50.0, 45.0, 49.5, 54.45]),
            "GLD": _frame([100.0, 90.0, 99.0, 89.1]),
            "BTC-USD": _frame([100.0, 90.0, 99.0, 108.9]),
        },
        enabled_groups={"indices", "mega_cap_tech"},
    )

    spy_overlay = _row(result, "SPY", "group_rebound")
    same_day_lookahead_return = (1.0 + 0.75 * 0.10) ** 2 - 1.0

    assert spy_overlay["strategy_return"] == pytest.approx(0.075)
    assert spy_overlay["strategy_return"] != pytest.approx(same_day_lookahead_return)


def test_drawdown_gate_allows_rebound_only_after_prior_selloff():
    result = compare_group_rebound_overlay(
        {
            "SPY": _frame([100.0, 90.0, 99.0, 108.9]),
            "QQQ": _frame([100.0, 102.0, 104.0, 106.0]),
        },
        asset_groups={
            "indices": ["SPY", "QQQ"],
        },
        overlay_enabled_groups={"indices"},
        fast_window=1,
        drawdown_lookback=3,
        min_rebound=0.05,
        min_fast_return=0.01,
        require_above_fast_ma=False,
        rebound_exposure=0.75,
        require_prior_drawdown=True,
        drawdown_gate_lookback=3,
        min_prior_drawdown=-0.05,
    )

    spy_overlay = _row(result, "SPY", "group_rebound")
    qqq_current = _row(result, "QQQ", "current")
    qqq_overlay = _row(result, "QQQ", "group_rebound")

    assert spy_overlay["strategy_return"] == pytest.approx(0.075)
    assert spy_overlay["drawdown_gate_share"] > 0
    assert qqq_overlay["strategy_return"] == pytest.approx(qqq_current["strategy_return"])
    assert qqq_overlay["drawdown_gate_share"] == 0.0


def test_survival_guard_blocks_rebound_overlay_in_fragile_state():
    fragile = _frame([100.0, 90.0, 99.0, 108.9])
    fragile["survival_regime"] = "Fragile"

    result = compare_group_rebound_overlay(
        {
            "SPY": fragile,
        },
        asset_groups={
            "indices": ["SPY"],
        },
        overlay_enabled_groups={"indices"},
        fast_window=1,
        drawdown_lookback=3,
        min_rebound=0.05,
        min_fast_return=0.05,
        require_above_fast_ma=False,
        rebound_exposure=0.75,
        require_prior_drawdown=False,
    )

    current = _row(result, "SPY", "current")
    overlay = _row(result, "SPY", "group_rebound")

    assert overlay["strategy_return"] == pytest.approx(current["strategy_return"])
    assert overlay["rebound_signal_count"] > 0
    assert overlay["survival_gate_share"] == 0.0
    assert overlay["raw_overlay_signal_count"] == 0


def test_overlay_max_holding_period_exits_before_reversal_day():
    result = compare_group_rebound_overlay(
        {
            "SPY": _frame([100.0, 90.0, 99.0, 108.9, 98.01]),
        },
        asset_groups={
            "indices": ["SPY"],
        },
        overlay_enabled_groups={"indices"},
        fast_window=1,
        drawdown_lookback=3,
        min_rebound=0.05,
        min_fast_return=0.05,
        require_above_fast_ma=False,
        rebound_exposure=0.75,
        require_prior_drawdown=True,
        drawdown_gate_lookback=3,
        min_prior_drawdown=-0.05,
        max_overlay_holding_period=1,
        overlay_stop_loss=-0.50,
    )

    overlay = _row(result, "SPY", "group_rebound")

    assert overlay["strategy_return"] == pytest.approx(0.075)
    assert overlay["overlay_max_hold_count"] >= 1
    assert overlay["overlay_active_share"] < overlay["raw_overlay_signal_share"]


def test_recent_walk_forward_windows_are_named_and_ordered():
    windows = build_recent_walk_forward_windows(
        "2024-04-30",
        window_specs=(
            ("recent", 30, 0),
            ("prior", 60, 31),
        ),
    )

    assert windows == [
        {
            "name": "recent",
            "start": pd.Timestamp("2024-03-31"),
            "end": pd.Timestamp("2024-04-30"),
        },
        {
            "name": "prior",
            "start": pd.Timestamp("2024-03-01"),
            "end": pd.Timestamp("2024-03-30"),
        },
    ]


def test_promotion_criteria_require_eligible_improvement_and_protected_safety():
    passing = pd.DataFrame(
        [
            {
                "group": "indices",
                "variant": "group_rebound",
                "improved_vs_current": True,
                "drawdown_delta_vs_current": -0.005,
                "strategy_return_delta_vs_current": 0.02,
            },
            {
                "group": "mega_cap_tech",
                "variant": "group_rebound",
                "improved_vs_current": True,
                "drawdown_delta_vs_current": 0.0,
                "strategy_return_delta_vs_current": 0.01,
            },
            {
                "group": "diversifiers",
                "variant": "group_rebound",
                "improved_vs_current": False,
                "drawdown_delta_vs_current": 0.0,
                "strategy_return_delta_vs_current": 0.0,
            },
        ]
    )

    assert evaluate_group_rebound_promotion(passing)["promote"] is True

    failing = passing.copy()
    failing.loc[failing["group"] == "diversifiers", "strategy_return_delta_vs_current"] = -0.01

    evaluation = evaluate_group_rebound_promotion(failing)

    assert evaluation["promote"] is False
    assert "protected groups degraded" in evaluation["reasons"]


def _research_gate_frame(transaction_cost_bps=5.0, recent_candidate_posture="selective"):
    rows = []
    for window in ("recent_1m", "prior_3m", "prior_6m"):
        candidate_posture = recent_candidate_posture if window == "recent_1m" else "fully_invested"
        rows.append(
            {
                "window": window,
                "asset": "SPY",
                "group": "indices",
                "variant": "current",
                "improved_vs_current": False,
                "drawdown_delta_vs_current": 0.0,
                "strategy_return_delta_vs_current": 0.0,
                "strategy_return": 0.01,
                "buy_hold_return": 0.04,
                "excess_return": -0.03,
                "mean_exposure": 0.25,
                "market_story": "risk_on_rebound",
                "strategy_posture": "defensive",
                "transaction_cost_bps": transaction_cost_bps,
                "observations": 20,
            }
        )
        rows.append(
            {
                "window": window,
                "asset": "SPY",
                "group": "indices",
                "variant": "group_rebound",
                "improved_vs_current": True,
                "drawdown_delta_vs_current": -0.005,
                "strategy_return_delta_vs_current": 0.02,
                "strategy_return": 0.03,
                "buy_hold_return": 0.04,
                "excess_return": -0.01,
                "mean_exposure": 0.75,
                "market_story": "risk_on_rebound",
                "strategy_posture": candidate_posture,
                "transaction_cost_bps": transaction_cost_bps,
                "observations": 20,
            }
        )
    rows.append(
        {
            "window": "recent_1m",
            "asset": "GLD",
            "group": "diversifiers",
            "variant": "group_rebound",
            "improved_vs_current": False,
            "drawdown_delta_vs_current": 0.0,
            "strategy_return_delta_vs_current": 0.0,
            "strategy_return": 0.0,
            "buy_hold_return": 0.0,
            "excess_return": 0.0,
            "mean_exposure": 0.0,
            "market_story": "choppy_or_rangebound",
            "strategy_posture": "defensive",
            "transaction_cost_bps": transaction_cost_bps,
            "observations": 20,
        }
    )
    return pd.DataFrame(rows)


def test_research_grade_promotion_requires_costs_narrative_and_stability():
    comparison = _research_gate_frame()

    evaluation = evaluate_research_grade_rebound_promotion(
        comparison,
        parameter_results={
            "lower_exposure": comparison,
            "slower_confirmation": comparison,
            "stricter_risk_gate": comparison,
        },
    )

    assert evaluation["promote"] is True
    assert evaluation["windows_tested"] == 3
    assert evaluation["transaction_cost_checked"] is True
    assert evaluation["recent_narrative_alignment_score"] == pytest.approx(0.7)
    assert evaluation["parameter_pass_share"] == 1.0


def test_research_grade_promotion_blocks_unstable_or_cost_free_results():
    cost_free = _research_gate_frame(transaction_cost_bps=0.0)

    evaluation = evaluate_research_grade_rebound_promotion(
        cost_free,
        parameter_results={"lower_exposure": cost_free},
    )

    assert evaluation["promote"] is False
    assert "transaction cost check below required bps" in evaluation["reasons"]
    assert "parameter stability pass share below threshold" in evaluation["reasons"]


def test_research_grade_promotion_blocks_bad_recent_narrative_alignment():
    comparison = _research_gate_frame(recent_candidate_posture="defensive")

    evaluation = evaluate_research_grade_rebound_promotion(
        comparison,
        parameter_results={
            "lower_exposure": comparison,
            "slower_confirmation": comparison,
            "stricter_risk_gate": comparison,
        },
    )

    assert evaluation["promote"] is False
    assert "recent narrative alignment below threshold" in evaluation["reasons"]


def test_narrative_alignment_scores_rebound_selloff_and_chop():
    assert narrative_alignment_score("risk_on_rebound", "fully_invested") == 1.0
    assert narrative_alignment_score("risk_on_rebound", "defensive") == 0.0
    assert narrative_alignment_score("risk_off_selloff", "defensive") == 1.0
    assert narrative_alignment_score("choppy_or_rangebound", "selective") == 1.0


def test_promotion_ignores_empty_walk_forward_windows():
    comparison = pd.DataFrame(
        [
            {
                "group": "indices",
                "variant": "group_rebound",
                "improved_vs_current": False,
                "drawdown_delta_vs_current": -0.50,
                "strategy_return_delta_vs_current": -0.10,
                "observations": 0,
            },
            {
                "group": "indices",
                "variant": "group_rebound",
                "improved_vs_current": True,
                "drawdown_delta_vs_current": -0.005,
                "strategy_return_delta_vs_current": 0.02,
                "observations": 10,
            },
            {
                "group": "diversifiers",
                "variant": "group_rebound",
                "improved_vs_current": False,
                "drawdown_delta_vs_current": 0.0,
                "strategy_return_delta_vs_current": -0.20,
                "observations": 0,
            },
        ]
    )

    evaluation = evaluate_group_rebound_promotion(comparison)

    assert evaluation["promote"] is True
    assert evaluation["eligible_count"] == 1
    assert evaluation["protected_count"] == 0


def test_group_rebound_summary_ignores_empty_walk_forward_windows():
    def row(window, variant, observations, strategy_return, delta, improved):
        return {
            "window": window,
            "asset": "SPY",
            "group": "indices",
            "variant": variant,
            "observations": observations,
            "strategy_return": strategy_return,
            "buy_hold_return": 0.04,
            "excess_return": strategy_return - 0.04,
            "mean_exposure": 0.25,
            "strategy_return_delta_vs_current": delta,
            "improved_vs_current": improved,
            "overlay_enabled": variant == "group_rebound",
            "market_story": "risk_on_rebound",
            "strategy_posture": "selective",
        }

    comparison = pd.DataFrame(
        [
            row("empty", "current", 0, 0.0, 0.0, False),
            row("empty", "group_rebound", 0, 0.0, -0.10, False),
            row("recent", "current", 5, 0.01, 0.0, False),
            row("recent", "group_rebound", 5, 0.03, 0.02, True),
        ]
    )

    summary = summarize_group_rebound_results(comparison)
    candidate = summary[(summary["window"] == "recent") & (summary["variant"] == "group_rebound")]

    assert set(summary["window"]) == {"recent"}
    assert candidate.iloc[0]["mean_strategy_return"] == pytest.approx(0.03)
