import pandas as pd

import main

from main import (
    DEFAULT_DIAGNOSTIC_SYMBOLS,
    _format_percentage_points,
    parse_arguments,
    print_walk_forward_diagnostics_report,
    run_walk_forward_diagnostics_report,
)


def test_percentage_points_are_not_scaled_twice():
    assert _format_percentage_points(52.5).strip() == "52.50%"


def test_default_diagnostic_symbols_use_yahoo_crypto_tickers():
    assert "BTC-USD" in DEFAULT_DIAGNOSTIC_SYMBOLS
    assert "ETH-USD" in DEFAULT_DIAGNOSTIC_SYMBOLS
    assert "BTCUSDT" not in DEFAULT_DIAGNOSTIC_SYMBOLS


def test_walk_forward_report_prints_promotion_gate(capsys):
    comparison = pd.DataFrame(
        {
            "window": ["recent_1m"],
            "group": ["indices"],
            "variant": ["group_rebound"],
        }
    )
    summary = pd.DataFrame(
        {
            "window": ["recent_1m"],
            "group": ["indices"],
            "variant": ["group_rebound"],
            "asset_count": [1],
            "mean_strategy_return": [0.02],
            "mean_buy_hold_return": [0.03],
            "mean_excess_return": [-0.01],
            "median_exposure": [0.75],
            "mean_strategy_delta_vs_current": [0.01],
            "improved_count": [1],
            "overlay_enabled_count": [1],
        }
    )
    promotion = {
        "promote": False,
        "eligible_improved_share": 0.5,
        "protected_degraded_share": 0.0,
        "worst_drawdown_delta": -0.01,
        "reasons": ["eligible improvement share below threshold"],
    }

    print_walk_forward_diagnostics_report(["SPY"], comparison, promotion, summary)

    output = capsys.readouterr().out

    assert "Promotion Gate" in output
    assert "eligible improvement share below threshold" in output
    assert "recent_1m / indices" in output


def test_walk_forward_calibration_arguments_parse(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "main.py",
            "--walk-forward-diagnostics",
            "--overlay-enabled-groups",
            "indices",
            "mega_cap_tech",
            "--rebound-exposure",
            "0.5",
            "--rebound-fast-window",
            "7",
            "--rebound-lookback",
            "30",
            "--min-rebound",
            "0.05",
            "--min-fast-return",
            "0.03",
            "--no-rebound-fast-ma",
            "--no-drawdown-gate",
            "--drawdown-gate-lookback",
            "40",
            "--min-prior-drawdown",
            "-0.08",
            "--max-overlay-holding-period",
            "3",
            "--overlay-stop-loss",
            "-0.04",
            "--transaction-cost-bps",
            "7",
            "--no-parameter-stability-check",
        ],
    )

    args = parse_arguments()

    assert args.walk_forward_diagnostics is True
    assert args.overlay_enabled_groups == ["indices", "mega_cap_tech"]
    assert args.rebound_exposure == 0.5
    assert args.rebound_fast_window == 7
    assert args.rebound_lookback == 30
    assert args.min_rebound == 0.05
    assert args.min_fast_return == 0.03
    assert args.no_rebound_fast_ma is True
    assert args.no_drawdown_gate is True
    assert args.drawdown_gate_lookback == 40
    assert args.min_prior_drawdown == -0.08
    assert args.max_overlay_holding_period == 3
    assert args.overlay_stop_loss == -0.04
    assert args.transaction_cost_bps == 7
    assert args.no_parameter_stability_check is True


def test_walk_forward_report_uses_enabled_groups_as_promotion_eligible(monkeypatch):
    fake_df = pd.DataFrame(
        {
            "Close": [100.0, 101.0],
            "regime_name": ["Bear Quiet", "Bear Quiet"],
        },
        index=pd.date_range("2024-01-01", periods=2),
    )

    def fake_run_analysis(**kwargs):
        return {"df": fake_df}

    def fake_run_walk_forward(*args, **kwargs):
        return pd.DataFrame(
            {
                "group": ["indices"],
                "variant": ["group_rebound"],
                "improved_vs_current": [True],
                "drawdown_delta_vs_current": [0.0],
                "strategy_return_delta_vs_current": [0.01],
                "strategy_return": [0.01],
                "buy_hold_return": [0.01],
                "excess_return": [0.0],
                "mean_exposure": [0.5],
                "market_story": ["risk_on_rebound"],
                "strategy_posture": ["selective"],
                "transaction_cost_bps": [5.0],
                "observations": [20],
                "window": ["recent_1m"],
            }
        )

    captured = {}

    def fake_evaluate(comparison, eligible_groups, **kwargs):
        captured["eligible_groups"] = eligible_groups
        captured["parameter_results"] = kwargs["parameter_results"]
        return {
            "promote": True,
            "eligible_improved_share": 1.0,
            "protected_degraded_share": 0.0,
            "worst_drawdown_delta": 0.0,
            "reasons": [],
        }

    def fake_summary(comparison):
        return pd.DataFrame()

    monkeypatch.setattr(main, "run_analysis", fake_run_analysis)
    monkeypatch.setattr(main, "run_walk_forward_diagnostics", fake_run_walk_forward)
    monkeypatch.setattr(main, "evaluate_research_grade_rebound_promotion", fake_evaluate)
    monkeypatch.setattr(main, "summarize_group_rebound_results", fake_summary)

    run_walk_forward_diagnostics_report(["SPY"], overlay_enabled_groups=["indices"])

    assert captured["eligible_groups"] == ("indices",)
    assert set(captured["parameter_results"]) == {
        "lower_exposure",
        "slower_confirmation",
        "stricter_risk_gate",
    }
