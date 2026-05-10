import pandas as pd
import pytest

from juliams.analysis import (
    backtest_rebound_overlay,
    compare_exposure_maps,
    compare_rebound_overlay,
    compute_rebound_signal,
    compute_regime_forward_diagnostics,
    measure_bullish_turn_lag,
    summarize_group_narratives,
    summarize_variant_comparison_by_group,
)


def _frame(close, regimes):
    index = pd.date_range("2024-01-01", periods=len(close), freq="D")
    return pd.DataFrame({"Close": close, "regime_name": regimes}, index=index)


def test_forward_diagnostics_show_regime_return_separation():
    df = _frame(
        [100.0, 110.0, 121.0, 100.0, 90.0, 81.0],
        ["Bull Quiet", "Bull Quiet", "Bear Quiet", "Bear Quiet", "Sideways Quiet", "Bull Quiet"],
    )

    diagnostics = compute_regime_forward_diagnostics(df, horizons=(1,))
    bull = diagnostics[diagnostics["regime"] == "Bull Quiet"].iloc[0]
    bear = diagnostics[diagnostics["regime"] == "Bear Quiet"].iloc[0]

    assert bull["horizon"] == 1
    assert bull["count"] == 2
    assert bull["mean_return"] > 0
    assert bear["mean_return"] < 0
    assert bull["mean_return"] > bear["mean_return"]


def test_bullish_turn_lag_measures_delay_after_local_low():
    df = _frame(
        [100.0, 90.0, 80.0, 88.0, 96.0],
        ["Bear Quiet", "Bear Quiet", "Sideways Quiet", "Sideways Quiet", "Bull Quiet"],
    )

    lag = measure_bullish_turn_lag(df)

    assert lag["trough_date"] == pd.Timestamp("2024-01-03")
    assert lag["first_bull_date"] == pd.Timestamp("2024-01-05")
    assert lag["lag_periods"] == 2
    assert lag["missed_return"] == pytest.approx(0.20)


def test_bullish_turn_lag_reports_when_no_bull_signal_arrives():
    df = _frame(
        [100.0, 95.0, 90.0],
        ["Bear Quiet", "Sideways Quiet", "Sideways Quiet"],
    )

    lag = measure_bullish_turn_lag(df)

    assert lag["first_bull_date"] is None
    assert lag["lag_periods"] is None
    assert pd.isna(lag["missed_return"])


def test_compare_exposure_maps_uses_same_regimes_for_strategy_variants():
    df = _frame(
        [100.0, 110.0, 121.0],
        ["Bull Quiet", "Bull Quiet", "Bull Quiet"],
    )

    comparison = compare_exposure_maps(
        df,
        exposure_maps={
            "full": {"Bull Quiet": 1.0},
            "flat": {"Bull Quiet": 0.0},
        },
    )

    assert comparison["map"].tolist() == ["full", "flat"]
    assert comparison.loc[comparison["map"] == "full", "strategy_return"].iloc[0] == pytest.approx(0.21)
    assert comparison.loc[comparison["map"] == "flat", "strategy_return"].iloc[0] == 0.0


def test_rebound_signal_uses_only_current_and_past_prices():
    base = _frame(
        [100.0, 90.0, 95.0, 99.0, 100.0, 101.0],
        ["Bear Quiet"] * 6,
    )
    changed_future = base.copy()
    changed_future.loc[pd.Timestamp("2024-01-06"), "Close"] = 150.0

    base_signal = compute_rebound_signal(
        base,
        fast_window=2,
        drawdown_lookback=4,
        min_rebound=0.05,
        min_fast_return=0.04,
        require_above_fast_ma=False,
    )
    changed_signal = compute_rebound_signal(
        changed_future,
        fast_window=2,
        drawdown_lookback=4,
        min_rebound=0.05,
        min_fast_return=0.04,
        require_above_fast_ma=False,
    )

    assert base_signal.iloc[:5].tolist() == changed_signal.iloc[:5].tolist()


def test_rebound_overlay_uses_prior_day_signal_without_lookahead():
    df = _frame(
        [100.0, 90.0, 99.0, 108.9],
        ["Bear Quiet", "Bear Quiet", "Bear Quiet", "Bear Quiet"],
    )

    result = backtest_rebound_overlay(
        df,
        fast_window=1,
        drawdown_lookback=3,
        min_rebound=0.05,
        min_fast_return=0.05,
        require_above_fast_ma=False,
        rebound_exposure=0.75,
    )

    assert result.data["position"].tolist() == [0.0, 0.0, 0.0, 0.75]
    assert result.data["strategy_return"].tolist() == pytest.approx([0.0, 0.0, 0.0, 0.075])
    assert result.metrics["rebound_signal_count"] == 2


def test_compare_rebound_overlay_reports_current_and_overlay_variants():
    df = _frame(
        [100.0, 90.0, 99.0, 108.9],
        ["Bear Quiet", "Bear Quiet", "Bear Quiet", "Bear Quiet"],
    )

    comparison = compare_rebound_overlay(
        df,
        fast_window=1,
        drawdown_lookback=3,
        min_rebound=0.05,
        min_fast_return=0.05,
        require_above_fast_ma=False,
        rebound_exposure=0.75,
    )

    assert comparison["variant"].tolist() == ["current", "rebound_overlay"]
    assert comparison.loc[comparison["variant"] == "current", "strategy_return"].iloc[0] == 0.0
    assert comparison.loc[comparison["variant"] == "rebound_overlay", "strategy_return"].iloc[0] == pytest.approx(0.075)


def test_group_variant_summary_identifies_where_rebound_overlay_generalizes():
    rebound_asset_1 = _frame(
        [100.0, 90.0, 99.0, 108.9],
        ["Bear Quiet", "Bear Quiet", "Bear Quiet", "Bear Quiet"],
    )
    rebound_asset_2 = _frame(
        [100.0, 92.0, 101.2, 111.32],
        ["Bear Quiet", "Bear Quiet", "Bear Quiet", "Bear Quiet"],
    )
    whipsaw_asset = _frame(
        [100.0, 90.0, 99.0, 89.1],
        ["Bear Quiet", "Bear Quiet", "Bear Quiet", "Bear Quiet"],
    )

    comparisons = []
    for asset, frame in {
        "SPY": rebound_asset_1,
        "QQQ": rebound_asset_2,
        "GLD": whipsaw_asset,
    }.items():
        comparisons.append(
            compare_rebound_overlay(
                frame,
                asset_name=asset,
                fast_window=1,
                drawdown_lookback=3,
                min_rebound=0.05,
                min_fast_return=0.05,
                require_above_fast_ma=False,
                rebound_exposure=0.75,
            )
        )

    summary = summarize_variant_comparison_by_group(
        pd.concat(comparisons, ignore_index=True),
        {
            "indices": ["SPY", "QQQ"],
            "diversifiers": ["GLD"],
        },
    )

    indices = summary[summary["group"] == "indices"].iloc[0]
    diversifiers = summary[summary["group"] == "diversifiers"].iloc[0]

    assert indices["asset_count"] == 2
    assert indices["improved_count"] == 2
    assert indices["degraded_count"] == 0
    assert indices["mean_improvement"] > 0

    assert diversifiers["asset_count"] == 1
    assert diversifiers["improved_count"] == 0
    assert diversifiers["degraded_count"] == 1
    assert diversifiers["mean_improvement"] < 0


def test_group_narrative_summary_aggregates_asset_clusters():
    metrics_by_asset = {
        "SPY": {
            "strategy_return": 0.01,
            "buy_hold_return": 0.10,
            "excess_return": -0.09,
            "mean_exposure": 0.2,
            "market_story": "risk_on_rebound",
            "strategy_posture": "defensive",
            "relative_result": "lagged",
        },
        "QQQ": {
            "strategy_return": 0.02,
            "buy_hold_return": 0.12,
            "excess_return": -0.10,
            "mean_exposure": 0.3,
            "market_story": "risk_on_rebound",
            "strategy_posture": "defensive",
            "relative_result": "lagged",
        },
        "TLT": {
            "strategy_return": 0.00,
            "buy_hold_return": 0.01,
            "excess_return": -0.01,
            "mean_exposure": 0.0,
            "market_story": "choppy_or_rangebound",
            "strategy_posture": "defensive",
            "relative_result": "matched",
        },
    }

    summary = summarize_group_narratives(
        metrics_by_asset,
        {
            "indices": ["SPY", "QQQ"],
            "diversifiers": ["TLT", "GLD"],
        },
    )

    indices = summary[summary["group"] == "indices"].iloc[0]
    diversifiers = summary[summary["group"] == "diversifiers"].iloc[0]

    assert indices["asset_count"] == 2
    assert indices["top_story"] == "risk_on_rebound"
    assert indices["top_relative_result"] == "lagged"
    assert indices["mean_buy_hold_return"] == pytest.approx(0.11)
    assert diversifiers["asset_count"] == 1
    assert diversifiers["top_story"] == "choppy_or_rangebound"
