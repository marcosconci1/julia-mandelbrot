import pytest
import pandas as pd

from juliams.analysis import backtest_regime_strategy, regime_exposure


def _frame(close, regimes):
    index = pd.date_range("2024-01-01", periods=len(close), freq="D")
    return pd.DataFrame({"Close": close, "regime_name": regimes}, index=index)


def test_regime_backtest_uses_prior_day_signal_without_lookahead():
    df = _frame(
        [100.0, 110.0, 99.0, 108.9],
        ["Bull Quiet", "Bear Quiet", "Bull Quiet", "Bear Quiet"],
    )

    result = backtest_regime_strategy(df)

    assert result.data["position"].tolist() == [0.0, 1.0, 0.0, 1.0]
    assert result.data["strategy_return"].tolist() == pytest.approx([0.0, 0.10, 0.0, 0.10])
    assert result.metrics["strategy_return"] == pytest.approx(0.21)


def test_regime_backtest_evaluation_window_keeps_pre_window_signal():
    df = _frame(
        [100.0, 110.0, 121.0],
        ["Bull Quiet", "Bull Quiet", "Bear Quiet"],
    )

    result = backtest_regime_strategy(df, evaluation_start="2024-01-02")

    assert result.metrics["start_date"] == pd.Timestamp("2024-01-02")
    assert result.metrics["mean_exposure"] == 1.0
    assert result.metrics["strategy_return"] == pytest.approx(0.21)


def test_regime_backtest_reports_net_returns_after_transaction_costs():
    df = _frame(
        [100.0, 110.0, 99.0, 108.9],
        ["Bull Quiet", "Bear Quiet", "Bull Quiet", "Bear Quiet"],
    )

    result = backtest_regime_strategy(df, transaction_cost_bps=10)

    assert result.data["turnover"].tolist() == pytest.approx([0.0, 1.0, 1.0, 1.0])
    assert result.data["transaction_cost"].tolist() == pytest.approx([0.0, 0.001, 0.001, 0.001])
    assert result.data["strategy_return"].tolist() == pytest.approx([0.0, 0.099, -0.001, 0.099])
    assert result.metrics["gross_strategy_return"] == pytest.approx(0.21)
    assert result.metrics["transaction_cost_bps"] == 10
    assert result.metrics["total_transaction_cost"] == pytest.approx(0.003)
    assert result.metrics["strategy_return"] < result.metrics["gross_strategy_return"]


def test_regime_backtest_narrative_flags_missed_rebound():
    df = _frame(
        [100.0, 102.0, 104.0, 106.0, 108.0, 110.0],
        [
            "Sideways Volatile",
            "Sideways Volatile",
            "Sideways Volatile",
            "Sideways Volatile",
            "Sideways Volatile",
            "Bull Quiet",
        ],
    )

    result = backtest_regime_strategy(df, asset_name="SPY")

    assert result.narrative["market_story"] == "risk_on_rebound"
    assert result.narrative["posture"] == "defensive"
    assert result.narrative["relative_result"] == "lagged"
    assert "captured less upside" in result.narrative["interpretation"]


def test_regime_backtest_narrative_flags_defensive_selloff():
    df = _frame(
        [100.0, 98.0, 96.0, 94.0, 92.0, 90.0],
        [
            "Bear Volatile",
            "Bear Volatile",
            "Bear Volatile",
            "Bear Volatile",
            "Bear Volatile",
            "Bear Volatile",
        ],
    )

    result = backtest_regime_strategy(df, asset_name="SPY")

    assert result.narrative["market_story"] == "risk_off_selloff"
    assert result.narrative["posture"] == "defensive"
    assert result.narrative["relative_result"] == "outperformed"
    assert "capital preservation" in result.narrative["interpretation"]


def test_regime_exposure_clips_custom_weights_to_long_only_bounds():
    regimes = pd.Series(["Bull Quiet", "Sideways Quiet", "Bear Quiet"])

    exposure = regime_exposure(
        regimes,
        {
            "Bull Quiet": 1.5,
            "Sideways Quiet": 0.5,
            "Bear Quiet": -0.5,
        },
    )

    assert exposure.tolist() == [1.0, 0.5, 0.0]
