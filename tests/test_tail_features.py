import numpy as np
import pandas as pd
import pytest

from juliams.features.tail import (
    compute_tail_risk_features,
    conditional_value_at_risk,
    hill_tail_index,
    value_at_risk,
)


def test_tail_loss_metrics_are_positive_loss_magnitudes():
    returns = pd.Series([0.01, -0.05, 0.02, -0.10])

    assert value_at_risk(returns, alpha=0.75) > 0
    assert conditional_value_at_risk(returns, alpha=0.75) == pytest.approx(0.10)


def test_hill_tail_index_uses_left_tail_losses():
    returns = pd.Series([0.02, -0.01, -0.02, -0.03, -0.05, -0.08, -0.13])

    estimate = hill_tail_index(returns, tail_fraction=0.5, min_tail_observations=3)

    assert np.isfinite(estimate)
    assert estimate > 0


def test_compute_tail_risk_features_flags_fragile_survival_state():
    log_returns = pd.Series([0.01, 0.01, -0.01, 0.02, -0.08, 0.01, -0.02, 0.01, 0.02, -0.10])
    close = 100.0 * np.exp(log_returns.cumsum())
    df = pd.DataFrame({"Close": close, "log_return": log_returns})

    result = compute_tail_risk_features(
        df,
        {
            "tail_risk_window": 5,
            "tail_var_alpha": 0.80,
            "tail_cvar_alpha": 0.80,
            "hill_tail_fraction": 0.5,
            "hill_min_tail_observations": 2,
            "fragile_cvar": 0.05,
            "stressed_cvar": 0.02,
            "fragile_tail_index": 1.0,
            "stressed_tail_index": 2.0,
            "fragile_drawdown": -0.05,
            "stressed_drawdown": -0.02,
        },
    )

    assert {
        "loss_var",
        "loss_cvar",
        "tail_index",
        "excess_kurtosis",
        "rolling_max_drawdown",
        "survival_regime",
        "survival_score",
    }.issubset(result.columns)
    assert result["survival_regime"].iloc[-1] == "Fragile"
    assert result["survival_score"].iloc[-1] > 0
