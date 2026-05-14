import numpy as np
import pandas as pd

from juliams.features.volatility import (
    compute_atr,
    compute_realized_variance,
    compute_realized_volatility,
    compute_volatility,
    compute_volatility_features,
    compute_volatility_regime,
)


def test_compute_atr_uses_wilder_smoothing():
    df = pd.DataFrame(
        {
            "High": [11.0, 13.0, 14.0],
            "Low": [9.0, 10.0, 13.0],
            "Close": [10.0, 12.0, 13.0],
        }
    )

    atr = compute_atr(df, window=3)

    assert np.allclose(atr.to_numpy(), [2.0, 2.3333333333, 2.2222222222])


def test_percentile_volatility_regime_keeps_unavailable_values_unknown():
    volatility = pd.Series([0.1] * 10)

    regimes = compute_volatility_regime(volatility, method="percentile", threshold=0.67)

    assert regimes.tolist() == ["Unknown"] * 10


def test_realized_variance_is_rolling_quadratic_variation():
    df = pd.DataFrame({"log_return": [np.nan, 0.01, -0.02, 0.03]})

    realized_variance = compute_realized_variance(df, window=2)
    realized_volatility = compute_realized_volatility(df, window=2)

    assert np.isnan(realized_variance.iloc[0])
    assert np.isnan(realized_variance.iloc[1])
    assert np.isclose(realized_variance.iloc[2], 0.0005)
    assert np.isclose(realized_variance.iloc[3], 0.0013)
    assert np.isclose(realized_volatility.iloc[3], np.sqrt(0.0013))


def test_compute_volatility_can_use_realized_method():
    df = pd.DataFrame({"log_return": [np.nan, 0.01, -0.02, 0.03]})

    volatility = compute_volatility(df, window=2, method="realized")

    assert np.isclose(volatility.iloc[3], np.sqrt(0.0013))


def test_volatility_features_can_classify_with_realized_volatility():
    df = pd.DataFrame(
        {
            "Close": [100.0, 101.0, 99.0, 102.0],
            "log_return": [np.nan, 0.01, -0.02, 0.03],
        }
    )

    features = compute_volatility_features(
        df,
        config={
            "volatility_window": 2,
            "volatility_method": "realized",
            "volatility_baseline_window": 2,
        },
    )

    assert "realized_variance" in features.columns
    assert "realized_volatility" in features.columns
    assert np.isclose(features["volatility"].iloc[3], features["realized_volatility"].iloc[3])


def test_volatility_percentile_lookback_is_configurable():
    close = np.linspace(100.0, 120.0, 40)
    df = pd.DataFrame({"Close": close})

    features = compute_volatility_features(
        df,
        config={
            "volatility_window": 2,
            "volatility_percentile_lookback": 10,
            "volatility_baseline_window": 10,
        },
    )

    assert features["volatility_percentile_lookback"].iloc[-1] == 10
    assert pd.notna(features["volatility_percentile"].iloc[-1])
    assert features["volatility_regime"].iloc[-1] in {"High", "Low"}
