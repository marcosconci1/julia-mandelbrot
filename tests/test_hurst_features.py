import numpy as np
import pandas as pd

from juliams.features.hurst import (
    classify_confirmed_hurst_regime,
    classify_modified_rs,
    modified_rescaled_range_statistic,
)


def test_modified_rs_statistic_matches_manual_no_lag_case():
    statistic = modified_rescaled_range_statistic(
        pd.Series([1.0, 2.0, 3.0, 4.0, 5.0]),
        lags=0,
        min_periods=2,
    )

    assert np.isclose(statistic, 3 / np.sqrt(10))


def test_modified_rs_classification_uses_short_memory_band():
    statistics = pd.Series([0.5, 1.0, 2.0, np.nan])

    regimes = classify_modified_rs(statistics, confidence_level=0.95)

    assert regimes.tolist() == [
        "Anti-Persistent",
        "Short-Memory",
        "Long-Memory",
        "Unknown",
    ]


def test_confirmed_hurst_regime_requires_modified_rs_support():
    hurst_regimes = pd.Series(["Trending", "Mean-Reverting", "Random", "Unknown"])
    modified_rs_regimes = pd.Series([
        "Short-Memory",
        "Anti-Persistent",
        "Long-Memory",
        "Unknown",
    ])

    confirmed = classify_confirmed_hurst_regime(hurst_regimes, modified_rs_regimes)

    assert confirmed.tolist() == [
        "Short-Memory",
        "Mean-Reverting",
        "Random",
        "Unknown",
    ]
