import pytest
import pandas as pd

import juliams.regimes.fuzzy as fuzzy


def test_manual_trapezoidal_membership_handles_shoulder_boundaries(monkeypatch):
    monkeypatch.setattr(fuzzy, "SKFUZZY_AVAILABLE", False)
    classifier = fuzzy.FuzzyRegimeClassifier(
        trend_range=(-3.0, 3.0),
        volatility_range=(0.0, 0.5),
    )

    assert classifier.get_trend_memberships(-3.0)["Down"] == pytest.approx(1.0)
    assert classifier.get_trend_memberships(3.0)["Up"] == pytest.approx(1.0)
    assert classifier.get_volatility_memberships(0.0)["Low"] == pytest.approx(1.0)
    assert classifier.get_volatility_memberships(0.5)["High"] == pytest.approx(1.0)


def test_fuzzy_can_use_volatility_percentile():
    df = pd.DataFrame(
        {
            "trend_strength": [1.0],
            "volatility": [0.01],
            "volatility_percentile": [0.9],
        }
    )

    raw = fuzzy.compute_fuzzy_features(df, {"fuzzy_volatility_source": "volatility"})
    percentile = fuzzy.compute_fuzzy_features(
        df, {"fuzzy_volatility_source": "volatility_percentile"}
    )

    assert raw["fuzzy_primary_regime"].iloc[0] == "Up-LowVol"
    assert percentile["fuzzy_primary_regime"].iloc[0] == "Up-HighVol"


def test_percentile_fuzzy_mode_is_scale_neutral():
    frames = [
        pd.DataFrame(
            {
                "trend_strength": [1.0],
                "volatility": [raw_vol],
                "volatility_percentile": [0.85],
            }
        )
        for raw_vol in (0.01, 0.20)
    ]

    labels = [
        fuzzy.compute_fuzzy_features(frame, {"fuzzy_volatility_source": "percentile"})[
            "fuzzy_primary_regime"
        ].iloc[0]
        for frame in frames
    ]

    assert labels == ["Up-HighVol", "Up-HighVol"]


def test_percentile_fuzzy_mode_requires_percentile_column():
    df = pd.DataFrame({"trend_strength": [1.0], "volatility": [0.01]})

    with pytest.raises(ValueError, match="volatility_percentile"):
        fuzzy.compute_fuzzy_features(df, {"fuzzy_volatility_source": "percentile"})
