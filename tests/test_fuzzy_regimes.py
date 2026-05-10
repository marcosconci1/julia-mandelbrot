import pytest

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
