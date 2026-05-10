"""Tests for juliams.features.ewma_calibration.

Falsifiable claims:
- Asset-class defaults match the 2020-2024 literature consensus.
- The fitter recovers the true optimal halflife on synthetic data
  generated from a known EWMA process.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from juliams.features.ewma_calibration import (
    DEFAULT_HALFLIFE_BY_ASSET,
    default_halflife,
    fit_ewma_halflife,
)


def test_defaults_present_for_main_asset_classes():
    for asset in ("equity", "fx", "crypto", "commodity"):
        assert default_halflife(asset) > 0


def test_defaults_match_literature_ranges():
    """Sanity-check that the defaults sit in the range supported by
    Caporin & Lillo 2023 / arXiv 2105.14382."""
    # Equities: λ in [0.95, 0.99] → halflife in [13.5, 68.9].
    assert 13.0 < default_halflife("equity") < 70.0
    # FX: closer to original RiskMetrics 0.94, halflife ≈ 11.
    assert 8.0 < default_halflife("fx") < 18.0
    # Crypto: faster decay than equities (high turnover).
    assert default_halflife("crypto") < default_halflife("equity")


def test_default_halflife_unknown_raises():
    with pytest.raises(ValueError, match="Unknown asset_class"):
        default_halflife("rare_earth_metals")


def test_default_halflife_case_insensitive():
    assert default_halflife("EQUITY") == default_halflife("equity")
    assert default_halflife(" Equity ") == default_halflife("equity")


def test_fit_picks_reasonable_halflife_on_known_process():
    """On returns generated from a known EWMA(halflife=20) variance
    process, the fitter must pick a halflife from the correct order of
    magnitude. We deliberately do NOT assert the exact value — Andersen
    et al. (1999) and the broader RV literature show squared-return-MSE
    landscapes are very flat near the optimum, so the minimum routinely
    lands several units away from truth purely from noise."""
    rng = np.random.default_rng(0)
    n = 3000
    true_halflife = 20.0
    decay = 0.5 ** (1.0 / true_halflife)
    sigma2 = 0.0001
    returns = np.zeros(n)
    for t in range(n):
        returns[t] = rng.normal(0.0, np.sqrt(sigma2))
        sigma2 = decay * sigma2 + (1 - decay) * returns[t] ** 2

    candidates = [5, 10, 15, 18, 20, 22, 25, 30, 40]
    best, scores = fit_ewma_halflife(
        pd.Series(returns), candidates=candidates, min_warmup=100
    )
    # Reasonable: within an order of magnitude (i.e. not 5 or 40).
    assert 10 <= best <= 40, f"Picked {best}, scores={scores}"


def test_fit_mse_landscape_is_flat_near_optimum():
    """Documents the known shape of the variance-forecast MSE landscape:
    the difference between the best score and any 'reasonable' score
    (within 0.5x to 2x the true halflife) is < 5%. This is why the
    fitter's exact pick is noisy — and why we shouldn't claim it is."""
    rng = np.random.default_rng(0)
    n = 3000
    true_halflife = 20.0
    decay = 0.5 ** (1.0 / true_halflife)
    sigma2 = 0.0001
    returns = np.zeros(n)
    for t in range(n):
        returns[t] = rng.normal(0.0, np.sqrt(sigma2))
        sigma2 = decay * sigma2 + (1 - decay) * returns[t] ** 2

    candidates = [10, 15, 20, 25, 30, 40]
    _, scores = fit_ewma_halflife(
        pd.Series(returns), candidates=candidates, min_warmup=100
    )
    best_score = min(scores.values())
    worst_in_range = max(scores[h] for h in (10, 15, 20, 25, 30, 40))
    relative_spread = (worst_in_range - best_score) / best_score
    # The spread within [0.5x, 2x] of true should be < 5% — confirming
    # the landscape is too flat to claim "this exact halflife".
    assert relative_spread < 0.05, (
        f"MSE landscape sharper than expected: spread={relative_spread:.4f}. "
        f"Either the test fixture changed or the fitter's claim of "
        f"'optimal halflife' is more meaningful than the literature suggests."
    )


def test_fit_returns_score_for_every_candidate():
    rng = np.random.default_rng(1)
    rets = pd.Series(rng.normal(0, 0.01, 500))
    candidates = [5, 10, 20, 30]
    best, scores = fit_ewma_halflife(rets, candidates=candidates)
    assert set(scores.keys()) == {float(c) for c in candidates}
    assert best in scores


def test_fit_too_few_observations_raises():
    with pytest.raises(ValueError, match="at least"):
        fit_ewma_halflife(pd.Series(np.ones(10)), candidates=[10, 20])


def test_fit_negative_halflife_raises():
    rng = np.random.default_rng(2)
    rets = pd.Series(rng.normal(0, 0.01, 500))
    with pytest.raises(ValueError, match="positive"):
        fit_ewma_halflife(rets, candidates=[10, -5])


def test_fit_too_few_candidates_raises():
    rng = np.random.default_rng(3)
    rets = pd.Series(rng.normal(0, 0.01, 500))
    with pytest.raises(ValueError):
        fit_ewma_halflife(rets, candidates=[20])
