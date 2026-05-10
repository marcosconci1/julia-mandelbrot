"""Regression tests for gold-like regime detection.

Pinned on a frozen synthetic fixture that emulates the salient
features of gold's 2026 trajectory: long calm period, a fat-tail
crash in a single day, then a recovery during which implied vol
stays elevated even as realized vol normalises.

Why this fixture matters
------------------------
The May 2026 web validation of the live XAUUSD run showed:
- BOCPD with default Gaussian/lambda=100 reported run_length=502, missing the
  Feb 2026 crash entirely (acknowledged limitation; see test docstrings).
- Univariate Markov said Low vol while GVZ was elevated, because
  realized vol normalised after the crash; multivariate Markov with
  GVZ as second channel correctly tracks the ongoing high-vol regime.

These tests pin those behaviours so we notice regressions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

hmmlearn = pytest.importorskip("hmmlearn")

from juliams.regimes.markov import (
    fit_markov_variance_regime,
    fit_multivariate_markov_regime,
)


def _gold_like_fixture(seed: int = 42) -> pd.DataFrame:
    """Synthetic series mimicking 2026 gold structure: 250 calm days,
    one fat-tail crash day, 50 chop days where realized vol normalises
    but implied vol stays elevated."""
    rng = np.random.default_rng(seed)
    n_calm = 250
    n_post = 50

    # Calm regime: small returns, low implied vol.
    calm_ret = rng.normal(0.0, 0.005, n_calm)
    calm_iv = rng.normal(0.16, 0.01, n_calm)

    # Single fat-tail crash day.
    crash_ret = np.array([-0.05])  # -5% in one day, ~10 sigma vs calm
    crash_iv = np.array([0.40])  # implied vol spikes

    # Post-crash recovery: realized vol normalises (similar magnitude
    # to calm) but implied vol stays elevated (the gold pattern).
    post_ret = rng.normal(0.0, 0.008, n_post)
    post_iv = rng.normal(0.30, 0.02, n_post)

    return pd.DataFrame(
        {
            "log_return": np.concatenate([calm_ret, crash_ret, post_ret]),
            "implied_vol": np.concatenate([calm_iv, crash_iv, post_iv]),
        }
    )


def test_multivariate_catches_post_crash_high_vol_that_univariate_misses():
    """The headline scenario from the May 2026 gold validation:
    after a fat-tail crash, realized vol normalises but implied vol
    stays elevated. Univariate-on-returns-only will revert to low-vol
    state; multivariate-with-implied-vol must keep flagging high-vol."""
    df = _gold_like_fixture()
    crash_idx = 250
    post_window = slice(crash_idx + 25, crash_idx + 50)  # near the end

    uni = fit_markov_variance_regime(df["log_return"])
    multi = fit_multivariate_markov_regime(df)

    uni_high = uni.smoothed_prob_high.iloc[post_window].mean()
    multi_high = multi.smoothed_prob_high.iloc[post_window].mean()

    # Multivariate should keep elevated P(high) thanks to the iv channel.
    assert multi_high > 0.5, (
        f"Multivariate failed to track elevated implied vol post-crash: "
        f"P(high)={multi_high:.3f}"
    )
    # Multivariate should be materially HIGHER than univariate on this
    # post-crash window — the whole point of the second channel.
    assert multi_high > uni_high + 0.2, (
        f"Multivariate did not exploit implied-vol channel: "
        f"univariate P(high)={uni_high:.3f}, multivariate={multi_high:.3f}"
    )


def test_both_models_flag_crash_day_as_high_vol():
    """Sanity check: both detectors should put HIGH probability of
    high-vol state at the crash day itself. This is the easy case."""
    df = _gold_like_fixture()
    uni = fit_markov_variance_regime(df["log_return"])
    multi = fit_multivariate_markov_regime(df)
    crash_idx = 250
    assert uni.smoothed_prob_high.iloc[crash_idx] > 0.5
    assert multi.smoothed_prob_high.iloc[crash_idx] > 0.5


def test_both_models_flag_pre_crash_calm_period_as_low_vol():
    """Pre-crash days should be mostly low-vol under both models."""
    df = _gold_like_fixture()
    uni = fit_markov_variance_regime(df["log_return"])
    multi = fit_multivariate_markov_regime(df)
    pre_window = slice(50, 240)
    assert uni.smoothed_prob_high.iloc[pre_window].mean() < 0.1
    assert multi.smoothed_prob_high.iloc[pre_window].mean() < 0.1


def test_state_alignment_is_consistent_across_models_on_gold_fixture():
    """On the same data, low-vol state in both models should correspond
    to the calm period. This guards against label-switching bugs."""
    df = _gold_like_fixture()
    uni = fit_markov_variance_regime(df["log_return"])
    multi = fit_multivariate_markov_regime(df)
    # Both should assign calm period to the LOW state.
    assert uni.smoothed_prob_high.iloc[100] < 0.1
    assert multi.smoothed_prob_high.iloc[100] < 0.1
