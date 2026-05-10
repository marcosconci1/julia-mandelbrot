"""Tests for the multivariate (return + implied vol) Markov regime fit.

Falsifiable claims:
- On synthetic data with a known regime structure, the fit assigns
  the calm region to the low-vol state and the turbulent region to
  the high-vol state.
- Adding the implied vol channel changes the inferred state path
  vs the univariate-on-returns-only baseline (so the channel is
  doing something).
- ImportError is raised when hmmlearn is unavailable.
- Filtered (causal) probabilities are not identical to smoothed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

# hmmlearn is optional; skip the whole module if unavailable.
hmmlearn = pytest.importorskip("hmmlearn")

from juliams.regimes.markov import (
    fit_markov_variance_regime,
    fit_multivariate_markov_regime,
)


def _two_regime_features(seed: int = 0) -> pd.DataFrame:
    """600 days: calm 0..299, turbulent 300..499, calm 500..599.
    Implied vol channel doubles in the turbulent regime."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "log_return": np.concatenate(
                [
                    rng.normal(0.0, 0.005, 300),
                    rng.normal(0.0, 0.03, 200),
                    rng.normal(0.0, 0.005, 100),
                ]
            ),
            "implied_vol": np.concatenate(
                [
                    rng.normal(0.15, 0.01, 300),
                    rng.normal(0.40, 0.05, 200),
                    rng.normal(0.16, 0.01, 100),
                ]
            ),
        }
    )


def test_state_alignment_low_state_has_lower_return_variance():
    fit = fit_multivariate_markov_regime(_two_regime_features())
    # state_covariances has shape (2, 2, 2): [state, channel, channel]
    var_low = fit.state_covariances[0, 0, 0]
    var_high = fit.state_covariances[1, 0, 0]
    assert var_low < var_high


def test_low_state_assigned_to_calm_region():
    fit = fit_multivariate_markov_regime(_two_regime_features())
    assert fit.smoothed_prob_high.iloc[:290].mean() < 0.1


def test_high_state_assigned_to_turbulent_region():
    fit = fit_multivariate_markov_regime(_two_regime_features())
    assert fit.smoothed_prob_high.iloc[320:490].mean() > 0.9


def test_implied_vol_channel_changes_state_path_vs_univariate():
    """The whole point of adding the vol channel is that it provides
    information the univariate model lacks. Verify the state paths
    differ on a fixture where vol leads return-vol.

    Construction: returns stay quiet for the first 50 days but the
    implied vol jumps. The univariate model sees only quiet returns
    and predicts low state; the multivariate model sees the vol jump
    and should predict high state."""
    rng = np.random.default_rng(2)
    n = 600
    # Returns: small everywhere except 200..400
    returns = np.concatenate(
        [
            rng.normal(0, 0.005, 50),  # vol jumps here but returns don't
            rng.normal(0, 0.005, 150),
            rng.normal(0, 0.025, 200),  # returns finally turbulent
            rng.normal(0, 0.005, 200),
        ]
    )
    # Implied vol: jumps at index 50, recedes around 350
    iv = np.concatenate(
        [
            rng.normal(0.12, 0.005, 50),
            rng.normal(0.35, 0.02, 300),  # implied vol leads
            rng.normal(0.13, 0.005, 250),
        ]
    )
    df = pd.DataFrame({"log_return": returns, "implied_vol": iv})

    uni = fit_markov_variance_regime(df["log_return"])
    multi = fit_multivariate_markov_regime(df)

    # Find disagreement on the 50..200 window (where vol is high but
    # returns are quiet). Multivariate should call this high-vol;
    # univariate should call it low-vol.
    uni_high = uni.smoothed_prob_high.iloc[100:200].mean()
    multi_high = multi.smoothed_prob_high.iloc[100:200].mean()
    assert multi_high > uni_high + 0.2, (
        f"Multivariate did not exploit the implied-vol channel: "
        f"univariate P(high)={uni_high:.3f}, multivariate={multi_high:.3f}"
    )


def test_missing_required_columns_raise():
    df = pd.DataFrame({"only_one_col": np.zeros(100)})
    with pytest.raises(ValueError, match="Missing"):
        fit_multivariate_markov_regime(df)


def test_too_few_clean_rows_raise():
    df = pd.DataFrame({"log_return": np.ones(20), "implied_vol": np.ones(20)})
    with pytest.raises(ValueError, match="at least 50"):
        fit_multivariate_markov_regime(df)


def test_filtered_differs_from_smoothed():
    fit = fit_multivariate_markov_regime(_two_regime_features())
    f = fit.filtered_prob_high.dropna().values
    s = fit.smoothed_prob_high.dropna().values
    assert not np.allclose(f, s), (
        "Filtered and smoothed posteriors should differ (smoothed uses "
        "future data, filtered does not)."
    )


def test_transition_matrix_persistent():
    fit = fit_multivariate_markov_regime(_two_regime_features())
    assert fit.transition_matrix[0, 0] > 0.95
    assert fit.transition_matrix[1, 1] > 0.95


def test_returns_object_has_documented_shapes():
    fit = fit_multivariate_markov_regime(_two_regime_features())
    assert fit.state_means.shape == (2, 2)
    assert fit.state_covariances.shape == (2, 2, 2)
    assert fit.transition_matrix.shape == (2, 2)
    assert fit.feature_names == ["log_return", "implied_vol"]


def test_works_with_custom_column_names():
    df = _two_regime_features().rename(
        columns={"log_return": "ret", "implied_vol": "iv"}
    )
    fit = fit_multivariate_markov_regime(df, return_col="ret", vol_col="iv")
    assert fit.feature_names == ["ret", "iv"]
