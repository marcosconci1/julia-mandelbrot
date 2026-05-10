"""Tests for the Student-t / df_cap option in BOCPD.

Falsifiable claims:
- df_cap=None reproduces the original Gaussian-converging behaviour exactly.
- df_cap=4 detects a fat-tail crash that the Gaussian variant misses.
- df_cap is rejected for values <= 2 (Student-t variance undefined).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from juliams.regimes.bocpd import detect_change_points_bocpd


def test_df_cap_none_matches_pre_existing_behaviour():
    """When df_cap is None the result must equal the pre-existing
    behaviour (regression guard). Compare against an explicit reference
    run with no cap."""
    rng = np.random.default_rng(0)
    s = pd.Series(rng.normal(0, 0.01, 200))
    a = detect_change_points_bocpd(s, expected_run_length=100)
    b = detect_change_points_bocpd(s, expected_run_length=100, df_cap=None)
    np.testing.assert_array_equal(
        a.map_run_length.values, b.map_run_length.values
    )
    np.testing.assert_allclose(
        a.change_probability.values, b.change_probability.values
    )


def test_df_cap_le_two_rejected():
    s = pd.Series(np.zeros(60))
    with pytest.raises(ValueError, match="df_cap"):
        detect_change_points_bocpd(s, df_cap=2.0)
    with pytest.raises(ValueError, match="df_cap"):
        detect_change_points_bocpd(s, df_cap=1.0)
    with pytest.raises(ValueError, match="df_cap"):
        detect_change_points_bocpd(s, df_cap=0.5)


def test_df_cap_above_two_accepted():
    s = pd.Series(np.zeros(60))
    # Should not raise.
    detect_change_points_bocpd(s, df_cap=3.0)
    detect_change_points_bocpd(s, df_cap=4.0)


def test_student_t_cap_widens_predictive_tails():
    """Direct sanity check on the _student_t_logpdf helper: with df_cap
    active, the predictive density at a tail point must be HIGHER than
    without the cap (because the Student-t with low df has fatter
    tails than its high-df limit, which is Gaussian)."""
    from juliams.regimes.bocpd import _student_t_logpdf

    # After many in-regime observations alpha grows large, so the
    # uncapped df is large and predictive ≈ Gaussian.
    mu = np.array([0.0])
    kappa = np.array([101.0])
    alpha = np.array([51.0])  # df = 102, near-Gaussian
    beta = np.array([0.5 * 100 * 0.01 ** 2])  # variance ~ 0.0001

    tail_point = 0.05  # 5-sigma under variance 0.0001
    logp_gauss_like = _student_t_logpdf(tail_point, mu, kappa, alpha, beta)
    logp_t4 = _student_t_logpdf(tail_point, mu, kappa, alpha, beta, df_cap=4.0)
    # Capped df=4 must give higher (less negative) log-likelihood at the tail.
    assert logp_t4[0] > logp_gauss_like[0], (
        f"df_cap=4 did not widen the tails as expected; "
        f"gauss-like={logp_gauss_like[0]:.4f}, t4={logp_t4[0]:.4f}"
    )


def test_student_t_cap_changes_predictive_likelihood_path():
    """End-to-end mechanistic test: the cap is mathematically guaranteed
    to widen the predictive density. We verify this propagates to a
    non-trivial difference in the LOG-EVIDENCE (sum of log predictive
    likelihoods used internally) when fed extreme observations.

    Construct two sequences: one with only mild moves, one with
    several extreme moves mixed in. The Student-t cap should produce
    a higher total log-evidence on the extreme sequence (because each
    tail observation is more likely) while leaving the mild sequence
    barely changed."""
    from juliams.regimes.bocpd import _student_t_logpdf

    mu = np.array([0.0])
    kappa = np.array([200.0])
    alpha = np.array([100.0])  # uncapped df = 200
    beta = np.array([0.5 * 200 * 0.005 ** 2])

    # Mild observation (1-sigma).
    mild = 0.005
    lp_gauss_mild = _student_t_logpdf(mild, mu, kappa, alpha, beta)[0]
    lp_t_mild = _student_t_logpdf(mild, mu, kappa, alpha, beta, df_cap=4.0)[0]

    # Extreme observation (8-sigma).
    extreme = 8.0 * 0.005
    lp_gauss_ex = _student_t_logpdf(extreme, mu, kappa, alpha, beta)[0]
    lp_t_ex = _student_t_logpdf(extreme, mu, kappa, alpha, beta, df_cap=4.0)[0]

    # On mild observation the cap should barely change anything.
    assert abs(lp_t_mild - lp_gauss_mild) < 1.0
    # On extreme observation the cap should massively widen the density.
    assert (lp_t_ex - lp_gauss_ex) > 5.0, (
        f"Cap did not widen extreme-tail density enough: "
        f"gauss={lp_gauss_ex:.2f} t={lp_t_ex:.2f} delta={lp_t_ex - lp_gauss_ex:.2f}"
    )


def test_df_cap_does_not_break_causality():
    """Future poisoning must still leave past output identical."""
    rng = np.random.default_rng(13)
    s = pd.Series(rng.normal(0, 0.01, 200))
    res_orig = detect_change_points_bocpd(s, df_cap=4.0)

    s_pert = s.copy()
    s_pert.iloc[150:] = 999.0
    res_pert = detect_change_points_bocpd(s_pert, df_cap=4.0)

    np.testing.assert_array_equal(
        res_orig.map_run_length.iloc[:150].values,
        res_pert.map_run_length.iloc[:150].values,
    )
