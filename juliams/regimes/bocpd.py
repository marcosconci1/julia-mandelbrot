"""
Bayesian Online Change-Point Detection (BOCPD).

Online detector for regime changes that returns, at each time step, a
posterior over the current "run length" (the number of observations
since the last change point). Unlike the Markov-switching model in
``markov.py``, BOCPD does not require a preset number of states and
operates in a strictly online (causal) fashion.

References
----------
- Adams & MacKay (2007), "Bayesian Online Changepoint Detection",
  arXiv:0710.3742. The foundational algorithm.
- Tsaknaki, Lillo, Mazzarisi (2024), "Online learning of order flow and
  market impact with Bayesian change-point detection methods",
  *Quantitative Finance*. Recent finance-domain validation; extends to
  score-driven hazard for time-varying parameters within regimes. We
  implement only the Adams-MacKay baseline here — the score-driven
  extension is a follow-up if empirical results warrant it.

Model
-----
Observation model: Gaussian with unknown mean and variance, conjugate
Normal-Inverse-Gamma prior:

    μ, σ² ~ NIG(μ_0, κ_0, α_0, β_0)
    x_t | μ, σ² ~ N(μ, σ²)

Constant hazard rate ``hazard = 1 / λ`` so the prior on the run length
is geometric with mean λ. λ ≈ 100-250 days is a reasonable starting
point for daily equity vol regimes.

What this returns
-----------------
For each time step, ``run_length_posterior`` is a (T, T+1) lower-tri
matrix where row t gives P(r_t = k | x_{1:t}) for k = 0..t. Two
derived series help downstream code:

- ``map_run_length``: argmax of each posterior row.
- ``change_probability``: P(r_t = 0 | x_{1:t}), the probability the
  most recent observation marked a change.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.special import gammaln


@dataclass(frozen=True)
class BOCPDResult:
    map_run_length: pd.Series
    change_probability: pd.Series
    run_length_posterior: np.ndarray  # (T, T+1) lower-triangular


def _student_t_logpdf(
    x: float, mu: np.ndarray, kappa: np.ndarray, alpha: np.ndarray, beta: np.ndarray
) -> np.ndarray:
    """Log pdf of the predictive Student-t under the NIG prior.

    Standard derivation (Murphy 2007 conjugate-prior tutorial section 3):
    posterior predictive of NIG-Gaussian is
        Student-t(2α, μ, β(κ+1)/(α κ)).
    """
    df = 2.0 * alpha
    scale_sq = beta * (kappa + 1.0) / (alpha * kappa)
    z = (x - mu) ** 2 / scale_sq
    log_norm = (
        gammaln((df + 1.0) / 2.0)
        - gammaln(df / 2.0)
        - 0.5 * np.log(df * np.pi * scale_sq)
    )
    return log_norm - ((df + 1.0) / 2.0) * np.log1p(z / df)


def detect_change_points_bocpd(
    series: pd.Series,
    expected_run_length: float = 100.0,
    prior_mu: float = 0.0,
    prior_kappa: float = 1.0,
    prior_alpha: float = 1.0,
    prior_beta: float = 1.0,
) -> BOCPDResult:
    """Run Adams-MacKay BOCPD on a 1-D series with Gaussian observations.

    Parameters
    ----------
    series
        Observations; NaN rows are dropped before processing and the
        result series are reindexed back to the original index.
    expected_run_length
        Mean of the geometric run-length prior. Hazard = 1/λ. Larger
        values bias toward longer regimes (fewer change points). For
        daily equity returns 100-250 is a reasonable starting range.
    prior_mu, prior_kappa, prior_alpha, prior_beta
        Normal-Inverse-Gamma prior hyperparameters. Defaults are weakly
        informative; in practice users should set ``prior_mu`` near the
        sample mean and ``prior_beta / prior_alpha`` near the sample
        variance, but the algorithm is robust to mild misspecification
        for diagnostic use.

    Returns
    -------
    BOCPDResult
    """
    if expected_run_length <= 0:
        raise ValueError(f"expected_run_length must be positive, got {expected_run_length}")
    if prior_kappa <= 0 or prior_alpha <= 0 or prior_beta <= 0:
        raise ValueError("NIG prior parameters must be positive")

    clean = series.dropna().astype(float)
    n = len(clean)
    if n == 0:
        return BOCPDResult(
            map_run_length=pd.Series(dtype=float),
            change_probability=pd.Series(dtype=float),
            run_length_posterior=np.zeros((0, 1)),
        )

    hazard = 1.0 / expected_run_length

    # NIG posterior parameters per run length, growing each step.
    mu = np.array([prior_mu], dtype=float)
    kappa = np.array([prior_kappa], dtype=float)
    alpha = np.array([prior_alpha], dtype=float)
    beta = np.array([prior_beta], dtype=float)

    # Run-length posterior at time t over r = 0..t. Initial: P(r_0 = 0) = 1.
    run_post = np.zeros((n, n + 1))
    run_post[0, 0] = 1.0

    log_post = np.array([0.0])  # log P(r_t = 0 | x_{1:0}) = 0 (single mass)

    for t in range(n):
        x = clean.iloc[t]
        # Predictive log-likelihood under each existing run length.
        log_pred = _student_t_logpdf(x, mu, kappa, alpha, beta)

        # Growth probabilities (run length increases by 1).
        log_growth = log_post + log_pred + np.log(1.0 - hazard)
        # Change-point probability (run length resets to 0).
        log_cp = np.logaddexp.reduce(log_post + log_pred + np.log(hazard))

        # New unnormalised log posterior over r_t+1 in {0, 1, ..., t+1}.
        new_log_post = np.empty(len(log_growth) + 1)
        new_log_post[0] = log_cp
        new_log_post[1:] = log_growth

        # Normalise.
        log_norm = np.logaddexp.reduce(new_log_post)
        new_log_post -= log_norm

        # Record posterior at this step.
        post = np.exp(new_log_post)
        run_post[t, : len(post)] = post

        # Update sufficient statistics.
        # NIG conjugate update for a single Gaussian observation:
        #   κ' = κ + 1
        #   μ' = (κ μ + x) / κ'
        #   α' = α + 1/2
        #   β' = β + κ (x - μ)² / (2 κ')
        new_mu = (kappa * mu + x) / (kappa + 1.0)
        new_beta = beta + (kappa * (x - mu) ** 2) / (2.0 * (kappa + 1.0))
        new_kappa = kappa + 1.0
        new_alpha = alpha + 0.5

        # Prepend the prior (for the new "run reset to 0" hypothesis).
        mu = np.concatenate([[prior_mu], new_mu])
        kappa = np.concatenate([[prior_kappa], new_kappa])
        alpha = np.concatenate([[prior_alpha], new_alpha])
        beta = np.concatenate([[prior_beta], new_beta])
        log_post = new_log_post

    # Build output series, reindexed back to original index.
    map_rl = np.argmax(run_post, axis=1).astype(float)
    change_prob = run_post[:, 0]

    map_series = pd.Series(np.nan, index=series.index, dtype=float)
    cp_series = pd.Series(np.nan, index=series.index, dtype=float)
    map_series.loc[clean.index] = map_rl
    cp_series.loc[clean.index] = change_prob

    return BOCPDResult(
        map_run_length=map_series,
        change_probability=cp_series,
        run_length_posterior=run_post,
    )
