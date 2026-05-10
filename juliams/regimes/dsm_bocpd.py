"""
Diffusion Score Matching BOCPD for heavy-tailed financial returns.

Port of the 1-D Gaussian-unknown-mean variant from
Altamirano, Briol, Knoblauch, "Robust and Scalable Bayesian Online
Changepoint Detection", ICML 2023, arXiv:2302.04759.

Reference implementation: github.com/maltamiranomontero/DSM-bocd
(specifically ``models.py::DSMGaussianUnknownMean`` and ``bocpd.py``).

What problem this solves
------------------------
Standard Adams-MacKay BOCPD with a Gaussian likelihood treats outliers
as evidence for a regime change *or* as anomalous draws from the
current regime, with the relative weight set by the predictive
density. On heavy-tailed financial returns the likelihood-update
mechanism is too rigid: a fat-tail event either over-drives a change
hypothesis (false positive) or gets absorbed into the existing run as
an unlikely sample, inflating the posterior variance and making the
detector less responsive afterwards.

Diffusion Score Matching (DSM) replaces the Bayes update with a
generalised-Bayes update that minimises the Fisher divergence between
the data and the model. The practical consequence is that the update
to the posterior mean and precision is *weighted by the local
robustness function* ``m(x)``: observations far from the predicted
mean get weighted lower, so a single fat-tail event does not ruin the
sufficient statistics for the current run length. The Bayes-versus-DSM
trade-off is governed by ``omega``: ω = 0 is no update at all, larger ω
is closer to vanilla Bayes when ``m(x) = 1`` everywhere, and the
robustness function ``m(x)`` is what makes DSM actually robust.

Scope of this port
------------------
Only the 1-D Gaussian-unknown-mean variant (their
``DSMGaussianUnknownMean``) is implemented. The multivariate / Gamma
/ Wishart variants in the reference code rely on a truncated
multivariate Gaussian sampler we do not need for daily univariate
returns. The ``bocpd`` top-K loop is ported verbatim except for the
addition of NaN handling and pandas-friendly I/O.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import pandas as pd
from scipy.special import logsumexp
from scipy.stats import norm


@dataclass(frozen=True)
class DSMBOCPDResult:
    """Result of a DSM-BOCPD run.

    Attributes
    ----------
    map_run_length
        Argmax over the per-step run-length posterior. Reindexed to
        the input series index, NaN for rows where input was NaN.
    change_probability
        Posterior P(run_length = 0 | data through t).
    run_length_posterior
        ``(T_clean, K + 1)`` matrix of the top-K posteriors per step.
        Truncation parameter is ``K`` (default 50). Index 0 of the
        second axis is the change-point branch.
    """

    map_run_length: pd.Series
    change_probability: pd.Series
    run_length_posterior: np.ndarray


def gaussian_robustness_m(c: float) -> Callable[[float, float], float]:
    """Default robustness function used in the paper's experiments.

    ``m(x; mu) = exp(-((x - mu) / c) ** 2 / 2)``

    Returns 1 when ``x`` equals the predicted mean, decays toward 0
    as ``|x - mu|`` grows beyond ``c``. The bandwidth ``c`` controls
    how aggressively outliers are downweighted; smaller ``c`` is more
    aggressive.

    Returns a closure ``m(x, mu)`` which takes the observation and the
    current posterior-mean prediction.
    """
    if c <= 0:
        raise ValueError(f"Bandwidth c must be positive, got {c}")
    inv_c2 = 1.0 / (c ** 2)

    def m(x: float, mu: float) -> float:
        return float(np.exp(-0.5 * inv_c2 * (x - mu) ** 2))

    return m


def detect_change_points_dsm_bocpd(
    series: pd.Series,
    expected_run_length: float = 100.0,
    varx: float = 1e-4,
    omega: float = 1.0,
    robustness_bandwidth: float = 3.0,
    prior_mean: float = 0.0,
    prior_var: float = 1.0,
    K: int = 50,
) -> DSMBOCPDResult:
    """Run DSM-BOCPD on a 1-D series.

    Parameters
    ----------
    series
        Input observations. NaN rows are dropped before processing
        and the result is reindexed back.
    expected_run_length
        Mean of the geometric run-length prior; same role as in the
        standard BOCPD.
    varx
        Assumed observation noise variance. For daily log returns,
        a reasonable value is the realised variance over a long
        in-sample window (e.g. ``returns.var()``).
    omega
        Robustness weight ω from the paper. ω = 1 with bandwidth
        ``robustness_bandwidth >> 1`` recovers near-standard Bayes.
        Smaller ω + smaller bandwidth = more robust (slower updates).
    robustness_bandwidth
        Bandwidth ``c`` of the Gaussian robustness function ``m``.
        In units of ``sqrt(varx)``: ``c = 3`` means observations more
        than 3 noise-standard-deviations from the prediction are
        heavily downweighted.
    prior_mean, prior_var
        Initial posterior mean and variance for run length 0.
    K
        Top-K truncation of the run-length posterior (Altamirano et
        al. 2023 default 50). Trades accuracy for O(T*K) runtime.

    Returns
    -------
    DSMBOCPDResult
    """
    if expected_run_length <= 0:
        raise ValueError(f"expected_run_length must be positive, got {expected_run_length}")
    if varx <= 0:
        raise ValueError(f"varx must be positive, got {varx}")
    if omega < 0:
        raise ValueError(f"omega must be non-negative, got {omega}")
    if K < 1:
        raise ValueError(f"K must be >= 1, got {K}")

    clean = series.dropna().astype(float)
    n = len(clean)
    if n == 0:
        return DSMBOCPDResult(
            map_run_length=pd.Series(dtype=float),
            change_probability=pd.Series(dtype=float),
            run_length_posterior=np.zeros((0, K + 1)),
        )

    hazard = 1.0 / expected_run_length
    log_h = np.log(hazard)
    log_1mh = np.log(1.0 - hazard)
    inv_varx = 1.0 / varx
    inv_c2 = 1.0 / (robustness_bandwidth ** 2 * varx)

    # Posterior parameters per run length. We follow the reference's
    # design: full growing arrays for mean/prec; max_indices tracks
    # the top-K positions in the GROWING (untruncated) joint, and we
    # evaluate predictives only at those positions to keep cost O(T*K).
    mean = np.array([prior_mean], dtype=float)
    prec = np.array([1.0 / prior_var], dtype=float)

    log_message = np.array([0.0])  # log P(r_t = i) at surviving positions
    max_indices = np.array([0])

    run_post = np.zeros((n, K + 1))
    map_rl = np.empty(n, dtype=float)
    cp_prob = np.empty(n, dtype=float)

    for t in range(n):
        x = clean.iloc[t]

        # 1. Predictive Gaussian density per surviving run length.
        surviving_means = mean[max_indices]
        surviving_var = 1.0 / prec[max_indices] + varx
        log_pred = norm.logpdf(x, loc=surviving_means, scale=np.sqrt(surviving_var))

        # 2. Branch probabilities.
        log_growth = log_pred + log_message + log_1mh
        log_cp = logsumexp(log_pred + log_message + log_h)

        # 3. Build the new joint over r in {0, 1, ..., t+1}. The growth
        # branches go at positions max_indices + 1 (one day older).
        new_size = len(mean) + 1
        new_log_joint = np.full(new_size, -np.inf)
        new_log_joint[0] = log_cp
        new_log_joint[max_indices + 1] = log_growth

        # 4. Track top-K positions for the NEXT step.
        if new_size > K:
            top_idx = np.argpartition(-new_log_joint, K - 1)[:K]
            top_idx = top_idx[np.argsort(-new_log_joint[top_idx])]
        else:
            top_idx = np.argsort(-new_log_joint)

        # 5. Normalise the FULL joint for recording (use the truncated
        # set for the message-passing log probability — this matches
        # the reference, which normalises within the kept set).
        log_norm = logsumexp(new_log_joint[top_idx])
        log_post_full = new_log_joint - log_norm
        log_message = new_log_joint[top_idx] - log_norm

        # 6. Record per-step outputs (top-K).
        post_arr = np.exp(log_post_full[top_idx])
        run_post[t, : len(post_arr)] = post_arr
        map_rl[t] = top_idx[np.argmax(log_post_full[top_idx])]
        cp_prob[t] = float(np.exp(log_post_full[0]))

        # 7. DSM update of sufficient statistics (per existing run length).
        delta = x - mean
        m_sq = np.exp(-inv_c2 * delta ** 2)
        update_strength = 2.0 * omega * m_sq * inv_varx ** 2
        new_prec_growth = prec + update_strength
        new_mean_growth = (prec * mean + update_strength * x) / new_prec_growth

        # 8. Grow arrays by prepending the prior (the change-point branch).
        mean = np.concatenate([[prior_mean], new_mean_growth])
        prec = np.concatenate([[1.0 / prior_var], new_prec_growth])

        # 9. max_indices now points at positions in the new (growing) arrays.
        max_indices = top_idx

    map_series = pd.Series(np.nan, index=series.index, dtype=float)
    cp_series = pd.Series(np.nan, index=series.index, dtype=float)
    map_series.loc[clean.index] = map_rl
    cp_series.loc[clean.index] = cp_prob

    return DSMBOCPDResult(
        map_run_length=map_series,
        change_probability=cp_series,
        run_length_posterior=run_post,
    )
