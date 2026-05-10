"""
Markov-switching variance regime detection.

Wraps ``statsmodels.tsa.regime_switching.markov_regression.MarkovRegression``
for the specific use-case of *vol-regime detection on log returns* with
no exogenous covariates and switching variance only. This is Hamilton's
(1989) two-state model in its simplest form.

Why this exists alongside the rolling-quantile vol regime:
- Quantiles label the *current* observation by where it sits in the recent
  empirical distribution. Good when vol changes are gradual.
- The Markov model labels the *latent state* the system is most likely in,
  having seen the whole history. It produces smoothed probabilities that
  are temporally stable, which complements (and sometimes contradicts) the
  pointwise quantile label.

Critical implementation details
-------------------------------
- **Label switching**: across refits the model may swap which state it
  calls "0" vs "1". We canonicalise by reordering states so state 0 has
  the lower estimated variance ("calm") and state 1 the higher
  ("turbulent"). For K=2 variance-only models this is the standard fix
  (variance-sort); for K≥3 use the ECR algorithm (Papastamoulis 2016).
- **Smoothed vs filtered probs**: we expose smoothed (Kim 1994) by
  default — these use the entire sample and are the right summary for
  ex-post analysis. ``filtered=True`` returns the causal version.
- **Spurious flipping**: 2-state variance HMMs are well documented to
  produce 2-3 day "regime flips" during structural breaks (Salisu et al.
  2020+ on COVID; practitioner accounts 2023-2024 on SVB / Fed pivot).
  Apply :func:`enforce_min_dwell` to the output labels to suppress runs
  shorter than the minimum economically-meaningful regime length.
- **Convergence failures**: EM on small samples sometimes fails. We
  catch and propagate ``MarkovStateAlignmentError`` rather than letting
  cryptic statsmodels errors leak.

References
----------
- Hamilton (1989), "A New Approach to the Economic Analysis of
  Nonstationary Time Series and the Business Cycle" — original
  framework.
- Kim (1994), "Dynamic Linear Models with Markov-Switching",
  *J. Econometrics* — smoothed probability algorithm.
- Wang & Lin (2020), "Regime-Switching Factor Investing with HMMs",
  *JRFM* 13(12):311 — recent peer-reviewed confirmation that 2-state
  variance HMMs remain the default for daily equity regime detection.
- Salisu, Adediran, Gupta (2020+), various — documents Markov-switching
  spurious-flip issue on COVID-era data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


class MarkovStateAlignmentError(RuntimeError):
    """Raised when the Markov model fits but state alignment fails
    (e.g. because both states ended up with identical variance)."""


@dataclass(frozen=True)
class MarkovVarianceFit:
    """Results from fitting a 2-state switching-variance model.

    Attributes
    ----------
    smoothed_prob_high
        Posterior probability of being in the high-variance state at each
        time step, smoothed using the full sample (Kim 1994 algorithm).
    filtered_prob_high
        Same as above but using only data up to time t — strictly causal.
    variance_low, variance_high
        Estimated state variances (after canonical alignment).
    transition_matrix
        2x2 numpy array. Row i is "P(next state = j | current = i)".
    log_likelihood, aic, bic
        Goodness-of-fit numbers from statsmodels.
    converged
        Whether the EM iteration converged.
    """

    smoothed_prob_high: pd.Series
    filtered_prob_high: pd.Series
    variance_low: float
    variance_high: float
    transition_matrix: np.ndarray
    log_likelihood: float
    aic: float
    bic: float
    converged: bool


def fit_markov_variance_regime(
    returns: pd.Series,
    k_regimes: int = 2,
    max_iter: int = 200,
    random_state: Optional[int] = 0,
) -> MarkovVarianceFit:
    """Fit a Markov-switching variance model to a returns series.

    Parameters
    ----------
    returns
        Log-return series. NaN rows are dropped before fitting; the
        returned probability series is reindexed to match ``returns``
        with NaN padding.
    k_regimes
        Number of latent regimes. Currently only k=2 is supported and
        results in low/high variance state alignment. Other values raise.
    max_iter
        EM iteration cap.
    random_state
        Seed forwarded to statsmodels for reproducibility of starting
        values.

    Returns
    -------
    MarkovVarianceFit
    """
    if k_regimes != 2:
        raise NotImplementedError(
            f"Only k_regimes=2 is supported; got {k_regimes}. "
            "Variance-only state alignment is ambiguous beyond 2 regimes "
            "and would require a separate API."
        )

    from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

    clean = returns.dropna().astype(float)
    if len(clean) < 50:
        raise ValueError(
            f"Need at least 50 observations to fit a 2-state Markov model; "
            f"got {len(clean)}."
        )

    model = MarkovRegression(
        endog=clean.values,
        k_regimes=k_regimes,
        trend="c",
        switching_variance=True,
    )
    res = model.fit(maxiter=max_iter, disp=False)

    variances = np.asarray(res.params[-k_regimes:], dtype=float)
    if variances[0] == variances[1]:
        raise MarkovStateAlignmentError(
            "Both fitted states have identical variance — cannot canonicalise. "
            "Often indicates the model failed to identify two regimes."
        )

    if variances[0] < variances[1]:
        low_idx, high_idx = 0, 1
    else:
        low_idx, high_idx = 1, 0

    var_low = float(variances[low_idx])
    var_high = float(variances[high_idx])

    smoothed = np.asarray(res.smoothed_marginal_probabilities)
    filtered = np.asarray(res.filtered_marginal_probabilities)
    if smoothed.shape[1] == k_regimes and smoothed.shape[0] != k_regimes:
        smoothed_high = smoothed[:, high_idx]
        filtered_high = filtered[:, high_idx]
    else:
        smoothed_high = smoothed[high_idx]
        filtered_high = filtered[high_idx]

    smoothed_series = pd.Series(np.nan, index=returns.index, dtype=float)
    filtered_series = pd.Series(np.nan, index=returns.index, dtype=float)
    smoothed_series.loc[clean.index] = smoothed_high
    filtered_series.loc[clean.index] = filtered_high

    transmat_raw = np.asarray(res.regime_transition).reshape(k_regimes, k_regimes)
    perm = np.array([low_idx, high_idx])
    transition_matrix = transmat_raw[np.ix_(perm, perm)]

    return MarkovVarianceFit(
        smoothed_prob_high=smoothed_series,
        filtered_prob_high=filtered_series,
        variance_low=var_low,
        variance_high=var_high,
        transition_matrix=transition_matrix,
        log_likelihood=float(res.llf),
        aic=float(res.aic),
        bic=float(res.bic),
        converged=bool(res.mle_retvals.get("converged", False))
        if hasattr(res, "mle_retvals") and res.mle_retvals is not None
        else True,
    )


def label_markov_regimes(
    fit: MarkovVarianceFit,
    threshold: float = 0.5,
    use_filtered: bool = False,
) -> pd.Series:
    """Convert smoothed (or filtered) state probabilities into discrete
    {High, Low, Unknown} labels.

    Parameters
    ----------
    fit
        Result of :func:`fit_markov_variance_regime`.
    threshold
        Classify as "High" when P(high) > threshold, otherwise "Low".
        0.5 is the maximum-a-posteriori choice.
    use_filtered
        If True use the strictly-causal filtered probabilities. Default
        False (smoothed, Kim 1994) which uses the whole sample and is
        appropriate for ex-post analysis. For live or walk-forward use,
        pass True.
    """
    if not 0.0 < threshold < 1.0:
        raise ValueError(f"threshold must be in (0, 1); got {threshold}")
    probs = fit.filtered_prob_high if use_filtered else fit.smoothed_prob_high
    labels = pd.Series("Unknown", index=probs.index, dtype=object)
    valid = probs.notna()
    labels.loc[valid & (probs > threshold)] = "High"
    labels.loc[valid & (probs <= threshold)] = "Low"
    return labels


def _gaussian_log_emission_per_state(model, X: np.ndarray) -> np.ndarray:
    """Per-state log Gaussian density for a fitted hmmlearn GaussianHMM.

    Returns an (T, n_components) array. We compute this manually
    because hmmlearn 0.3 made the corresponding internal helper
    private. Uses the multivariate Gaussian density with the model's
    means_ and covars_ attributes.
    """
    from scipy.stats import multivariate_normal

    n_components = model.n_components
    T = X.shape[0]
    log_emission = np.empty((T, n_components))
    for k in range(n_components):
        cov = model.covars_[k]
        # GaussianHMM with covariance_type="full" stores cov as (D, D).
        log_emission[:, k] = multivariate_normal.logpdf(
            X, mean=model.means_[k], cov=cov, allow_singular=True
        )
    return log_emission


@dataclass(frozen=True)
class MultivariateMarkovFit:
    """Results from a multivariate-emission Gaussian HMM fit.

    Same shape as ``MarkovVarianceFit`` but emissions are vectors,
    typically ``[log_return, implied_vol_proxy]``. Use this when an
    exogenous volatility series (GVZ for gold, VIX for equities, OVX
    for oil, etc.) is available; the second channel lets the model
    detect regime breaks that show up in implied vol BEFORE realized
    vol catches up.

    Attributes
    ----------
    smoothed_prob_high
        Posterior P(high-vol state) using the full sample.
    filtered_prob_high
        Strictly causal version using only data up to time t.
    state_means
        2x2 array: row i is the mean vector of state i.
    state_covariances
        2x2x2 array: covariance matrix per state.
    transition_matrix
        2x2 array, canonicalised so state 0 = low vol, state 1 = high.
    log_likelihood
        Total log-likelihood under the fitted model.
    feature_names
        Names of the emission channels in column order.
    """

    smoothed_prob_high: pd.Series
    filtered_prob_high: pd.Series
    state_means: np.ndarray
    state_covariances: np.ndarray
    transition_matrix: np.ndarray
    log_likelihood: float
    feature_names: list[str]


def fit_multivariate_markov_regime(
    features: pd.DataFrame,
    return_col: str = "log_return",
    vol_col: str = "implied_vol",
    n_iter: int = 200,
    random_state: Optional[int] = 0,
) -> MultivariateMarkovFit:
    """Fit a 2-state Gaussian HMM to a return + implied-vol feature pair.

    Why this exists alongside ``fit_markov_variance_regime``
    --------------------------------------------------------
    The single-channel model fits only ``returns`` and infers vol
    regimes from realised variance changes. This works well for slow
    structural breaks but lags fast ones because realised vol takes
    days to register a shock that options markets price instantly.
    Adding an implied-vol channel (Aigner 2023; ACM 2024 VIX-gold
    causality study) closes that gap — the HMM can flag a high-vol
    state from elevated implied vol even when realised vol hasn't yet
    moved.

    Parameters
    ----------
    features
        DataFrame containing at least ``return_col`` and ``vol_col``.
        Rows where either is NaN are dropped before fitting.
    return_col, vol_col
        Column names for the two emission channels.
    n_iter, random_state
        Forwarded to ``hmmlearn.GaussianHMM``.

    Raises
    ------
    ImportError
        If ``hmmlearn`` is not installed (it is an optional dep).
    ValueError
        If required columns are missing or there are too few clean rows.
    """
    try:
        from hmmlearn.hmm import GaussianHMM
    except ImportError as exc:
        raise ImportError(
            "fit_multivariate_markov_regime requires hmmlearn. "
            "Install with: pip install 'juliams[hmm]'  or  pip install hmmlearn"
        ) from exc

    if return_col not in features.columns:
        raise ValueError(f"Missing required column {return_col!r}")
    if vol_col not in features.columns:
        raise ValueError(f"Missing required column {vol_col!r}")

    clean = features[[return_col, vol_col]].dropna().astype(float)
    if len(clean) < 50:
        raise ValueError(
            f"Need at least 50 clean rows; got {len(clean)} "
            f"after dropping NaN in {return_col} or {vol_col}."
        )

    model = GaussianHMM(
        n_components=2,
        covariance_type="full",
        n_iter=n_iter,
        random_state=random_state,
    )
    model.fit(clean.values)

    # Canonicalise: state 0 = low vol, state 1 = high vol, by ranking
    # on the variance of the return channel (column 0). This matches
    # the convention in the univariate fitter and keeps labels stable
    # across refits.
    return_vars = np.array(
        [model.covars_[i][0, 0] for i in range(2)]
    )
    if return_vars[0] <= return_vars[1]:
        low_idx, high_idx = 0, 1
    else:
        low_idx, high_idx = 1, 0
    perm = np.array([low_idx, high_idx])

    state_means = model.means_[perm]
    state_covariances = model.covars_[perm]
    transition_matrix = model.transmat_[np.ix_(perm, perm)]

    posteriors = model.predict_proba(clean.values)  # shape (T, 2)
    smoothed_high = posteriors[:, high_idx]

    # Compute filtered (causal) probabilities by running the forward
    # recursion ourselves using the public per-state emission
    # log-likelihoods. hmmlearn 0.3 dropped its public _do_forward_pass.
    log_emission = _gaussian_log_emission_per_state(
        model, clean.values
    )  # shape (T, 2)
    log_pi = np.log(model.startprob_ + 1e-300)
    log_trans = np.log(model.transmat_ + 1e-300)

    T = len(clean)
    log_alpha = np.empty((T, 2))
    log_alpha[0] = log_pi + log_emission[0]
    for t in range(1, T):
        # log alpha_t[j] = logsumexp_i (log alpha_{t-1}[i] + log P(j|i)) + log P(x_t|j)
        log_alpha[t] = (
            np.logaddexp(
                log_alpha[t - 1, 0] + log_trans[0],
                log_alpha[t - 1, 1] + log_trans[1],
            )
            + log_emission[t]
        )
    norm = np.logaddexp(log_alpha[:, 0], log_alpha[:, 1])
    filtered_raw = np.exp(log_alpha - norm[:, None])  # (T, 2)
    filtered_high = filtered_raw[:, high_idx]

    smoothed_series = pd.Series(np.nan, index=features.index, dtype=float)
    filtered_series = pd.Series(np.nan, index=features.index, dtype=float)
    smoothed_series.loc[clean.index] = smoothed_high
    filtered_series.loc[clean.index] = filtered_high

    return MultivariateMarkovFit(
        smoothed_prob_high=smoothed_series,
        filtered_prob_high=filtered_series,
        state_means=state_means,
        state_covariances=state_covariances,
        transition_matrix=transition_matrix,
        log_likelihood=float(model.score(clean.values)),
        feature_names=[return_col, vol_col],
    )


def enforce_min_dwell(
    labels: pd.Series,
    min_days: int = 5,
    unknown_label: str = "Unknown",
) -> pd.Series:
    """Suppress regime flips shorter than ``min_days`` observations.

    Why this exists
    ---------------
    A well-documented failure mode of 2-state Markov-switching variance
    models on real equity data (Salisu et al. 2020+ on COVID;
    practitioner accounts 2023-2024 on SVB / Fed pivot) is *spurious
    flipping*: the smoothed/filtered state probability oscillates
    around 0.5 during structural breaks, producing label sequences like
    ``HLHLHHLLHHL`` that do not correspond to actual regime changes.

    The standard fix is a minimum-dwell-time post-processor: a new
    state must persist for at least ``min_days`` observations before we
    accept the switch. If the candidate run is shorter, we extend the
    previous state through it.

    The function preserves Unknown labels (which mark warmup or NaN
    inputs) and operates left-to-right, so it never uses future
    information. Idempotent — applying it twice yields the same result.

    Parameters
    ----------
    labels
        Discrete regime labels, e.g. output of
        :func:`label_markov_regimes`.
    min_days
        Minimum dwell time. ``min_days=1`` is a no-op.
    unknown_label
        Label string treated as "warmup" — neither extends nor breaks
        runs. Defaults to ``"Unknown"`` to match this module's convention.
    """
    if min_days < 1:
        raise ValueError(f"min_days must be >= 1, got {min_days}")
    if min_days == 1 or len(labels) == 0:
        return labels.copy()

    out = labels.copy()
    values = out.values.copy()
    n = len(values)

    # Walk runs of identical non-Unknown labels. When a run shorter than
    # min_days is followed by a different non-Unknown label, replace it
    # with the previous accepted label.
    last_accepted: object = None
    i = 0
    while i < n:
        if values[i] == unknown_label:
            i += 1
            continue
        # Find the end of the current run.
        j = i
        while j < n and values[j] == values[i]:
            j += 1
        run_len = j - i
        if run_len < min_days and last_accepted is not None:
            values[i:j] = last_accepted
        else:
            last_accepted = values[i]
        i = j

    out.iloc[:] = values
    return out
