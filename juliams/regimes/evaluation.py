"""
Evaluation metrics for comparing regime classification methods.

Used to make adaptive-vs-hardcoded comparisons measurable rather than
hand-wavy. Three primary metrics:

- regime_stability: fraction of consecutive days sharing the same label.
  High values mean low whipsaw; pair with information_ratio so we don't
  reward a classifier that simply never changes its mind.

- regime_distribution_entropy: Shannon entropy of regime frequencies.
  Catches degenerate classifiers that collapse all days into one bucket.

- forward_return_information_ratio: mean / std of forward returns
  conditioned on regime, averaged across regimes (weighted by support).
  Higher = the regime label carries more predictive content.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def regime_stability(labels: pd.Series) -> float:
    """Fraction of consecutive-day pairs that share the same regime label.

    NaN labels are dropped before comparison. Returns NaN when fewer than
    two valid labels remain.
    """
    s = labels.dropna()
    if len(s) < 2:
        return float("nan")
    same = (s.values[1:] == s.values[:-1]).sum()
    return float(same) / float(len(s) - 1)


def regime_distribution_entropy(labels: pd.Series, base: float = 2.0) -> float:
    """Shannon entropy of the regime frequency distribution.

    Zero when one regime dominates entirely; log(K) when all K regimes
    are equally frequent. Use to flag degenerate classifiers.
    """
    s = labels.dropna()
    if len(s) == 0:
        return float("nan")
    counts = s.value_counts(normalize=True)
    p = counts.values
    p = p[p > 0]
    return float(-(p * np.log(p) / np.log(base)).sum())


def forward_return_information_ratio(
    labels: pd.Series,
    forward_returns: pd.Series,
    min_support: int = 10,
) -> float:
    """Support-weighted mean(|mean / std|) of forward returns by regime.

    Higher values mean the regime label carries more predictive content
    about the next forward-return window. We take absolute value because
    a regime that reliably predicts negative returns is just as useful as
    one that predicts positive returns.

    Regimes with fewer than `min_support` observations are dropped so
    rare-label noise doesn't dominate the score.
    """
    df = pd.DataFrame({"label": labels, "fr": forward_returns}).dropna()
    if df.empty:
        return float("nan")
    grouped = df.groupby("label")["fr"]
    means = grouped.mean()
    stds = grouped.std()
    counts = grouped.count()

    keep = counts >= min_support
    if not keep.any():
        return float("nan")

    means = means[keep]
    stds = stds[keep]
    counts = counts[keep]

    safe_stds = stds.where(stds > 0, np.nan)
    irs = (means.abs() / safe_stds).dropna()
    if irs.empty:
        return float("nan")

    weights = counts.loc[irs.index].astype(float)
    return float((irs * weights).sum() / weights.sum())


def compare_classifiers(
    baseline_labels: pd.Series,
    candidate_labels: pd.Series,
    forward_returns: pd.Series,
    min_support: int = 10,
) -> pd.DataFrame:
    """Side-by-side metric table for two classifiers on identical data."""
    rows = []
    for name, labels in [("baseline", baseline_labels), ("candidate", candidate_labels)]:
        rows.append(
            {
                "classifier": name,
                "stability": regime_stability(labels),
                "entropy": regime_distribution_entropy(labels),
                "information_ratio": forward_return_information_ratio(
                    labels, forward_returns, min_support=min_support
                ),
                "n_regimes": int(labels.dropna().nunique()),
            }
        )
    return pd.DataFrame(rows).set_index("classifier")
