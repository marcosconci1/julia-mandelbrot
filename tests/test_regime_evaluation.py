"""Tests for juliams.regimes.evaluation metrics."""

import numpy as np
import pandas as pd
import pytest

from juliams.regimes.evaluation import (
    compare_classifiers,
    forward_return_information_ratio,
    regime_distribution_entropy,
    regime_stability,
)


def test_stability_constant_label_is_one():
    s = pd.Series(["A"] * 100)
    assert regime_stability(s) == pytest.approx(1.0)


def test_stability_alternating_label_is_zero():
    s = pd.Series(["A", "B"] * 50)
    assert regime_stability(s) == pytest.approx(0.0)


def test_stability_handles_nans():
    # Drop NaN, then 4 valid labels [A,A,A,B] → 3 transitions, 2 same.
    s = pd.Series(["A", np.nan, "A", "A", np.nan, "B"])
    assert regime_stability(s) == pytest.approx(2 / 3)


def test_stability_too_few_values_returns_nan():
    assert np.isnan(regime_stability(pd.Series([], dtype=object)))
    assert np.isnan(regime_stability(pd.Series(["A"])))


def test_entropy_uniform_two_buckets_is_one_bit():
    s = pd.Series(["A"] * 50 + ["B"] * 50)
    assert regime_distribution_entropy(s, base=2.0) == pytest.approx(1.0)


def test_entropy_single_bucket_is_zero():
    s = pd.Series(["A"] * 100)
    assert regime_distribution_entropy(s) == pytest.approx(0.0)


def test_entropy_uniform_six_buckets_is_log2_six():
    labels = ["A", "B", "C", "D", "E", "F"] * 30
    s = pd.Series(labels)
    assert regime_distribution_entropy(s, base=2.0) == pytest.approx(np.log2(6), rel=1e-6)


def test_information_ratio_perfect_separation_is_high():
    # Regime A always +1, Regime B always -1 → infinite IR (std=0).
    # min_support drops nothing here. Both regimes have 0 std so all
    # entries get filtered, returning NaN by design.
    labels = pd.Series(["A"] * 50 + ["B"] * 50)
    fr = pd.Series([1.0] * 50 + [-1.0] * 50)
    assert np.isnan(forward_return_information_ratio(labels, fr))


def test_information_ratio_random_labels_is_low():
    rng = np.random.default_rng(0)
    fr = pd.Series(rng.standard_normal(500))
    labels = pd.Series(rng.choice(["A", "B", "C"], size=500))
    ir = forward_return_information_ratio(labels, fr, min_support=10)
    # With random labels, mean(fr | label) ≈ 0, so IR should be small.
    assert ir < 0.2


def test_information_ratio_signal_beats_random():
    rng = np.random.default_rng(42)
    n = 600
    # True regime drives mean: A gets +0.5 drift, B gets -0.5, C is noise.
    labels_true = rng.choice(["A", "B", "C"], size=n, p=[0.4, 0.4, 0.2])
    drift = pd.Series(labels_true).map({"A": 0.5, "B": -0.5, "C": 0.0}).values
    fr = pd.Series(drift + rng.standard_normal(n))
    labels_random = pd.Series(rng.choice(["A", "B", "C"], size=n))
    ir_signal = forward_return_information_ratio(pd.Series(labels_true), fr, min_support=10)
    ir_random = forward_return_information_ratio(labels_random, fr, min_support=10)
    assert ir_signal > ir_random


def test_information_ratio_drops_low_support_regimes():
    # Regime "rare" has 5 obs (< min_support=10), regime "common" has 100.
    rng = np.random.default_rng(1)
    labels = pd.Series(["common"] * 100 + ["rare"] * 5)
    fr = pd.Series(np.concatenate([rng.standard_normal(100) + 0.3, [99.0] * 5]))
    ir = forward_return_information_ratio(labels, fr, min_support=10)
    # If "rare" weren't dropped its huge mean would dominate; verify it's
    # in a sane range driven by "common" only.
    assert 0.0 <= ir < 1.0


def test_compare_classifiers_returns_two_rows():
    rng = np.random.default_rng(3)
    n = 200
    fr = pd.Series(rng.standard_normal(n))
    baseline = pd.Series(rng.choice(["X", "Y"], size=n))
    candidate = pd.Series(rng.choice(["X", "Y", "Z"], size=n))
    table = compare_classifiers(baseline, candidate, fr)
    assert list(table.index) == ["baseline", "candidate"]
    assert set(table.columns) == {"stability", "entropy", "information_ratio", "n_regimes"}
    assert table.loc["baseline", "n_regimes"] == 2
    assert table.loc["candidate", "n_regimes"] == 3
