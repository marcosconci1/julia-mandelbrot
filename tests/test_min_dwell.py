"""Tests for juliams.regimes.markov.enforce_min_dwell.

Falsifiable claims:
- Spurious 1-day flips are eliminated.
- Real regime runs (>= min_days) are preserved.
- Idempotent: applying twice yields same result as once.
- Causal: result at index t uses only labels through t.
- Unknown labels (warmup) are not affected.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from juliams.regimes.markov import enforce_min_dwell


def test_eliminates_single_day_flips():
    labels = pd.Series(["L", "L", "L", "H", "L", "L", "L"])
    out = enforce_min_dwell(labels, min_days=3)
    # The single 'H' is shorter than min_days=3, should be suppressed.
    assert out.tolist() == ["L"] * 7


def test_preserves_runs_at_or_above_min_days():
    labels = pd.Series(["L", "L", "L", "H", "H", "H", "L", "L", "L"])
    out = enforce_min_dwell(labels, min_days=3)
    assert out.tolist() == labels.tolist()


def test_suppresses_short_run_below_threshold_keeps_long_one():
    """A 2-day H run between L's should be suppressed when min=3, but
    a 5-day H run elsewhere should survive. Note: the trailing 2-day L
    run is also below threshold, so it gets absorbed into the
    preceding accepted state H."""
    labels = pd.Series(
        ["L", "L", "L", "H", "H", "L", "L", "L", "H", "H", "H", "H", "H", "L", "L"]
    )
    out = enforce_min_dwell(labels, min_days=3)
    expected = ["L"] * 8 + ["H"] * 7
    assert out.tolist() == expected


def test_min_days_one_is_noop():
    labels = pd.Series(["L", "H", "L", "H", "L", "H"])
    out = enforce_min_dwell(labels, min_days=1)
    assert out.tolist() == labels.tolist()


def test_idempotent():
    rng = np.random.default_rng(0)
    labels = pd.Series(rng.choice(["L", "H"], size=200))
    once = enforce_min_dwell(labels, min_days=4)
    twice = enforce_min_dwell(once, min_days=4)
    assert once.equals(twice)


def test_unknown_labels_preserved():
    labels = pd.Series(["Unknown", "Unknown", "L", "L", "H", "L", "L"])
    out = enforce_min_dwell(labels, min_days=3)
    assert out.iloc[:2].tolist() == ["Unknown", "Unknown"]
    # The lone H is suppressed back to L (the accepted predecessor).
    assert out.iloc[2:].tolist() == ["L", "L", "L", "L", "L"]


def test_causal_no_future_leakage():
    """Result at index t must depend only on labels through index t —
    we verify by poisoning the future and confirming earlier output is
    unchanged."""
    rng = np.random.default_rng(7)
    labels = pd.Series(rng.choice(["L", "H"], size=200))
    out_orig = enforce_min_dwell(labels, min_days=4)

    perturbed = labels.copy()
    perturbed.iloc[150:] = "H"
    out_pert = enforce_min_dwell(perturbed, min_days=4)

    # Indices 0..149 must be identical.
    assert out_orig.iloc[:150].equals(out_pert.iloc[:150])


def test_first_run_below_threshold_is_kept_when_no_predecessor():
    """If the very first non-Unknown run is too short, there's no
    predecessor to fall back to — keep it rather than relabel arbitrarily."""
    labels = pd.Series(["Unknown", "Unknown", "L", "L", "H", "H", "H", "H"])
    out = enforce_min_dwell(labels, min_days=3)
    # 'L' run is 2 days < 3, but there's no predecessor; keep it.
    assert out.iloc[2:4].tolist() == ["L", "L"]
    assert out.iloc[4:].tolist() == ["H"] * 4


def test_invalid_min_days_raises():
    with pytest.raises(ValueError):
        enforce_min_dwell(pd.Series(["L"]), min_days=0)
    with pytest.raises(ValueError):
        enforce_min_dwell(pd.Series(["L"]), min_days=-1)


def test_empty_input_returns_empty():
    out = enforce_min_dwell(pd.Series([], dtype=object), min_days=3)
    assert len(out) == 0


def test_reduces_total_label_changes_on_noisy_series():
    """Headline behavioural test: on a synthetic series with realistic
    spurious flipping, enforce_min_dwell must materially reduce the
    number of label transitions."""
    rng = np.random.default_rng(11)
    n = 500
    # 70% chance of staying in the same state, 30% of flipping → lots
    # of short runs.
    states = ["L"]
    for _ in range(n - 1):
        states.append(states[-1] if rng.random() < 0.7 else ("L" if states[-1] == "H" else "H"))
    labels = pd.Series(states)
    raw_changes = (labels.values[1:] != labels.values[:-1]).sum()

    smoothed = enforce_min_dwell(labels, min_days=5)
    smoothed_changes = (smoothed.values[1:] != smoothed.values[:-1]).sum()

    assert smoothed_changes < raw_changes / 2, (
        f"min_dwell did not materially reduce flip count: "
        f"raw={raw_changes} smoothed={smoothed_changes}"
    )
