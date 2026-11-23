"""Offline synthetic tests for causal transition dynamics and conditional stats (spec section 6.3)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from julia_mandelbrot.cells import cell_frame
from julia_mandelbrot.config import IndicatorConfig
from julia_mandelbrot.transitions import (
    conditional_stats,
    next_cell_probabilities,
    state_duration,
    transition_counts,
)


def _cell_series(values: list[object], name: str = "cell") -> pd.Series:
    index = pd.date_range("2024-01-01", periods=len(values), freq="B", tz="UTC")
    return pd.Series(values, index=index, name=name, dtype=object)


def _return_series(values: list[float], index: pd.Index) -> pd.Series:
    return pd.Series(values, index=index, name="returns", dtype=float)


def test_hand_built_sequence_exact_transition_counts_and_probabilities():
    # Sequence of cells at t = 0..6:
    # t=0: A
    # t=1: B  (transition u=0->1: A->B, usable at t >= 2)
    # t=2: A  (transition u=1->2: B->A, usable at t >= 3)
    # t=3: A  (transition u=2->3: A->A, usable at t >= 4)
    # t=4: B  (transition u=3->4: A->B, usable at t >= 5)
    # t=5: A  (transition u=4->5: B->A, usable at t >= 6)
    # t=6: A
    cells = _cell_series(["A", "B", "A", "A", "B", "A", "A"])

    counts = transition_counts(cells, window=3)
    assert counts.index.equals(cells.index)
    assert list(counts.columns) == [("A", "A"), ("A", "B"), ("B", "A"), ("B", "B")]

    # Row-by-row exact counts with trailing window=3 over u+1 <= t-1:
    # t=0: no transitions -> [0, 0, 0, 0]
    # t=1: u+1 <= 0 (empty) -> [0, 0, 0, 0]
    # t=2: u+1 in [1..1] = {A->B} -> [0, 1, 0, 0]
    # t=3: u+1 in [1..2] = {A->B, B->A} -> [0, 1, 1, 0]
    # t=4: u+1 in [1..3] = {A->B, B->A, A->A} -> [1, 1, 1, 0]
    # t=5: u+1 in [2..4] = {B->A, A->A, A->B} -> [1, 1, 1, 0]
    # t=6: u+1 in [3..5] = {A->A, A->B, B->A} -> [1, 1, 1, 0]
    expected_counts = pd.DataFrame(
        [
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 1, 1, 0],
            [1, 1, 1, 0],
            [1, 1, 1, 0],
            [1, 1, 1, 0],
        ],
        index=cells.index,
        columns=counts.columns,
        dtype="int64",
    )
    pd.testing.assert_frame_equal(counts, expected_counts)

    # With floor=2 and window=3:
    # At t=4 (cell=B): outgoing from B in window is B->A: 1 (total 1 < 2, pooled also 1 < 2)
    # -> p_next_A = NaN, p_next_B = NaN, pooled = True.
    # At t=5 and t=6 (cell=A): outgoing from A in window are A->A: 1, A->B: 1 (total 2 >= 2)
    # -> p_next_A = 0.5, p_next_B = 0.5, pooled = False.
    probs = next_cell_probabilities(cells, window=3, floor=2)
    assert list(probs.columns) == ["p_next_A", "p_next_B", "pooled"]
    assert probs.index.equals(cells.index)

    assert probs.loc[cells.index[4], ["p_next_A", "p_next_B"]].isna().all()
    assert probs.loc[cells.index[4], "pooled"] == True  # noqa: E712

    for t_idx in (5, 6):
        assert probs.loc[cells.index[t_idx], "p_next_A"] == pytest.approx(0.5)
        assert probs.loc[cells.index[t_idx], "p_next_B"] == pytest.approx(0.5)
        assert probs.loc[cells.index[t_idx], "pooled"] == False  # noqa: E712


def test_causality_perturbing_and_appending_last_cell_leaves_earlier_rows_unchanged():
    base_values = ["low|low", "mid|low", "low|low", "high|high", "low|low", "mid|low", "low|low"]
    cells = _cell_series(base_values)
    returns = _return_series([0.01, -0.02, 0.015, -0.03, 0.005, 0.02, -0.01], cells.index)

    base_counts = transition_counts(cells, window=3)
    base_probs = next_cell_probabilities(cells, window=3, floor=1)
    base_duration = state_duration(cells)
    base_cond = conditional_stats(cells, returns, floor=2)

    # 1. Perturb the LAST cell value.
    perturbed = cells.copy()
    perturbed.iloc[-1] = "high|high"
    perturbed_returns = returns.copy()
    perturbed_returns.iloc[-1] = -0.50

    pert_counts = transition_counts(perturbed, window=3)
    pert_probs = next_cell_probabilities(perturbed, window=3, floor=1)
    pert_duration = state_duration(perturbed)
    pert_cond = conditional_stats(perturbed, perturbed_returns, floor=2)

    # All earlier rows (t < T-1) are completely unchanged.
    pd.testing.assert_frame_equal(base_counts.iloc[:-1], pert_counts.iloc[:-1])
    pd.testing.assert_frame_equal(base_probs.iloc[:-1], pert_probs.iloc[:-1])
    pd.testing.assert_series_equal(base_duration.iloc[:-1], pert_duration.iloc[:-1])
    pd.testing.assert_frame_equal(base_cond.iloc[:-1], pert_cond.iloc[:-1])

    # Furthermore, transition_counts at the last row t = T-1 never uses cells[T-1]
    # (only transitions u+1 <= T-2), so even row T-1 of counts is unchanged!
    pd.testing.assert_series_equal(base_counts.iloc[-1], pert_counts.iloc[-1])

    # And conditional_stats at t = T-1 never uses returns[T-1] (only u <= T-2).
    cond_same_cell_pert_ret = conditional_stats(cells, perturbed_returns, floor=2)
    pd.testing.assert_frame_equal(base_cond, cond_same_cell_pert_ret)

    # 2. Append a new cell value at the end; earlier rows match identically.
    extended = _cell_series([*base_values, "high|high"])
    ext_returns = _return_series([0.01, -0.02, 0.015, -0.03, 0.005, 0.02, -0.01, 0.04], extended.index)

    ext_counts = transition_counts(extended, window=3)
    ext_probs = next_cell_probabilities(extended, window=3, floor=1)
    ext_duration = state_duration(extended)
    ext_cond = conditional_stats(extended, ext_returns, floor=2)

    pd.testing.assert_frame_equal(base_counts, ext_counts.iloc[:-1])
    pd.testing.assert_frame_equal(base_probs, ext_probs.iloc[:-1])
    pd.testing.assert_series_equal(base_duration, ext_duration.iloc[:-1])
    pd.testing.assert_frame_equal(base_cond, ext_cond.iloc[:-1])


def test_floor_behavior_windowed_vs_pooled_fallback_and_below_pooled_floor_nan():
    # Construct a sequence where cell "A" accumulates 3 transitions early on:
    # u=0->1: A->B (1st out of A)
    # u=1->2: B->A
    # u=2->3: A->A (2nd out of A)
    # u=3->4: A->B (3rd out of A; historical out of A: A->A=1, A->B=2, total=3)
    # Then spend several transitions inside B so A's transitions drop out of a short window=2:
    # u=4->5: B->B
    # u=5->6: B->B
    # u=6->7: B->A
    # At t=8, cells[8] = "A":
    # - Usable transitions have u+1 <= 7.
    # - Trailing window=2 covers u+1 in {6, 7}: {B->B, B->A}, which has 0 transitions out of A!
    # - Pooled history through t-1=7 covers u+1 in {1..7}, which has 3 transitions out of A
    #   (A->A: 1, A->B: 2).
    cells = _cell_series(["A", "B", "A", "A", "B", "B", "B", "A", "A"])
    probs = next_cell_probabilities(cells, window=2, floor=3)

    # 1. Early rows where even pooled history out of current cell has < 3 transitions -> NaN.
    for early_t in (0, 2, 3, 4):
        assert probs.iloc[early_t][["p_next_A", "p_next_B"]].isna().all()
        assert probs.iloc[early_t]["pooled"] == True  # noqa: E712

    # 2. At t=7 and t=8, current cell is "A". Windowed count out of "A" is 0 (< floor=3),
    # while pooled all-history count out of "A" through t-1 is 3 (>= floor=3).
    # Triggers pooled=True and returns the pooled estimates (1/3 and 2/3).
    for t_idx in (7, 8):
        assert probs.iloc[t_idx]["pooled"] == True  # noqa: E712
        assert probs.iloc[t_idx]["p_next_A"] == pytest.approx(1.0 / 3.0)
        assert probs.iloc[t_idx]["p_next_B"] == pytest.approx(2.0 / 3.0)

    # 3. With a large window (window=10) and floor=3, at t=7 and t=8 the windowed total
    # out of "A" is 3 (>= floor=3), so pooled=False.
    unpooled = next_cell_probabilities(cells, window=10, floor=3)
    assert unpooled.iloc[7]["pooled"] == False  # noqa: E712
    assert unpooled.iloc[7]["p_next_A"] == pytest.approx(1.0 / 3.0)
    assert unpooled.iloc[7]["p_next_B"] == pytest.approx(2.0 / 3.0)


def test_conditional_stats_known_cvar_and_per_cell_floor():
    # Build 40 historical observations in cell "A" (u = 0..39) followed by 1 observation
    # in cell "B" (u = 40), then query cell "A" at t = 41 and cell "B" at t = 42.
    # For cell "A" with 40 observations and default tail_quantile=0.95 (worst 5% -> k = 2):
    # Let the two worst returns in "A" be -0.10 and -0.06, and the other 38 returns be +0.01.
    a_returns = [-0.10, 0.01, -0.06] + [0.01] * 37
    cells_list = ["A"] * 40 + ["B", "A", "B"]
    returns_list = a_returns + [-0.05, 0.99, -0.99]

    cells = _cell_series(cells_list)
    returns = _return_series(returns_list, cells.index)

    stats = conditional_stats(cells, returns, floor=30)
    assert list(stats.columns) == ["cond_mean", "cond_cvar"]
    assert stats.index.equals(cells.index)

    # Rows t = 0..29 for cell "A" have fewer than 30 prior observations (u <= t-1) -> NaN.
    assert stats.iloc[:30].isna().all().all()

    # Row t = 30 is the first row where cell "A" has 30 prior observations (u = 0..29).
    assert stats.iloc[30].notna().all()

    # At row t = 41 (cell = "A"), prior observations of "A" at u <= 40 are exactly the 40
    # values in `a_returns` (note returns[41] = 0.99 is at u = t and must NOT be included).
    expected_mean = float(np.mean(a_returns))
    expected_cvar = (-0.10 + -0.06) / 2.0  # worst 5% of 40 observations = worst 2
    assert stats.iloc[41]["cond_mean"] == pytest.approx(expected_mean)
    assert stats.iloc[41]["cond_cvar"] == pytest.approx(expected_cvar)

    # At row t = 40 and t = 42 (cell = "B"), cell "B" has only 0 and 1 prior observation
    # (< floor = 30), even though total observations across all cells exceed 30.
    # Per-cell floor -> NaN.
    assert stats.iloc[40].isna().all()
    assert stats.iloc[42].isna().all()


def test_state_duration_counts_runs_inclusively_and_handles_missing_cells():
    cells = _cell_series(["A", "A", "A", "B", "B", "A", "C", "C"])
    duration = state_duration(cells)

    assert duration.name == "state_duration"
    assert duration.index.equals(cells.index)
    assert duration.tolist() == [1, 2, 3, 1, 2, 1, 1, 2]

    # Missing cell values produce NaN and reset consecutive run counts.
    with_nulls = _cell_series([None, "A", "A", None, "A", "A", "A"])
    null_duration = state_duration(with_nulls)
    assert np.isnan(null_duration.iloc[0])
    assert np.isnan(null_duration.iloc[3])
    assert null_duration.iloc[1:3].tolist() == [1.0, 2.0]
    assert null_duration.iloc[4:7].tolist() == [1.0, 2.0, 3.0]


def test_integration_alignment_with_cell_frame_and_docstrings():
    rng = np.random.default_rng(42)
    log_returns = rng.normal(0.0005, 0.01, size=80)
    prices = pd.Series(
        100.0 * np.exp(np.cumsum(log_returns)),
        index=pd.date_range("2023-01-02", periods=80, freq="B", tz="UTC"),
        name="Close",
    )
    frame = cell_frame(prices, window=5, percentile_window=10)
    returns = np.log(prices).diff()

    counts = transition_counts(frame["cell"], window=20)
    probs = next_cell_probabilities(frame["cell"], window=20, floor=3)
    duration = state_duration(frame["cell"])
    cond = conditional_stats(frame["cell"], returns, floor=3)

    assert counts.index.equals(prices.index)
    assert probs.index.equals(prices.index)
    assert duration.index.equals(prices.index)
    assert cond.index.equals(prices.index)

    # Invalid warm-up rows in cell_frame have cell=None -> NaN probabilities, duration, and cond stats.
    invalid_mask = ~frame["valid"]
    assert probs.loc[invalid_mask].drop(columns=["pooled"]).isna().all().all()
    assert not probs.loc[invalid_mask, "pooled"].any()
    assert duration.loc[invalid_mask].isna().all()
    assert cond.loc[invalid_mask].isna().all().all()

    # Valid probability rows sum to 1.0 across p_next_<cell> columns.
    prob_only = probs.drop(columns=["pooled"])
    valid_prob_rows = prob_only.dropna(how="all")
    assert not valid_prob_rows.empty
    np.testing.assert_allclose(valid_prob_rows.sum(axis=1), 1.0, rtol=1e-12)

    # Every public function states its causality rule.
    for fn in (
        transition_counts,
        next_cell_probabilities,
        state_duration,
        conditional_stats,
    ):
        doc = fn.__doc__ or ""
        assert "causality rule" in doc.lower()
        assert "causal" in doc.lower()
        assert "t - 1" in doc or "t-1" in doc or "<= t" in doc

    # Default config parameters are respected.
    cfg = IndicatorConfig()
    assert cfg.transition_window == 756
    assert cfg.transition_count_floor == 30
