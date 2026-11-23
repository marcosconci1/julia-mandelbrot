"""Causal regime transition counts, next-cell probabilities, duration, and conditional return statistics.

Every calculation in this module is strictly causal (no lookahead):
- :func:`transition_counts` at row ``t`` counts only observed transitions
  ``(cells[u] -> cells[u + 1])`` with arrival index ``u + 1 <= t - 1`` within
  the trailing window of transitions (the last usable transition at row ``t``
  is ``cells[t - 2] -> cells[t - 1]``; ``cells[t]`` is never used).
- :func:`next_cell_probabilities` at row ``t`` conditions on the active cell
  ``cells[t]`` and normalizes the outgoing causal transition counts from
  ``u + 1 <= t - 1`` when the windowed outgoing total meets the count floor,
  falling back to the pooled (all-history through ``t - 1``) outgoing estimate
  with ``pooled=True`` when the windowed total is below the floor, and
  emitting ``NaN`` when the pooled total is also below the floor.
- :func:`state_duration` at row ``t`` counts consecutive sessions in the same
  cell up to and including ``t`` using only ``cells[0..t]``.
- :func:`conditional_stats` at row ``t`` computes the in-cell mean return
  (``cond_mean``) and left-tail conditional Value-at-Risk (``cond_cvar``) for
  ``cells[t]`` using only past observations ``u <= t - 1`` where
  ``cells[u] == cells[t]``, returning ``NaN`` whenever the per-cell historical
  observation count through ``t - 1`` is below the count floor.
"""

from __future__ import annotations

import math
import operator
from typing import Any

import pandas as pd

from .cells import TREND_STATES, VOL_STATES
from .config import IndicatorConfig

CANONICAL_CELLS: tuple[str, ...] = tuple(
    f"{trend}|{vol}" for trend in TREND_STATES for vol in VOL_STATES
)


def _validate_series(series: pd.Series, name: str) -> pd.Series:
    if not isinstance(series, pd.Series):
        raise TypeError(f"{name} must be a pandas Series")
    return series


def _validate_positive_int(value: int, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer, not bool")
    try:
        int_val = operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if int_val < 1:
        raise ValueError(f"{name} must be an integer of at least 1")
    return int(int_val)


def _validate_tail_fraction(tail_quantile: float) -> float:
    if isinstance(tail_quantile, bool) or not isinstance(tail_quantile, (int, float)):
        raise TypeError("tail_quantile must be a float")
    q = float(tail_quantile)
    if not math.isfinite(q) or not (0.0 < q < 1.0):
        raise ValueError("tail_quantile must be finite and strictly between 0 and 1")
    alpha = 1.0 - q if q > 0.5 else q
    return round(alpha, 10)


def _discover_cells(cells: pd.Series) -> list[Any]:
    non_null = cells.dropna()
    if non_null.empty:
        return []
    unique_values = list(pd.unique(non_null))
    unique_set = set(unique_values)
    if unique_set.issubset(CANONICAL_CELLS):
        return [cell for cell in CANONICAL_CELLS if cell in unique_set]
    try:
        return sorted(unique_values)
    except TypeError:
        return unique_values


def transition_counts(
    cells: pd.Series,
    window: int | None = None,
) -> pd.DataFrame:
    """Compute rolling causal transition counts for each ``(from_cell, to_cell)`` pair.

    Causality rule:
    At row ``t``, only transitions ``(cells[u] -> cells[u + 1])`` with arrival
    index ``u + 1 <= t - 1`` are eligible. Consequently, the last transition
    usable at row ``t`` is ``cells[t - 2] -> cells[t - 1]`` and row ``t`` never
    uses ``cells[t]`` or any future observation. Within ``u + 1 <= t - 1``,
    counts are summed over the trailing ``window`` transition slots
    ``u + 1 in [max(1, t - window), t - 1]`` (defaulting to
    ``IndicatorConfig().transition_window = 756``). Rows ``t = 0`` and ``t = 1``
    have no prior completed transitions with ``u + 1 <= t - 1`` and therefore
    have zero counts across all pairs.

    Parameters
    ----------
    cells : pd.Series
        Sequence of discrete regime cell labels indexed by session timestamp.
    window : int, optional
        Trailing window of transitions to count. Defaults to
        ``IndicatorConfig().transition_window`` (756).

    Returns
    -------
    pd.DataFrame
        DataFrame indexed like ``cells`` with a two-level column
        :class:`pandas.MultiIndex` ``(from_cell, to_cell)`` containing integer
        rolling counts.
    """
    _validate_series(cells, "cells")
    win = (
        IndicatorConfig().transition_window
        if window is None
        else _validate_positive_int(window, "window")
    )

    states = _discover_cells(cells)
    columns = pd.MultiIndex.from_product(
        [states, states], names=["from_cell", "to_cell"]
    )
    if not states:
        return pd.DataFrame(index=cells.index, columns=columns, dtype="int64")

    from_cells = cells.shift(1)
    to_cells = cells
    valid_step = from_cells.notna() & to_cells.notna()

    pair_indicators = pd.DataFrame(0, index=cells.index, columns=columns, dtype="int64")
    for from_cell, to_cell in columns:
        matched = valid_step & (from_cells == from_cell) & (to_cells == to_cell)
        pair_indicators[(from_cell, to_cell)] = matched.astype("int64")

    if len(cells) <= 1:
        return pair_indicators

    # Arrival indices v = u + 1 start at positional index 1. Rolling over
    # pair_indicators.iloc[1:] sums the trailing `win` transitions ending at v,
    # and shifting by 1 aligns v = t - 1 onto row t so row t only sees u + 1 <= t - 1.
    rolling_at_v = (
        pair_indicators.iloc[1:]
        .rolling(window=win, min_periods=1)
        .sum()
        .astype("int64")
    )
    counts = (
        rolling_at_v.reindex(cells.index, fill_value=0)
        .shift(1, fill_value=0)
        .astype("int64")
    )
    return counts


def next_cell_probabilities(
    cells: pd.Series,
    window: int | None = None,
    floor: int | None = None,
) -> pd.DataFrame:
    """Estimate causal next-cell transition probabilities with a pooled all-history fallback.

    Causality rule:
    At row ``t``, the outgoing transition distribution ``p_next_<cell>`` is
    conditioned on the current cell ``s_t = cells[t]`` using only transition
    counts observed at arrival indices ``u + 1 <= t - 1`` (whose last transition
    is ``cells[t - 2] -> cells[t - 1]``). No transition involving ``cells[t]`` as
    a destination (``u + 1 = t``) or origin (``u + 1 = t + 1``) enters the
    counts at row ``t``.

    Count-floor mitigation:
    Let ``windowed_total`` be the total number of transitions out of ``cells[t]``
    in the trailing ``window`` transitions through ``t - 1``.
    - When ``windowed_total >= floor``, ``p_next_<cell> = windowed_count / windowed_total``
      and ``pooled = False``.
    - When ``windowed_total < floor``, estimation falls back to the pooled
      all-history estimate through ``t - 1`` (all transitions out of ``cells[t]``
      with ``1 <= u + 1 <= t - 1``) and marks ``pooled = True``. If that pooled
      total meets ``floor``, ``p_next_<cell> = pooled_count / pooled_total``;
      if the pooled total is also below ``floor``, ``p_next_<cell>`` is ``NaN``
      (not identifiable at the floor, never guessed).

    Parameters
    ----------
    cells : pd.Series
        Sequence of discrete regime cell labels.
    window : int, optional
        Trailing transition window size. Defaults to
        ``IndicatorConfig().transition_window`` (756).
    floor : int, optional
        Minimum required outgoing transition count. Defaults to
        ``IndicatorConfig().transition_count_floor`` (30).

    Returns
    -------
    pd.DataFrame
        DataFrame indexed like ``cells`` with columns ``p_next_<cell>`` for each
        discovered cell and a boolean ``pooled`` column.
    """
    _validate_series(cells, "cells")
    cfg = IndicatorConfig()
    win = cfg.transition_window if window is None else _validate_positive_int(window, "window")
    flr = (
        cfg.transition_count_floor
        if floor is None
        else _validate_positive_int(floor, "floor")
    )

    states = _discover_cells(cells)
    prob_cols = [f"p_next_{state}" for state in states]
    result = pd.DataFrame(
        float("nan"), index=cells.index, columns=prob_cols, dtype="float64"
    )
    pooled = pd.Series(False, index=cells.index, name="pooled", dtype=bool)

    if not states:
        result["pooled"] = pooled
        return result

    windowed_counts = transition_counts(cells, window=win)
    full_history_window = max(len(cells), 1)
    pooled_counts = transition_counts(cells, window=full_history_window)

    for state in states:
        in_state = cells.notna() & (cells == state)
        if not in_state.any():
            continue

        win_sub = windowed_counts[state]
        win_total = win_sub.sum(axis=1)

        pool_sub = pooled_counts[state]
        pool_total = pool_sub.sum(axis=1)

        use_window = in_state & (win_total >= flr)
        below_window_floor = in_state & (win_total < flr)
        use_pooled = below_window_floor & (pool_total >= flr)

        pooled = pooled | below_window_floor

        for to_state in states:
            col = f"p_next_{to_state}"
            if use_window.any():
                result.loc[use_window, col] = (
                    win_sub.loc[use_window, to_state] / win_total.loc[use_window]
                ).astype("float64")
            if use_pooled.any():
                result.loc[use_pooled, col] = (
                    pool_sub.loc[use_pooled, to_state] / pool_total.loc[use_pooled]
                ).astype("float64")

    result["pooled"] = pooled
    return result


def state_duration(cells: pd.Series) -> pd.Series:
    """Count consecutive sessions in the current cell as of ``t``, inclusive of ``t``.

    Causality rule:
    At row ``t``, ``state_duration[t]`` depends only on the trailing contiguous
    run of identical non-null labels ending at ``cells[t]`` (using observations
    ``<= t`` and never any future observation ``> t``). The count is inclusive
    of ``t`` so the first session in a cell has duration ``1``. Rows where
    ``cells[t]`` is null (``None`` or ``NaN``) yield ``NaN`` and break any run.

    Parameters
    ----------
    cells : pd.Series
        Sequence of discrete regime cell labels.

    Returns
    -------
    pd.Series
        Series named ``"state_duration"`` indexed like ``cells``. Integer dtype
        when all rows are non-null, float with ``NaN`` at missing rows otherwise.
    """
    _validate_series(cells, "cells")
    if cells.empty:
        return pd.Series([], index=cells.index, name="state_duration", dtype="int64")

    valid = cells.notna()
    prev_valid = valid.shift(1, fill_value=False)
    is_new_run = (~valid) | (~prev_valid) | (cells != cells.shift(1))
    run_id = is_new_run.cumsum()
    durations = cells.groupby(run_id).cumcount() + 1

    if valid.all():
        return durations.astype("int64").rename("state_duration")
    return durations.where(valid).astype("float64").rename("state_duration")


def conditional_stats(
    cells: pd.Series,
    returns: pd.Series,
    floor: int | None = None,
    *,
    tail_quantile: float | None = None,
) -> pd.DataFrame:
    """Compute causal per-cell conditional mean return and left-tail CVaR through ``t - 1``.

    Causality rule:
    At row ``t``, the statistics for the current cell ``c = cells[t]`` are
    estimated strictly from historical observations ``u <= t - 1`` for which
    ``cells[u] == c`` and ``returns[u]`` is non-null. Neither ``returns[t]`` nor
    any future observation ``u >= t`` is ever included on row ``t``.

    Left-tail CVaR definition and per-cell floor:
    - ``tail_quantile`` defaults to ``IndicatorConfig().tail_quantile`` (``0.95``),
      which maps to the left-tail fraction ``alpha = 1 - tail_quantile = 0.05``
      (the worst 5% of historical in-cell return observations). Given ``n``
      causal in-cell return observations at ``u <= t - 1``, ``cond_cvar`` is the
      arithmetic mean of the ``k = max(1, ceil(alpha * n))`` smallest (most
      negative / worst) returns observed in cell ``c``.
    - ``floor`` defaults to ``IndicatorConfig().transition_count_floor`` (``30``)
      and applies per cell: when cell ``c`` has fewer than ``floor`` valid return
      observations in ``u <= t - 1``, both ``cond_mean`` and ``cond_cvar`` at
      row ``t`` are ``NaN`` (not identifiable at the floor).

    Parameters
    ----------
    cells : pd.Series
        Sequence of discrete regime cell labels.
    returns : pd.Series
        One-period return series aligned to ``cells.index``.
    floor : int, optional
        Minimum number of causal observations ``u <= t - 1`` required in the
        active cell before emitting non-null statistics. Defaults to
        ``IndicatorConfig().transition_count_floor`` (30).
    tail_quantile : float, optional
        Confidence quantile from :class:`IndicatorConfig` (default ``0.95``,
        mapped to the worst ``1 - q = 5%`` fraction of in-cell observations).

    Returns
    -------
    pd.DataFrame
        DataFrame indexed like ``cells`` with columns ``cond_mean`` and
        ``cond_cvar``.
    """
    _validate_series(cells, "cells")
    _validate_series(returns, "returns")
    if not returns.index.equals(cells.index):
        raise ValueError("returns must share the cells index")

    cfg = IndicatorConfig()
    flr = (
        cfg.transition_count_floor
        if floor is None
        else _validate_positive_int(floor, "floor")
    )
    alpha = _validate_tail_fraction(
        cfg.tail_quantile if tail_quantile is None else tail_quantile
    )

    numeric_returns = pd.to_numeric(returns, errors="coerce").astype("float64")
    cond_mean = pd.Series(float("nan"), index=cells.index, name="cond_mean", dtype="float64")
    cond_cvar = pd.Series(float("nan"), index=cells.index, name="cond_cvar", dtype="float64")

    for t in range(1, len(cells)):
        current_cell = cells.iloc[t]
        if pd.isna(current_cell):
            continue
        prior_cells = cells.iloc[:t]
        prior_returns = numeric_returns.iloc[:t]
        in_cell_obs = prior_returns.loc[prior_cells == current_cell].dropna()
        n_obs = len(in_cell_obs)
        if n_obs < flr:
            continue
        cond_mean.iloc[t] = float(in_cell_obs.mean())
        k_worst = max(1, math.ceil(round(alpha * n_obs, 9)))
        cond_cvar.iloc[t] = float(in_cell_obs.nsmallest(k_worst).mean())

    return pd.DataFrame({"cond_mean": cond_mean, "cond_cvar": cond_cvar}, index=cells.index)
