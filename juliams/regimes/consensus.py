"""
Two-detector consensus gating for regime change detection.

Combines a local detector (DSM-BOCPD via its run-length output) with a
global detector (Markov HMM via its high-vol-state probability) and
declares a "consensus event" only when both flag a change within a
short time window.

Why two detectors instead of one
--------------------------------
A single detector trades off detection delay against false alarms.
BOCPD reacts fast (online, sees one observation at a time) but its
sensitivity to fat-tail events can produce singletons. The HMM is
slower to react (its state probabilities are smoothed over the data)
but more robust because it requires a sustained shift in the emission
distribution. Requiring both to fire within a 5-day window gives the
delay of the faster detector with the false-positive resistance of the
slower one. This is the "AND-gating with cooldown" approach noted in
the 2024 change-point detection survey literature.

What constitutes a "fire" per detector
--------------------------------------
- BOCPD: the MAP run length drops by at least ``rl_drop_threshold``
  observations from one day to the next. This is the same proxy used
  in ``scripts/validate_dsm_across_assets.py``.
- Markov: the high-vol-state probability rises by at least
  ``prob_high_jump_threshold`` over a 3-day window, OR crosses 0.5 from
  below. The 3-day window absorbs the smoothing of the filtered
  posterior.

After a consensus event fires, a ``cooldown_days`` window suppresses
new events to prevent runs of consecutive declarations from a single
underlying break.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _bocpd_fire_days(
    bocpd_run_length: pd.Series, rl_drop_threshold: int
) -> pd.Series:
    """Boolean Series: True on days where MAP run length drops by at
    least ``rl_drop_threshold`` from the previous day."""
    rl = bocpd_run_length.ffill().fillna(0)
    drop = rl.shift(1) - rl
    return (drop >= rl_drop_threshold).fillna(False)


def _markov_fire_days(
    markov_prob_high: pd.Series,
    prob_high_jump_threshold: float,
    crossing_threshold: float = 0.5,
) -> pd.Series:
    """Boolean Series: True on days where the Markov P(high) jumps by
    at least ``prob_high_jump_threshold`` over a 3-day window OR
    crosses ``crossing_threshold`` from below."""
    p = markov_prob_high.astype(float)
    # Three-day jump.
    jumped_3d = (p - p.shift(3)) >= prob_high_jump_threshold
    # Crosses 0.5 from below.
    crossed = (p > crossing_threshold) & (p.shift(1) <= crossing_threshold)
    return (jumped_3d.fillna(False)) | (crossed.fillna(False))


def detect_consensus_change_points(
    bocpd_run_length: pd.Series,
    markov_prob_high: pd.Series,
    window_days: int = 5,
    rl_drop_threshold: int = 30,
    prob_high_jump_threshold: float = 0.3,
    cooldown_days: int = 10,
) -> pd.Series:
    """Return a boolean Series of consensus change-point events.

    Both detectors must fire within ``window_days`` of each other for
    a consensus event to be declared. The window is symmetric: a
    Markov fire on day t triggers a consensus check against BOCPD
    fires in [t - window_days, t + window_days]. The event is anchored
    at the LATER of the two fire days so the consensus is declared no
    earlier than the slowest detector's signal.

    A ``cooldown_days`` window suppresses subsequent events.

    Parameters
    ----------
    bocpd_run_length, markov_prob_high
        Aligned Series from the BOCPD and Markov overlays. NaN values
        are treated as no-signal.
    window_days
        Maximum allowed delay between the two detectors' fires.
    rl_drop_threshold, prob_high_jump_threshold
        Per-detector firing thresholds; see module docstring.
    cooldown_days
        Number of observations after a consensus event during which no
        new event is declared.

    Returns
    -------
    pd.Series of bool, same index as the inputs.
    """
    if window_days < 0 or cooldown_days < 0:
        raise ValueError("window_days and cooldown_days must be non-negative")
    if rl_drop_threshold < 1:
        raise ValueError("rl_drop_threshold must be >= 1")
    if not 0 < prob_high_jump_threshold < 1:
        raise ValueError("prob_high_jump_threshold must be in (0, 1)")

    if not bocpd_run_length.index.equals(markov_prob_high.index):
        raise ValueError("bocpd_run_length and markov_prob_high must be aligned")

    bocpd_fires = _bocpd_fire_days(bocpd_run_length, rl_drop_threshold)
    markov_fires = _markov_fire_days(markov_prob_high, prob_high_jump_threshold)

    bocpd_idx = np.where(bocpd_fires.values)[0]
    markov_idx = np.where(markov_fires.values)[0]

    out = pd.Series(False, index=bocpd_run_length.index)
    if len(bocpd_idx) == 0 or len(markov_idx) == 0:
        return out

    # Greedy match: walk through Markov fires and find the closest
    # BOCPD fire within window. Anchor the consensus at the later of
    # the two so we are not declaring a signal before both have fired.
    last_event = -10**9
    for m in markov_idx:
        candidates = bocpd_idx[np.abs(bocpd_idx - m) <= window_days]
        if len(candidates) == 0:
            continue
        anchor = int(max(candidates.max(), m))
        if anchor - last_event < cooldown_days:
            continue
        if anchor < len(out):
            out.iloc[anchor] = True
            last_event = anchor

    return out
