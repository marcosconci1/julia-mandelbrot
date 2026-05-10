"""Regression: DSM-BOCPD does not over-detect on a stationary quiet series.

This test is the deterministic counterpart to ``scripts/validate_dsm_across_assets.py``.
On the live multi-asset run (May 2026), DSM-BOCPD produced 0 resets on
IEF and 1 on TLT (long-duration treasury bonds with no real regime
break in the period). The synthetic fixture below emulates that
behaviour: a long stationary low-vol series with no real regime
change. We assert DSM produces a small bounded number of resets so
that future tuning changes don't silently introduce false positives.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from juliams.regimes.dsm_bocpd import detect_change_points_dsm_bocpd


def _stationary_low_vol_series(n: int = 600, sigma: float = 0.004, seed: int = 0) -> pd.Series:
    """Stationary Gaussian returns mimicking a quiet bond ETF."""
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(0.0, sigma, n))


def _count_resets(map_run_length: pd.Series, drop_threshold: int = 30) -> int:
    rl = map_run_length.dropna().values
    return int(((rl[:-1] - rl[1:]) >= drop_threshold).sum())


def test_dsm_no_more_than_two_resets_on_stationary_low_vol_series():
    """DSM-BOCPD with default tuning must produce at most 2 resets on
    600 days of stationary low-vol Gaussian returns. This is the
    falsifiable false-positive guard.

    Bound chosen from empirical runs: synthetic seeds 0..5 produce
    0-1 resets each. We allow up to 2 to absorb sampling noise."""
    s = _stationary_low_vol_series()
    varx = float(s.var())
    res = detect_change_points_dsm_bocpd(
        s,
        expected_run_length=100,
        varx=varx,
        omega=1.0,
        robustness_bandwidth=3.0,
    )
    n_resets = _count_resets(res.map_run_length)
    assert n_resets <= 2, (
        f"DSM-BOCPD over-detected on a stationary low-vol series; "
        f"got {n_resets} resets, expected <= 2."
    )


def test_dsm_grows_run_length_to_at_least_half_of_observations():
    """On a stationary series, the MAP run length at the end should
    reflect a long surviving run, not many recent resets."""
    s = _stationary_low_vol_series()
    varx = float(s.var())
    res = detect_change_points_dsm_bocpd(
        s,
        expected_run_length=100,
        varx=varx,
        omega=1.0,
        robustness_bandwidth=3.0,
    )
    final_rl = int(res.map_run_length.iloc[-1])
    assert final_rl >= len(s) // 2, (
        f"DSM-BOCPD failed to keep a long run length on a stationary series; "
        f"final MAP rl = {final_rl}, expected >= {len(s) // 2}."
    )


def test_quiet_series_consistent_across_seeds():
    """Run the same fixture across 5 seeds. The reset count distribution
    should be tight (max - min <= 3) so behaviour is reproducible."""
    counts = []
    for seed in range(5):
        s = _stationary_low_vol_series(seed=seed)
        varx = float(s.var())
        res = detect_change_points_dsm_bocpd(
            s, expected_run_length=100, varx=varx, omega=1.0, robustness_bandwidth=3.0,
        )
        counts.append(_count_resets(res.map_run_length))
    assert max(counts) - min(counts) <= 3, (
        f"DSM-BOCPD reset count varies wildly across seeds on a quiet "
        f"series; counts={counts}"
    )
    assert max(counts) <= 3
