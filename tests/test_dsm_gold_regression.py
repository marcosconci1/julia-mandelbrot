"""Regression: DSM-BOCPD detects the 2026 gold crash that vanilla BOCPD missed.

This test pins the practical fix for the May 2026 XAUUSD validation
failure (BOCPD reported run_length=502, missing the worst gold rout
since 1983 in early Feb 2026). On a synthetic gold-like fixture
reproducing the salient features of the real data, DSM-BOCPD with
reasonable hyperparameters resets the MAP run length around the
crash day, while vanilla BOCPD does not.

We use a synthetic fixture rather than live Yahoo data so the test
is deterministic and offline-runnable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from juliams.regimes.bocpd import detect_change_points_bocpd
from juliams.regimes.dsm_bocpd import detect_change_points_dsm_bocpd


def _gold_like_returns(seed: int = 0) -> pd.Series:
    """500 days emulating gold 2024-2026: long calm run, then an
    extreme (8-sigma vs full-sample sigma) crash, then continued
    moderate vol. Calibrated so that:
    - max(|ret|) / std(ret) is roughly the real-gold ratio (~8.3 in
      the May 2026 yfinance pull).
    - The post-crash regime has slightly elevated vol but is mostly
      stationary; DSM should detect the single crash day, not flag
      every subsequent observation.
    """
    rng = np.random.default_rng(seed)
    n_pre = 260
    n_post = 240
    sigma = 0.012
    pre = rng.normal(0.0005, sigma, n_pre)
    crash = np.array([-10.0 * sigma])  # very large crash day
    post = rng.normal(-0.0002, sigma * 1.2, n_post - 1)
    return pd.Series(np.concatenate([pre, crash, post]))


def test_dsm_bocpd_detects_crash_day_when_standard_bocpd_misses_it():
    s = _gold_like_returns()
    varx = float(s.var())
    crash_idx = 260

    std = detect_change_points_bocpd(s, expected_run_length=100)
    dsm = detect_change_points_dsm_bocpd(
        s, expected_run_length=100, varx=varx,
        omega=1.0, robustness_bandwidth=3.0,
    )

    rl_std_before = int(std.map_run_length.iloc[crash_idx - 1])
    rl_std_after = int(std.map_run_length.iloc[crash_idx + 1])
    rl_dsm_before = int(dsm.map_run_length.iloc[crash_idx - 1])
    rl_dsm_after = int(dsm.map_run_length.iloc[crash_idx + 1])

    # Standard BOCPD will NOT reset at the crash (this is the observed
    # bug from May 2026 gold validation).
    assert rl_std_after > rl_std_before - 5, (
        f"Standard BOCPD unexpectedly reset on the crash day: "
        f"before={rl_std_before}, after={rl_std_after}. "
        f"If this assertion starts failing, the standard detector may "
        f"have been tuned more aggressively (which is fine) — update the "
        f"test to reflect that."
    )

    # DSM-BOCPD MUST reset at the crash.
    assert rl_dsm_after < 10, (
        f"DSM-BOCPD failed to reset on the crash day; "
        f"before={rl_dsm_before}, after={rl_dsm_after}"
    )


def test_dsm_bocpd_subsequent_run_length_grows_smoothly_after_crash():
    """After detecting the crash, DSM-BOCPD should accumulate a fresh
    run length without oscillating wildly. Measure: 100 days after
    the crash, MAP run length should be at least 50 (smooth growth)."""
    s = _gold_like_returns()
    varx = float(s.var())
    dsm = detect_change_points_dsm_bocpd(
        s, expected_run_length=100, varx=varx,
        omega=1.0, robustness_bandwidth=3.0,
    )
    crash_idx = 260
    rl_at_360 = int(dsm.map_run_length.iloc[crash_idx + 100])
    assert rl_at_360 >= 50, (
        f"DSM run length failed to recover smoothly post-crash; "
        f"rl 100 days after crash = {rl_at_360}"
    )
