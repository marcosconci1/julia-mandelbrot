"""Tests for the two-detector consensus gating.

Falsifiable claims:
- Both detectors fire within window -> consensus event fires.
- Only one detector fires -> consensus does NOT fire.
- After firing, the cooldown window suppresses subsequent events.
- On a gold-like fixture, consensus fires near the real crash and
  produces no spurious events elsewhere.
"""

from __future__ import annotations

import io
import sys
from contextlib import redirect_stdout
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

import main as cli_main
from juliams import JMSConfig, JuliaMandelbrotSystem
from juliams.regimes.consensus import detect_consensus_change_points
from juliams.regimes.overlays import apply_consensus_overlay


# -- Core consensus logic --------------------------------------------------

def test_both_detectors_fire_within_window_triggers_event():
    """BOCPD fires at t=10 (rl drops 100->1), Markov P(high) jumps from
    0.1 to 0.9 at t=11. Within the 5-day window -> consensus."""
    n = 30
    idx = pd.RangeIndex(n)
    rl = pd.Series([100] * 10 + [1] * 20, index=idx, dtype=float)
    p_high = pd.Series([0.1] * 11 + [0.9] * 19, index=idx, dtype=float)

    events = detect_consensus_change_points(rl, p_high, window_days=5)
    assert events.any()
    fire_idx = int(np.where(events.values)[0][0])
    # Anchored at the later of (10, 11) = 11.
    assert fire_idx == 11


def test_bocpd_only_fire_does_not_trigger_consensus():
    n = 30
    idx = pd.RangeIndex(n)
    rl = pd.Series([100] * 10 + [1] * 20, index=idx, dtype=float)
    # Markov stays low everywhere.
    p_high = pd.Series([0.1] * n, index=idx, dtype=float)

    events = detect_consensus_change_points(rl, p_high, window_days=5)
    assert not events.any()


def test_markov_only_fire_does_not_trigger_consensus():
    n = 30
    idx = pd.RangeIndex(n)
    # BOCPD never resets.
    rl = pd.Series(np.arange(1, n + 1), index=idx, dtype=float)
    p_high = pd.Series([0.1] * 10 + [0.9] * 20, index=idx, dtype=float)

    events = detect_consensus_change_points(rl, p_high, window_days=5)
    assert not events.any()


def test_fires_outside_window_do_not_trigger_consensus():
    """BOCPD fires at t=10, Markov fires at t=20. Window is 5 days
    so they should NOT consense."""
    n = 30
    idx = pd.RangeIndex(n)
    rl = pd.Series([100] * 10 + [1] + [2, 3, 4, 5, 6, 7, 8, 9, 10] +
                   [11] * 10, index=idx, dtype=float)
    p_high = pd.Series([0.1] * 20 + [0.9] * 10, index=idx, dtype=float)

    events = detect_consensus_change_points(rl, p_high, window_days=5)
    assert not events.any()


def test_cooldown_suppresses_subsequent_events():
    """Two consensus opportunities 7 days apart with cooldown=10 should
    fire only once. First opportunity anchors at index 10; second
    opportunity would anchor at index 17 (within cooldown), so it must
    be suppressed."""
    n = 50
    idx = pd.RangeIndex(n)
    # BOCPD fires at t=10 (rl drops 100->1) and again at t=17.
    rl_vals = (
        [100] * 10
        + [1] * 7
        + [80] + [1] * 32
    )
    # Markov P(high) jumps at t=11 then again at t=18.
    p_high_vals = (
        [0.1] * 11 + [0.9] * 7 + [0.1] + [0.9] * 31
    )
    rl = pd.Series(rl_vals, index=idx, dtype=float)
    p_high = pd.Series(p_high_vals, index=idx, dtype=float)

    events = detect_consensus_change_points(
        rl, p_high, window_days=5, cooldown_days=15
    )
    # Exactly one event; the cooldown of 15 days swallows the second
    # opportunity (which would have anchored ~10 days after the first).
    assert events.sum() == 1, (
        f"Expected exactly 1 event after cooldown; got {events.sum()} "
        f"at indices {list(np.where(events)[0])}"
    )


def test_invalid_args_raise():
    rl = pd.Series([1.0])
    p = pd.Series([0.5])
    with pytest.raises(ValueError):
        detect_consensus_change_points(rl, p, window_days=-1)
    with pytest.raises(ValueError):
        detect_consensus_change_points(rl, p, prob_high_jump_threshold=0.0)
    with pytest.raises(ValueError):
        detect_consensus_change_points(rl, p, prob_high_jump_threshold=1.0)
    with pytest.raises(ValueError):
        detect_consensus_change_points(rl, p, rl_drop_threshold=0)


def test_misaligned_inputs_raise():
    rl = pd.Series([1.0, 2.0], index=[0, 1])
    p = pd.Series([0.5, 0.5], index=[0, 2])
    with pytest.raises(ValueError, match="aligned"):
        detect_consensus_change_points(rl, p)


# -- apply_consensus_overlay -----------------------------------------------

def test_overlay_requires_bocpd_and_markov_columns():
    df = pd.DataFrame({"Close": [1.0, 2.0, 3.0]})
    with pytest.raises(ValueError, match="apply_consensus_overlay requires"):
        apply_consensus_overlay(df)


def test_overlay_adds_consensus_event_column():
    n = 30
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    df = pd.DataFrame(
        {
            "bocpd_run_length": [100] * 10 + [1] * 20,
            "markov_prob_high": [0.1] * 11 + [0.9] * 19,
        },
        index=idx,
    )
    out = apply_consensus_overlay(df, window_days=5)
    assert "consensus_event" in out.columns
    assert out["consensus_event"].dtype == bool
    assert out["consensus_event"].sum() == 1


# -- Wiring through classify_regimes ---------------------------------------

def _synthetic_ohlcv(n: int = 600, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    half = n // 2
    rets = np.concatenate(
        [
            rng.normal(0.0, 0.005, half),
            rng.normal(0.0, 0.025, n - half),
        ]
    )
    close = 100.0 * np.exp(np.cumsum(rets))
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    return pd.DataFrame(
        {
            "Open": close, "High": close * 1.005, "Low": close * 0.995,
            "Close": close, "Volume": rng.integers(1_000_000, 5_000_000, n),
        },
        index=dates,
    )


def test_classify_regimes_consensus_auto_enables_dependencies():
    """Setting consensus_overlay=True should auto-enable bocpd and
    markov so the resulting df has all four columns."""
    pytest.importorskip("hmmlearn")
    sys = JuliaMandelbrotSystem()
    sys.df = _synthetic_ohlcv()
    sys.ticker = "TEST"
    sys.compute_features()
    out = sys.classify_regimes(use_fuzzy=False, consensus_overlay=True)
    expected = {
        "bocpd_run_length", "bocpd_change_prob",
        "markov_prob_high", "markov_state",
        "consensus_event",
    }
    assert expected.issubset(set(out.columns))


def test_consensus_via_config_default_off():
    cfg = JMSConfig()
    assert cfg.consensus_overlay is False
    d = cfg.to_dict()
    assert d["consensus_overlay"] is False
    assert d["consensus_window_days"] == 5
    assert d["consensus_cooldown_days"] == 10


# -- CLI ----------------------------------------------------------------

class _StubSource:
    def __init__(self, df: pd.DataFrame):
        self._df = df

    def fetch_data(self, symbol, period=None, start=None, end=None, interval="1d"):
        return self._df.copy()


@pytest.fixture
def stub_fetcher_factory():
    df = _synthetic_ohlcv()
    with patch.object(
        cli_main.DataFetcherFactory, "create", return_value=_StubSource(df)
    ):
        yield df


def test_cli_parser_accepts_consensus_flags():
    argv = [
        "main.py", "GC=F", "--no-plot", "--consensus-overlay",
        "--consensus-window-days", "7", "--consensus-cooldown-days", "15",
    ]
    with patch.object(sys, "argv", argv):
        args = cli_main.parse_arguments()
    assert args.consensus_overlay is True
    assert args.consensus_window_days == 7
    assert args.consensus_cooldown_days == 15


def test_cli_consensus_default_off():
    argv = ["main.py", "AAPL", "--no-plot"]
    with patch.object(sys, "argv", argv):
        args = cli_main.parse_arguments()
    assert args.consensus_overlay is False


def test_run_analysis_consensus_produces_event_column(stub_fetcher_factory):
    pytest.importorskip("hmmlearn")
    artefacts = cli_main.run_analysis(
        symbol="GC=F", period="2y", use_fuzzy=False,
        consensus_overlay=True,
    )
    df = artefacts["df"]
    assert "consensus_event" in df.columns
    assert any("consensus" in tag for tag in artefacts["overlays_used"])


def test_main_entry_point_with_consensus(stub_fetcher_factory):
    pytest.importorskip("hmmlearn")
    argv = [
        "main.py", "GC=F",
        "--no-plot", "--no-fuzzy",
        "--consensus-overlay",
    ]
    buf = io.StringIO()
    with patch.object(sys, "argv", argv), redirect_stdout(buf):
        cli_main.main()
    out = buf.getvalue()
    assert "Consensus:" in out


# -- Stop condition regression --------------------------------------------

def test_consensus_fires_exactly_once_on_gold_like_fixture():
    """The Phase 3 stop condition: on the gold-like fixture used by the
    DSM-BOCPD regression tests, consensus must fire exactly once and
    that fire must be near the crash (within +/- 10 days)."""
    pytest.importorskip("hmmlearn")
    from juliams.regimes.dsm_bocpd import detect_change_points_dsm_bocpd
    from juliams.regimes.markov import fit_multivariate_markov_regime

    rng = np.random.default_rng(0)
    n_pre, n_post, sigma = 260, 240, 0.012
    pre_r = rng.normal(0.0005, sigma, n_pre)
    crash_r = np.array([-10.0 * sigma])
    post_r = rng.normal(-0.0002, sigma * 1.2, n_post - 1)
    pre_iv = rng.normal(0.16, 0.01, n_pre)
    crash_iv = np.array([0.40])
    post_iv = rng.normal(0.30, 0.02, n_post - 1)

    df = pd.DataFrame(
        {
            "log_return": np.concatenate([pre_r, crash_r, post_r]),
            "implied_vol": np.concatenate([pre_iv, crash_iv, post_iv]),
        }
    )
    ret = df["log_return"]
    varx = float(ret.var())
    bocpd = detect_change_points_dsm_bocpd(
        ret, expected_run_length=100, varx=varx,
        omega=1.0, robustness_bandwidth=3.0,
    )
    multi = fit_multivariate_markov_regime(df)
    events = detect_consensus_change_points(
        bocpd.map_run_length, multi.filtered_prob_high,
        window_days=5, cooldown_days=10,
    )
    n_events = int(events.sum())
    assert n_events == 1, f"Expected exactly 1 consensus event, got {n_events}"
    fire_idx = int(np.where(events.values)[0][0])
    crash_idx = 260
    assert abs(fire_idx - crash_idx) <= 10, (
        f"Consensus event at index {fire_idx} is more than 10 days from "
        f"the crash at {crash_idx}"
    )
