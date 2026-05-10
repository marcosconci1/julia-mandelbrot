"""Integration tests for JuliaMandelbrotSystem.classify_regimes adaptive wiring.

Backwards-compatibility contract
--------------------------------
Calling ``classify_regimes()`` (or with ``use_fuzzy=True/False``) and no
new flags must produce IDENTICAL output to the pre-adaptive behaviour:
- The ``regime`` column matches the legacy RegimeClassifier output.
- No new columns appear unless explicitly requested.

Opt-in surface
--------------
Each adaptive feature is gated on a config flag (or per-call kwarg):
- ``adaptive_thresholds=True`` → adds ``regime_adaptive`` column
- ``ewma_halflife=...``       → adds ``trend_strength_ewma`` column
- ``markov_overlay=True``     → adds ``markov_prob_high`` & ``markov_state``
- ``bocpd_overlay=True``      → adds ``bocpd_run_length`` & ``bocpd_change_prob``
- ``min_dwell_days>1``        → applied to ``markov_state`` if present
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _synthetic_ohlcv(n: int = 500, seed: int = 0) -> pd.DataFrame:
    """Realistic OHLCV with a vol regime shift halfway through."""
    rng = np.random.default_rng(seed)
    half = n // 2
    rets = np.concatenate(
        [
            rng.normal(0.0005, 0.012, half),
            rng.normal(-0.0002, 0.028, n - half),
        ]
    )
    close = 100.0 * np.exp(np.cumsum(rets))
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    df = pd.DataFrame(
        {
            "Open": close * (1 + rng.normal(0, 0.001, n)),
            "High": close * (1 + np.abs(rng.normal(0, 0.005, n))),
            "Low": close * (1 - np.abs(rng.normal(0, 0.005, n))),
            "Close": close,
            "Volume": rng.integers(1_000_000, 5_000_000, n),
        },
        index=dates,
    )
    return df


def _build_system_with_data(df: pd.DataFrame):
    """Construct a JuliaMandelbrotSystem and inject pre-fetched data,
    bypassing the network DataFetcher."""
    from juliams import JuliaMandelbrotSystem
    sys = JuliaMandelbrotSystem()
    sys.df = df.copy()
    sys.ticker = "TEST"
    return sys


# -- Backwards compatibility ------------------------------------------------

def test_default_call_produces_legacy_regime_column():
    sys = _build_system_with_data(_synthetic_ohlcv())
    sys.compute_features()
    out = sys.classify_regimes(use_fuzzy=False)
    assert "regime" in out.columns
    legal = {
        "Up-LowVol", "Up-HighVol",
        "Sideways-LowVol", "Sideways-HighVol",
        "Down-LowVol", "Down-HighVol",
        "Unknown",
    }
    assert set(out["regime"].dropna().unique()).issubset(legal)


def test_default_call_does_not_add_adaptive_columns():
    sys = _build_system_with_data(_synthetic_ohlcv())
    sys.compute_features()
    out = sys.classify_regimes(use_fuzzy=False)
    forbidden = {
        "regime_adaptive",
        "trend_strength_ewma",
        "markov_prob_high",
        "markov_state",
        "bocpd_run_length",
        "bocpd_change_prob",
    }
    assert not (forbidden & set(out.columns)), (
        f"Default classify_regimes leaked opt-in columns: {forbidden & set(out.columns)}"
    )


# -- Opt-in: adaptive thresholds -------------------------------------------

def test_adaptive_thresholds_flag_adds_regime_adaptive():
    sys = _build_system_with_data(_synthetic_ohlcv())
    sys.compute_features()
    out = sys.classify_regimes(use_fuzzy=False, adaptive_thresholds=True)
    assert "regime_adaptive" in out.columns
    assert set(out["regime_adaptive"].dropna().unique()).issubset(
        {"Up", "Down", "Sideways", "Unknown"}
    )


def test_adaptive_thresholds_does_not_replace_legacy_regime():
    sys = _build_system_with_data(_synthetic_ohlcv())
    sys.compute_features()
    out = sys.classify_regimes(use_fuzzy=False, adaptive_thresholds=True)
    assert "regime" in out.columns  # legacy still there
    assert "regime_adaptive" in out.columns  # new is additive


# -- Opt-in: EWMA halflife --------------------------------------------------

def test_ewma_halflife_flag_adds_trend_strength_ewma():
    sys = _build_system_with_data(_synthetic_ohlcv())
    sys.compute_features()
    out = sys.classify_regimes(use_fuzzy=False, ewma_halflife=20.0)
    assert "trend_strength_ewma" in out.columns
    # Must have non-NaN values past warmup.
    assert out["trend_strength_ewma"].iloc[100:].notna().any()


# -- Opt-in: Markov overlay -------------------------------------------------

def test_markov_overlay_adds_prob_and_state_columns():
    sys = _build_system_with_data(_synthetic_ohlcv())
    sys.compute_features()
    out = sys.classify_regimes(use_fuzzy=False, markov_overlay=True)
    assert "markov_prob_high" in out.columns
    assert "markov_state" in out.columns
    # P(high) should be in [0, 1].
    p = out["markov_prob_high"].dropna()
    assert ((p >= 0) & (p <= 1)).all()
    # The first row's log_return is NaN, so its label is "Unknown".
    assert set(out["markov_state"].dropna().unique()).issubset(
        {"High", "Low", "Unknown"}
    )


def test_min_dwell_smooths_markov_state():
    """When min_dwell_days > 1 and markov_overlay=True, the post-processed
    state should have at most as many transitions as the raw."""
    from juliams.regimes.markov import enforce_min_dwell

    sys = _build_system_with_data(_synthetic_ohlcv(n=600, seed=42))
    sys.compute_features()
    raw_out = sys.classify_regimes(use_fuzzy=False, markov_overlay=True)
    smoothed_out = sys.classify_regimes(
        use_fuzzy=False, markov_overlay=True, min_dwell_days=10
    )
    raw_changes = (
        raw_out["markov_state"].dropna().values[1:]
        != raw_out["markov_state"].dropna().values[:-1]
    ).sum()
    smoothed_changes = (
        smoothed_out["markov_state"].dropna().values[1:]
        != smoothed_out["markov_state"].dropna().values[:-1]
    ).sum()
    assert smoothed_changes <= raw_changes


# -- Opt-in: BOCPD overlay --------------------------------------------------

def test_bocpd_overlay_adds_run_length_and_change_prob_columns():
    sys = _build_system_with_data(_synthetic_ohlcv())
    sys.compute_features()
    out = sys.classify_regimes(use_fuzzy=False, bocpd_overlay=True)
    assert "bocpd_run_length" in out.columns
    assert "bocpd_change_prob" in out.columns
    # change_prob in [0, 1]
    cp = out["bocpd_change_prob"].dropna()
    assert ((cp >= 0) & (cp <= 1)).all()


# -- Combined opt-ins -------------------------------------------------------

def test_all_overlays_can_be_enabled_together():
    sys = _build_system_with_data(_synthetic_ohlcv())
    sys.compute_features()
    out = sys.classify_regimes(
        use_fuzzy=False,
        adaptive_thresholds=True,
        ewma_halflife=20.0,
        markov_overlay=True,
        bocpd_overlay=True,
        min_dwell_days=5,
    )
    expected_columns = {
        "regime", "regime_adaptive", "trend_strength_ewma",
        "markov_prob_high", "markov_state",
        "bocpd_run_length", "bocpd_change_prob",
    }
    missing = expected_columns - set(out.columns)
    assert not missing, f"Missing columns when all overlays enabled: {missing}"


# -- Config-driven defaults -------------------------------------------------

def test_config_flags_propagate_when_kwargs_omitted():
    """If the JMSConfig has adaptive_thresholds=True, calling
    classify_regimes() without kwargs should still produce regime_adaptive."""
    from juliams import JMSConfig, JuliaMandelbrotSystem

    cfg = JMSConfig()
    cfg.adaptive_thresholds = True

    sys = JuliaMandelbrotSystem(config=cfg)
    sys.df = _synthetic_ohlcv().copy()
    sys.ticker = "TEST"
    sys.compute_features()
    out = sys.classify_regimes(use_fuzzy=False)
    assert "regime_adaptive" in out.columns


def test_config_to_dict_round_trip_preserves_adaptive_flags():
    """JMSConfig.to_dict() must include the new adaptive fields so that
    serialised configs (e.g. for export to JSON) survive a round trip."""
    from juliams import JMSConfig

    cfg = JMSConfig()
    cfg.adaptive_thresholds = True
    cfg.ewma_halflife = 25.0
    cfg.markov_overlay = True
    cfg.bocpd_overlay = True
    cfg.min_dwell_days = 7
    cfg.adaptive_q_up = 0.75
    cfg.adaptive_q_down = 0.25

    d = cfg.to_dict()
    assert d["adaptive_thresholds"] is True
    assert d["ewma_halflife"] == 25.0
    assert d["markov_overlay"] is True
    assert d["bocpd_overlay"] is True
    assert d["min_dwell_days"] == 7
    assert d["adaptive_q_up"] == 0.75
    assert d["adaptive_q_down"] == 0.25
    assert d["bocpd_expected_run_length"] == 100.0  # default preserved


def test_kwarg_overrides_config_flag():
    """Per-call kwarg should override the config setting."""
    from juliams import JMSConfig, JuliaMandelbrotSystem

    cfg = JMSConfig()
    cfg.adaptive_thresholds = True

    sys = JuliaMandelbrotSystem(config=cfg)
    sys.df = _synthetic_ohlcv().copy()
    sys.ticker = "TEST"
    sys.compute_features()
    out = sys.classify_regimes(use_fuzzy=False, adaptive_thresholds=False)
    assert "regime_adaptive" not in out.columns
