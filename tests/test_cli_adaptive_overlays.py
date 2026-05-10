"""End-to-end CLI tests for adaptive overlay flags.

Mocks the network fetcher so the test runs offline. Verifies:
- Argparse accepts every new flag
- Defaults preserve legacy behaviour (no overlay columns)
- Each individual flag adds the documented columns
- All flags together work
- print_summary surfaces overlay status without crashing
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


def _synthetic_ohlcv(n: int = 600, seed: int = 0) -> pd.DataFrame:
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
    return pd.DataFrame(
        {
            "Open": close * (1 + rng.normal(0, 0.001, n)),
            "High": close * (1 + np.abs(rng.normal(0, 0.005, n))),
            "Low": close * (1 - np.abs(rng.normal(0, 0.005, n))),
            "Close": close,
            "Volume": rng.integers(1_000_000, 5_000_000, n),
        },
        index=dates,
    )


class _StubSource:
    """Quacks like a juliams DataSource: just returns fixed OHLCV."""

    def __init__(self, df: pd.DataFrame):
        self._df = df

    def fetch_data(
        self,
        symbol: str,
        period=None,
        start=None,
        end=None,
    ):
        return self._df.copy()


@pytest.fixture
def stub_fetcher_factory():
    """Patch DataFetcherFactory.create to return a stub source."""
    df = _synthetic_ohlcv()
    with patch.object(
        cli_main.DataFetcherFactory,
        "create",
        return_value=_StubSource(df),
    ):
        yield df


# -- Parser surface ---------------------------------------------------------

def test_parser_accepts_all_adaptive_flags():
    argv = [
        "main.py", "AAPL",
        "--adaptive-thresholds",
        "--adaptive-q-up", "0.75",
        "--adaptive-q-down", "0.25",
        "--adaptive-floor", "0.15",
        "--adaptive-window", "200",
        "--ewma-halflife", "25",
        "--markov-overlay",
        "--bocpd-overlay",
        "--bocpd-expected-run-length", "150",
        "--min-dwell-days", "5",
        "--no-plot",
    ]
    with patch.object(sys, "argv", argv):
        args = cli_main.parse_arguments()
    assert args.adaptive_thresholds is True
    assert args.adaptive_q_up == 0.75
    assert args.adaptive_q_down == 0.25
    assert args.adaptive_floor == 0.15
    assert args.adaptive_window == 200
    assert args.ewma_halflife == 25.0
    assert args.markov_overlay is True
    assert args.bocpd_overlay is True
    assert args.bocpd_expected_run_length == 150.0
    assert args.min_dwell_days == 5


def test_parser_defaults_preserve_legacy_behaviour():
    argv = ["main.py", "AAPL", "--no-plot"]
    with patch.object(sys, "argv", argv):
        args = cli_main.parse_arguments()
    assert args.adaptive_thresholds is False
    assert args.markov_overlay is False
    assert args.bocpd_overlay is False
    assert args.ewma_halflife is None
    assert args.min_dwell_days == 1


# -- run_analysis with overlays --------------------------------------------

def test_run_analysis_default_omits_overlay_columns(stub_fetcher_factory):
    artefacts = cli_main.run_analysis(symbol="AAPL", period="2y", use_fuzzy=False)
    df = artefacts["df"]
    forbidden = {
        "regime_adaptive",
        "trend_strength_ewma",
        "markov_prob_high",
        "markov_state",
        "bocpd_run_length",
        "bocpd_change_prob",
    }
    assert not (forbidden & set(df.columns))
    assert artefacts["overlays_used"] == []


def test_run_analysis_adaptive_thresholds_adds_column(stub_fetcher_factory):
    artefacts = cli_main.run_analysis(
        symbol="AAPL",
        period="2y",
        use_fuzzy=False,
        adaptive_thresholds=True,
    )
    df = artefacts["df"]
    assert "regime_adaptive" in df.columns
    assert "adaptive_thresholds" in artefacts["overlays_used"]


def test_run_analysis_ewma_overlay_adds_column(stub_fetcher_factory):
    artefacts = cli_main.run_analysis(
        symbol="AAPL",
        period="2y",
        use_fuzzy=False,
        ewma_halflife=25.0,
    )
    df = artefacts["df"]
    assert "trend_strength_ewma" in df.columns
    assert any("ewma" in tag for tag in artefacts["overlays_used"])


def test_run_analysis_markov_overlay_adds_columns(stub_fetcher_factory):
    artefacts = cli_main.run_analysis(
        symbol="AAPL",
        period="2y",
        use_fuzzy=False,
        markov_overlay=True,
    )
    df = artefacts["df"]
    assert {"markov_prob_high", "markov_state"} <= set(df.columns)
    assert "markov" in artefacts["overlays_used"][0]


def test_run_analysis_bocpd_overlay_adds_columns(stub_fetcher_factory):
    artefacts = cli_main.run_analysis(
        symbol="AAPL",
        period="2y",
        use_fuzzy=False,
        bocpd_overlay=True,
        bocpd_expected_run_length=150.0,
    )
    df = artefacts["df"]
    assert {"bocpd_run_length", "bocpd_change_prob"} <= set(df.columns)
    assert any("bocpd" in tag for tag in artefacts["overlays_used"])


def test_run_analysis_min_dwell_smooths_markov(stub_fetcher_factory):
    """min_dwell > 1 with markov_overlay enabled must not increase
    the number of state transitions."""
    raw = cli_main.run_analysis(
        symbol="AAPL", period="2y", use_fuzzy=False, markov_overlay=True
    )
    smoothed = cli_main.run_analysis(
        symbol="AAPL",
        period="2y",
        use_fuzzy=False,
        markov_overlay=True,
        min_dwell_days=10,
    )
    raw_changes = (
        raw["df"]["markov_state"].dropna().values[1:]
        != raw["df"]["markov_state"].dropna().values[:-1]
    ).sum()
    smoothed_changes = (
        smoothed["df"]["markov_state"].dropna().values[1:]
        != smoothed["df"]["markov_state"].dropna().values[:-1]
    ).sum()
    assert smoothed_changes <= raw_changes


def test_run_analysis_all_overlays_together(stub_fetcher_factory):
    artefacts = cli_main.run_analysis(
        symbol="AAPL",
        period="2y",
        use_fuzzy=False,
        adaptive_thresholds=True,
        ewma_halflife=20.0,
        markov_overlay=True,
        bocpd_overlay=True,
        min_dwell_days=5,
    )
    df = artefacts["df"]
    expected = {
        "regime",
        "regime_adaptive",
        "trend_strength_ewma",
        "markov_prob_high",
        "markov_state",
        "bocpd_run_length",
        "bocpd_change_prob",
    }
    assert expected <= set(df.columns)
    assert len(artefacts["overlays_used"]) == 4


# -- print_summary smoke tests ---------------------------------------------

def test_print_summary_handles_overlay_artefacts(stub_fetcher_factory):
    """The summary printer must handle overlay columns without crashing
    and must mention overlays in the output."""
    artefacts = cli_main.run_analysis(
        symbol="AAPL",
        period="2y",
        use_fuzzy=False,
        adaptive_thresholds=True,
        markov_overlay=True,
        bocpd_overlay=True,
    )
    buf = io.StringIO()
    with redirect_stdout(buf):
        cli_main.print_summary("AAPL", artefacts)
    output = buf.getvalue()
    assert "Adaptive Overlays" in output
    assert "Adaptive:" in output
    assert "Markov:" in output
    assert "BOCPD:" in output


def test_print_summary_omits_overlay_section_when_no_overlays(stub_fetcher_factory):
    artefacts = cli_main.run_analysis(symbol="AAPL", period="2y", use_fuzzy=False)
    buf = io.StringIO()
    with redirect_stdout(buf):
        cli_main.print_summary("AAPL", artefacts)
    output = buf.getvalue()
    assert "Adaptive Overlays" not in output


# -- End-to-end main() entry point ----------------------------------------

def test_main_entry_point_with_adaptive_flags(stub_fetcher_factory):
    """Run the actual main() with adaptive flags and confirm it exits 0."""
    argv = [
        "main.py",
        "AAPL",
        "--no-plot",
        "--no-fuzzy",
        "--adaptive-thresholds",
        "--markov-overlay",
        "--bocpd-overlay",
        "--min-dwell-days", "5",
    ]
    buf = io.StringIO()
    with patch.object(sys, "argv", argv), redirect_stdout(buf):
        cli_main.main()
    output = buf.getvalue()
    assert "Adaptive Overlays" in output
    assert "Julia Mandelbrot System" in output
