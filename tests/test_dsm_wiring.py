"""End-to-end tests for the DSM-BOCPD wiring through overlays, JMSConfig,
classify_regimes, and the CLI."""

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
from juliams.regimes.overlays import apply_bocpd_overlay


def _synthetic_ohlcv(n: int = 600, seed: int = 0) -> pd.DataFrame:
    """OHLCV with a vol regime shift halfway through."""
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
            "Open": close,
            "High": close * 1.005,
            "Low": close * 0.995,
            "Close": close,
            "Volume": rng.integers(1_000_000, 5_000_000, n),
        },
        index=dates,
    )


# -- Overlay function dispatch ---------------------------------------------

def test_apply_bocpd_overlay_standard_path():
    df = _synthetic_ohlcv()
    df["log_return"] = np.log(df["Close"]).diff()
    out = apply_bocpd_overlay(df, method="standard", expected_run_length=100)
    assert "bocpd_run_length" in out.columns
    assert "bocpd_change_prob" in out.columns


def test_apply_bocpd_overlay_dsm_path():
    df = _synthetic_ohlcv()
    df["log_return"] = np.log(df["Close"]).diff()
    out = apply_bocpd_overlay(
        df,
        method="dsm",
        expected_run_length=100,
        omega=1.0,
        robustness_bandwidth=3.0,
    )
    assert "bocpd_run_length" in out.columns
    assert "bocpd_change_prob" in out.columns


def test_apply_bocpd_overlay_dsm_and_standard_produce_different_outputs():
    """On a series with regime shifts the two algorithms should yield
    different MAP run lengths somewhere along the path."""
    df = _synthetic_ohlcv(n=600, seed=42)
    df["log_return"] = np.log(df["Close"]).diff()
    std = apply_bocpd_overlay(df, method="standard", expected_run_length=100)
    dsm = apply_bocpd_overlay(
        df, method="dsm", expected_run_length=100,
        omega=1.0, robustness_bandwidth=3.0,
    )
    diff = (std["bocpd_run_length"] != dsm["bocpd_run_length"]).sum()
    assert diff > 0, (
        "Standard and DSM BOCPD produced identical run-length series; "
        "the two algorithms should differ on regime-shifted synthetic data."
    )


def test_apply_bocpd_overlay_unknown_method_raises():
    df = _synthetic_ohlcv()
    df["log_return"] = np.log(df["Close"]).diff()
    with pytest.raises(ValueError, match="Unknown bocpd method"):
        apply_bocpd_overlay(df, method="bogus")


def test_apply_bocpd_overlay_dsm_uses_sample_var_when_varx_omitted():
    """DSM should derive varx from the returns series when not given."""
    df = _synthetic_ohlcv()
    df["log_return"] = np.log(df["Close"]).diff()
    # Should not raise.
    out = apply_bocpd_overlay(df, method="dsm")
    assert out["bocpd_run_length"].dropna().notna().any()


# -- Config + classify_regimes -----------------------------------------------

def test_config_bocpd_method_default_is_standard():
    cfg = JMSConfig()
    assert cfg.bocpd_method == "standard"


def test_config_to_dict_includes_dsm_fields():
    cfg = JMSConfig()
    cfg.bocpd_method = "dsm"
    cfg.bocpd_omega = 0.8
    cfg.bocpd_robustness_bandwidth = 2.5
    d = cfg.to_dict()
    assert d["bocpd_method"] == "dsm"
    assert d["bocpd_omega"] == 0.8
    assert d["bocpd_robustness_bandwidth"] == 2.5


def _system_with_data(df: pd.DataFrame) -> JuliaMandelbrotSystem:
    sys = JuliaMandelbrotSystem()
    sys.df = df.copy()
    sys.ticker = "TEST"
    return sys


def test_classify_regimes_default_uses_standard_bocpd():
    sys = _system_with_data(_synthetic_ohlcv())
    sys.compute_features()
    # Use bocpd_overlay=True so the column is produced; default method
    # should be 'standard'.
    out = sys.classify_regimes(use_fuzzy=False, bocpd_overlay=True)
    assert "bocpd_run_length" in out.columns


def test_classify_regimes_dsm_method_via_kwarg():
    sys = _system_with_data(_synthetic_ohlcv())
    sys.compute_features()
    out = sys.classify_regimes(
        use_fuzzy=False, bocpd_overlay=True, bocpd_method="dsm"
    )
    assert "bocpd_run_length" in out.columns


def test_classify_regimes_dsm_method_via_config():
    cfg = JMSConfig()
    cfg.bocpd_overlay = True
    cfg.bocpd_method = "dsm"
    sys = JuliaMandelbrotSystem(config=cfg)
    sys.df = _synthetic_ohlcv()
    sys.ticker = "TEST"
    sys.compute_features()
    out = sys.classify_regimes(use_fuzzy=False)
    assert "bocpd_run_length" in out.columns


def test_classify_regimes_kwarg_overrides_config_for_method():
    cfg = JMSConfig()
    cfg.bocpd_overlay = True
    cfg.bocpd_method = "dsm"
    sys = JuliaMandelbrotSystem(config=cfg)
    sys.df = _synthetic_ohlcv()
    sys.ticker = "TEST"
    sys.compute_features()
    # Per-call override back to standard.
    out_std = sys.classify_regimes(use_fuzzy=False, bocpd_method="standard")
    sys.compute_features()
    out_dsm = sys.classify_regimes(use_fuzzy=False, bocpd_method="dsm")
    # The two should not match exactly.
    diff = (
        out_std["bocpd_run_length"].fillna(-1)
        != out_dsm["bocpd_run_length"].fillna(-1)
    ).sum()
    assert diff > 0


# -- CLI parser + run_analysis path ----------------------------------------

class _StubSource:
    def __init__(self, df: pd.DataFrame):
        self._df = df

    def fetch_data(self, symbol, period=None, start=None, end=None):
        return self._df.copy()


@pytest.fixture
def stub_fetcher_factory():
    df = _synthetic_ohlcv()
    with patch.object(
        cli_main.DataFetcherFactory, "create", return_value=_StubSource(df)
    ):
        yield df


def test_cli_parser_accepts_dsm_method():
    argv = [
        "main.py", "AAPL",
        "--no-plot",
        "--bocpd-overlay",
        "--bocpd-method", "dsm",
        "--bocpd-omega", "0.8",
        "--bocpd-robustness-bandwidth", "2.5",
    ]
    with patch.object(sys, "argv", argv):
        args = cli_main.parse_arguments()
    assert args.bocpd_method == "dsm"
    assert args.bocpd_omega == 0.8
    assert args.bocpd_robustness_bandwidth == 2.5


def test_cli_parser_default_bocpd_method_is_standard():
    argv = ["main.py", "AAPL", "--no-plot"]
    with patch.object(sys, "argv", argv):
        args = cli_main.parse_arguments()
    assert args.bocpd_method == "standard"


def test_cli_parser_rejects_unknown_method():
    argv = ["main.py", "AAPL", "--no-plot", "--bocpd-method", "bogus"]
    with patch.object(sys, "argv", argv):
        with pytest.raises(SystemExit):
            cli_main.parse_arguments()


def test_run_analysis_dsm_path_adds_columns(stub_fetcher_factory):
    artefacts = cli_main.run_analysis(
        symbol="AAPL",
        period="2y",
        use_fuzzy=False,
        bocpd_overlay=True,
        bocpd_method="dsm",
    )
    df = artefacts["df"]
    assert "bocpd_run_length" in df.columns
    assert any("bocpd-dsm" in tag for tag in artefacts["overlays_used"])


def test_main_entry_point_with_dsm(stub_fetcher_factory):
    argv = [
        "main.py", "AAPL",
        "--no-plot", "--no-fuzzy",
        "--bocpd-overlay",
        "--bocpd-method", "dsm",
    ]
    buf = io.StringIO()
    with patch.object(sys, "argv", argv), redirect_stdout(buf):
        cli_main.main()
    output = buf.getvalue()
    assert "Adaptive Overlays" in output
    assert "BOCPD:" in output
    assert "bocpd-dsm" in output
