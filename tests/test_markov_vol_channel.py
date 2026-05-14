"""End-to-end tests for the implied-vol auto-fetch wiring through
overlays, JMSConfig, classify_regimes, and the CLI."""

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
from juliams.regimes.overlays import apply_markov_overlay


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
            "Open": close, "High": close * 1.005, "Low": close * 0.995,
            "Close": close, "Volume": rng.integers(1_000_000, 5_000_000, n),
        },
        index=dates,
    )


def _synthetic_vol_series(idx: pd.DatetimeIndex, seed: int = 1) -> pd.Series:
    """Synthetic implied-vol series matching the OHLCV index."""
    rng = np.random.default_rng(seed)
    half = len(idx) // 2
    vol = np.concatenate(
        [
            rng.normal(0.16, 0.01, half),
            rng.normal(0.32, 0.03, len(idx) - half),
        ]
    )
    return pd.Series(vol, index=idx)


# -- Direct overlay function ------------------------------------------------

def test_overlay_with_series_vol_channel_uses_multivariate_path():
    """Pre-aligned Series vol channel should produce different output
    than the univariate path on regime-shift data."""
    pytest.importorskip("hmmlearn")
    df = _synthetic_ohlcv(n=600, seed=0)
    df["log_return"] = np.log(df["Close"]).diff()
    iv = _synthetic_vol_series(df.index)

    uni = apply_markov_overlay(df)
    multi = apply_markov_overlay(df, vol_channel=iv)

    # The probability series must differ.
    diff_count = (
        uni["markov_prob_high"].fillna(-1)
        != multi["markov_prob_high"].fillna(-1)
    ).sum()
    assert diff_count > 0


def test_overlay_falls_back_to_univariate_when_fetch_fails():
    """When auto-fetch returns None (mocked failure), the overlay must
    still produce the standard columns and not crash."""
    df = _synthetic_ohlcv()
    df["log_return"] = np.log(df["Close"]).diff()

    with patch(
        "juliams.regimes.overlays._fetch_vol_series", return_value=None
    ):
        out = apply_markov_overlay(df, vol_channel="^GVZ")
    assert "markov_prob_high" in out.columns
    assert "markov_state" in out.columns


def test_overlay_rejects_invalid_vol_channel_type():
    df = _synthetic_ohlcv()
    df["log_return"] = np.log(df["Close"]).diff()
    with pytest.raises(TypeError, match="vol_channel"):
        apply_markov_overlay(df, vol_channel=12345)


# -- Config + classify_regimes ----------------------------------------------

def test_config_defaults_for_vol_channel_are_off():
    cfg = JMSConfig()
    assert cfg.markov_vol_channel is None
    assert cfg.markov_auto_vol_channel is False


def test_config_to_dict_includes_vol_channel_fields():
    cfg = JMSConfig()
    cfg.markov_vol_channel = "^GVZ"
    cfg.markov_auto_vol_channel = True
    d = cfg.to_dict()
    assert d["markov_vol_channel"] == "^GVZ"
    assert d["markov_auto_vol_channel"] is True


def test_classify_regimes_respects_explicit_vol_channel():
    """When the user passes a Series via the underlying overlay function,
    the multivariate path runs. Through classify_regimes the user passes
    a ticker string; we mock the fetch to return a deterministic Series
    so the test is offline."""
    pytest.importorskip("hmmlearn")
    df = _synthetic_ohlcv()
    iv = _synthetic_vol_series(df.index)

    sys = JuliaMandelbrotSystem()
    sys.df = df.copy()
    sys.ticker = "GC=F"
    sys.compute_features()

    with patch(
        "juliams.regimes.overlays._fetch_vol_series", return_value=iv
    ):
        out = sys.classify_regimes(
            use_fuzzy=False,
            markov_overlay=True,
            markov_vol_channel="^GVZ",
        )
    assert "markov_prob_high" in out.columns


def test_classify_regimes_auto_vol_channel_uses_ticker():
    """With markov_auto_vol_channel=True and self.ticker='GC=F', the
    auto-fetch should call _fetch_vol_series with '^GVZ'."""
    pytest.importorskip("hmmlearn")
    df = _synthetic_ohlcv()
    iv = _synthetic_vol_series(df.index)

    sys = JuliaMandelbrotSystem()
    sys.df = df.copy()
    sys.ticker = "GC=F"
    sys.compute_features()

    with patch(
        "juliams.regimes.overlays._fetch_vol_series", return_value=iv
    ) as mock_fetch:
        sys.classify_regimes(
            use_fuzzy=False,
            markov_overlay=True,
            markov_auto_vol_channel=True,
        )
    mock_fetch.assert_called_once()
    # First positional arg is the ticker.
    args, _ = mock_fetch.call_args
    assert args[0] == "^GVZ"


def test_auto_vol_channel_is_no_op_for_unknown_symbol():
    """Auto on AAPL (no IV index in our map) should leave markov as
    univariate and not call the fetcher."""
    pytest.importorskip("hmmlearn")
    df = _synthetic_ohlcv()

    sys = JuliaMandelbrotSystem()
    sys.df = df.copy()
    sys.ticker = "AAPL"
    sys.compute_features()

    with patch(
        "juliams.regimes.overlays._fetch_vol_series"
    ) as mock_fetch:
        sys.classify_regimes(
            use_fuzzy=False,
            markov_overlay=True,
            markov_auto_vol_channel=True,
        )
    mock_fetch.assert_not_called()


# -- CLI parser + run_analysis ----------------------------------------------

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


def test_cli_parser_accepts_vol_channel_flags():
    argv = [
        "main.py", "GC=F",
        "--no-plot",
        "--markov-overlay",
        "--markov-vol-channel", "^GVZ",
        "--markov-auto-vol-channel",
    ]
    with patch.object(sys, "argv", argv):
        args = cli_main.parse_arguments()
    assert args.markov_vol_channel == "^GVZ"
    assert args.markov_auto_vol_channel is True


def test_cli_defaults_keep_vol_channel_off():
    argv = ["main.py", "AAPL", "--no-plot"]
    with patch.object(sys, "argv", argv):
        args = cli_main.parse_arguments()
    assert args.markov_vol_channel is None
    assert args.markov_auto_vol_channel is False


def test_run_analysis_auto_vol_channel_for_gold(stub_fetcher_factory):
    """End-to-end: run_analysis on GC=F with --markov-auto-vol-channel
    should call the auto-fetch and tag overlays_used with vol info."""
    pytest.importorskip("hmmlearn")
    df = stub_fetcher_factory
    iv = _synthetic_vol_series(df.index)
    with patch(
        "juliams.regimes.overlays._fetch_vol_series", return_value=iv
    ):
        artefacts = cli_main.run_analysis(
            symbol="GC=F",
            period="2y",
            use_fuzzy=False,
            markov_overlay=True,
            markov_auto_vol_channel=True,
        )
    assert "markov_state" in artefacts["df"].columns
    assert any("vol=^GVZ" in tag for tag in artefacts["overlays_used"])


def test_run_analysis_auto_vol_channel_skipped_for_unmapped_symbol(
    stub_fetcher_factory,
):
    """AAPL is not in our vol-ticker map; auto should not request a fetch."""
    with patch(
        "juliams.regimes.overlays._fetch_vol_series"
    ) as mock_fetch:
        artefacts = cli_main.run_analysis(
            symbol="AAPL",
            period="2y",
            use_fuzzy=False,
            markov_overlay=True,
            markov_auto_vol_channel=True,
        )
    mock_fetch.assert_not_called()
    # And the markov overlay still produced columns via univariate path.
    assert "markov_state" in artefacts["df"].columns


def test_main_entry_point_with_auto_vol_channel(stub_fetcher_factory):
    pytest.importorskip("hmmlearn")
    df = stub_fetcher_factory
    iv = _synthetic_vol_series(df.index)
    argv = [
        "main.py", "GC=F",
        "--no-plot", "--no-fuzzy",
        "--markov-overlay",
        "--markov-auto-vol-channel",
    ]
    buf = io.StringIO()
    with patch(
        "juliams.regimes.overlays._fetch_vol_series", return_value=iv
    ), patch.object(sys, "argv", argv), redirect_stdout(buf):
        cli_main.main()
    out = buf.getvalue()
    assert "Adaptive Overlays" in out
    assert "vol=^GVZ" in out
