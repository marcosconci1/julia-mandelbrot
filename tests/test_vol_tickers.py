"""Tests for the symbol-to-vol-index lookup."""

from __future__ import annotations

import pytest

from juliams.regimes.vol_tickers import VOL_INDEX_BY_PATTERN, auto_vol_channel


def test_gold_maps_to_gvz():
    assert auto_vol_channel("GC=F") == "^GVZ"
    assert auto_vol_channel("GLD") == "^GVZ"
    assert auto_vol_channel("XAUUSD=X") == "^GVZ"


def test_spy_maps_to_vix():
    assert auto_vol_channel("SPY") == "^VIX"
    assert auto_vol_channel("^GSPC") == "^VIX"
    assert auto_vol_channel("ES=F") == "^VIX"


def test_qqq_maps_to_vxn():
    assert auto_vol_channel("QQQ") == "^VXN"


def test_oil_maps_to_ovx():
    assert auto_vol_channel("CL=F") == "^OVX"
    assert auto_vol_channel("USO") == "^OVX"


def test_unknown_symbol_returns_none():
    assert auto_vol_channel("AAPL") is None
    assert auto_vol_channel("MSFT") is None
    assert auto_vol_channel("BTC-USD") is None
    assert auto_vol_channel("EURUSD=X") is None


def test_case_insensitive_and_strips_whitespace():
    assert auto_vol_channel("gc=f") == "^GVZ"
    assert auto_vol_channel(" Spy ") == "^VIX"


def test_empty_symbol_returns_none():
    assert auto_vol_channel("") is None


def test_map_is_a_dict_with_str_values():
    """Sanity: every value should be a yfinance-compatible ticker string."""
    assert isinstance(VOL_INDEX_BY_PATTERN, dict)
    for k, v in VOL_INDEX_BY_PATTERN.items():
        assert isinstance(k, str) and isinstance(v, str)
        # Convention: vol index tickers start with '^'.
        assert v.startswith("^")
