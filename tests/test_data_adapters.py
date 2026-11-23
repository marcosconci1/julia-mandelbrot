"""Offline tests for provider parsing and the on-disk price cache."""

from datetime import datetime, timezone
import os
import time

import pandas as pd
import pytest

from julia_mandelbrot import data_adapters as adapters
from julia_mandelbrot.data_adapters import (
    BinanceAdapter,
    DataAdapterError,
    PriceSeries,
    YahooFinanceAdapter,
)


def _mock_payload(monkeypatch, payload):
    def request(url):
        return payload

    monkeypatch.setattr(adapters, "_http_json", request)


def test_yahoo_maps_daily_ohlcv_and_utc_index(monkeypatch):
    _mock_payload(
        monkeypatch,
        {"chart": {"result": [{"timestamp": [1704067200], "indicators": {"quote": [{
            "open": [10], "high": [12], "low": [9], "close": [11], "volume": [100],
        }]}}], "error": None}},
    )
    series = YahooFinanceAdapter(cache_dir="/tmp/no-cache").fetch("ABC", "2024-01-01", "2024-01-03")
    assert list(series.data.columns) == ["Open", "High", "Low", "Close", "Volume"]
    assert list(series.data.iloc[0]) == [10, 12, 9, 11, 100]
    assert str(series.data.index.tz) == "UTC"
    assert series.source == "yahoo"


def test_binance_maps_klines(monkeypatch):
    _mock_payload(monkeypatch, [[1704067200000, "10", "12", "9", "11", "100", 0]])
    series = BinanceAdapter(cache_dir="/tmp/no-cache").fetch("btcusdt", "2024-01-01", "2024-01-03")
    assert list(series.data.columns) == ["Open", "High", "Low", "Close", "Volume"]
    assert list(series.data.iloc[0]) == [10, 12, 9, 11, 100]
    assert series.source == "binance"


@pytest.mark.parametrize("payload", [{"chart": {"result": None, "error": None}}, {"chart": {"result": [{"timestamp": [], "indicators": {}}], "error": None}}])
def test_yahoo_malformed_or_empty_response_is_clear(monkeypatch, payload):
    _mock_payload(monkeypatch, payload)
    with pytest.raises(DataAdapterError, match="Yahoo"):
        YahooFinanceAdapter(cache_dir="/tmp/no-cache").fetch("ABC", "2024-01-01", "2024-01-03")


def test_binance_malformed_response_is_clear(monkeypatch):
    _mock_payload(monkeypatch, [[1, "bad"]])
    with pytest.raises(DataAdapterError, match="Binance"):
        BinanceAdapter(cache_dir="/tmp/no-cache").fetch("BTCUSDT", "2024-01-01", "2024-01-03")


def test_cache_roundtrip_and_fresh_cache_precedes_network(tmp_path, monkeypatch):
    frame = pd.DataFrame(
        [[1, 2, 0.5, 1.5, 10]],
        columns=["Open", "High", "Low", "Close", "Volume"],
        index=pd.DatetimeIndex([datetime(2024, 1, 1, tzinfo=timezone.utc)], name="Date"),
    )
    adapter = YahooFinanceAdapter(cache_dir=tmp_path)
    calls = []

    def fetch(ticker, start, end, *, interval="1d"):
        calls.append(ticker)
        return PriceSeries(frame, "yahoo")

    monkeypatch.setattr(adapter, "fetch", fetch)
    first = adapter.get_prices("ABC", "2024-01-01", "2024-01-02")
    second = adapter.get_prices("ABC", "2024-01-01", "2024-01-02")
    assert calls == ["ABC"]
    pd.testing.assert_frame_equal(first.data, second.data)
    assert list(tmp_path.glob("*.csv"))


def test_cache_staleness_and_refresh_override(tmp_path, monkeypatch):
    frame = pd.DataFrame([[1, 2, 0.5, 1.5, 10]], columns=["Open", "High", "Low", "Close", "Volume"], index=pd.DatetimeIndex(["2024-01-01"], tz="UTC"))
    adapter = YahooFinanceAdapter(cache_dir=tmp_path, max_age_days=10)
    calls = []

    def fetch(*args, **kwargs):
        calls.append(True)
        return PriceSeries(frame, "yahoo")

    monkeypatch.setattr(adapter, "fetch", fetch)
    adapter.get_prices("ABC", "2024-01-01", "2024-01-02")
    cache_path = next(tmp_path.glob("*.csv"))
    old = time.time() - 3 * 86400
    os.utime(cache_path, (old, old))
    adapter.get_prices("ABC", "2024-01-01", "2024-01-02", max_age_days=1)
    adapter.get_prices("ABC", "2024-01-01", "2024-01-02", refresh=True)
    assert len(calls) == 3


def test_binance_pagination_concatenates_pages_and_stops(monkeypatch):
    base_ms = 1577836800000  # 2020-01-01T00:00:00Z
    day_ms = 86_400_000
    page1 = [
        [base_ms + i * day_ms, "10", "12", "9", str(10 + i), "100", 0]
        for i in range(500)
    ]
    page2 = [
        [base_ms + (500 + i) * day_ms, "20", "22", "19", str(510 + i), "200", 0]
        for i in range(120)
    ]
    pages = [page1, page2]
    urls: list[str] = []

    def request(url: str):
        urls.append(url)
        return pages[len(urls) - 1]

    monkeypatch.setattr(adapters, "_http_json", request)
    series = BinanceAdapter(cache_dir="/tmp/no-cache").fetch("BTCUSDT", "2020-01-01", "2022-01-01")
    assert len(urls) == 2
    expected_second_start = base_ms + 500 * day_ms
    assert f"startTime={expected_second_start}" in urls[1]
    assert len(series.data) == 620
    assert series.data["Close"].iloc[0] == 10.0
    assert series.data["Close"].iloc[499] == 509.0
    assert series.data["Close"].iloc[500] == 510.0
    assert series.data["Close"].iloc[-1] == 629.0


def test_binance_single_short_page_makes_one_request(monkeypatch):
    calls: list[str] = []

    def request(url: str):
        calls.append(url)
        return [
            [1704067200000, "10", "12", "9", "11", "100", 0],
            [1704153600000, "11", "13", "10", "12", "150", 0],
        ]

    monkeypatch.setattr(adapters, "_http_json", request)
    series = BinanceAdapter(cache_dir="/tmp/no-cache").fetch("BTCUSDT", "2024-01-01", "2024-01-10")
    assert len(calls) == 1
    assert len(series.data) == 2


def test_binance_pagination_request_cap_raises(monkeypatch):
    calls = 0
    full_page = [[1577836800000, "10", "12", "9", "11", "100", 0]] * 500

    def request(url: str):
        nonlocal calls
        calls += 1
        return full_page

    monkeypatch.setattr(adapters, "_http_json", request)
    with pytest.raises(DataAdapterError, match="50"):
        BinanceAdapter(cache_dir="/tmp/no-cache").fetch("BTCUSDT", "2020-01-01", "2025-01-01")
    assert calls == 50


def test_yahoo_incomplete_range_coverage_raises(monkeypatch):
    base_ts = 1672531200  # 2023-01-01T00:00:00Z
    short_timestamps = [base_ts + i * 86400 for i in range(10)]
    _mock_payload(
        monkeypatch,
        {
            "chart": {
                "result": [
                    {
                        "timestamp": short_timestamps,
                        "indicators": {
                            "quote": [
                                {
                                    "open": [10.0] * 10,
                                    "high": [12.0] * 10,
                                    "low": [9.0] * 10,
                                    "close": [11.0] * 10,
                                    "volume": [100.0] * 10,
                                }
                            ]
                        },
                    }
                ],
                "error": None,
            }
        },
    )
    with pytest.raises(DataAdapterError, match="Yahoo"):
        YahooFinanceAdapter(cache_dir="/tmp/no-cache").fetch("SPY", "2023-01-01", "2024-01-01")


def test_cache_key_includes_start_and_end_window(tmp_path, monkeypatch):
    adapter = YahooFinanceAdapter(cache_dir=tmp_path)
    calls: list[tuple[str, str, str]] = []

    def fetch(ticker, start, end, *, interval="1d"):
        calls.append((ticker, str(start), str(end)))
        idx = pd.DatetimeIndex([pd.Timestamp(start, tz="UTC")], name="Date")
        frame = pd.DataFrame(
            [[1.0, 2.0, 0.5, float(len(calls)), 10.0]],
            columns=["Open", "High", "Low", "Close", "Volume"],
            index=idx,
        )
        return PriceSeries(frame, "yahoo")

    monkeypatch.setattr(adapter, "fetch", fetch)

    win1_first = adapter.get_prices("ABC", "2022-01-01", "2023-01-01")
    win1_second = adapter.get_prices("ABC", "2022-01-01", "2023-01-01")
    win2 = adapter.get_prices("ABC", "2023-01-01", "2024-01-01")

    assert len(calls) == 2
    assert calls == [
        ("ABC", "2022-01-01", "2023-01-01"),
        ("ABC", "2023-01-01", "2024-01-01"),
    ]
    assert len(list(tmp_path.glob("*.csv"))) == 2
    pd.testing.assert_frame_equal(win1_first.data, win1_second.data)
    assert win1_first.data["Close"].iloc[0] == 1.0
    assert win2.data["Close"].iloc[0] == 2.0


