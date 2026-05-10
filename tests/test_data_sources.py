import pandas as pd
import pytest

from juliams.config import JMSConfig
from juliams.data.base import DataNotAvailableError
from juliams.data.crypto import BinanceSource
from juliams.data.stock import YahooFinanceSource
from juliams.data.utils import detect_source_type, validate_crypto_symbol


def _no_cache_config() -> JMSConfig:
    config = JMSConfig()
    config.cache_data = False
    return config


def test_validate_crypto_symbol_requires_full_match():
    assert validate_crypto_symbol("BTCUSDT")
    assert not validate_crypto_symbol("BTCUSDTXYZ")


def test_readme_symbol_examples_route_to_expected_sources():
    assert detect_source_type("AAPL") == "stock"
    assert detect_source_type("^GSPC") == "stock"
    assert detect_source_type("BRL=X") == "stock"
    assert detect_source_type("GC=F") == "stock"

    assert detect_source_type("BTCUSDT") == "crypto"
    assert detect_source_type("ETHUSDT") == "crypto"


def test_yahoo_finance_ticker_model_symbols_stay_on_stock_source():
    assert detect_source_type("BTC-USD") == "stock"
    assert detect_source_type("ETH-USD") == "stock"

    source = YahooFinanceSource(config=_no_cache_config())

    assert source._normalize_symbol("BTC-USD") == "BTC-USD"
    assert source._normalize_symbol("EUR-USD") == "EURUSD=X"


def test_yahoo_fetch_multiple_skips_data_source_errors():
    class FailingYahooSource(YahooFinanceSource):
        def fetch_data(self, *args, **kwargs):
            raise DataNotAvailableError("missing")

    source = FailingYahooSource(config=_no_cache_config())

    assert source.fetch_multiple(["AAPL"]) == {}


def test_binance_period_sets_timestamp_range_without_network():
    class RecordingBinanceSource(BinanceSource):
        def _fetch_klines(self, symbol, interval, start_ts, end_ts):
            self.request = {
                "symbol": symbol,
                "interval": interval,
                "start_ts": start_ts,
                "end_ts": end_ts,
            }
            index = pd.date_range("2024-01-01", periods=1, tz="UTC")
            return pd.DataFrame(
                {
                    "Open": [100.0],
                    "High": [101.0],
                    "Low": [99.0],
                    "Close": [100.0],
                    "Volume": [1.0],
                },
                index=index,
            )

    source = RecordingBinanceSource(config=_no_cache_config())
    source.fetch_data("BTCUSDT", period="1mo")

    assert source.request["interval"] == "1h"
    assert source.request["start_ts"] is not None
    assert source.request["end_ts"] is not None
    assert source.request["end_ts"] > source.request["start_ts"]
    assert source.request["end_ts"] - source.request["start_ts"] == pytest.approx(
        30 * 24 * 60 * 60 * 1000,
        rel=0.001,
    )
