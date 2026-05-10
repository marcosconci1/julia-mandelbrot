"""
Yahoo Finance data source implementation.
"""

from __future__ import annotations

import hashlib
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import yfinance as yf

from .base import (
    APIConnectionError,
    DataNotAvailableError,
    DataSource,
    DataSourceError,
    InvalidSymbolError,
)
from .utils import validate_stock_symbol, yfinance_rate_limit, _CRYPTO_TICKERS

logger = logging.getLogger(__name__)


class YahooFinanceSource(DataSource):
    """Fetch historical market data from Yahoo Finance."""

    def _normalize_symbol(self, symbol: str) -> str:
        symbol_upper = symbol.upper().strip()
        if "-" in symbol_upper:
            parts = symbol_upper.split("-")
            if len(parts) == 2:
                base, quote = parts
                if quote in {"USD", "EUR", "GBP", "JPY", "CHF", "CAD", "AUD", "NZD"}:
                    if base and base not in _CRYPTO_TICKERS:
                        return f"{base}{quote}=X"
        return symbol_upper

    def __init__(self, config=None, cache_dir: Optional[str] = None):
        super().__init__(config=config, cache_dir=cache_dir)
        if self.config.cache_data:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _validate_symbol(self, symbol: str) -> bool:
        if not validate_stock_symbol(symbol):
            raise InvalidSymbolError(f"Invalid stock symbol format: {symbol}")
        return True

    def _cache_key(
        self,
        symbol: str,
        start: Optional[str],
        end: Optional[str],
        period: Optional[str],
        interval: str,
    ) -> str:
        normalized = self._normalize_symbol(symbol)
        key_str = f"yfinance_{normalized}_{start}_{end}_{period}_{interval}"
        return hashlib.md5(key_str.encode()).hexdigest()

    def _load_from_cache(self, cache_key: str) -> Optional[pd.DataFrame]:
        if not self.config.cache_data:
            return None
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        if cache_file.exists():
            cache_age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
            if cache_age < timedelta(hours=24):
                try:
                    return pd.read_pickle(cache_file)
                except Exception as exc:  # pragma: no cover - corrupted cache
                    logger.warning("Failed to load cache %s: %s", cache_key, exc)
        return None

    def _save_to_cache(self, cache_key: str, data: pd.DataFrame) -> None:
        if not self.config.cache_data:
            return
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        try:
            data.to_pickle(cache_file)
        except Exception as exc:  # pragma: no cover
            logger.warning("Failed to save cache %s: %s", cache_file, exc)

    @yfinance_rate_limit
    def fetch_data(
        self,
        symbol: str,
        start: Optional[Union[str, datetime]] = None,
        end: Optional[Union[str, datetime]] = None,
        period: Optional[str] = None,
        interval: str = "1d",
    ) -> pd.DataFrame:
        self._validate_symbol(symbol)

        if start is None and end is None and period is None:
            period = self.config.stock.default_period if self.config and self.config.stock else "2y"

        cache_key = self._cache_key(symbol, str(start), str(end), period, interval)
        cached = self._load_from_cache(cache_key)
        if cached is not None:
            return cached

        normalized_symbol = self._normalize_symbol(symbol)
        if normalized_symbol != symbol:
            logger.info("Normalized Yahoo Finance symbol %s -> %s", symbol, normalized_symbol)

        try:
            logger.info("Fetching %s from Yahoo Finance...", symbol)
            ticker = yf.Ticker(normalized_symbol)
            if period:
                df = ticker.history(period=period, interval=interval)
            else:
                df = ticker.history(start=start, end=end, interval=interval)
        except Exception as exc:
            raise APIConnectionError(f"Failed to fetch data for {symbol}: {exc}") from exc

        if df.empty:
            raise DataNotAvailableError(f"No data available for ticker {symbol}")

        df = self._clean_data(df)
        df["log_price"] = np.log(df["Close"])
        df["log_return"] = df["log_price"].diff()
        df["return"] = df["Close"].pct_change()

        self._save_to_cache(cache_key, df)
        return df

    def fetch_multiple(
        self,
        symbols: List[str],
        start: Optional[Union[str, datetime]] = None,
        end: Optional[Union[str, datetime]] = None,
        period: Optional[str] = None,
        interval: str = "1d",
    ) -> Dict[str, pd.DataFrame]:
        results: Dict[str, pd.DataFrame] = {}
        for symbol in symbols:
            try:
                results[symbol] = self.fetch_data(symbol, start=start, end=end, period=period, interval=interval)
            except DataSourceError as exc:
                logger.error("Failed to fetch %s: %s", symbol, exc)
        return results

    def get_latest_price(self, symbol: str) -> Optional[float]:
        try:
            self._validate_symbol(symbol)
            normalized_symbol = self._normalize_symbol(symbol)
            ticker = yf.Ticker(normalized_symbol)
            info = ticker.fast_info if hasattr(ticker, "fast_info") else ticker.info
            return info.get("last_price") or info.get("regularMarketPrice")
        except Exception as exc:
            logger.error("Failed to get latest price for %s: %s", symbol, exc)
            return None
