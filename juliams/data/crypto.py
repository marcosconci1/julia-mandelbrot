"""
Binance data source implementation for cryptocurrency pairs.
"""

from __future__ import annotations

import hashlib
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests
import numpy as np

from .base import (
    APIConnectionError,
    DataNotAvailableError,
    DataSource,
    ExchangeMaintenanceError,
    InvalidSymbolError,
    RateLimitError,
)
from .utils import validate_crypto_symbol, retry_on_failure

logger = logging.getLogger(__name__)


class BinanceSource(DataSource):
    BASE_URL = "https://api.binance.com"
    MAX_LIMIT = 1000

    # Mapping from requested period to (interval, days)
    PERIOD_MAPPING: Dict[str, Tuple[str, int]] = {
        "1d": ("1m", 1),
        "5d": ("5m", 5),
        "1mo": ("1h", 30),
        "3mo": ("4h", 90),
        "6mo": ("1d", 180),
        "1y": ("1d", 365),
        "2y": ("1d", 730),
        "5y": ("1d", 1825),
        "max": ("1d", None),  # type: ignore
    }

    def __init__(self, config=None, cache_dir: Optional[str] = None):
        super().__init__(config=config, cache_dir=cache_dir)
        self.session = requests.Session()
        if self.config.cache_data:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _validate_symbol(self, symbol: str) -> bool:
        if not validate_crypto_symbol(symbol):
            raise InvalidSymbolError(
                f"Invalid crypto symbol format: {symbol}. Expected format like BTCUSDT or ETHBUSD."
            )
        return True

    def _cache_key(
        self,
        symbol: str,
        interval: str,
        start: Optional[int],
        end: Optional[int],
    ) -> str:
        return hashlib.md5(f"binance_{symbol}_{interval}_{start}_{end}".encode()).hexdigest()

    def _load_from_cache(self, cache_key: str) -> Optional[pd.DataFrame]:
        if not self.config.cache_data:
            return None
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        if cache_file.exists():
            cache_age = datetime.utcnow() - datetime.utcfromtimestamp(cache_file.stat().st_mtime)
            if cache_age < timedelta(hours=1):
                try:
                    return pd.read_pickle(cache_file)
                except Exception as exc:  # pragma: no cover
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

    def _parse_period(self, period: Optional[str]) -> Tuple[str, Optional[int]]:
        if period and period in self.PERIOD_MAPPING:
            return self.PERIOD_MAPPING[period]
        default = self.config.crypto.default_period if self.config and self.config.crypto else "1y"
        return self.PERIOD_MAPPING.get(default, ("1d", 365))

    @retry_on_failure(max_retries=3, backoff=2.0, exceptions=(requests.RequestException,))
    def _request(self, endpoint: str, params: Dict[str, object]) -> requests.Response:
        url = f"{self.BASE_URL}{endpoint}"
        response = self.session.get(url, params=params, timeout=10)
        if response.status_code == 429:
            raise RateLimitError("Binance rate limit exceeded.")
        if response.status_code == 418:
            raise ExchangeMaintenanceError("Binance indicates IP banned (418).")
        if response.status_code != 200:
            raise APIConnectionError(f"Binance API error: {response.status_code} {response.text}")
        return response

    def _fetch_klines(
        self,
        symbol: str,
        interval: str,
        start_ts: Optional[int],
        end_ts: Optional[int],
    ) -> pd.DataFrame:
        params: Dict[str, object] = {
            "symbol": symbol.upper(),
            "interval": interval,
            "limit": self.MAX_LIMIT,
        }
        if start_ts is not None:
            params["startTime"] = start_ts
        if end_ts is not None:
            params["endTime"] = end_ts

        frames: List[pd.DataFrame] = []
        while True:
            response = self._request("/api/v3/klines", params)
            data = response.json()
            if not data:
                break
            frame = pd.DataFrame(
                data,
                columns=[
                    "open_time",
                    "Open",
                    "High",
                    "Low",
                    "Close",
                    "Volume",
                    "close_time",
                    "QuoteVolume",
                    "Trades",
                    "TakerBaseVolume",
                    "TakerQuoteVolume",
                    "Ignore",
                ],
            )
            frames.append(frame)
            last_open_time = int(data[-1][0])
            params["startTime"] = last_open_time + 1
            if len(data) < self.MAX_LIMIT:
                break

        if not frames:
            raise DataNotAvailableError(f"No data returned for {symbol} interval {interval}")

        df = pd.concat(frames, ignore_index=True)
        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
        df.set_index("open_time", inplace=True)
        numeric_cols = ["Open", "High", "Low", "Close", "Volume"]
        df[numeric_cols] = df[numeric_cols].astype(float)
        df = df.rename(columns={"Open": "Open", "High": "High", "Low": "Low", "Close": "Close", "Volume": "Volume"})
        return df[["Open", "High", "Low", "Close", "Volume"]]

    def fetch_data(
        self,
        symbol: str,
        start: Optional[str] = None,
        end: Optional[str] = None,
        period: Optional[str] = None,
        interval: str = "1d",
    ) -> pd.DataFrame:
        self._validate_symbol(symbol)

        if interval == "1d" and period is not None:
            interval, _ = self._parse_period(period)

        start_ts = int(pd.Timestamp(start).timestamp() * 1000) if start else None
        end_ts = int(pd.Timestamp(end).timestamp() * 1000) if end else None

        cache_key = self._cache_key(symbol, interval, start_ts, end_ts)
        cached = self._load_from_cache(cache_key)
        if cached is not None:
            return cached

        df = self._fetch_klines(symbol, interval, start_ts, end_ts)
        df = self._clean_data(df)
        df["return"] = df["Close"].pct_change()
        df["log_price"] = np.log(df["Close"])
        df["log_return"] = df["log_price"].diff()

        self._save_to_cache(cache_key, df)
        return df

    def get_latest_price(self, symbol: str) -> Optional[float]:
        try:
            response = self._request("/api/v3/ticker/price", {"symbol": symbol.upper()})
            data = response.json()
            return float(data["price"])
        except Exception as exc:
            logger.error("Failed to get latest price for %s: %s", symbol, exc)
            return None
