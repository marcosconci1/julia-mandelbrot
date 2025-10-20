"""
Base abstractions and shared utilities for data sources.
"""

from __future__ import annotations

import abc
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from ..config import JMSConfig, DEFAULT_CONFIG

logger = logging.getLogger(__name__)


class DataSourceError(Exception):
    """Base exception for data source errors."""


class InvalidSymbolError(DataSourceError):
    """Raised when a symbol does not match the expected format."""


class RateLimitError(DataSourceError):
    """Raised when an API rate limit is exceeded."""


class ExchangeMaintenanceError(DataSourceError):
    """Raised when an exchange is under maintenance."""


class DataNotAvailableError(DataSourceError):
    """Raised when requested data is not available."""


class APIConnectionError(DataSourceError):
    """Raised for network/API connectivity issues."""


class DataSource(abc.ABC):
    """
    Common interface for all data sources.
    """

    def __init__(self, config: Optional[JMSConfig] = None, cache_dir: Optional[str] = None):
        self.config = config or DEFAULT_CONFIG
        self.cache_dir = Path(cache_dir) if cache_dir else Path.home() / ".juliams_cache"
        if self.config.cache_data:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    @abc.abstractmethod
    def _validate_symbol(self, symbol: str) -> bool:
        """Validate the symbol format."""

    @abc.abstractmethod
    def fetch_data(
        self,
        symbol: str,
        start: Optional[str] = None,
        end: Optional[str] = None,
        period: Optional[str] = None,
        interval: str = "1d",
    ) -> pd.DataFrame:
        """Fetch data for a single symbol."""

    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply standard cleaning steps to OHLCV dataframes.
        """
        df = df.copy()

        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

        df = df.sort_index()
        df = df[~df.index.duplicated(keep="first")]

        if self.config.fill_missing_data == "ffill":
            df = df.ffill()
        elif self.config.fill_missing_data == "drop":
            df = df.dropna()
        elif self.config.fill_missing_data == "interpolate":
            df = df.interpolate(method="linear")

        required_columns = ["Open", "High", "Low", "Close", "Volume"]
        for col in required_columns:
            if col not in df.columns:
                logger.warning("Missing column '%s' in data. Filling with fallback.", col)
                if col == "Volume":
                    df[col] = 0
                else:
                    df[col] = df.get("Close", 0)

        return df

    def _timestamp_from_cache(self, cache_path: Path) -> datetime:
        return datetime.fromtimestamp(cache_path.stat().st_mtime)
