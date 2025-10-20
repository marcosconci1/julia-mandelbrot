"""
Data module for the Julia Mandelbrot System.
Provides unified data fetching from multiple sources (stocks, crypto).
"""

from typing import Optional, Literal
import logging

from .base import (
    DataSource,
    DataSourceError,
    InvalidSymbolError,
    RateLimitError,
    ExchangeMaintenanceError,
    DataNotAvailableError,
    APIConnectionError,
)
from .stock import YahooFinanceSource
from .crypto import BinanceSource
from .utils import detect_source_type
from ..config import JMSConfig

logger = logging.getLogger(__name__)


class DataFetcherFactory:
    """
    Factory for creating appropriate data source instances.
    Supports automatic source detection based on symbol format.
    """

    @staticmethod
    def create(
        symbol: Optional[str] = None,
        source_type: Optional[Literal["stock", "crypto"]] = None,
        config: Optional[JMSConfig] = None,
        cache_dir: Optional[str] = None,
    ) -> DataSource:
        """
        Create appropriate data source instance.

        Args:
            symbol: Symbol/ticker (used for auto-detection if source_type not specified)
            source_type: Explicit source type ('stock' or 'crypto')
            config: Configuration object
            cache_dir: Cache directory path

        Returns:
            DataSource instance (YahooFinanceSource or BinanceSource)
        """
        if source_type is None and symbol is not None:
            source_type = detect_source_type(symbol)
            logger.info("Auto-detected source type '%s' for symbol '%s'", source_type, symbol)

        source_type = source_type or "stock"

        if source_type == "stock":
            return YahooFinanceSource(config=config, cache_dir=cache_dir)
        if source_type == "crypto":
            return BinanceSource(config=config, cache_dir=cache_dir)

        raise ValueError(f"Unknown source type: {source_type}. Must be 'stock' or 'crypto'.")


class DataFetcher(YahooFinanceSource):
    """
    Backwards compatible DataFetcher class.

    Existing code importing `DataFetcher` will continue to receive
    the Yahoo Finance implementation.
    """

    pass


def fetch_data(
    symbol: str,
    start: Optional[str] = None,
    end: Optional[str] = None,
    period: Optional[str] = None,
    config: Optional[JMSConfig] = None,
    source_type: Optional[Literal["stock", "crypto"]] = None,
) -> "pd.DataFrame":
    """
    Convenience function to fetch data with automatic source detection.

    Args:
        symbol: Ticker/crypto pair to fetch
        start: Start date (YYYY-MM-DD)
        end: End date (YYYY-MM-DD)
        period: Alternative to start/end (e.g., '1y', 'max')
        config: JMSConfig overrides
        source_type: Optionally force source ('stock' or 'crypto')

    Returns:
        pandas.DataFrame with OHLCV data
    """
    import pandas as pd  # Local import to avoid optional dependency at module load

    fetcher = DataFetcherFactory.create(
        symbol=symbol,
        source_type=source_type,
        config=config,
    )
    return fetcher.fetch_data(symbol, start=start, end=end, period=period)


__all__ = [
    "DataFetcherFactory",
    "DataFetcher",
    "fetch_data",
    "DataSource",
    "YahooFinanceSource",
    "BinanceSource",
    "DataSourceError",
    "InvalidSymbolError",
    "RateLimitError",
    "ExchangeMaintenanceError",
    "DataNotAvailableError",
    "APIConnectionError",
    "detect_source_type",
]
