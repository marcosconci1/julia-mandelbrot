"""
Data fetching module for the Julia Mandelbrot System.
Handles data ingestion from Yahoo Finance API with error handling and caching.
"""

import logging
from datetime import datetime, timedelta
from typing import Optional, Union, List, Dict
import numpy as np
import pandas as pd
import yfinance as yf
from pathlib import Path
import pickle
import hashlib

from .config import JMSConfig, DEFAULT_CONFIG

# Set up logging
logger = logging.getLogger(__name__)


class DataFetcher:
    """
    Fetches and manages historical stock price data from Yahoo Finance.
    Supports caching to minimize API calls and handles missing data.
    """
    
    def __init__(self, config: Optional[JMSConfig] = None, cache_dir: Optional[str] = None):
        """
        Initialize the DataFetcher.
        
        Args:
            config: Configuration object with data settings
            cache_dir: Directory for caching fetched data
        """
        self.config = config or DEFAULT_CONFIG
        self.cache_dir = Path(cache_dir) if cache_dir else Path.home() / ".juliams_cache"
        
        if self.config.cache_data:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_cache_key(self, ticker: str, start: Optional[str], end: Optional[str], 
                      period: Optional[str]) -> str:
        """Generate a unique cache key for the data request."""
        key_str = f"{ticker}_{start}_{end}_{period}"
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _load_from_cache(self, cache_key: str) -> Optional[pd.DataFrame]:
        """Load data from cache if available and not expired."""
        if not self.config.cache_data:
            return None
        
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        if cache_file.exists():
            # Check if cache is less than 24 hours old
            cache_age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
            if cache_age < timedelta(hours=24):
                try:
                    with open(cache_file, 'rb') as f:
                        data = pickle.load(f)
                    logger.info(f"Loaded data from cache: {cache_key}")
                    return data
                except Exception as e:
                    logger.warning(f"Failed to load cache: {e}")
        return None
    
    def _save_to_cache(self, cache_key: str, data: pd.DataFrame):
        """Save data to cache."""
        if not self.config.cache_data:
            return
        
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(data, f)
            logger.info(f"Saved data to cache: {cache_key}")
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")
    
    def fetch_data(self, 
                   ticker: str,
                   start: Optional[Union[str, datetime]] = None,
                   end: Optional[Union[str, datetime]] = None,
                   period: Optional[str] = None,
                   interval: str = "1d") -> pd.DataFrame:
        """
        Fetch historical stock data from Yahoo Finance.
        
        Args:
            ticker: Stock ticker symbol (e.g., 'AAPL', 'MSFT')
            start: Start date for data (YYYY-MM-DD format or datetime)
            end: End date for data (YYYY-MM-DD format or datetime)
            period: Alternative to start/end (e.g., '1y', '2y', 'max')
            interval: Data frequency ('1d' for daily, '1h' for hourly, etc.)
        
        Returns:
            DataFrame with OHLCV data indexed by date
        
        Raises:
            ValueError: If ticker is invalid or data cannot be fetched
        """
        # Use default period if no dates specified
        if start is None and end is None and period is None:
            period = self.config.default_period
            logger.info(f"Using default period: {period}")
        
        # Check cache first
        cache_key = self._get_cache_key(ticker, str(start), str(end), period)
        cached_data = self._load_from_cache(cache_key)
        if cached_data is not None:
            return cached_data
        
        try:
            # Fetch data from Yahoo Finance
            logger.info(f"Fetching data for {ticker} from Yahoo Finance...")
            stock = yf.Ticker(ticker)
            
            if period:
                df = stock.history(period=period, interval=interval)
            else:
                df = stock.history(start=start, end=end, interval=interval)
            
            if df.empty:
                raise ValueError(f"No data available for ticker {ticker}")
            
            # Clean and prepare the data
            df = self._clean_data(df)
            
            # Add log price and log returns
            df['log_price'] = np.log(df['Close'])
            df['log_return'] = df['log_price'].diff()
            df['return'] = df['Close'].pct_change()
            
            # Save to cache
            self._save_to_cache(cache_key, df)
            
            logger.info(f"Successfully fetched {len(df)} rows of data for {ticker}")
            return df
            
        except Exception as e:
            logger.error(f"Failed to fetch data for {ticker}: {e}")
            raise ValueError(f"Failed to fetch data for {ticker}: {e}")
    
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and prepare the fetched data.
        
        Args:
            df: Raw DataFrame from yfinance
        
        Returns:
            Cleaned DataFrame
        """
        # Remove any duplicate indices
        df = df[~df.index.duplicated(keep='first')]
        
        # Sort by date
        df = df.sort_index()
        
        # Handle missing data based on configuration
        if self.config.fill_missing_data == 'ffill':
            df = df.ffill()
        elif self.config.fill_missing_data == 'drop':
            df = df.dropna()
        elif self.config.fill_missing_data == 'interpolate':
            df = df.interpolate(method='linear')
        
        # Ensure we have the essential columns
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in required_columns:
            if col not in df.columns:
                logger.warning(f"Missing column {col} in data")
                if col == 'Volume':
                    df[col] = 0  # Volume can be zero for some instruments
                else:
                    # For price columns, forward fill or use Close
                    df[col] = df.get(col, df['Close'])
        
        # Log any remaining NaN values
        nan_counts = df[required_columns].isna().sum()
        if nan_counts.any():
            logger.warning(f"NaN values remaining after cleaning: {nan_counts[nan_counts > 0].to_dict()}")
        
        return df
    
    def fetch_multiple(self, 
                      tickers: List[str],
                      start: Optional[Union[str, datetime]] = None,
                      end: Optional[Union[str, datetime]] = None,
                      period: Optional[str] = None,
                      interval: str = "1d") -> Dict[str, pd.DataFrame]:
        """
        Fetch data for multiple tickers.
        
        Args:
            tickers: List of ticker symbols
            start: Start date for data
            end: End date for data
            period: Alternative to start/end
            interval: Data frequency
        
        Returns:
            Dictionary mapping ticker to DataFrame
        """
        results = {}
        
        for ticker in tickers:
            try:
                results[ticker] = self.fetch_data(ticker, start, end, period, interval)
            except Exception as e:
                logger.error(f"Failed to fetch {ticker}: {e}")
                continue
        
        return results
    
    def get_latest_price(self, ticker: str) -> Optional[float]:
        """
        Get the latest closing price for a ticker.
        
        Args:
            ticker: Stock ticker symbol
        
        Returns:
            Latest closing price or None if unavailable
        """
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            return info.get('regularMarketPrice', info.get('previousClose'))
        except Exception as e:
            logger.error(f"Failed to get latest price for {ticker}: {e}")
            return None
    
    def get_ticker_info(self, ticker: str) -> dict:
        """
        Get detailed information about a ticker.
        
        Args:
            ticker: Stock ticker symbol
        
        Returns:
            Dictionary with ticker information
        """
        try:
            stock = yf.Ticker(ticker)
            return stock.info
        except Exception as e:
            logger.error(f"Failed to get info for {ticker}: {e}")
            return {}
    
    def clear_cache(self):
        """Clear all cached data."""
        if self.cache_dir.exists():
            for cache_file in self.cache_dir.glob("*.pkl"):
                cache_file.unlink()
            logger.info("Cache cleared")


def fetch_data(ticker: str,
               start: Optional[Union[str, datetime]] = None,
               end: Optional[Union[str, datetime]] = None,
               period: Optional[str] = None,
               config: Optional[JMSConfig] = None) -> pd.DataFrame:
    """
    Convenience function to fetch data without instantiating DataFetcher.
    
    Args:
        ticker: Stock ticker symbol
        start: Start date for data
        end: End date for data
        period: Alternative to start/end (e.g., '1y', '2y', 'max')
        config: Configuration object
    
    Returns:
        DataFrame with OHLCV data
    """
    fetcher = DataFetcher(config)
    return fetcher.fetch_data(ticker, start, end, period)
