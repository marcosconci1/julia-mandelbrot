"""
Volatility analysis features for the Julia Mandelbrot System.
Computes rolling volatility, ATR, and volatility regime classification.
"""

import numpy as np
import pandas as pd
from typing import Optional, Union
import logging

logger = logging.getLogger(__name__)


def compute_volatility(df: pd.DataFrame,
                      window: int = 20,
                      method: str = 'std',
                      annualize: bool = False) -> pd.Series:
    """
    Compute rolling volatility using various methods.
    
    Args:
        df: DataFrame with price data
        window: Rolling window size in periods
        method: Method for volatility calculation ('std', 'ewm', 'parkinson')
        annualize: Whether to annualize the volatility (assumes 252 trading days)
    
    Returns:
        Series with volatility values
    """
    if 'log_return' not in df.columns:
        if 'log_price' not in df.columns:
            df['log_price'] = np.log(df['Close'])
        log_returns = df['log_price'].diff()
    else:
        log_returns = df['log_return']
    
    if method == 'std':
        # Standard deviation of log returns
        volatility = log_returns.rolling(window=window, min_periods=max(2, window//2)).std()
    
    elif method == 'ewm':
        # Exponentially weighted moving volatility
        volatility = log_returns.ewm(span=window, min_periods=max(2, window//2)).std()
    
    elif method == 'parkinson':
        # Parkinson volatility estimator using high-low range
        if 'High' in df.columns and 'Low' in df.columns:
            hl_ratio = np.log(df['High'] / df['Low'])
            factor = 1 / (4 * np.log(2))
            volatility = np.sqrt(
                factor * hl_ratio.pow(2).rolling(window=window, min_periods=max(2, window//2)).mean()
            )
        else:
            logger.warning("High/Low columns not found, falling back to standard deviation")
            volatility = log_returns.rolling(window=window, min_periods=max(2, window//2)).std()
    
    else:
        raise ValueError(f"Unknown volatility method: {method}")
    
    # Annualize if requested
    if annualize:
        volatility = volatility * np.sqrt(252)
    
    return volatility


def compute_atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """
    Compute Average True Range (ATR) indicator.
    
    ATR measures volatility by considering the full range of price movement.
    
    Args:
        df: DataFrame with OHLC data
        window: Period for ATR calculation (typically 14)
    
    Returns:
        Series with ATR values
    """
    if not all(col in df.columns for col in ['High', 'Low', 'Close']):
        raise ValueError("DataFrame must contain High, Low, and Close columns")
    
    # Calculate True Range
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    
    # Calculate ATR as exponential moving average of True Range
    atr = true_range.ewm(span=window, min_periods=1).mean()
    
    return atr


def compute_volatility_percentile(volatility: pd.Series,
                                 lookback: int = 252) -> pd.Series:
    """
    Compute the percentile rank of current volatility within a lookback period.
    
    Args:
        volatility: Series of volatility values
        lookback: Number of periods to look back for percentile calculation
    
    Returns:
        Series with percentile values (0-1)
    """
    percentile = volatility.rolling(window=lookback, min_periods=max(20, lookback//10)).apply(
        lambda x: (x.iloc[-1] > x).sum() / len(x) if len(x) > 0 else 0.5,
        raw=False
    )
    return percentile


def compute_volatility_regime(volatility: pd.Series,
                             method: str = 'percentile',
                             threshold: float = 0.67,
                             baseline_window: int = 100) -> pd.Series:
    """
    Classify volatility into High/Low regimes.
    
    Args:
        volatility: Series of volatility values
        method: Method for classification ('percentile', 'absolute', 'adaptive')
        threshold: Threshold for classification
        baseline_window: Window for adaptive baseline calculation
    
    Returns:
        Series with volatility regime classification ('High' or 'Low')
    """
    if method == 'percentile':
        # Use percentile of historical volatility
        vol_percentile = compute_volatility_percentile(volatility, lookback=252)
        regime = vol_percentile.apply(lambda x: 'High' if x > threshold else 'Low')
    
    elif method == 'absolute':
        # Use absolute threshold
        regime = volatility.apply(lambda x: 'High' if x > threshold else 'Low')
    
    elif method == 'adaptive':
        # Use adaptive baseline (rolling mean)
        baseline = volatility.rolling(window=baseline_window, min_periods=max(20, baseline_window//5)).mean()
        regime = pd.Series(
            np.where(volatility > baseline * (1 + (threshold - 0.5)), 'High', 'Low'),
            index=volatility.index
        )
    
    else:
        raise ValueError(f"Unknown volatility regime method: {method}")
    
    # Handle NaN values
    regime = regime.fillna('Unknown')
    
    return regime


def compute_volatility_features(df: pd.DataFrame,
                               config: Optional[dict] = None) -> pd.DataFrame:
    """
    Compute all volatility-related features and add them to the DataFrame.
    
    Args:
        df: DataFrame with price data
        config: Configuration dictionary with parameters
    
    Returns:
        DataFrame with added volatility features
    """
    if config is None:
        config = {
            'volatility_window': 20,
            'volatility_baseline_window': 100,
            'volatility_percentile': 0.67,
            'atr_window': 14
        }
    
    df = df.copy()
    
    # Ensure log returns exist
    if 'log_return' not in df.columns:
        if 'log_price' not in df.columns:
            df['log_price'] = np.log(df['Close'])
        df['log_return'] = df['log_price'].diff()
    
    # Compute basic volatility (realized volatility)
    df['volatility'] = compute_volatility(
        df,
        window=config.get('volatility_window', 20),
        method='std'
    )
    
    # Compute ATR if OHLC data is available
    if all(col in df.columns for col in ['High', 'Low', 'Close']):
        df['atr'] = compute_atr(df, window=config.get('atr_window', 14))
        # Normalized ATR (ATR as percentage of close price)
        df['atr_pct'] = df['atr'] / df['Close'] * 100
    
    # Compute volatility percentile rank
    df['volatility_percentile'] = compute_volatility_percentile(
        df['volatility'],
        lookback=252
    )
    
    # Compute volatility baseline (adaptive)
    baseline_window = config.get('volatility_baseline_window', 100)
    df['volatility_baseline'] = df['volatility'].rolling(
        window=baseline_window,
        min_periods=max(20, baseline_window//5)
    ).mean()
    
    # Classify volatility regime
    df['volatility_regime'] = compute_volatility_regime(
        df['volatility'],
        method='percentile',
        threshold=config.get('volatility_percentile', 0.67)
    )
    
    # Additional volatility metrics
    vol_window = config.get('volatility_window', 20)
    
    # Volatility of volatility (vol of vol)
    df['vol_of_vol'] = df['volatility'].rolling(
        window=vol_window,
        min_periods=max(2, vol_window//2)
    ).std()
    
    # Volatility change rate
    df['volatility_change'] = df['volatility'].pct_change()
    
    # Volatility z-score (standardized volatility)
    vol_mean = df['volatility'].rolling(window=252, min_periods=20).mean()
    vol_std = df['volatility'].rolling(window=252, min_periods=20).std()
    # Avoid division by zero
    vol_std = vol_std.replace(0, np.nan)
    df['volatility_zscore'] = (df['volatility'] - vol_mean) / vol_std
    
    logger.info(f"Computed volatility features with window={config.get('volatility_window', 20)}")
    
    return df


def get_volatility_summary(df: pd.DataFrame) -> dict:
    """
    Get summary statistics for volatility features.
    
    Args:
        df: DataFrame with volatility features
    
    Returns:
        Dictionary with volatility summary statistics
    """
    summary = {}
    
    if 'volatility' in df.columns:
        summary['volatility'] = {
            'mean': df['volatility'].mean(),
            'std': df['volatility'].std(),
            'min': df['volatility'].min(),
            'max': df['volatility'].max(),
            'current': df['volatility'].iloc[-1] if len(df) > 0 else None
        }
    
    if 'volatility_percentile' in df.columns:
        summary['current_volatility_percentile'] = df['volatility_percentile'].iloc[-1] if len(df) > 0 else None
    
    if 'volatility_regime' in df.columns:
        regime_counts = df['volatility_regime'].value_counts()
        total = len(df[df['volatility_regime'] != 'Unknown'])
        summary['volatility_regime_distribution'] = {
            regime: count / total if total > 0 else 0
            for regime, count in regime_counts.items()
            if regime != 'Unknown'
        }
        summary['current_volatility_regime'] = df['volatility_regime'].iloc[-1] if len(df) > 0 else None
    
    if 'atr' in df.columns:
        summary['atr'] = {
            'mean': df['atr'].mean(),
            'current': df['atr'].iloc[-1] if len(df) > 0 else None
        }
    
    if 'vol_of_vol' in df.columns:
        summary['volatility_stability'] = {
            'vol_of_vol_mean': df['vol_of_vol'].mean(),
            'vol_of_vol_current': df['vol_of_vol'].iloc[-1] if len(df) > 0 else None
        }
    
    return summary


def identify_volatility_clusters(df: pd.DataFrame,
                                high_vol_threshold: float = 0.7) -> pd.DataFrame:
    """
    Identify volatility clusters (periods of sustained high/low volatility).
    
    Args:
        df: DataFrame with volatility data
        high_vol_threshold: Percentile threshold for high volatility
    
    Returns:
        DataFrame with volatility cluster information
    """
    if 'volatility_percentile' not in df.columns:
        df['volatility_percentile'] = compute_volatility_percentile(df['volatility'])
    
    # Identify high volatility periods
    df['is_high_vol'] = df['volatility_percentile'] > high_vol_threshold
    
    # Find cluster boundaries
    cluster_changes = df['is_high_vol'].ne(df['is_high_vol'].shift())
    df['cluster_id'] = cluster_changes.cumsum()
    
    # Calculate cluster statistics
    cluster_stats = df.groupby('cluster_id').agg({
        'is_high_vol': 'first',
        'volatility': ['mean', 'max', 'min'],
        'Close': ['first', 'last']
    })
    
    return cluster_stats
