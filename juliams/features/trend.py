"""
Trend analysis features for the Julia Mandelbrot System.
Computes rolling OLS slope on log-prices and normalized trend strength.
"""

import numpy as np
import pandas as pd
from typing import Optional, Union
import logging

logger = logging.getLogger(__name__)


def ols_slope(y: np.ndarray) -> float:
    """
    Calculate the OLS (Ordinary Least Squares) slope for a given array.
    
    Args:
        y: Array of values (e.g., log prices)
    
    Returns:
        Slope coefficient from linear regression
    """
    if len(y) < 2:
        return np.nan
    
    x = np.arange(len(y))
    # Remove any NaN values
    mask = ~np.isnan(y)
    if mask.sum() < 2:
        return np.nan
    
    x_clean = x[mask]
    y_clean = y[mask]
    
    # Calculate slope using numpy polyfit (degree 1 for linear)
    try:
        slope, _ = np.polyfit(x_clean, y_clean, 1)
        return slope
    except:
        return np.nan


def compute_rolling_ols_slope(df: pd.DataFrame, 
                             window: int = 20,
                             price_col: str = 'Close') -> pd.Series:
    """
    Compute rolling OLS slope on log-prices.
    
    Args:
        df: DataFrame with price data
        window: Rolling window size in periods
        price_col: Column name for price data
    
    Returns:
        Series with rolling OLS slope values
    """
    if price_col not in df.columns:
        raise ValueError(f"Column '{price_col}' not found in DataFrame")
    
    # Calculate log price if not already present
    if 'log_price' not in df.columns:
        log_price = np.log(df[price_col])
    else:
        log_price = df['log_price']
    
    # Calculate rolling slope
    slope = log_price.rolling(window=window, min_periods=max(2, window//2)).apply(
        ols_slope, raw=True
    )
    
    return slope


def compute_trend_strength(df: pd.DataFrame,
                          window: int = 20,
                          volatility_col: Optional[str] = None,
                          price_col: str = 'Close') -> pd.Series:
    """
    Compute normalized trend strength (slope divided by volatility).
    
    This creates a dimensionless indicator of trend direction and strength,
    similar to a t-statistic or Sharpe ratio of the trend.
    
    Args:
        df: DataFrame with price and volatility data
        window: Rolling window size for trend calculation
        volatility_col: Column name for volatility (if pre-computed)
        price_col: Column name for price data
    
    Returns:
        Series with normalized trend strength values
    """
    # Compute rolling OLS slope
    slope = compute_rolling_ols_slope(df, window, price_col)
    
    # Get or compute volatility for normalization
    if volatility_col and volatility_col in df.columns:
        volatility = df[volatility_col]
    else:
        # Compute volatility from log returns
        if 'log_return' not in df.columns:
            if 'log_price' not in df.columns:
                log_price = np.log(df[price_col])
            else:
                log_price = df['log_price']
            log_return = log_price.diff()
        else:
            log_return = df['log_return']
        
        volatility = log_return.rolling(window=window, min_periods=max(2, window//2)).std()
    
    # Normalize slope by volatility to get trend strength
    # Avoid division by zero or very small volatility
    min_vol = 1e-8
    trend_strength = slope / np.maximum(volatility, min_vol)
    
    # Handle infinite values
    trend_strength = trend_strength.replace([np.inf, -np.inf], np.nan)
    
    return trend_strength


def classify_trend_regime(trend_strength: Union[float, pd.Series],
                         threshold_up: float = 0.2,
                         threshold_down: float = -0.2) -> Union[str, pd.Series]:
    """
    Classify trend regime based on trend strength.
    
    Args:
        trend_strength: Trend strength value(s)
        threshold_up: Threshold for uptrend classification
        threshold_down: Threshold for downtrend classification
    
    Returns:
        Trend regime classification: 'Up', 'Down', or 'Sideways'
    """
    def classify_single(value):
        if pd.isna(value):
            return 'Unknown'
        elif value > threshold_up:
            return 'Up'
        elif value < threshold_down:
            return 'Down'
        else:
            return 'Sideways'
    
    if isinstance(trend_strength, pd.Series):
        return trend_strength.apply(classify_single)
    else:
        return classify_single(trend_strength)


def compute_trend_features(df: pd.DataFrame,
                          config: Optional[dict] = None) -> pd.DataFrame:
    """
    Compute all trend-related features and add them to the DataFrame.
    
    Args:
        df: DataFrame with price data
        config: Configuration dictionary with parameters
    
    Returns:
        DataFrame with added trend features
    """
    if config is None:
        config = {
            'trend_window': 20,
            'trend_threshold_up': 0.2,
            'trend_threshold_down': -0.2
        }
    
    df = df.copy()
    
    # Ensure we have log price and log returns
    if 'log_price' not in df.columns:
        df['log_price'] = np.log(df['Close'])
    if 'log_return' not in df.columns:
        df['log_return'] = df['log_price'].diff()
    
    # Compute rolling OLS slope
    df['slope'] = compute_rolling_ols_slope(
        df, 
        window=config.get('trend_window', 20)
    )
    
    # Compute trend strength (normalized slope)
    df['trend_strength'] = compute_trend_strength(
        df,
        window=config.get('trend_window', 20)
    )
    
    # Classify trend regime
    df['trend_regime'] = classify_trend_regime(
        df['trend_strength'],
        threshold_up=config.get('trend_threshold_up', 0.2),
        threshold_down=config.get('trend_threshold_down', -0.2)
    )
    
    # Add additional trend metrics
    # Moving average of trend strength for smoothing
    df['trend_strength_ma'] = df['trend_strength'].rolling(
        window=5, min_periods=1
    ).mean()
    
    # Trend acceleration (change in slope)
    df['trend_acceleration'] = df['slope'].diff()
    
    # Trend consistency (how many of last N periods had same trend direction)
    window = config.get('trend_window', 20)
    # Convert trend regime to numeric for rolling calculation
    trend_map = {'Up': 1, 'Down': -1, 'Sideways': 0, 'Unknown': np.nan}
    trend_numeric = df['trend_regime'].map(trend_map)
    
    def calc_consistency(x):
        # Remove NaN values
        valid = x[~np.isnan(x)]
        if len(valid) == 0:
            return np.nan
        # Check consistency with the last value
        last_val = valid[-1]
        return (valid == last_val).sum() / len(valid)
    
    df['trend_consistency'] = trend_numeric.rolling(
        window=window, min_periods=1
    ).apply(calc_consistency, raw=True)
    
    logger.info(f"Computed trend features with window={config.get('trend_window', 20)}")
    
    return df


def get_trend_summary(df: pd.DataFrame) -> dict:
    """
    Get summary statistics for trend features.
    
    Args:
        df: DataFrame with trend features
    
    Returns:
        Dictionary with trend summary statistics
    """
    summary = {}
    
    if 'trend_strength' in df.columns:
        summary['trend_strength'] = {
            'mean': df['trend_strength'].mean(),
            'std': df['trend_strength'].std(),
            'min': df['trend_strength'].min(),
            'max': df['trend_strength'].max(),
            'current': df['trend_strength'].iloc[-1] if len(df) > 0 else None
        }
    
    if 'trend_regime' in df.columns:
        regime_counts = df['trend_regime'].value_counts()
        total = len(df[df['trend_regime'] != 'Unknown'])
        summary['trend_regime_distribution'] = {
            regime: count / total if total > 0 else 0
            for regime, count in regime_counts.items()
            if regime != 'Unknown'
        }
        summary['current_trend_regime'] = df['trend_regime'].iloc[-1] if len(df) > 0 else None
    
    if 'trend_consistency' in df.columns:
        summary['average_trend_consistency'] = df['trend_consistency'].mean()
    
    return summary
