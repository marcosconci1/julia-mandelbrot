"""
Hurst exponent computation for the Julia Mandelbrot System.
Implements fractal analysis to measure long-term memory in price series.
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple, Union
import logging
import warnings

logger = logging.getLogger(__name__)

# Try to import nolds library for Hurst calculation
try:
    import nolds
    NOLDS_AVAILABLE = True
except ImportError:
    NOLDS_AVAILABLE = False
    logger.warning("nolds library not available. Using fallback Hurst implementation.")


def rescaled_range(ts: np.ndarray) -> float:
    """
    Calculate Hurst exponent using Rescaled Range (R/S) analysis.
    
    This is a fallback implementation when nolds is not available.
    
    Args:
        ts: Time series array
    
    Returns:
        Hurst exponent estimate
    """
    # Ensure we have enough data points
    if len(ts) < 10:
        return np.nan
    
    # Remove NaN values
    ts = ts[~np.isnan(ts)]
    if len(ts) < 10:
        return np.nan
    
    lags = range(2, min(100, len(ts)//2))
    tau = [np.sqrt(np.std(np.subtract(ts[lag:], ts[:-lag]))) for lag in lags]
    
    # Avoid log of zero or negative values
    valid_indices = [(i, t) for i, t in enumerate(tau) if t > 0]
    if len(valid_indices) < 2:
        return np.nan
    
    indices, valid_tau = zip(*valid_indices)
    
    # Perform linear regression on log-log plot
    try:
        poly = np.polyfit(np.log([lags[i] for i in indices]), np.log(valid_tau), 1)
        return poly[0] * 2.0  # Hurst exponent
    except:
        return np.nan


def hurst_rs(series: Union[np.ndarray, pd.Series], 
             min_window: int = 10,
             max_window: Optional[int] = None) -> float:
    """
    Calculate Hurst exponent using Rescaled Range (R/S) method.
    
    Args:
        series: Price or return series
        min_window: Minimum window size for R/S calculation
        max_window: Maximum window size (default: len(series)//2)
    
    Returns:
        Hurst exponent (0 < H < 1)
        H > 0.5: Persistent/trending
        H < 0.5: Anti-persistent/mean-reverting
        H ≈ 0.5: Random walk
    """
    if isinstance(series, pd.Series):
        series = series.values
    
    # Remove NaN values
    series = series[~np.isnan(series)]
    
    if len(series) < min_window:
        return np.nan
    
    if NOLDS_AVAILABLE:
        try:
            # Use nolds library if available
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                h = nolds.hurst_rs(series, fit='poly')
            return h
        except Exception as e:
            logger.debug(f"nolds hurst_rs failed: {e}, using fallback")
            return rescaled_range(series)
    else:
        # Use fallback implementation
        return rescaled_range(series)


def detrended_fluctuation_analysis(series: np.ndarray, 
                                  min_scale: int = 4,
                                  max_scale: Optional[int] = None) -> float:
    """
    Calculate Hurst exponent using Detrended Fluctuation Analysis (DFA).
    
    DFA is often more robust than R/S for non-stationary series.
    
    Args:
        series: Time series data
        min_scale: Minimum scale for DFA
        max_scale: Maximum scale (default: len(series)//4)
    
    Returns:
        Hurst exponent estimate
    """
    N = len(series)
    if N < min_scale * 2:
        return np.nan
    
    if max_scale is None:
        max_scale = N // 4
    
    # Integrate the series (cumulative sum)
    y = np.cumsum(series - np.mean(series))
    
    # Calculate fluctuation for different scales
    scales = []
    flucts = []
    
    for scale in range(min_scale, min(max_scale, N//2)):
        # Number of segments
        n_segments = N // scale
        if n_segments < 2:
            continue
        
        # Calculate fluctuation for this scale
        fluctuation = 0
        for i in range(n_segments):
            segment = y[i*scale:(i+1)*scale]
            x = np.arange(len(segment))
            
            # Fit linear trend
            if len(segment) > 1:
                coeffs = np.polyfit(x, segment, 1)
                fit = np.polyval(coeffs, x)
                fluctuation += np.sum((segment - fit) ** 2)
        
        if n_segments > 0:
            fluctuation = np.sqrt(fluctuation / (n_segments * scale))
            scales.append(scale)
            flucts.append(fluctuation)
    
    if len(scales) < 2:
        return np.nan
    
    # Fit log-log plot to get Hurst exponent
    try:
        coeffs = np.polyfit(np.log(scales), np.log(flucts), 1)
        return coeffs[0]  # This is the Hurst exponent
    except:
        return np.nan


def compute_rolling_hurst(df: pd.DataFrame,
                         window: int = 100,
                         method: str = 'rs',
                         price_col: str = 'Close',
                         step: int = 1) -> pd.Series:
    """
    Compute rolling Hurst exponent over a specified window.
    
    Args:
        df: DataFrame with price data
        window: Rolling window size
        method: Method for Hurst calculation ('rs' or 'dfa')
        price_col: Column name for price data
        step: Step size for rolling calculation (for performance)
    
    Returns:
        Series with rolling Hurst exponent values
    """
    if price_col not in df.columns:
        raise ValueError(f"Column '{price_col}' not found in DataFrame")
    
    prices = df[price_col]
    hurst_values = pd.Series(index=prices.index, dtype=float)
    
    def calculate_hurst(price_window):
        if len(price_window) < window // 2:
            return np.nan
        
        # Remove NaN values
        clean_prices = price_window.dropna()
        if len(clean_prices) < window // 2:
            return np.nan
        
        if method == 'rs':
            return hurst_rs(clean_prices.values)
        elif method == 'dfa':
            # For DFA, use returns instead of prices
            returns = np.diff(np.log(clean_prices.values))
            if len(returns) < 10:
                return np.nan
            return detrended_fluctuation_analysis(returns)
        else:
            raise ValueError(f"Unknown Hurst method: {method}")
    
    # Calculate rolling Hurst with optional step size for performance
    if step > 1:
        # Calculate only at step intervals and forward fill
        for i in range(window, len(prices), step):
            start_idx = max(0, i - window)
            end_idx = i
            price_window = prices.iloc[start_idx:end_idx]
            hurst_val = calculate_hurst(price_window)
            hurst_values.iloc[i] = hurst_val
        
        # Forward fill the missing values
        hurst_values = hurst_values.ffill()
    else:
        # Standard rolling calculation
        hurst_values = prices.rolling(window=window, min_periods=max(20, window//2)).apply(
            calculate_hurst, raw=False
        )
    
    return hurst_values


def classify_hurst_regime(hurst: Union[float, pd.Series],
                         indeterminate_range: Tuple[float, float] = (0.45, 0.55)) -> Union[str, pd.Series]:
    """
    Classify market behavior based on Hurst exponent.
    
    Args:
        hurst: Hurst exponent value(s)
        indeterminate_range: Range for random walk classification
    
    Returns:
        Classification: 'Trending', 'Mean-Reverting', or 'Random'
    """
    def classify_single(h):
        if pd.isna(h):
            return 'Unknown'
        elif h > indeterminate_range[1]:
            return 'Trending'  # Persistent, long memory
        elif h < indeterminate_range[0]:
            return 'Mean-Reverting'  # Anti-persistent
        else:
            return 'Random'  # Random walk
    
    if isinstance(hurst, pd.Series):
        return hurst.apply(classify_single)
    else:
        return classify_single(hurst)


def compute_hurst_features(df: pd.DataFrame,
                          config: Optional[dict] = None) -> pd.DataFrame:
    """
    Compute all Hurst-related features and add them to the DataFrame.
    
    Args:
        df: DataFrame with price data
        config: Configuration dictionary with parameters
    
    Returns:
        DataFrame with added Hurst features
    """
    if config is None:
        config = {
            'hurst_window': 100,
            'hurst_indeterminate_range': (0.45, 0.55),
            'hurst_method': 'rs'
        }
    
    df = df.copy()
    
    # Compute rolling Hurst exponent
    df['hurst'] = compute_rolling_hurst(
        df,
        window=config.get('hurst_window', 100),
        method=config.get('hurst_method', 'rs')
    )
    
    # Classify Hurst regime
    df['hurst_regime'] = classify_hurst_regime(
        df['hurst'],
        indeterminate_range=config.get('hurst_indeterminate_range', (0.45, 0.55))
    )
    
    # Additional Hurst metrics
    # Hurst deviation from 0.5 (random walk)
    df['hurst_deviation'] = np.abs(df['hurst'] - 0.5)
    
    # Hurst trend (is Hurst increasing or decreasing)
    df['hurst_change'] = df['hurst'].diff()
    
    # Moving average of Hurst for smoothing
    df['hurst_ma'] = df['hurst'].rolling(window=20, min_periods=1).mean()
    
    # Persistence strength (how far from 0.5)
    df['persistence_strength'] = df['hurst'] - 0.5
    
    logger.info(f"Computed Hurst features with window={config.get('hurst_window', 100)}")
    
    return df


def compute_segment_hurst(df: pd.DataFrame,
                         start_idx: int,
                         end_idx: int,
                         price_col: str = 'Close',
                         method: str = 'rs') -> float:
    """
    Compute Hurst exponent for a specific segment of data.
    
    Args:
        df: DataFrame with price data
        start_idx: Start index of segment
        end_idx: End index of segment
        price_col: Column name for price data
        method: Method for Hurst calculation
    
    Returns:
        Hurst exponent for the segment
    """
    segment_prices = df[price_col].iloc[start_idx:end_idx]
    
    if len(segment_prices) < 20:
        return np.nan
    
    if method == 'rs':
        return hurst_rs(segment_prices.values)
    elif method == 'dfa':
        returns = np.diff(np.log(segment_prices.values))
        if len(returns) < 10:
            return np.nan
        return detrended_fluctuation_analysis(returns)
    else:
        raise ValueError(f"Unknown Hurst method: {method}")


def get_hurst_summary(df: pd.DataFrame) -> dict:
    """
    Get summary statistics for Hurst features.
    
    Args:
        df: DataFrame with Hurst features
    
    Returns:
        Dictionary with Hurst summary statistics
    """
    summary = {}
    
    if 'hurst' in df.columns:
        summary['hurst'] = {
            'mean': df['hurst'].mean(),
            'std': df['hurst'].std(),
            'min': df['hurst'].min(),
            'max': df['hurst'].max(),
            'current': df['hurst'].iloc[-1] if len(df) > 0 else None
        }
    
    if 'hurst_regime' in df.columns:
        regime_counts = df['hurst_regime'].value_counts()
        total = len(df[df['hurst_regime'] != 'Unknown'])
        summary['hurst_regime_distribution'] = {
            regime: count / total if total > 0 else 0
            for regime, count in regime_counts.items()
            if regime != 'Unknown'
        }
        summary['current_hurst_regime'] = df['hurst_regime'].iloc[-1] if len(df) > 0 else None
    
    if 'persistence_strength' in df.columns:
        summary['average_persistence'] = df['persistence_strength'].mean()
        summary['current_persistence'] = df['persistence_strength'].iloc[-1] if len(df) > 0 else None
    
    # Interpretation
    if 'hurst' in df.columns and len(df) > 0:
        current_h = df['hurst'].iloc[-1]
        if not pd.isna(current_h):
            if current_h > 0.55:
                summary['interpretation'] = f"Market shows persistent trending behavior (H={current_h:.3f})"
            elif current_h < 0.45:
                summary['interpretation'] = f"Market shows mean-reverting behavior (H={current_h:.3f})"
            else:
                summary['interpretation'] = f"Market behaves like a random walk (H={current_h:.3f})"
    
    return summary
