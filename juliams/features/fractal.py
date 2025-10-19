"""
Fractal memory filter for the Julia Mandelbrot System.
Creates filtered price series highlighting persistent trending behavior based on Hurst exponent.
"""

import numpy as np
import pandas as pd
from typing import Optional, Union, Tuple
import logging

logger = logging.getLogger(__name__)


def create_fractal_mask(hurst: pd.Series,
                        threshold: float = 0.55,
                        smooth: bool = True,
                        smooth_window: int = 5) -> pd.Series:
    """
    Create a binary mask for fractal memory filtering.
    
    Args:
        hurst: Series of Hurst exponent values
        threshold: Threshold above which fractal memory is considered strong
        smooth: Whether to smooth the mask to avoid rapid switching
        smooth_window: Window for smoothing the mask
    
    Returns:
        Binary mask series (1 for strong fractal memory, 0 otherwise)
    """
    # Create basic mask
    mask = (hurst > threshold).astype(float)
    
    # Handle NaN values
    mask = mask.fillna(0)
    
    if smooth:
        # Smooth the mask to avoid rapid on/off switching
        # Use a rolling mean and threshold at 0.5
        smoothed = mask.rolling(window=smooth_window, min_periods=1, center=True).mean()
        mask = (smoothed >= 0.5).astype(float)
    
    return mask


def compute_fractal_filtered_price(df: pd.DataFrame,
                                  hurst_col: str = 'hurst',
                                  price_col: str = 'Close',
                                  threshold: float = 0.55,
                                  method: str = 'accumulate') -> pd.Series:
    """
    Compute fractal memory-filtered price series.
    
    This creates a price series that only moves during periods of strong
    fractal memory (persistent trending behavior).
    
    Args:
        df: DataFrame with price and Hurst data
        hurst_col: Column name for Hurst exponent
        price_col: Column name for price data
        threshold: Hurst threshold for fractal memory
        method: Filtering method ('accumulate', 'hold', 'weighted')
    
    Returns:
        Filtered price series
    """
    if hurst_col not in df.columns:
        raise ValueError(f"Column '{hurst_col}' not found in DataFrame")
    if price_col not in df.columns:
        raise ValueError(f"Column '{price_col}' not found in DataFrame")
    
    # Create fractal memory mask
    mask = create_fractal_mask(df[hurst_col], threshold)
    
    if method == 'accumulate':
        # Accumulate returns only during high-Hurst periods
        if 'log_return' not in df.columns:
            log_returns = np.log(df[price_col]).diff()
        else:
            log_returns = df['log_return']
        
        # Apply mask to returns
        filtered_returns = mask * log_returns
        
        # Reconstruct price from filtered returns
        # Start from the first valid price
        initial_price = df[price_col].iloc[0]
        filtered_log_price = filtered_returns.cumsum() + np.log(initial_price)
        filtered_price = np.exp(filtered_log_price)
        
    elif method == 'hold':
        # Hold price constant during low-Hurst periods
        filtered_price = df[price_col].copy()
        
        # Forward fill prices when mask is 0
        for i in range(1, len(filtered_price)):
            if mask.iloc[i] == 0:
                filtered_price.iloc[i] = filtered_price.iloc[i-1]
    
    elif method == 'weighted':
        # Weight price changes by Hurst value
        # Higher Hurst = more weight to price change
        if 'log_return' not in df.columns:
            log_returns = np.log(df[price_col]).diff()
        else:
            log_returns = df['log_return']
        
        # Use Hurst as weight (normalized to [0, 1])
        hurst_normalized = df[hurst_col].clip(0, 1)
        weighted_returns = hurst_normalized * log_returns
        
        # Reconstruct price
        initial_price = df[price_col].iloc[0]
        filtered_log_price = weighted_returns.cumsum() + np.log(initial_price)
        filtered_price = np.exp(filtered_log_price)
    
    else:
        raise ValueError(f"Unknown filtering method: {method}")
    
    return filtered_price


def compute_fractal_filtered_returns(df: pd.DataFrame,
                                    hurst_col: str = 'hurst',
                                    threshold: float = 0.55) -> pd.Series:
    """
    Compute returns filtered by fractal memory.
    
    Args:
        df: DataFrame with returns and Hurst data
        hurst_col: Column name for Hurst exponent
        threshold: Hurst threshold for fractal memory
    
    Returns:
        Filtered returns series
    """
    if 'log_return' not in df.columns:
        if 'log_price' not in df.columns:
            df['log_price'] = np.log(df['Close'])
        log_returns = df['log_price'].diff()
    else:
        log_returns = df['log_return']
    
    # Create fractal memory mask
    mask = create_fractal_mask(df[hurst_col], threshold)
    
    # Apply mask to returns
    filtered_returns = mask * log_returns
    
    return filtered_returns


def identify_fractal_segments(df: pd.DataFrame,
                             hurst_col: str = 'hurst',
                             threshold: float = 0.55,
                             min_length: int = 5) -> pd.DataFrame:
    """
    Identify contiguous segments with strong fractal memory.
    
    Args:
        df: DataFrame with Hurst data
        hurst_col: Column name for Hurst exponent
        threshold: Hurst threshold for fractal memory
        min_length: Minimum segment length to consider
    
    Returns:
        DataFrame with segment information
    """
    # Create fractal memory mask
    mask = create_fractal_mask(df[hurst_col], threshold, smooth=False)
    
    # Find segment boundaries
    segment_changes = mask.ne(mask.shift())
    segment_id = segment_changes.cumsum()
    
    # Add segment info to dataframe
    df_segments = df.copy()
    df_segments['fractal_segment_id'] = segment_id
    df_segments['has_fractal_memory'] = mask.astype(bool)
    
    # Calculate segment statistics
    segments = []
    for seg_id in df_segments['fractal_segment_id'].unique():
        segment_data = df_segments[df_segments['fractal_segment_id'] == seg_id]
        
        if len(segment_data) >= min_length:
            segment_info = {
                'segment_id': seg_id,
                'start_date': segment_data.index[0],
                'end_date': segment_data.index[-1],
                'length': len(segment_data),
                'has_fractal_memory': segment_data['has_fractal_memory'].iloc[0],
                'avg_hurst': segment_data[hurst_col].mean(),
                'min_hurst': segment_data[hurst_col].min(),
                'max_hurst': segment_data[hurst_col].max(),
            }
            
            # Add price change if available
            if 'Close' in segment_data.columns:
                segment_info['price_change'] = (
                    segment_data['Close'].iloc[-1] / segment_data['Close'].iloc[0] - 1
                )
            
            segments.append(segment_info)
    
    return pd.DataFrame(segments)


def compute_fractal_features(df: pd.DataFrame,
                            config: Optional[dict] = None) -> pd.DataFrame:
    """
    Compute all fractal-related features and add them to the DataFrame.
    
    Args:
        df: DataFrame with price and Hurst data
        config: Configuration dictionary with parameters
    
    Returns:
        DataFrame with added fractal features
    """
    if config is None:
        config = {
            'hurst_threshold': 0.55,
            'fractal_filter_method': 'accumulate'
        }
    
    df = df.copy()
    
    # Ensure we have Hurst values
    if 'hurst' not in df.columns:
        logger.warning("Hurst column not found. Fractal features require Hurst computation first.")
        return df
    
    # Create fractal memory mask
    df['fractal_mask'] = create_fractal_mask(
        df['hurst'],
        threshold=config.get('hurst_threshold', 0.55)
    )
    
    # Compute fractal-filtered price
    df['fractal_price'] = compute_fractal_filtered_price(
        df,
        threshold=config.get('hurst_threshold', 0.55),
        method=config.get('fractal_filter_method', 'accumulate')
    )
    
    # Compute fractal-filtered returns
    df['fractal_returns'] = compute_fractal_filtered_returns(
        df,
        threshold=config.get('hurst_threshold', 0.55)
    )
    
    # Compute cumulative fractal returns
    df['cumulative_fractal_returns'] = df['fractal_returns'].cumsum()
    
    # Fractal trend strength (filtered trend)
    if 'trend_strength' in df.columns:
        df['fractal_trend_strength'] = df['trend_strength'] * df['fractal_mask']
    
    # Fractal volatility (volatility during fractal periods)
    if 'volatility' in df.columns:
        df['fractal_volatility'] = df['volatility'] * df['fractal_mask']
        df['fractal_volatility'] = df['fractal_volatility'].replace(0, np.nan)
    
    # Fractal efficiency ratio (how much of the move was during fractal periods)
    total_return = df['log_return'].sum() if 'log_return' in df.columns else 0
    fractal_return = df['fractal_returns'].sum()
    df['fractal_efficiency'] = fractal_return / total_return if total_return != 0 else 0
    
    logger.info(f"Computed fractal features with threshold={config.get('hurst_threshold', 0.55)}")
    
    return df


def get_fractal_summary(df: pd.DataFrame) -> dict:
    """
    Get summary statistics for fractal features.
    
    Args:
        df: DataFrame with fractal features
    
    Returns:
        Dictionary with fractal summary statistics
    """
    summary = {}
    
    if 'fractal_mask' in df.columns:
        fractal_periods = df['fractal_mask'].sum()
        total_periods = len(df)
        summary['fractal_coverage'] = fractal_periods / total_periods if total_periods > 0 else 0
        summary['current_has_fractal_memory'] = bool(df['fractal_mask'].iloc[-1]) if len(df) > 0 else False
    
    if 'fractal_price' in df.columns and 'Close' in df.columns:
        # Compare fractal-filtered price to actual price
        actual_return = (df['Close'].iloc[-1] / df['Close'].iloc[0] - 1) if len(df) > 1 else 0
        fractal_return = (df['fractal_price'].iloc[-1] / df['fractal_price'].iloc[0] - 1) if len(df) > 1 else 0
        
        summary['total_return'] = actual_return
        summary['fractal_filtered_return'] = fractal_return
        summary['fractal_capture_ratio'] = fractal_return / actual_return if actual_return != 0 else 0
    
    if 'fractal_returns' in df.columns:
        # Statistics on fractal returns
        fractal_rets = df['fractal_returns'][df['fractal_returns'] != 0]
        if len(fractal_rets) > 0:
            summary['fractal_return_stats'] = {
                'mean': fractal_rets.mean(),
                'std': fractal_rets.std(),
                'sharpe': fractal_rets.mean() / fractal_rets.std() if fractal_rets.std() > 0 else 0
            }
    
    if 'fractal_volatility' in df.columns:
        fractal_vol = df['fractal_volatility'].dropna()
        if len(fractal_vol) > 0:
            summary['avg_fractal_volatility'] = fractal_vol.mean()
    
    # Interpretation
    if 'fractal_mask' in df.columns and len(df) > 0:
        if df['fractal_mask'].iloc[-1] == 1:
            summary['interpretation'] = "Market currently exhibits strong fractal memory (persistent trending)"
        else:
            summary['interpretation'] = "Market currently lacks fractal memory (random or mean-reverting)"
    
    return summary


def plot_fractal_filter(df: pd.DataFrame,
                       price_col: str = 'Close',
                       fractal_col: str = 'fractal_price') -> Tuple[pd.Series, pd.Series]:
    """
    Prepare data for plotting fractal-filtered price vs actual price.
    
    Args:
        df: DataFrame with price and fractal data
        price_col: Column name for actual price
        fractal_col: Column name for fractal-filtered price
    
    Returns:
        Tuple of (actual_price, fractal_price) series for plotting
    """
    if price_col not in df.columns or fractal_col not in df.columns:
        raise ValueError("Required columns not found in DataFrame")
    
    return df[price_col], df[fractal_col]
