"""
Crisp regime classification for the Julia Mandelbrot System.
Classifies market into six regimes based on trend and volatility.
"""

import numpy as np
import pandas as pd
from typing import Optional, Union, Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)


class RegimeClassifier:
    """
    Classifies market regimes based on trend and volatility indicators.
    
    Six regimes:
    1. Up-LowVol (Bull Quiet)
    2. Up-HighVol (Bull Volatile)
    3. Sideways-LowVol (Sideways Quiet)
    4. Sideways-HighVol (Sideways Volatile)
    5. Down-LowVol (Bear Quiet)
    6. Down-HighVol (Bear Volatile)
    """
    
    # Define the six market regimes
    REGIMES = [
        "Up-LowVol",
        "Up-HighVol",
        "Sideways-LowVol",
        "Sideways-HighVol",
        "Down-LowVol",
        "Down-HighVol"
    ]
    
    # Friendly names for regimes
    REGIME_NAMES = {
        "Up-LowVol": "Bull Quiet",
        "Up-HighVol": "Bull Volatile",
        "Sideways-LowVol": "Sideways Quiet",
        "Sideways-HighVol": "Sideways Volatile",
        "Down-LowVol": "Bear Quiet",
        "Down-HighVol": "Bear Volatile"
    }
    
    def __init__(self, 
                 trend_threshold_up: float = 0.2,
                 trend_threshold_down: float = -0.2,
                 volatility_threshold: float = 0.67,
                 volatility_method: str = 'percentile'):
        """
        Initialize the regime classifier.
        
        Args:
            trend_threshold_up: Threshold for uptrend classification
            trend_threshold_down: Threshold for downtrend classification
            volatility_threshold: Threshold for high/low volatility
            volatility_method: Method for volatility classification
        """
        self.trend_threshold_up = trend_threshold_up
        self.trend_threshold_down = trend_threshold_down
        self.volatility_threshold = volatility_threshold
        self.volatility_method = volatility_method
    
    def classify_trend(self, trend_strength: Union[float, pd.Series]) -> Union[str, pd.Series]:
        """
        Classify trend direction based on trend strength.
        
        Args:
            trend_strength: Trend strength value(s)
        
        Returns:
            Trend classification: 'Up', 'Down', or 'Sideways'
        """
        def classify_single(value):
            if pd.isna(value):
                return 'Unknown'
            elif value > self.trend_threshold_up:
                return 'Up'
            elif value < self.trend_threshold_down:
                return 'Down'
            else:
                return 'Sideways'
        
        if isinstance(trend_strength, pd.Series):
            return trend_strength.apply(classify_single)
        else:
            return classify_single(trend_strength)
    
    def classify_volatility(self, 
                          volatility: Union[float, pd.Series],
                          volatility_percentile: Optional[Union[float, pd.Series]] = None) -> Union[str, pd.Series]:
        """
        Classify volatility level.
        
        Args:
            volatility: Volatility value(s)
            volatility_percentile: Volatility percentile if using percentile method
        
        Returns:
            Volatility classification: 'High' or 'Low'
        """
        if self.volatility_method == 'percentile' and volatility_percentile is not None:
            # Use percentile-based classification
            def classify_single(pct):
                if pd.isna(pct):
                    return 'Unknown'
                return 'High' if pct > self.volatility_threshold else 'Low'
            
            if isinstance(volatility_percentile, pd.Series):
                return volatility_percentile.apply(classify_single)
            else:
                return classify_single(volatility_percentile)
        else:
            # Use absolute threshold
            def classify_single(vol):
                if pd.isna(vol):
                    return 'Unknown'
                return 'High' if vol > self.volatility_threshold else 'Low'
            
            if isinstance(volatility, pd.Series):
                return volatility.apply(classify_single)
            else:
                return classify_single(volatility)
    
    def classify_regime(self, 
                       trend: Union[str, pd.Series],
                       volatility: Union[str, pd.Series]) -> Union[str, pd.Series]:
        """
        Combine trend and volatility to determine market regime.
        
        Args:
            trend: Trend classification ('Up', 'Down', 'Sideways')
            volatility: Volatility classification ('High', 'Low')
        
        Returns:
            Market regime classification
        """
        def combine_single(t, v):
            if t == 'Unknown' or v == 'Unknown':
                return 'Unknown'
            
            vol_suffix = 'HighVol' if v == 'High' else 'LowVol'
            return f"{t}-{vol_suffix}"
        
        if isinstance(trend, pd.Series) and isinstance(volatility, pd.Series):
            return pd.Series([combine_single(t, v) for t, v in zip(trend, volatility)],
                           index=trend.index)
        else:
            return combine_single(trend, volatility)
    
    def classify(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Classify market regimes for the entire DataFrame.
        
        Args:
            df: DataFrame with trend_strength and volatility columns
        
        Returns:
            DataFrame with added regime classification columns
        """
        df = df.copy()
        
        # Check required columns
        if 'trend_strength' not in df.columns:
            raise ValueError("DataFrame must contain 'trend_strength' column")
        if 'volatility' not in df.columns:
            raise ValueError("DataFrame must contain 'volatility' column")
        
        # Classify trend
        df['trend_regime'] = self.classify_trend(df['trend_strength'])
        
        # Classify volatility
        if 'volatility_percentile' in df.columns and self.volatility_method == 'percentile':
            df['volatility_regime'] = self.classify_volatility(
                df['volatility'], 
                df['volatility_percentile']
            )
        else:
            df['volatility_regime'] = self.classify_volatility(df['volatility'])
        
        # Combine to get market regime
        df['regime'] = self.classify_regime(df['trend_regime'], df['volatility_regime'])
        
        # Add friendly name
        df['regime_name'] = df['regime'].map(self.REGIME_NAMES).fillna('Unknown')
        
        logger.info(f"Classified {len(df)} periods into market regimes")
        
        return df
    
    def get_regime_code(self, regime: str) -> int:
        """
        Get numeric code for a regime.
        
        Args:
            regime: Regime string
        
        Returns:
            Numeric code (0-5) or -1 for unknown
        """
        try:
            return self.REGIMES.index(regime)
        except ValueError:
            return -1


def classify_market_regime(df: pd.DataFrame,
                          config: Optional[dict] = None) -> pd.DataFrame:
    """
    Convenience function to classify market regimes.
    
    Args:
        df: DataFrame with required features
        config: Configuration dictionary
    
    Returns:
        DataFrame with regime classifications
    """
    if config is None:
        config = {
            'trend_threshold_up': 0.2,
            'trend_threshold_down': -0.2,
            'volatility_percentile': 0.67
        }
    
    classifier = RegimeClassifier(
        trend_threshold_up=config.get('trend_threshold_up', 0.2),
        trend_threshold_down=config.get('trend_threshold_down', -0.2),
        volatility_threshold=config.get('volatility_percentile', 0.67),
        volatility_method='percentile' if 'volatility_percentile' in df.columns else 'absolute'
    )
    
    return classifier.classify(df)


def identify_regime_segments(df: pd.DataFrame,
                            regime_col: str = 'regime',
                            min_length: int = 1) -> List[Dict]:
    """
    Identify contiguous segments of the same regime.
    
    Args:
        df: DataFrame with regime classifications
        regime_col: Column name for regime
        min_length: Minimum segment length to include
    
    Returns:
        List of segment dictionaries
    """
    if regime_col not in df.columns:
        raise ValueError(f"Column '{regime_col}' not found in DataFrame")
    
    segments = []
    current_regime = None
    segment_start = None
    
    for i, (date, regime) in enumerate(df[regime_col].items()):
        if pd.isna(regime) or regime == 'Unknown':
            continue
        
        if regime != current_regime:
            # End previous segment
            if current_regime is not None and segment_start is not None:
                segment_end_idx = i - 1
                segment_length = segment_end_idx - segment_start + 1
                
                if segment_length >= min_length:
                    segment_data = df.iloc[segment_start:segment_end_idx + 1]
                    
                    segment_info = {
                        'regime': current_regime,
                        'regime_name': RegimeClassifier.REGIME_NAMES.get(current_regime, current_regime),
                        'start_date': df.index[segment_start],
                        'end_date': df.index[segment_end_idx],
                        'start_idx': segment_start,
                        'end_idx': segment_end_idx,
                        'length': segment_length,
                    }
                    
                    # Add price change if available
                    if 'Close' in segment_data.columns:
                        segment_info['price_change'] = (
                            segment_data['Close'].iloc[-1] / segment_data['Close'].iloc[0] - 1
                        )
                        segment_info['total_return'] = segment_info['price_change']
                    
                    # Add average indicators if available
                    if 'trend_strength' in segment_data.columns:
                        segment_info['avg_trend_strength'] = segment_data['trend_strength'].mean()
                    if 'volatility' in segment_data.columns:
                        segment_info['avg_volatility'] = segment_data['volatility'].mean()
                    if 'hurst' in segment_data.columns:
                        segment_info['avg_hurst'] = segment_data['hurst'].mean()
                    
                    segments.append(segment_info)
            
            # Start new segment
            current_regime = regime
            segment_start = i
    
    # Handle last segment
    if current_regime is not None and segment_start is not None:
        segment_end_idx = len(df) - 1
        segment_length = segment_end_idx - segment_start + 1
        
        if segment_length >= min_length:
            segment_data = df.iloc[segment_start:segment_end_idx + 1]
            
            segment_info = {
                'regime': current_regime,
                'regime_name': RegimeClassifier.REGIME_NAMES.get(current_regime, current_regime),
                'start_date': df.index[segment_start],
                'end_date': df.index[segment_end_idx],
                'start_idx': segment_start,
                'end_idx': segment_end_idx,
                'length': segment_length,
            }
            
            if 'Close' in segment_data.columns:
                segment_info['price_change'] = (
                    segment_data['Close'].iloc[-1] / segment_data['Close'].iloc[0] - 1
                )
            
            segments.append(segment_info)
    
    return segments


def get_regime_statistics(df: pd.DataFrame,
                         regime_col: str = 'regime') -> Dict:
    """
    Calculate statistics for each regime.
    
    Args:
        df: DataFrame with regime classifications
        regime_col: Column name for regime
    
    Returns:
        Dictionary with regime statistics
    """
    if regime_col not in df.columns:
        raise ValueError(f"Column '{regime_col}' not found in DataFrame")
    
    stats = {}
    
    # Overall distribution
    regime_counts = df[regime_col].value_counts()
    total = len(df[df[regime_col] != 'Unknown'])
    
    stats['distribution'] = {
        regime: {
            'count': count,
            'percentage': count / total * 100 if total > 0 else 0
        }
        for regime, count in regime_counts.items()
        if regime != 'Unknown'
    }
    
    # Statistics per regime
    for regime in RegimeClassifier.REGIMES:
        regime_data = df[df[regime_col] == regime]
        
        if len(regime_data) > 0:
            regime_stats = {
                'count': len(regime_data),
                'percentage': len(regime_data) / total * 100 if total > 0 else 0,
            }
            
            # Add return statistics if available
            if 'log_return' in regime_data.columns:
                returns = regime_data['log_return'].dropna()
                if len(returns) > 0:
                    regime_stats['return_stats'] = {
                        'mean': returns.mean(),
                        'std': returns.std(),
                        'min': returns.min(),
                        'max': returns.max(),
                        'sharpe': returns.mean() / returns.std() if returns.std() > 0 else 0
                    }
            
            # Add average indicators
            if 'trend_strength' in regime_data.columns:
                regime_stats['avg_trend_strength'] = regime_data['trend_strength'].mean()
            if 'volatility' in regime_data.columns:
                regime_stats['avg_volatility'] = regime_data['volatility'].mean()
            if 'hurst' in regime_data.columns:
                regime_stats['avg_hurst'] = regime_data['hurst'].mean()
            
            stats[regime] = regime_stats
    
    # Current regime
    if len(df) > 0:
        current = df[regime_col].iloc[-1]
        if current != 'Unknown':
            stats['current_regime'] = {
                'regime': current,
                'name': RegimeClassifier.REGIME_NAMES.get(current, current)
            }
    
    # Segment analysis
    segments = identify_regime_segments(df, regime_col)
    if segments:
        stats['segments'] = {
            'total_segments': len(segments),
            'avg_segment_length': np.mean([s['length'] for s in segments]),
            'max_segment_length': max(s['length'] for s in segments),
            'min_segment_length': min(s['length'] for s in segments),
        }
        
        # Average segment length per regime
        for regime in RegimeClassifier.REGIMES:
            regime_segments = [s for s in segments if s['regime'] == regime]
            if regime_segments:
                stats['segments'][f'{regime}_avg_length'] = np.mean([s['length'] for s in regime_segments])
    
    return stats


def plot_regime_timeline(df: pd.DataFrame,
                        regime_col: str = 'regime') -> pd.Series:
    """
    Create numeric encoding of regimes for plotting.
    
    Args:
        df: DataFrame with regime classifications
        regime_col: Column name for regime
    
    Returns:
        Series with numeric regime codes
    """
    classifier = RegimeClassifier()
    
    def encode_regime(regime):
        if pd.isna(regime) or regime == 'Unknown':
            return -1
        return classifier.get_regime_code(regime)
    
    return df[regime_col].apply(encode_regime)
