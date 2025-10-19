"""
Regime segment analysis for the Julia Mandelbrot System.
Analyzes contiguous periods of the same regime.
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)


def identify_regime_segments(df: pd.DataFrame,
                            regime_col: str = 'regime',
                            min_length: int = 1) -> List[Dict]:
    """
    Identify contiguous segments of the same regime.
    
    Args:
        df: DataFrame with regime classifications
        regime_col: Column name for regime classification
        min_length: Minimum segment length to include
    
    Returns:
        List of segment dictionaries with detailed information
    """
    if regime_col not in df.columns:
        raise ValueError(f"Column '{regime_col}' not found in DataFrame")
    
    segments = []
    current_regime = None
    segment_start_idx = None
    
    for i in range(len(df)):
        regime = df[regime_col].iloc[i]
        
        # Skip unknown regimes
        if pd.isna(regime) or regime == 'Unknown':
            # End current segment if exists
            if current_regime is not None and segment_start_idx is not None:
                segment_end_idx = i - 1
                if segment_end_idx >= segment_start_idx:
                    segment_length = segment_end_idx - segment_start_idx + 1
                    if segment_length >= min_length:
                        segment_data = df.iloc[segment_start_idx:segment_end_idx + 1]
                        segment_info = create_segment_info(
                            segment_data, current_regime, 
                            segment_start_idx, segment_end_idx
                        )
                        segments.append(segment_info)
                current_regime = None
                segment_start_idx = None
            continue
        
        if regime != current_regime:
            # End previous segment
            if current_regime is not None and segment_start_idx is not None:
                segment_end_idx = i - 1
                segment_length = segment_end_idx - segment_start_idx + 1
                
                if segment_length >= min_length:
                    segment_data = df.iloc[segment_start_idx:segment_end_idx + 1]
                    segment_info = create_segment_info(
                        segment_data, current_regime,
                        segment_start_idx, segment_end_idx
                    )
                    segments.append(segment_info)
            
            # Start new segment
            current_regime = regime
            segment_start_idx = i
    
    # Handle last segment
    if current_regime is not None and segment_start_idx is not None:
        segment_end_idx = len(df) - 1
        segment_length = segment_end_idx - segment_start_idx + 1
        
        if segment_length >= min_length:
            segment_data = df.iloc[segment_start_idx:segment_end_idx + 1]
            segment_info = create_segment_info(
                segment_data, current_regime,
                segment_start_idx, segment_end_idx
            )
            segments.append(segment_info)
    
    logger.info(f"Identified {len(segments)} regime segments")
    
    return segments


def create_segment_info(segment_data: pd.DataFrame,
                       regime: str,
                       start_idx: int,
                       end_idx: int) -> Dict:
    """
    Create detailed information dictionary for a regime segment.
    
    Args:
        segment_data: DataFrame slice for the segment
        regime: Regime classification
        start_idx: Starting index in original DataFrame
        end_idx: Ending index in original DataFrame
    
    Returns:
        Dictionary with segment information
    """
    segment_info = {
        'regime': regime,
        'start_date': segment_data.index[0],
        'end_date': segment_data.index[-1],
        'start_idx': start_idx,
        'end_idx': end_idx,
        'length': len(segment_data),
    }
    
    # Price information
    if 'Close' in segment_data.columns:
        segment_info['start_price'] = segment_data['Close'].iloc[0]
        segment_info['end_price'] = segment_data['Close'].iloc[-1]
        segment_info['price_change'] = (
            segment_data['Close'].iloc[-1] / segment_data['Close'].iloc[0] - 1
        )
        segment_info['max_price'] = segment_data['Close'].max()
        segment_info['min_price'] = segment_data['Close'].min()
        segment_info['price_range'] = (
            segment_data['Close'].max() - segment_data['Close'].min()
        ) / segment_data['Close'].iloc[0]
    
    # Return statistics
    if 'log_return' in segment_data.columns:
        returns = segment_data['log_return'].dropna()
        if len(returns) > 0:
            segment_info['total_return'] = returns.sum()
            segment_info['avg_daily_return'] = returns.mean()
            segment_info['return_std'] = returns.std()
            segment_info['sharpe'] = (
                returns.mean() / returns.std() if returns.std() > 0 else 0
            )
    
    # Average indicators
    if 'trend_strength' in segment_data.columns:
        segment_info['avg_trend_strength'] = segment_data['trend_strength'].mean()
    
    if 'volatility' in segment_data.columns:
        segment_info['avg_volatility'] = segment_data['volatility'].mean()
        segment_info['max_volatility'] = segment_data['volatility'].max()
        segment_info['min_volatility'] = segment_data['volatility'].min()
    
    if 'hurst' in segment_data.columns:
        hurst_values = segment_data['hurst'].dropna()
        if len(hurst_values) > 0:
            segment_info['avg_hurst'] = hurst_values.mean()
            segment_info['min_hurst'] = hurst_values.min()
            segment_info['max_hurst'] = hurst_values.max()
    
    # Volume information
    if 'Volume' in segment_data.columns:
        segment_info['total_volume'] = segment_data['Volume'].sum()
        segment_info['avg_volume'] = segment_data['Volume'].mean()
    
    return segment_info


def analyze_regime_segments(df: pd.DataFrame,
                           regime_col: str = 'regime',
                           min_length: int = 1) -> Dict:
    """
    Comprehensive analysis of regime segments.
    
    Args:
        df: DataFrame with regime classifications
        regime_col: Column name for regime classification
        min_length: Minimum segment length to analyze
    
    Returns:
        Dictionary with segment analysis results
    """
    segments = identify_regime_segments(df, regime_col, min_length)
    
    if not segments:
        return {'segments': [], 'statistics': {}}
    
    # Convert to DataFrame for easier analysis
    segments_df = pd.DataFrame(segments)
    
    results = {
        'segments': segments,
        'total_segments': len(segments),
        'statistics': {}
    }
    
    # Overall statistics
    results['statistics']['overall'] = {
        'total_segments': len(segments),
        'avg_segment_length': segments_df['length'].mean(),
        'median_segment_length': segments_df['length'].median(),
        'max_segment_length': segments_df['length'].max(),
        'min_segment_length': segments_df['length'].min(),
        'std_segment_length': segments_df['length'].std()
    }
    
    # Statistics by regime
    regime_stats = {}
    for regime in segments_df['regime'].unique():
        regime_segments = segments_df[segments_df['regime'] == regime]
        
        stats = {
            'count': len(regime_segments),
            'percentage': len(regime_segments) / len(segments) * 100,
            'avg_length': regime_segments['length'].mean(),
            'max_length': regime_segments['length'].max(),
            'min_length': regime_segments['length'].min(),
            'total_periods': regime_segments['length'].sum()
        }
        
        # Price change statistics
        if 'price_change' in regime_segments.columns:
            price_changes = regime_segments['price_change'].dropna()
            if len(price_changes) > 0:
                stats['avg_price_change'] = price_changes.mean()
                stats['median_price_change'] = price_changes.median()
                stats['best_price_change'] = price_changes.max()
                stats['worst_price_change'] = price_changes.min()
                stats['positive_segments'] = (price_changes > 0).sum()
                stats['win_rate'] = (price_changes > 0).sum() / len(price_changes) * 100
        
        # Average indicators
        if 'avg_volatility' in regime_segments.columns:
            stats['avg_volatility'] = regime_segments['avg_volatility'].mean()
        
        if 'avg_hurst' in regime_segments.columns:
            stats['avg_hurst'] = regime_segments['avg_hurst'].mean()
        
        regime_stats[regime] = stats
    
    results['statistics']['by_regime'] = regime_stats
    
    # Find notable segments
    if 'price_change' in segments_df.columns:
        price_changes = segments_df['price_change'].dropna()
        if len(price_changes) > 0:
            # Best performing segments
            best_idx = price_changes.nlargest(5).index
            results['best_segments'] = segments_df.loc[best_idx].to_dict('records')
            
            # Worst performing segments
            worst_idx = price_changes.nsmallest(5).index
            results['worst_segments'] = segments_df.loc[worst_idx].to_dict('records')
    
    # Longest segments
    longest_idx = segments_df['length'].nlargest(5).index
    results['longest_segments'] = segments_df.loc[longest_idx].to_dict('records')
    
    return results


def compute_segment_statistics(segments: List[Dict]) -> pd.DataFrame:
    """
    Compute summary statistics for regime segments.
    
    Args:
        segments: List of segment dictionaries
    
    Returns:
        DataFrame with segment statistics
    """
    if not segments:
        return pd.DataFrame()
    
    segments_df = pd.DataFrame(segments)
    
    # Group by regime
    stats = []
    for regime in segments_df['regime'].unique():
        regime_data = segments_df[segments_df['regime'] == regime]
        
        stat_row = {
            'Regime': regime,
            'Count': len(regime_data),
            'Avg Length': regime_data['length'].mean(),
            'Max Length': regime_data['length'].max(),
            'Total Days': regime_data['length'].sum()
        }
        
        if 'price_change' in regime_data.columns:
            stat_row['Avg Return'] = regime_data['price_change'].mean()
            stat_row['Win Rate %'] = (
                (regime_data['price_change'] > 0).sum() / len(regime_data) * 100
            )
        
        if 'avg_volatility' in regime_data.columns:
            stat_row['Avg Volatility'] = regime_data['avg_volatility'].mean()
        
        stats.append(stat_row)
    
    return pd.DataFrame(stats).sort_values('Count', ascending=False)


def analyze_segment_transitions(segments: List[Dict]) -> Dict:
    """
    Analyze transitions between regime segments.
    
    Args:
        segments: List of segment dictionaries
    
    Returns:
        Dictionary with transition analysis
    """
    if len(segments) < 2:
        return {}
    
    transitions = []
    for i in range(len(segments) - 1):
        current = segments[i]
        next_seg = segments[i + 1]
        
        transition = {
            'from_regime': current['regime'],
            'to_regime': next_seg['regime'],
            'from_length': current['length'],
            'to_length': next_seg['length'],
            'transition_date': next_seg['start_date']
        }
        
        # Add price change if available
        if 'price_change' in current and 'price_change' in next_seg:
            transition['from_return'] = current['price_change']
            transition['to_return'] = next_seg['price_change']
            transition['return_change'] = next_seg['price_change'] - current['price_change']
        
        transitions.append(transition)
    
    # Analyze patterns
    transition_counts = {}
    for t in transitions:
        key = (t['from_regime'], t['to_regime'])
        if key not in transition_counts:
            transition_counts[key] = 0
        transition_counts[key] += 1
    
    return {
        'transitions': transitions,
        'transition_counts': transition_counts,
        'total_transitions': len(transitions)
    }


def get_segment_summary(df: pd.DataFrame,
                       regime_col: str = 'regime') -> Dict:
    """
    Get comprehensive summary of regime segments.
    
    Args:
        df: DataFrame with regime classifications
        regime_col: Column name for regime classification
    
    Returns:
        Dictionary with segment summary
    """
    # Analyze segments
    segment_analysis = analyze_regime_segments(df, regime_col)
    
    summary = {
        'total_segments': segment_analysis['total_segments'],
        'statistics': segment_analysis['statistics']
    }
    
    # Add segment statistics table
    if segment_analysis['segments']:
        summary['segment_table'] = compute_segment_statistics(
            segment_analysis['segments']
        ).to_dict()
        
        # Add transition analysis
        transition_analysis = analyze_segment_transitions(
            segment_analysis['segments']
        )
        if transition_analysis:
            summary['transitions'] = transition_analysis
    
    # Current segment information
    if len(df) > 0 and regime_col in df.columns:
        current_regime = df[regime_col].iloc[-1]
        if not pd.isna(current_regime) and current_regime != 'Unknown':
            # Find current segment
            for segment in reversed(segment_analysis['segments']):
                if segment['end_idx'] == len(df) - 1:
                    summary['current_segment'] = segment
                    break
    
    return summary


def export_segments_to_dataframe(segments: List[Dict]) -> pd.DataFrame:
    """
    Export segment list to a formatted DataFrame.
    
    Args:
        segments: List of segment dictionaries
    
    Returns:
        DataFrame with segment information
    """
    if not segments:
        return pd.DataFrame()
    
    # Select key columns for export
    export_columns = [
        'regime', 'start_date', 'end_date', 'length',
        'start_price', 'end_price', 'price_change',
        'avg_volatility', 'avg_hurst', 'avg_trend_strength'
    ]
    
    rows = []
    for segment in segments:
        row = {}
        for col in export_columns:
            if col in segment:
                row[col] = segment[col]
            else:
                row[col] = None
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # Format columns
    if 'price_change' in df.columns:
        df['price_change'] = df['price_change'] * 100  # Convert to percentage
    
    return df


# Alias for backward compatibility
identify_segments = identify_regime_segments
