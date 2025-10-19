"""
Chart visualization module for price data with regime overlays.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.dates import DateFormatter
from typing import Dict, Optional, List, Tuple
import warnings

# Define consistent color scheme for regimes
REGIME_COLORS = {
    'Up-LowVol': '#2E7D32',      # Dark green
    'Up-HighVol': '#66BB6A',     # Light green
    'Sideways-LowVol': '#FFA726', # Orange
    'Sideways-HighVol': '#FF7043', # Dark orange
    'Down-LowVol': '#EF5350',     # Light red
    'Down-HighVol': '#C62828',    # Dark red
    'Unknown': '#9E9E9E'          # Gray
}

# Friendly names for regimes
REGIME_NAMES = {
    'Up-LowVol': 'Bull Quiet',
    'Up-HighVol': 'Bull Volatile',
    'Sideways-LowVol': 'Sideways Quiet',
    'Sideways-HighVol': 'Sideways Volatile',
    'Down-LowVol': 'Bear Quiet',
    'Down-HighVol': 'Bear Volatile',
    'Unknown': 'Unknown'
}


def plot_price_with_regimes(df: pd.DataFrame, 
                           ticker: str = '',
                           figsize: Tuple[int, int] = (15, 8),
                           save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot price chart with regime-colored background bands.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with 'Close' price and 'regime' columns
    ticker : str
        Stock ticker symbol for title
    figsize : tuple
        Figure size (width, height)
    save_path : str, optional
        Path to save the figure
        
    Returns:
    --------
    plt.Figure : The matplotlib figure object
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, 
                                   gridspec_kw={'height_ratios': [3, 1]})
    
    # Main price chart
    ax1.plot(df.index, df['Close'], color='black', linewidth=1.5, label='Close Price')
    
    # Add regime background colors
    if 'regime' in df.columns:
        _add_regime_backgrounds(ax1, df)
    
    # Add moving averages if available
    if 'sma_50' in df.columns:
        ax1.plot(df.index, df['sma_50'], color='blue', alpha=0.7, 
                linewidth=1, label='SMA 50')
    if 'sma_200' in df.columns:
        ax1.plot(df.index, df['sma_200'], color='red', alpha=0.7, 
                linewidth=1, label='SMA 200')
    
    # Add fractal price if available
    if 'fractal_price' in df.columns:
        ax1.plot(df.index, df['fractal_price'], color='purple', alpha=0.7,
                linewidth=2, linestyle='--', label='Fractal Memory')
    
    ax1.set_ylabel('Price ($)', fontsize=12)
    ax1.set_title(f'{ticker} Price with Market Regimes', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Volume subplot if available
    if 'Volume' in df.columns:
        ax2.bar(df.index, df['Volume'], color='gray', alpha=0.5)
        ax2.set_ylabel('Volume', fontsize=12)
        ax2.set_xlabel('Date', fontsize=12)
    else:
        # Use for regime timeline if no volume
        _plot_regime_strip(ax2, df)
        ax2.set_ylabel('Regime', fontsize=12)
        ax2.set_xlabel('Date', fontsize=12)
    
    # Format x-axis
    ax2.xaxis.set_major_formatter(DateFormatter('%Y-%m'))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_regime_timeline(df: pd.DataFrame,
                        figsize: Tuple[int, int] = (15, 4),
                        save_path: Optional[str] = None) -> plt.Figure:
    """
    Create a dedicated regime timeline chart.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with 'regime' column
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure
        
    Returns:
    --------
    plt.Figure : The matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    if 'regime' not in df.columns:
        ax.text(0.5, 0.5, 'No regime data available', 
               ha='center', va='center', transform=ax.transAxes)
        return fig
    
    # Create numeric encoding for regimes
    regime_map = {regime: i for i, regime in enumerate(REGIME_COLORS.keys())}
    regime_numeric = df['regime'].map(regime_map).fillna(-1)
    
    # Plot as colored segments
    for regime, color in REGIME_COLORS.items():
        mask = df['regime'] == regime
        if mask.any():
            # Find continuous segments
            segments = _find_continuous_segments(mask)
            for start, end in segments:
                ax.axvspan(df.index[start], df.index[end], 
                          color=color, alpha=0.6, label=REGIME_NAMES.get(regime, regime))
    
    # Remove duplicate labels
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), 
             loc='upper center', bbox_to_anchor=(0.5, -0.1),
             ncol=3, fontsize=10)
    
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Market Regime', fontsize=12)
    ax.set_title('Market Regime Timeline', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Format dates
    ax.xaxis.set_major_formatter(DateFormatter('%Y-%m'))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_technical_overlays(df: pd.DataFrame,
                          ticker: str = '',
                          indicators: List[str] = None,
                          figsize: Tuple[int, int] = (15, 10),
                          save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot price with multiple technical indicators.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with price and indicator columns
    ticker : str
        Stock ticker
    indicators : list
        List of indicators to plot ['trend_strength', 'volatility', 'hurst']
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure
        
    Returns:
    --------
    plt.Figure : The matplotlib figure object
    """
    if indicators is None:
        indicators = ['trend_strength', 'volatility', 'hurst']
    
    # Filter to available indicators
    available = [ind for ind in indicators if ind in df.columns]
    n_plots = len(available) + 1  # +1 for price
    
    fig, axes = plt.subplots(n_plots, 1, figsize=figsize, sharex=True)
    if n_plots == 1:
        axes = [axes]
    
    # Price chart
    ax = axes[0]
    ax.plot(df.index, df['Close'], color='black', linewidth=1.5)
    if 'regime' in df.columns:
        _add_regime_backgrounds(ax, df)
    ax.set_ylabel('Price ($)', fontsize=11)
    ax.set_title(f'{ticker} Technical Analysis Dashboard', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Plot each indicator
    for i, indicator in enumerate(available, 1):
        ax = axes[i]
        
        if indicator == 'trend_strength':
            ax.plot(df.index, df[indicator], color='blue', linewidth=1)
            ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            ax.axhline(y=1, color='green', linestyle='--', alpha=0.3)
            ax.axhline(y=-1, color='red', linestyle='--', alpha=0.3)
            ax.set_ylabel('Trend Strength', fontsize=11)
            ax.set_ylim(-3, 3)
            
        elif indicator == 'volatility':
            ax.plot(df.index, df[indicator], color='orange', linewidth=1)
            if 'vol_threshold' in df.columns:
                ax.plot(df.index, df['vol_threshold'], color='red', 
                       linestyle='--', alpha=0.5, label='Threshold')
            ax.set_ylabel('Volatility', fontsize=11)
            
        elif indicator == 'hurst':
            ax.plot(df.index, df[indicator], color='purple', linewidth=1)
            ax.axhline(y=0.5, color='black', linestyle='-', alpha=0.3, label='Random Walk')
            ax.axhline(y=0.55, color='green', linestyle='--', alpha=0.3, label='Trending')
            ax.axhline(y=0.45, color='red', linestyle='--', alpha=0.3, label='Mean Reverting')
            ax.set_ylabel('Hurst Exponent', fontsize=11)
            ax.set_ylim(0, 1)
            ax.legend(loc='upper right', fontsize=9)
        
        ax.grid(True, alpha=0.3)
    
    # Format x-axis on bottom plot
    axes[-1].set_xlabel('Date', fontsize=12)
    axes[-1].xaxis.set_major_formatter(DateFormatter('%Y-%m'))
    plt.setp(axes[-1].xaxis.get_majorticklabels(), rotation=45)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def _add_regime_backgrounds(ax, df):
    """Add colored background bands for regimes."""
    if 'regime' not in df.columns:
        return
    
    # Find regime changes
    regime_changes = df['regime'].ne(df['regime'].shift()).cumsum()
    
    for regime_id in regime_changes.unique():
        mask = regime_changes == regime_id
        if mask.sum() == 0:
            continue
            
        regime = df.loc[mask, 'regime'].iloc[0]
        if pd.isna(regime) or regime == 'Unknown':
            continue
            
        color = REGIME_COLORS.get(regime, '#9E9E9E')
        
        # Get start and end dates for this regime segment
        dates = df.index[mask]
        if len(dates) > 0:
            ax.axvspan(dates[0], dates[-1], alpha=0.2, color=color)


def _plot_regime_strip(ax, df):
    """Plot regime as colored strip."""
    if 'regime' not in df.columns:
        return
    
    # Create numeric encoding
    regime_map = {regime: i for i, regime in enumerate(REGIME_COLORS.keys())}
    regime_numeric = df['regime'].map(regime_map).fillna(-1)
    
    # Create color map
    colors = [REGIME_COLORS.get(df['regime'].iloc[i], '#9E9E9E') 
             for i in range(len(df))]
    
    # Plot as bar chart with minimal height
    ax.bar(df.index, np.ones(len(df)), color=colors, width=1, alpha=0.8)
    ax.set_ylim(0, 1.1)
    ax.set_yticks([])


def _find_continuous_segments(mask):
    """Find start and end indices of continuous True segments in a boolean mask."""
    segments = []
    in_segment = False
    start = None
    
    for i, val in enumerate(mask):
        if val and not in_segment:
            start = i
            in_segment = True
        elif not val and in_segment:
            segments.append((start, i-1))
            in_segment = False
    
    if in_segment:
        segments.append((start, len(mask)-1))
    
    return segments
