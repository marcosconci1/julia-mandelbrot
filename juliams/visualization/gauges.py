"""
Gauge visualization module for fuzzy probabilities and nowcast displays.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Wedge, Circle
from typing import Dict, Optional, Tuple, List
import warnings


def plot_fuzzy_gauge(probabilities: Dict[str, float],
                     figsize: Tuple[int, int] = (12, 8),
                     save_path: Optional[str] = None) -> plt.Figure:
    """
    Create gauge visualizations for fuzzy regime probabilities.
    
    Parameters:
    -----------
    probabilities : dict
        Dictionary with regime names as keys and probabilities as values
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure
        
    Returns:
    --------
    plt.Figure : The matplotlib figure object
    """
    fig = plt.figure(figsize=figsize)
    
    # Create 2x3 grid for six regimes
    positions = [
        (0.15, 0.55),  # Up-LowVol
        (0.5, 0.55),   # Up-HighVol
        (0.85, 0.55),  # Sideways-LowVol
        (0.15, 0.1),   # Sideways-HighVol
        (0.5, 0.1),    # Down-LowVol
        (0.85, 0.1)    # Down-HighVol
    ]
    
    regime_order = [
        'Up-LowVol', 'Up-HighVol', 
        'Sideways-LowVol', 'Sideways-HighVol',
        'Down-LowVol', 'Down-HighVol'
    ]
    
    regime_names = {
        'Up-LowVol': 'Bull Quiet',
        'Up-HighVol': 'Bull Volatile',
        'Sideways-LowVol': 'Sideways Quiet',
        'Sideways-HighVol': 'Sideways Volatile',
        'Down-LowVol': 'Bear Quiet',
        'Down-HighVol': 'Bear Volatile'
    }
    
    for i, regime in enumerate(regime_order):
        if i < len(positions):
            ax = fig.add_axes([positions[i][0] - 0.12, positions[i][1], 0.24, 0.3])
            prob = probabilities.get(regime, 0.0)
            _draw_single_gauge(ax, prob, regime_names.get(regime, regime))
    
    # Add title
    fig.suptitle('Market Regime Probability Gauges', fontsize=16, fontweight='bold', y=0.95)
    
    # Add timestamp if available
    fig.text(0.5, 0.02, f'Fuzzy Logic Nowcast', ha='center', fontsize=12, style='italic')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_regime_probabilities(probabilities: Dict[str, float],
                             figsize: Tuple[int, int] = (10, 6),
                             save_path: Optional[str] = None) -> plt.Figure:
    """
    Create bar chart of regime probabilities.
    
    Parameters:
    -----------
    probabilities : dict
        Dictionary with regime names as keys and probabilities as values
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure
        
    Returns:
    --------
    plt.Figure : The matplotlib figure object
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Sort regimes by probability
    sorted_regimes = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
    regimes = [r[0] for r in sorted_regimes]
    probs = [r[1] for r in sorted_regimes]
    
    # Define colors for each regime
    colors = {
        'Up-LowVol': '#2E7D32',
        'Up-HighVol': '#66BB6A',
        'Sideways-LowVol': '#FFA726',
        'Sideways-HighVol': '#FF7043',
        'Down-LowVol': '#EF5350',
        'Down-HighVol': '#C62828'
    }
    
    bar_colors = [colors.get(r, 'gray') for r in regimes]
    
    # Bar chart
    bars = ax1.barh(range(len(regimes)), probs, color=bar_colors, alpha=0.8)
    ax1.set_yticks(range(len(regimes)))
    ax1.set_yticklabels(regimes)
    ax1.set_xlabel('Probability', fontsize=12)
    ax1.set_title('Regime Probabilities', fontsize=14, fontweight='bold')
    ax1.set_xlim(0, 1)
    ax1.grid(True, alpha=0.3, axis='x')
    
    # Add percentage labels
    for i, (bar, prob) in enumerate(zip(bars, probs)):
        ax1.text(prob + 0.02, bar.get_y() + bar.get_height()/2, 
                f'{prob*100:.1f}%', va='center', fontsize=10)
    
    # Pie chart for trend/volatility breakdown
    trend_probs = {
        'Uptrend': sum(probabilities.get(r, 0) for r in ['Up-LowVol', 'Up-HighVol']),
        'Sideways': sum(probabilities.get(r, 0) for r in ['Sideways-LowVol', 'Sideways-HighVol']),
        'Downtrend': sum(probabilities.get(r, 0) for r in ['Down-LowVol', 'Down-HighVol'])
    }
    
    vol_probs = {
        'Low Volatility': sum(probabilities.get(r, 0) for r in ['Up-LowVol', 'Sideways-LowVol', 'Down-LowVol']),
        'High Volatility': sum(probabilities.get(r, 0) for r in ['Up-HighVol', 'Sideways-HighVol', 'Down-HighVol'])
    }
    
    # Create nested pie chart
    size = 0.3
    vals_outer = list(trend_probs.values())
    vals_inner = list(vol_probs.values())
    
    # Outer ring (trend)
    wedges, texts, autotexts = ax2.pie(vals_outer, labels=trend_probs.keys(),
                                        autopct=lambda p: f'{p:.1f}%' if p > 5 else '',
                                        colors=['green', 'orange', 'red'],
                                        radius=1, wedgeprops=dict(width=size, edgecolor='white'))
    
    # Inner ring (volatility)
    ax2.pie(vals_inner, labels=vol_probs.keys(),
           autopct=lambda p: f'{p:.1f}%' if p > 5 else '',
           colors=['lightblue', 'coral'],
           radius=1-size, wedgeprops=dict(width=size, edgecolor='white'))
    
    ax2.set_title('Trend vs Volatility Breakdown', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def create_nowcast_dashboard(df: pd.DataFrame,
                            probabilities: Dict[str, float],
                            current_stats: Dict[str, float],
                            figsize: Tuple[int, int] = (16, 10),
                            save_path: Optional[str] = None) -> plt.Figure:
    """
    Create comprehensive nowcast dashboard with multiple visualizations.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Recent data for mini charts
    probabilities : dict
        Current regime probabilities
    current_stats : dict
        Current indicator values (trend_strength, volatility, hurst, etc.)
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure
        
    Returns:
    --------
    plt.Figure : The matplotlib figure object
    """
    fig = plt.figure(figsize=figsize)
    
    # Layout: 
    # Top: Title and current date
    # Row 1: Trend gauge, Volatility gauge, Hurst gauge
    # Row 2: Regime probability bars
    # Row 3: Recent price chart with regime colors
    # Row 4: Key statistics table
    
    # Title
    fig.suptitle('Julia Mandelbrot System - Market Nowcast Dashboard', 
                fontsize=18, fontweight='bold', y=0.98)
    
    # Current date
    if not df.empty:
        current_date = df.index[-1].strftime('%Y-%m-%d')
        fig.text(0.5, 0.94, f'Analysis Date: {current_date}', 
                ha='center', fontsize=12)
    
    # Row 1: Three main gauges
    # Trend gauge
    ax1 = fig.add_subplot(3, 3, 1)
    trend_val = current_stats.get('trend_strength', 0)
    _draw_trend_gauge(ax1, trend_val)
    
    # Volatility gauge  
    ax2 = fig.add_subplot(3, 3, 2)
    vol_val = current_stats.get('volatility', 0)
    vol_percentile = current_stats.get('vol_percentile', 50)
    _draw_volatility_gauge(ax2, vol_percentile)
    
    # Hurst gauge
    ax3 = fig.add_subplot(3, 3, 3)
    hurst_val = current_stats.get('hurst', 0.5)
    _draw_hurst_gauge(ax3, hurst_val)
    
    # Row 2: Regime probabilities
    ax4 = fig.add_subplot(3, 1, 2)
    _draw_probability_bars(ax4, probabilities)
    
    # Row 3: Recent price with regime
    if not df.empty and 'Close' in df.columns:
        ax5 = fig.add_subplot(3, 1, 3)
        _draw_recent_price(ax5, df)
    
    # Add statistics text box
    stats_text = _format_statistics(current_stats, probabilities)
    fig.text(0.02, 0.02, stats_text, fontsize=9, 
            verticalalignment='bottom', family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.94])
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def _draw_single_gauge(ax, value, label):
    """Draw a single semicircular gauge."""
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-0.1, 1.2)
    ax.set_aspect('equal')
    
    # Draw semicircle background
    theta = np.linspace(0, np.pi, 100)
    x = np.cos(theta)
    y = np.sin(theta)
    
    # Color gradient based on value
    if value < 0.33:
        color = 'lightgray'
    elif value < 0.67:
        color = 'yellow'
    else:
        color = 'green'
    
    # Draw arc
    ax.plot(x, y, 'k-', linewidth=2)
    ax.fill_between(x, 0, y, alpha=0.1)
    
    # Draw needle
    angle = np.pi * (1 - value)
    needle_x = 0.9 * np.cos(angle)
    needle_y = 0.9 * np.sin(angle)
    ax.plot([0, needle_x], [0, needle_y], 'r-', linewidth=3)
    ax.plot(0, 0, 'ko', markersize=8)
    
    # Add percentage text
    ax.text(0, -0.3, f'{value*100:.1f}%', ha='center', fontsize=12, fontweight='bold')
    ax.text(0, -0.5, label, ha='center', fontsize=10)
    
    # Remove axes
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)


def _draw_trend_gauge(ax, trend_strength):
    """Draw trend strength gauge (-3 to +3 scale)."""
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-0.5, 1.2)
    
    # Normalize to 0-1 scale
    normalized = (trend_strength + 3) / 6
    
    # Draw arc with color zones
    theta = np.linspace(0, np.pi, 100)
    x = np.cos(theta)
    y = np.sin(theta)
    
    # Color zones
    bear_zone = theta[theta > 2*np.pi/3]
    neutral_zone = theta[(theta > np.pi/3) & (theta <= 2*np.pi/3)]
    bull_zone = theta[theta <= np.pi/3]
    
    ax.plot(np.cos(bear_zone), np.sin(bear_zone), 'r-', linewidth=8, alpha=0.3)
    ax.plot(np.cos(neutral_zone), np.sin(neutral_zone), 'y-', linewidth=8, alpha=0.3)
    ax.plot(np.cos(bull_zone), np.sin(bull_zone), 'g-', linewidth=8, alpha=0.3)
    
    # Draw needle
    angle = np.pi * (1 - normalized)
    needle_x = 0.9 * np.cos(angle)
    needle_y = 0.9 * np.sin(angle)
    ax.arrow(0, 0, needle_x, needle_y, head_width=0.1, head_length=0.05,
            fc='black', ec='black', linewidth=2)
    
    # Labels
    ax.text(-1, -0.2, 'Bear', fontsize=9)
    ax.text(0, 1.1, 'Neutral', fontsize=9, ha='center')
    ax.text(1, -0.2, 'Bull', fontsize=9)
    ax.text(0, -0.3, f'Trend: {trend_strength:.2f}', ha='center', fontsize=11, fontweight='bold')
    
    ax.set_aspect('equal')
    ax.axis('off')


def _draw_volatility_gauge(ax, percentile):
    """Draw volatility percentile gauge."""
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-0.5, 1.2)
    
    # Draw arc
    theta = np.linspace(0, np.pi, 100)
    x = np.cos(theta)
    y = np.sin(theta)
    
    # Color gradient
    colors = plt.cm.RdYlGn_r(np.linspace(0, 1, len(theta)))
    for i in range(len(theta)-1):
        ax.plot(x[i:i+2], y[i:i+2], color=colors[i], linewidth=8, alpha=0.5)
    
    # Draw needle
    angle = np.pi * (1 - percentile/100)
    needle_x = 0.9 * np.cos(angle)
    needle_y = 0.9 * np.sin(angle)
    ax.arrow(0, 0, needle_x, needle_y, head_width=0.1, head_length=0.05,
            fc='black', ec='black', linewidth=2)
    
    # Labels
    ax.text(-1, -0.2, 'Low', fontsize=9)
    ax.text(0, 1.1, 'Medium', fontsize=9, ha='center')
    ax.text(1, -0.2, 'High', fontsize=9)
    ax.text(0, -0.3, f'Vol Percentile: {percentile:.0f}%', 
           ha='center', fontsize=11, fontweight='bold')
    
    ax.set_aspect('equal')
    ax.axis('off')


def _draw_hurst_gauge(ax, hurst):
    """Draw Hurst exponent gauge."""
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-0.5, 1.2)
    
    # Draw arc with zones
    theta = np.linspace(0, np.pi, 100)
    x = np.cos(theta)
    y = np.sin(theta)
    
    # Zones: <0.45 mean reverting, 0.45-0.55 random, >0.55 trending
    mr_zone = theta[theta > 0.7*np.pi]
    random_zone = theta[(theta > 0.3*np.pi) & (theta <= 0.7*np.pi)]
    trend_zone = theta[theta <= 0.3*np.pi]
    
    ax.plot(np.cos(mr_zone), np.sin(mr_zone), 'r-', linewidth=8, alpha=0.3)
    ax.plot(np.cos(random_zone), np.sin(random_zone), 'gray', linewidth=8, alpha=0.3)
    ax.plot(np.cos(trend_zone), np.sin(trend_zone), 'g-', linewidth=8, alpha=0.3)
    
    # Draw needle (Hurst is 0-1)
    angle = np.pi * (1 - hurst)
    needle_x = 0.9 * np.cos(angle)
    needle_y = 0.9 * np.sin(angle)
    ax.arrow(0, 0, needle_x, needle_y, head_width=0.1, head_length=0.05,
            fc='black', ec='black', linewidth=2)
    
    # Labels
    ax.text(-1, -0.2, 'M.R.', fontsize=9)
    ax.text(0, 1.1, 'Random', fontsize=9, ha='center')
    ax.text(1, -0.2, 'Trend', fontsize=9)
    ax.text(0, -0.3, f'Hurst: {hurst:.3f}', ha='center', fontsize=11, fontweight='bold')
    
    ax.set_aspect('equal')
    ax.axis('off')


def _draw_probability_bars(ax, probabilities):
    """Draw horizontal probability bars."""
    regimes = list(probabilities.keys())
    probs = list(probabilities.values())
    
    colors = {
        'Up-LowVol': '#2E7D32',
        'Up-HighVol': '#66BB6A',
        'Sideways-LowVol': '#FFA726',
        'Sideways-HighVol': '#FF7043',
        'Down-LowVol': '#EF5350',
        'Down-HighVol': '#C62828'
    }
    
    y_pos = np.arange(len(regimes))
    bar_colors = [colors.get(r, 'gray') for r in regimes]
    
    bars = ax.barh(y_pos, probs, color=bar_colors, alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(regimes)
    ax.set_xlabel('Probability', fontsize=11)
    ax.set_title('Current Regime Probabilities', fontsize=12, fontweight='bold')
    ax.set_xlim(0, 1)
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add percentage labels
    for bar, prob in zip(bars, probs):
        width = bar.get_width()
        ax.text(width + 0.02, bar.get_y() + bar.get_height()/2,
               f'{prob*100:.1f}%', va='center', fontsize=9)


def _draw_recent_price(ax, df):
    """Draw recent price chart with regime coloring."""
    # Plot last 60 days or available data
    recent_df = df.tail(60)
    
    ax.plot(recent_df.index, recent_df['Close'], 'k-', linewidth=1.5)
    
    if 'regime' in recent_df.columns:
        # Add regime coloring
        regime_colors = {
            'Up-LowVol': 'green',
            'Up-HighVol': 'lightgreen',
            'Sideways-LowVol': 'orange',
            'Sideways-HighVol': 'darkorange',
            'Down-LowVol': 'lightcoral',
            'Down-HighVol': 'red'
        }
        
        for regime, color in regime_colors.items():
            mask = recent_df['regime'] == regime
            if mask.any():
                ax.fill_between(recent_df.index, recent_df['Close'].min(), 
                              recent_df['Close'].max(), 
                              where=mask, alpha=0.2, color=color)
    
    ax.set_xlabel('Date', fontsize=11)
    ax.set_ylabel('Price', fontsize=11)
    ax.set_title('Recent Price Action (60 days)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Format dates
    from matplotlib.dates import DateFormatter
    ax.xaxis.set_major_formatter(DateFormatter('%m/%d'))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)


def _format_statistics(stats, probabilities):
    """Format statistics for text display."""
    lines = ['Key Statistics:', '=' * 40]
    
    # Current indicators
    if 'trend_strength' in stats:
        lines.append(f"Trend Strength: {stats['trend_strength']:.3f}")
    if 'volatility' in stats:
        lines.append(f"Volatility: {stats['volatility']:.4f}")
    if 'hurst' in stats:
        lines.append(f"Hurst Exponent: {stats['hurst']:.3f}")
    
    lines.append('')
    
    # Dominant regime
    if probabilities:
        dominant = max(probabilities.items(), key=lambda x: x[1])
        lines.append(f"Dominant Regime: {dominant[0]}")
        lines.append(f"Confidence: {dominant[1]*100:.1f}%")
    
    # Entropy
    if 'entropy' in stats:
        lines.append(f"Classification Entropy: {stats['entropy']:.3f}")
    
    return '\n'.join(lines)
