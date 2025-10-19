"""
Statistical plots module for distributions, heatmaps, and analysis visualizations.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Optional, List, Tuple, Any
import warnings

# Set style for better-looking plots
sns.set_style("whitegrid")


def plot_forward_return_distributions(forward_stats: Dict[str, pd.DataFrame],
                                     horizons: List[int] = [5, 10],
                                     figsize: Tuple[int, int] = (15, 10),
                                     save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot forward return distributions for each regime.
    
    Parameters:
    -----------
    forward_stats : dict
        Dictionary with regime names as keys and DataFrames with forward returns
    horizons : list
        List of forward return horizons to plot
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure
        
    Returns:
    --------
    plt.Figure : The matplotlib figure object
    """
    n_regimes = len(forward_stats)
    n_horizons = len(horizons)
    
    fig, axes = plt.subplots(n_regimes, n_horizons, figsize=figsize, squeeze=False)
    
    regime_names = list(forward_stats.keys())
    
    for i, regime in enumerate(regime_names):
        df_regime = forward_stats[regime]
        
        for j, horizon in enumerate(horizons):
            ax = axes[i, j]
            col_name = f'fwd_{horizon}d'
            
            if col_name in df_regime.columns:
                returns = df_regime[col_name].dropna()
                
                if len(returns) > 0:
                    # Create histogram with KDE overlay
                    ax.hist(returns * 100, bins=30, alpha=0.7, density=True,
                           color='steelblue', edgecolor='black')
                    
                    # Add KDE if enough data
                    if len(returns) > 10:
                        returns_pct = returns * 100
                        kde_data = returns_pct[np.isfinite(returns_pct)]
                        if len(kde_data) > 1:
                            try:
                                kde = sns.kdeplot(data=kde_data, ax=ax, color='red', 
                                                linewidth=2, label='KDE')
                            except:
                                pass
                    
                    # Add vertical lines for mean and median
                    mean_val = returns.mean() * 100
                    median_val = returns.median() * 100
                    ax.axvline(mean_val, color='green', linestyle='--', 
                             linewidth=2, label=f'Mean: {mean_val:.2f}%')
                    ax.axvline(median_val, color='orange', linestyle='--', 
                             linewidth=2, label=f'Median: {median_val:.2f}%')
                    ax.axvline(0, color='black', linestyle='-', alpha=0.3)
                    
                    # Add statistics text
                    stats_text = f'n={len(returns)}\nStd={returns.std()*100:.2f}%'
                    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                           fontsize=9, verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                    
                    ax.legend(loc='upper right', fontsize=8)
                else:
                    ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                           transform=ax.transAxes)
            else:
                ax.text(0.5, 0.5, 'Not available', ha='center', va='center',
                       transform=ax.transAxes)
            
            # Labels
            if i == 0:
                ax.set_title(f'{horizon}-Day Forward Returns', fontsize=11, fontweight='bold')
            if j == 0:
                ax.set_ylabel(f'{regime}\nDensity', fontsize=10)
            if i == n_regimes - 1:
                ax.set_xlabel('Return (%)', fontsize=10)
            
            ax.grid(True, alpha=0.3)
    
    plt.suptitle('Forward Return Distributions by Regime', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_transition_matrix(transition_matrix: pd.DataFrame,
                          figsize: Tuple[int, int] = (10, 8),
                          save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot regime transition probability matrix as heatmap.
    
    Parameters:
    -----------
    transition_matrix : pd.DataFrame
        Transition probability matrix
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure
        
    Returns:
    --------
    plt.Figure : The matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create heatmap
    sns.heatmap(transition_matrix, annot=True, fmt='.2f', cmap='YlOrRd',
               cbar_kws={'label': 'Transition Probability'},
               linewidths=0.5, linecolor='gray',
               vmin=0, vmax=1, ax=ax)
    
    # Customize labels
    ax.set_xlabel('Next Regime', fontsize=12, fontweight='bold')
    ax.set_ylabel('Current Regime', fontsize=12, fontweight='bold')
    ax.set_title('Regime Transition Probability Matrix', fontsize=14, fontweight='bold')
    
    # Rotate labels for better readability
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    plt.setp(ax.get_yticklabels(), rotation=0)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_segment_statistics(segments: pd.DataFrame,
                          figsize: Tuple[int, int] = (15, 8),
                          save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot statistics for regime segments.
    
    Parameters:
    -----------
    segments : pd.DataFrame
        DataFrame with segment statistics
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure
        
    Returns:
    --------
    plt.Figure : The matplotlib figure object
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # 1. Average duration by regime
    ax = axes[0, 0]
    if 'regime' in segments.columns and 'duration' in segments.columns:
        avg_duration = segments.groupby('regime')['duration'].mean().sort_values()
        avg_duration.plot(kind='barh', ax=ax, color='steelblue')
        ax.set_xlabel('Average Duration (days)', fontsize=11)
        ax.set_ylabel('Regime', fontsize=11)
        ax.set_title('Average Segment Duration by Regime', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    # 2. Return distribution by regime
    ax = axes[0, 1]
    if 'regime' in segments.columns and 'return' in segments.columns:
        regime_returns = segments.groupby('regime')['return'].apply(list).to_dict()
        
        # Create box plot
        data_to_plot = []
        labels = []
        for regime, returns in regime_returns.items():
            if returns and len(returns) > 0:
                data_to_plot.append([r * 100 for r in returns if not pd.isna(r)])
                labels.append(regime)
        
        if data_to_plot:
            bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True)
            for patch in bp['boxes']:
                patch.set_facecolor('lightblue')
            ax.set_ylabel('Segment Return (%)', fontsize=11)
            ax.set_xlabel('Regime', fontsize=11)
            ax.set_title('Segment Return Distribution by Regime', fontsize=12, fontweight='bold')
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
            ax.grid(True, alpha=0.3)
            ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    
    # 3. Segment count by regime
    ax = axes[1, 0]
    if 'regime' in segments.columns:
        regime_counts = segments['regime'].value_counts()
        regime_counts.plot(kind='bar', ax=ax, color='coral')
        ax.set_xlabel('Regime', fontsize=11)
        ax.set_ylabel('Number of Segments', fontsize=11)
        ax.set_title('Segment Count by Regime', fontsize=12, fontweight='bold')
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        ax.grid(True, alpha=0.3)
    
    # 4. Cumulative return timeline
    ax = axes[1, 1]
    if 'start_date' in segments.columns and 'return' in segments.columns:
        segments_sorted = segments.sort_values('start_date')
        segments_sorted['cumulative_return'] = (1 + segments_sorted['return']).cumprod() - 1
        
        for regime in segments_sorted['regime'].unique():
            mask = segments_sorted['regime'] == regime
            regime_data = segments_sorted[mask]
            ax.plot(regime_data['start_date'], regime_data['cumulative_return'] * 100,
                   marker='o', label=regime, alpha=0.7)
        
        ax.set_xlabel('Date', fontsize=11)
        ax.set_ylabel('Cumulative Return (%)', fontsize=11)
        ax.set_title('Cumulative Returns by Regime Over Time', fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    plt.suptitle('Regime Segment Analysis', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_hurst_timeline(df: pd.DataFrame,
                       figsize: Tuple[int, int] = (15, 6),
                       save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot Hurst exponent over time with regime coloring.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with 'hurst' column
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure
        
    Returns:
    --------
    plt.Figure : The matplotlib figure object
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, 
                                   gridspec_kw={'height_ratios': [2, 1]})
    
    if 'hurst' not in df.columns:
        ax1.text(0.5, 0.5, 'No Hurst data available', 
                ha='center', va='center', transform=ax1.transAxes)
        return fig
    
    # Main Hurst plot
    ax1.plot(df.index, df['hurst'], color='purple', linewidth=1.5, label='Hurst Exponent')
    
    # Add reference lines
    ax1.axhline(y=0.5, color='black', linestyle='-', alpha=0.5, label='Random Walk (0.5)')
    ax1.axhline(y=0.55, color='green', linestyle='--', alpha=0.5, label='Trending (>0.55)')
    ax1.axhline(y=0.45, color='red', linestyle='--', alpha=0.5, label='Mean Reverting (<0.45)')
    
    # Fill areas
    ax1.fill_between(df.index, 0.55, 1, alpha=0.1, color='green')
    ax1.fill_between(df.index, 0.45, 0.55, alpha=0.1, color='gray')
    ax1.fill_between(df.index, 0, 0.45, alpha=0.1, color='red')
    
    ax1.set_ylabel('Hurst Exponent', fontsize=12)
    ax1.set_title('Hurst Exponent Timeline - Market Memory Analysis', 
                 fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1)
    
    # Hurst classification subplot
    if 'hurst_classification' in df.columns:
        # Map classification to numeric values
        class_map = {'Trending': 1, 'Random': 0, 'Mean-Reverting': -1}
        hurst_numeric = df['hurst_classification'].map(class_map).fillna(0)
        
        # Create color map
        colors = []
        for val in hurst_numeric:
            if val > 0:
                colors.append('green')
            elif val < 0:
                colors.append('red')
            else:
                colors.append('gray')
        
        ax2.bar(df.index, np.ones(len(df)), color=colors, width=1, alpha=0.6)
        ax2.set_ylabel('Memory Type', fontsize=11)
        ax2.set_ylim(0, 1.1)
        ax2.set_yticks([])
    else:
        # Show rolling average
        if len(df) > 50:
            ax2.plot(df.index, df['hurst'].rolling(50, min_periods=1).mean(),
                    color='darkblue', linewidth=2, label='50-day MA')
            ax2.set_ylabel('Hurst MA', fontsize=11)
            ax2.legend(loc='upper right', fontsize=9)
            ax2.grid(True, alpha=0.3)
    
    ax2.set_xlabel('Date', fontsize=12)
    
    # Format dates
    from matplotlib.dates import DateFormatter
    ax2.xaxis.set_major_formatter(DateFormatter('%Y-%m'))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_correlation_matrix(df: pd.DataFrame,
                          features: List[str] = None,
                          figsize: Tuple[int, int] = (10, 8),
                          save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot correlation matrix of features.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with feature columns
    features : list, optional
        List of features to include
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure
        
    Returns:
    --------
    plt.Figure : The matplotlib figure object
    """
    if features is None:
        features = ['trend_strength', 'volatility', 'hurst', 'volume_ratio']
    
    # Filter to available features
    available = [f for f in features if f in df.columns]
    
    if len(available) < 2:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, 'Insufficient features for correlation matrix',
               ha='center', va='center', transform=ax.transAxes)
        return fig
    
    # Calculate correlation matrix
    corr_matrix = df[available].corr()
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=figsize)
    
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
               center=0, vmin=-1, vmax=1,
               square=True, linewidths=1,
               cbar_kws={'label': 'Correlation Coefficient'},
               ax=ax)
    
    ax.set_title('Feature Correlation Matrix', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig
