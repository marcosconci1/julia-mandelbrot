"""
Forward returns analysis for the Julia Mandelbrot System.
Analyzes future returns associated with each market regime.
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, List, Union, Tuple
import logging

logger = logging.getLogger(__name__)


def compute_forward_returns(df: pd.DataFrame,
                           horizons: List[int] = [5, 10, 20],
                           price_col: str = 'Close',
                           method: str = 'simple') -> pd.DataFrame:
    """
    Compute forward returns for specified horizons.
    
    Args:
        df: DataFrame with price data
        horizons: List of forward-looking periods (in days)
        price_col: Column name for price data
        method: Return calculation method ('simple' or 'log')
    
    Returns:
        DataFrame with forward return columns added
    """
    df = df.copy()
    
    if price_col not in df.columns:
        raise ValueError(f"Column '{price_col}' not found in DataFrame")
    
    for horizon in horizons:
        col_name = f'fwd_{horizon}d_return'
        
        if method == 'simple':
            # Simple return: (future_price / current_price) - 1
            future_prices = df[price_col].shift(-horizon)
            df[col_name] = (future_prices / df[price_col]) - 1
        elif method == 'log':
            # Log return: log(future_price / current_price)
            future_prices = df[price_col].shift(-horizon)
            # Use np.log to handle potential NaN/inf values better
            df[col_name] = np.log(future_prices / df[price_col])
        else:
            raise ValueError(f"Unknown return method: {method}")
        
        # Also compute cumulative return over the period
        cum_col_name = f'fwd_{horizon}d_cumulative'
        future_prices = df[price_col].shift(-horizon)
        df[cum_col_name] = future_prices / df[price_col]
        
        # Remove invalid values (infinity, negative values)
        if method == 'log':
            df[col_name] = df[col_name].replace([np.inf, -np.inf], np.nan)
        
        # Only keep finite values
        mask = np.isfinite(df[col_name])
        df.loc[~mask, col_name] = np.nan
    
    logger.info(f"Computed forward returns for horizons: {horizons}")
    
    return df


def analyze_forward_returns_by_regime(df: pd.DataFrame,
                                     regime_col: str = 'regime',
                                     horizons: Optional[List[int]] = None) -> Dict:
    """
    Analyze forward return distributions by regime.
    
    Args:
        df: DataFrame with regime classifications and forward returns
        regime_col: Column name for regime classification
        horizons: List of horizons to analyze (auto-detect if None)
    
    Returns:
        Dictionary with forward return statistics per regime
    """
    if regime_col not in df.columns:
        raise ValueError(f"Column '{regime_col}' not found in DataFrame")
    
    # Auto-detect forward return columns if horizons not specified
    if horizons is None:
        fwd_cols = [col for col in df.columns if col.startswith('fwd_') and col.endswith('_return')]
        horizons = []
        for col in fwd_cols:
            try:
                horizon = int(col.split('_')[1].replace('d', ''))
                if horizon not in horizons:
                    horizons.append(horizon)
            except:
                continue
        horizons.sort()
    
    if not horizons:
        logger.warning("No forward return columns found")
        return {}
    
    # Get unique regimes
    regimes = df[regime_col].dropna().unique()
    regimes = [r for r in regimes if r != 'Unknown']
    
    results = {}
    
    for regime in regimes:
        regime_data = df[df[regime_col] == regime]
        regime_results = {
            'count': len(regime_data),
            'percentage': len(regime_data) / len(df) * 100
        }
        
        for horizon in horizons:
            col_name = f'fwd_{horizon}d_return'
            if col_name in regime_data.columns:
                returns = regime_data[col_name].dropna()
                
                if len(returns) > 0:
                    regime_results[f'{horizon}d'] = {
                        'count': len(returns),
                        'mean': returns.mean(),
                        'median': returns.median(),
                        'std': returns.std(),
                        'min': returns.min(),
                        'max': returns.max(),
                        'skew': returns.skew(),
                        'kurtosis': returns.kurtosis(),
                        'percentiles': {
                            5: returns.quantile(0.05),
                            25: returns.quantile(0.25),
                            75: returns.quantile(0.75),
                            95: returns.quantile(0.95)
                        },
                        'positive_pct': (returns > 0).sum() / len(returns) * 100,
                        'sharpe': returns.mean() / returns.std() if returns.std() > 0 else 0,
                        'sortino': calculate_sortino_ratio(returns)
                    }
        
        results[regime] = regime_results
    
    # Add overall statistics
    overall_results = {
        'count': len(df),
    }
    
    for horizon in horizons:
        col_name = f'fwd_{horizon}d_return'
        if col_name in df.columns:
            returns = df[col_name].dropna()
            
            if len(returns) > 0:
                overall_results[f'{horizon}d'] = {
                    'count': len(returns),
                    'mean': returns.mean(),
                    'median': returns.median(),
                    'std': returns.std(),
                    'min': returns.min(),
                    'max': returns.max(),
                    'positive_pct': (returns > 0).sum() / len(returns) * 100,
                    'sharpe': returns.mean() / returns.std() if returns.std() > 0 else 0
                }
    
    results['Overall'] = overall_results
    
    return results


def calculate_sortino_ratio(returns: pd.Series, target_return: float = 0) -> float:
    """
    Calculate Sortino ratio (uses downside deviation).
    
    Args:
        returns: Series of returns
        target_return: Minimum acceptable return
    
    Returns:
        Sortino ratio
    """
    excess_returns = returns - target_return
    downside_returns = excess_returns[excess_returns < 0]
    
    if len(downside_returns) == 0:
        return np.inf if excess_returns.mean() > 0 else 0
    
    downside_deviation = np.sqrt((downside_returns ** 2).mean())
    
    if downside_deviation == 0:
        return np.inf if excess_returns.mean() > 0 else 0
    
    return excess_returns.mean() / downside_deviation


def compute_conditional_forward_returns(df: pd.DataFrame,
                                       condition_col: str,
                                       condition_value: Union[str, float],
                                       horizons: List[int] = [5, 10]) -> Dict:
    """
    Compute forward returns conditional on a specific indicator value.
    
    Args:
        df: DataFrame with data
        condition_col: Column to condition on
        condition_value: Value to filter for
        horizons: Forward-looking horizons
    
    Returns:
        Dictionary with conditional forward return statistics
    """
    if condition_col not in df.columns:
        raise ValueError(f"Column '{condition_col}' not found in DataFrame")
    
    # Filter data based on condition
    if isinstance(condition_value, str):
        filtered_df = df[df[condition_col] == condition_value]
    else:
        # For numeric conditions, use threshold
        filtered_df = df[df[condition_col] > condition_value]
    
    results = {
        'condition': f"{condition_col} = {condition_value}",
        'count': len(filtered_df),
        'percentage': len(filtered_df) / len(df) * 100
    }
    
    for horizon in horizons:
        col_name = f'fwd_{horizon}d_return'
        if col_name in filtered_df.columns:
            returns = filtered_df[col_name].dropna()
            
            if len(returns) > 0:
                results[f'{horizon}d'] = {
                    'mean': returns.mean(),
                    'median': returns.median(),
                    'std': returns.std(),
                    'positive_pct': (returns > 0).sum() / len(returns) * 100
                }
    
    return results


def analyze_forward_returns_by_fuzzy_regime(df: pd.DataFrame,
                                           horizons: List[int] = [5, 10]) -> Dict:
    """
    Analyze forward returns weighted by fuzzy regime memberships.
    
    Args:
        df: DataFrame with fuzzy classifications and forward returns
        horizons: List of horizons to analyze
    
    Returns:
        Dictionary with weighted forward return statistics
    """
    # Check for fuzzy columns
    fuzzy_cols = [col for col in df.columns if col.startswith('fuzzy_') and 
                  col not in ['fuzzy_primary_regime', 'fuzzy_confidence', 'fuzzy_entropy']]
    
    if not fuzzy_cols:
        logger.warning("No fuzzy classification columns found")
        return {}
    
    results = {}
    
    for regime_col in fuzzy_cols:
        regime = regime_col.replace('fuzzy_', '')
        regime_results = {
            'avg_membership': df[regime_col].mean()
        }
        
        for horizon in horizons:
            return_col = f'fwd_{horizon}d_return'
            if return_col in df.columns:
                # Weight returns by fuzzy membership
                valid_mask = df[return_col].notna() & df[regime_col].notna()
                if valid_mask.sum() > 0:
                    weights = df.loc[valid_mask, regime_col]
                    returns = df.loc[valid_mask, return_col]
                    
                    # Weighted statistics
                    weighted_mean = np.average(returns, weights=weights) if weights.sum() > 0 else 0
                    
                    # Weighted standard deviation
                    if weights.sum() > 0:
                        variance = np.average((returns - weighted_mean) ** 2, weights=weights)
                        weighted_std = np.sqrt(variance)
                    else:
                        weighted_std = 0
                    
                    regime_results[f'{horizon}d'] = {
                        'weighted_mean': weighted_mean,
                        'weighted_std': weighted_std,
                        'effective_weight': weights.sum(),
                        'weighted_sharpe': weighted_mean / weighted_std if weighted_std > 0 else 0
                    }
        
        results[regime] = regime_results
    
    return results


def get_forward_return_statistics(df: pd.DataFrame,
                                 regime_col: str = 'regime') -> Dict:
    """
    Get comprehensive forward return statistics.
    
    Args:
        df: DataFrame with regime and forward return data
        regime_col: Column name for regime classification
    
    Returns:
        Dictionary with comprehensive statistics
    """
    stats = {}
    
    # Basic forward return analysis by regime
    regime_analysis = analyze_forward_returns_by_regime(df, regime_col)
    stats['by_regime'] = regime_analysis
    
    # Fuzzy-weighted analysis if available
    fuzzy_analysis = analyze_forward_returns_by_fuzzy_regime(df)
    if fuzzy_analysis:
        stats['fuzzy_weighted'] = fuzzy_analysis
    
    # Best and worst regimes for each horizon
    horizons = []
    for col in df.columns:
        if col.startswith('fwd_') and col.endswith('_return'):
            try:
                horizon = int(col.split('_')[1].replace('d', ''))
                if horizon not in horizons:
                    horizons.append(horizon)
            except:
                continue
    
    if horizons and 'by_regime' in stats:
        stats['best_worst'] = {}
        for horizon in horizons:
            best_regime = None
            best_return = -np.inf
            worst_regime = None
            worst_return = np.inf
            
            for regime, regime_stats in stats['by_regime'].items():
                if regime == 'Overall':
                    continue
                if f'{horizon}d' in regime_stats:
                    mean_return = regime_stats[f'{horizon}d']['mean']
                    if mean_return > best_return:
                        best_return = mean_return
                        best_regime = regime
                    if mean_return < worst_return:
                        worst_return = mean_return
                        worst_regime = regime
            
            if best_regime and worst_regime:
                stats['best_worst'][f'{horizon}d'] = {
                    'best_regime': best_regime,
                    'best_return': best_return,
                    'worst_regime': worst_regime,
                    'worst_return': worst_return,
                    'spread': best_return - worst_return
                }
    
    # Risk-adjusted returns
    if 'by_regime' in stats:
        stats['risk_adjusted'] = {}
        for regime, regime_stats in stats['by_regime'].items():
            if regime == 'Overall':
                continue
            
            regime_risk_adj = {}
            for key in regime_stats:
                if 'd' in key and isinstance(regime_stats[key], dict):
                    if 'sharpe' in regime_stats[key]:
                        regime_risk_adj[f'{key}_sharpe'] = regime_stats[key]['sharpe']
                    if 'sortino' in regime_stats[key]:
                        regime_risk_adj[f'{key}_sortino'] = regime_stats[key]['sortino']
            
            if regime_risk_adj:
                stats['risk_adjusted'][regime] = regime_risk_adj
    
    return stats


def create_forward_return_summary_table(analysis: Dict, 
                                       horizon: int = 5) -> pd.DataFrame:
    """
    Create a summary table of forward returns by regime.
    
    Args:
        analysis: Analysis dictionary from analyze_forward_returns_by_regime
        horizon: Horizon to summarize
    
    Returns:
        DataFrame with summary table
    """
    rows = []
    
    for regime, regime_stats in analysis.items():
        if regime == 'Overall':
            continue
        
        if f'{horizon}d' in regime_stats:
            stats = regime_stats[f'{horizon}d']
            row = {
                'Regime': regime,
                'Count': stats['count'],
                'Mean Return': stats['mean'],
                'Median Return': stats['median'],
                'Std Dev': stats['std'],
                'Sharpe Ratio': stats['sharpe'],
                'Win Rate %': stats['positive_pct'],
                'Min': stats['min'],
                'Max': stats['max']
            }
            rows.append(row)
    
    if rows:
        summary_df = pd.DataFrame(rows)
        summary_df = summary_df.sort_values('Mean Return', ascending=False)
        return summary_df
    else:
        return pd.DataFrame()
