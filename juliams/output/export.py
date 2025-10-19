"""
Export functionality for Julia Mandelbrot System analysis results.
"""

import pandas as pd
import numpy as np
import json
from typing import Dict, Optional, Any, List
from pathlib import Path
import datetime


def export_daily_regime_csv(df: pd.DataFrame, 
                           filepath: str,
                           include_fuzzy: bool = False) -> None:
    """
    Export daily regime classifications with indicators to CSV.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with daily data and regime classifications
    filepath : str
        Path to save CSV file
    include_fuzzy : bool
        Whether to include fuzzy membership columns
    """
    # Select columns to export
    columns = ['Close', 'Volume', 'log_return', 'trend_strength', 
               'volatility', 'hurst', 'regime']
    
    # Add trend and volatility regime columns if available
    if 'trend_regime' in df.columns:
        columns.append('trend_regime')
    if 'vol_regime' in df.columns:
        columns.append('vol_regime')
    
    # Add forward returns if available
    for col in df.columns:
        if col.startswith('fwd_'):
            columns.append(col)
    
    # Add fuzzy memberships if requested and available
    if include_fuzzy:
        fuzzy_cols = [col for col in df.columns if 'membership' in col or 'prob_' in col]
        columns.extend(fuzzy_cols)
    
    # Filter to available columns
    export_cols = [col for col in columns if col in df.columns]
    
    # Export to CSV
    export_df = df[export_cols].copy()
    export_df.index.name = 'Date'
    export_df.to_csv(filepath, float_format='%.6f')
    
    print(f"Daily regime data exported to {filepath}")


def export_segment_summary_csv(segments: pd.DataFrame, 
                              filepath: str) -> None:
    """
    Export regime segment summaries to CSV.
    
    Parameters:
    -----------
    segments : pd.DataFrame
        DataFrame with segment summaries
    filepath : str
        Path to save CSV file
    """
    if segments.empty:
        print("No segments to export")
        return
    
    # Ensure proper column order
    column_order = ['regime', 'start_date', 'end_date', 'duration', 
                   'return', 'volatility', 'avg_hurst']
    
    # Add any additional columns
    for col in segments.columns:
        if col not in column_order:
            column_order.append(col)
    
    # Filter to available columns
    export_cols = [col for col in column_order if col in segments.columns]
    
    # Export
    export_df = segments[export_cols].copy()
    export_df.to_csv(filepath, index=False, float_format='%.6f')
    
    print(f"Segment summary exported to {filepath}")


def export_transition_matrix_csv(transition_matrix: pd.DataFrame,
                                filepath: str,
                                include_counts: bool = True) -> None:
    """
    Export regime transition probability matrix to CSV.
    
    Parameters:
    -----------
    transition_matrix : pd.DataFrame
        Transition probability matrix
    filepath : str
        Path to save CSV file
    include_counts : bool
        Whether to also export transition counts
    """
    # Export probability matrix
    transition_matrix.to_csv(filepath, float_format='%.4f')
    print(f"Transition matrix exported to {filepath}")
    
    # Export counts if requested
    if include_counts and hasattr(transition_matrix, 'counts'):
        counts_path = filepath.replace('.csv', '_counts.csv')
        transition_matrix.counts.to_csv(counts_path)
        print(f"Transition counts exported to {counts_path}")


def export_forward_stats_csv(forward_stats: Dict[str, pd.DataFrame],
                            filepath: str) -> None:
    """
    Export forward return statistics by regime to CSV.
    
    Parameters:
    -----------
    forward_stats : dict
        Dictionary with regime names as keys and stats DataFrames as values
    filepath : str
        Path to save CSV file
    """
    # Combine all regime stats into one DataFrame
    all_stats = []
    
    for regime, stats in forward_stats.items():
        if isinstance(stats, pd.DataFrame) and not stats.empty:
            stats_copy = stats.copy()
            stats_copy['regime'] = regime
            all_stats.append(stats_copy)
    
    if all_stats:
        combined_df = pd.concat(all_stats, ignore_index=True)
        
        # Reorder columns with regime first
        cols = ['regime'] + [col for col in combined_df.columns if col != 'regime']
        combined_df = combined_df[cols]
        
        # Export
        combined_df.to_csv(filepath, index=False, float_format='%.6f')
        print(f"Forward return statistics exported to {filepath}")
    else:
        print("No forward return statistics to export")


def export_fuzzy_nowcast(probabilities: Dict[str, float],
                        current_stats: Dict[str, float],
                        filepath: str,
                        format: str = 'json',
                        ticker: str = '',
                        date: Optional[datetime.date] = None) -> None:
    """
    Export fuzzy nowcast output in JSON or text format.
    
    Parameters:
    -----------
    probabilities : dict
        Current regime probabilities
    current_stats : dict
        Current indicator values
    filepath : str
        Path to save file
    format : str
        'json' or 'text'
    ticker : str
        Stock ticker symbol
    date : datetime.date, optional
        Analysis date
    """
    if date is None:
        date = datetime.date.today()
    
    if format == 'json':
        # Create JSON structure
        nowcast = {
            'date': date.isoformat(),
            'ticker': ticker,
            'fuzzy_nowcast': probabilities,
            'indicators': {
                'trend_strength': current_stats.get('trend_strength', None),
                'volatility': current_stats.get('volatility', None),
                'hurst': current_stats.get('hurst', None)
            }
        }
        
        # Find primary regime
        if probabilities:
            primary = max(probabilities.items(), key=lambda x: x[1])
            nowcast['primary_regime'] = primary[0]
            nowcast['confidence'] = primary[1]
        
        # Add entropy if available
        if 'entropy' in current_stats:
            nowcast['entropy'] = current_stats['entropy']
        
        # Export to JSON
        with open(filepath, 'w') as f:
            json.dump(nowcast, f, indent=2)
        
        print(f"Fuzzy nowcast exported to {filepath}")
        
    elif format == 'text':
        # Create text report
        lines = []
        lines.append("=" * 60)
        lines.append("JULIA MANDELBROT SYSTEM - MARKET NOWCAST")
        lines.append("=" * 60)
        lines.append(f"Date: {date.isoformat()}")
        if ticker:
            lines.append(f"Ticker: {ticker}")
        lines.append("")
        
        # Current indicators
        lines.append("CURRENT INDICATORS:")
        lines.append("-" * 30)
        lines.append(f"Trend Strength: {current_stats.get('trend_strength', 'N/A'):.3f}")
        lines.append(f"Volatility: {current_stats.get('volatility', 'N/A'):.4f}")
        lines.append(f"Hurst Exponent: {current_stats.get('hurst', 'N/A'):.3f}")
        lines.append("")
        
        # Regime probabilities
        lines.append("REGIME PROBABILITIES:")
        lines.append("-" * 30)
        sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
        for regime, prob in sorted_probs:
            lines.append(f"{regime:20s}: {prob*100:6.2f}%")
        lines.append("")
        
        # Primary regime assessment
        if probabilities:
            primary = max(probabilities.items(), key=lambda x: x[1])
            lines.append("PRIMARY ASSESSMENT:")
            lines.append("-" * 30)
            lines.append(f"Dominant Regime: {primary[0]}")
            lines.append(f"Confidence: {primary[1]*100:.1f}%")
            
            # Interpretation
            if primary[1] > 0.6:
                confidence_text = "High confidence"
            elif primary[1] > 0.4:
                confidence_text = "Moderate confidence"
            else:
                confidence_text = "Low confidence - mixed regime"
            
            lines.append(f"Interpretation: {confidence_text}")
            
            # Market condition summary
            lines.append("")
            lines.append("MARKET CONDITION SUMMARY:")
            lines.append("-" * 30)
            lines.append(_generate_market_summary(primary[0], probabilities, current_stats))
        
        lines.append("")
        lines.append("=" * 60)
        
        # Write to file
        with open(filepath, 'w') as f:
            f.write('\n'.join(lines))
        
        print(f"Fuzzy nowcast text report exported to {filepath}")


def export_full_analysis(df: pd.DataFrame,
                        segments: pd.DataFrame,
                        transition_matrix: pd.DataFrame,
                        forward_stats: Dict[str, pd.DataFrame],
                        output_dir: str,
                        ticker: str = '',
                        include_fuzzy: bool = False) -> None:
    """
    Export complete analysis results to multiple files.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Daily data with all indicators and classifications
    segments : pd.DataFrame
        Segment summaries
    transition_matrix : pd.DataFrame
        Transition probability matrix
    forward_stats : dict
        Forward return statistics by regime
    output_dir : str
        Directory to save all files
    ticker : str
        Stock ticker symbol
    include_fuzzy : bool
        Whether to include fuzzy outputs
    """
    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Generate timestamp for filenames
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    prefix = f"{ticker}_{timestamp}" if ticker else timestamp
    
    # Export daily regime data
    daily_path = output_path / f"{prefix}_daily_regimes.csv"
    export_daily_regime_csv(df, str(daily_path), include_fuzzy)
    
    # Export segment summary
    if not segments.empty:
        segment_path = output_path / f"{prefix}_segments.csv"
        export_segment_summary_csv(segments, str(segment_path))
    
    # Export transition matrix
    if not transition_matrix.empty:
        trans_path = output_path / f"{prefix}_transitions.csv"
        export_transition_matrix_csv(transition_matrix, str(trans_path))
    
    # Export forward return stats
    if forward_stats:
        forward_path = output_path / f"{prefix}_forward_returns.csv"
        export_forward_stats_csv(forward_stats, str(forward_path))
    
    # Export fuzzy nowcast if available
    if include_fuzzy and 'fuzzy_probabilities' in df.columns:
        # Get latest probabilities
        latest_probs = df['fuzzy_probabilities'].iloc[-1] if 'fuzzy_probabilities' in df.columns else {}
        current_stats = {
            'trend_strength': df['trend_strength'].iloc[-1] if 'trend_strength' in df.columns else None,
            'volatility': df['volatility'].iloc[-1] if 'volatility' in df.columns else None,
            'hurst': df['hurst'].iloc[-1] if 'hurst' in df.columns else None
        }
        
        # JSON nowcast
        json_path = output_path / f"{prefix}_nowcast.json"
        export_fuzzy_nowcast(latest_probs, current_stats, str(json_path), 
                           format='json', ticker=ticker, date=df.index[-1])
        
        # Text nowcast
        text_path = output_path / f"{prefix}_nowcast.txt"
        export_fuzzy_nowcast(latest_probs, current_stats, str(text_path),
                           format='text', ticker=ticker, date=df.index[-1])
    
    # Generate summary report
    report_path = output_path / f"{prefix}_summary_report.txt"
    generate_text_report(df, segments, transition_matrix, forward_stats,
                        str(report_path), ticker)
    
    print(f"\nFull analysis exported to {output_dir}")
    print(f"Files created with prefix: {prefix}")


def generate_text_report(df: pd.DataFrame,
                        segments: pd.DataFrame,
                        transition_matrix: pd.DataFrame,
                        forward_stats: Dict[str, pd.DataFrame],
                        filepath: str,
                        ticker: str = '') -> None:
    """
    Generate comprehensive text summary report.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Daily data with indicators
    segments : pd.DataFrame
        Segment summaries
    transition_matrix : pd.DataFrame
        Transition matrix
    forward_stats : dict
        Forward return statistics
    filepath : str
        Path to save report
    ticker : str
        Stock ticker
    """
    lines = []
    lines.append("=" * 80)
    lines.append("JULIA MANDELBROT SYSTEM - COMPREHENSIVE ANALYSIS REPORT")
    lines.append("=" * 80)
    lines.append(f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    if ticker:
        lines.append(f"Ticker: {ticker}")
    lines.append(f"Analysis Period: {df.index[0].date()} to {df.index[-1].date()}")
    lines.append(f"Total Days: {len(df)}")
    lines.append("")
    
    # Data Overview
    lines.append("DATA OVERVIEW")
    lines.append("-" * 40)
    lines.append(f"Start Price: ${df['Close'].iloc[0]:.2f}")
    lines.append(f"End Price: ${df['Close'].iloc[-1]:.2f}")
    lines.append(f"Total Return: {((df['Close'].iloc[-1] / df['Close'].iloc[0]) - 1) * 100:.2f}%")
    lines.append(f"Volatility (annualized): {df['log_return'].std() * np.sqrt(252) * 100:.2f}%")
    lines.append("")
    
    # Regime Distribution
    if 'regime' in df.columns:
        lines.append("REGIME DISTRIBUTION")
        lines.append("-" * 40)
        regime_counts = df['regime'].value_counts()
        regime_pcts = df['regime'].value_counts(normalize=True) * 100
        
        for regime in regime_counts.index:
            lines.append(f"{regime:20s}: {regime_counts[regime]:5d} days ({regime_pcts[regime]:6.2f}%)")
        lines.append("")
    
    # Segment Analysis
    if not segments.empty:
        lines.append("SEGMENT ANALYSIS")
        lines.append("-" * 40)
        lines.append(f"Total Segments: {len(segments)}")
        lines.append(f"Average Segment Duration: {segments['duration'].mean():.1f} days")
        lines.append(f"Longest Segment: {segments['duration'].max()} days")
        lines.append(f"Shortest Segment: {segments['duration'].min()} days")
        lines.append("")
        
        # Average returns by regime
        lines.append("Average Segment Returns by Regime:")
        avg_returns = segments.groupby('regime')['return'].mean() * 100
        for regime, ret in avg_returns.items():
            lines.append(f"  {regime:20s}: {ret:7.3f}%")
        lines.append("")
    
    # Transition Analysis
    if not transition_matrix.empty:
        lines.append("REGIME PERSISTENCE (Diagonal Values)")
        lines.append("-" * 40)
        for regime in transition_matrix.index:
            if regime in transition_matrix.columns:
                persistence = transition_matrix.loc[regime, regime]
                lines.append(f"{regime:20s}: {persistence*100:6.2f}%")
        lines.append("")
    
    # Forward Return Analysis
    if forward_stats:
        lines.append("FORWARD RETURN ANALYSIS")
        lines.append("-" * 40)
        lines.append("Average 5-Day Forward Returns by Regime:")
        
        for regime, stats in forward_stats.items():
            if isinstance(stats, pd.DataFrame) and 'fwd_5d' in stats.columns:
                mean_ret = stats['fwd_5d'].mean() * 100
                std_ret = stats['fwd_5d'].std() * 100
                lines.append(f"  {regime:20s}: {mean_ret:7.3f}% (±{std_ret:.3f}%)")
        lines.append("")
    
    # Current State
    lines.append("CURRENT STATE (Latest Values)")
    lines.append("-" * 40)
    if 'trend_strength' in df.columns:
        lines.append(f"Trend Strength: {df['trend_strength'].iloc[-1]:.3f}")
    if 'volatility' in df.columns:
        lines.append(f"Volatility: {df['volatility'].iloc[-1]:.4f}")
    if 'hurst' in df.columns:
        lines.append(f"Hurst Exponent: {df['hurst'].iloc[-1]:.3f}")
    if 'regime' in df.columns:
        lines.append(f"Current Regime: {df['regime'].iloc[-1]}")
    lines.append("")
    
    # Key Insights
    lines.append("KEY INSIGHTS")
    lines.append("-" * 40)
    lines.extend(_generate_insights(df, segments, transition_matrix, forward_stats))
    
    lines.append("")
    lines.append("=" * 80)
    lines.append("END OF REPORT")
    lines.append("=" * 80)
    
    # Write to file
    with open(filepath, 'w') as f:
        f.write('\n'.join(lines))
    
    print(f"Summary report generated: {filepath}")


def _generate_market_summary(primary_regime: str, 
                            probabilities: Dict[str, float],
                            stats: Dict[str, float]) -> str:
    """Generate market condition summary text."""
    summary = []
    
    # Trend assessment
    up_prob = sum(probabilities.get(r, 0) for r in ['Up-LowVol', 'Up-HighVol'])
    down_prob = sum(probabilities.get(r, 0) for r in ['Down-LowVol', 'Down-HighVol'])
    
    if up_prob > 0.6:
        summary.append("Market shows strong upward trend momentum.")
    elif down_prob > 0.6:
        summary.append("Market shows strong downward trend pressure.")
    else:
        summary.append("Market is in a transitional or sideways phase.")
    
    # Volatility assessment
    high_vol_prob = sum(probabilities.get(r, 0) for r in ['Up-HighVol', 'Sideways-HighVol', 'Down-HighVol'])
    
    if high_vol_prob > 0.6:
        summary.append("Volatility is elevated, suggesting increased risk.")
    else:
        summary.append("Volatility is relatively contained.")
    
    # Hurst assessment
    if 'hurst' in stats and stats['hurst'] is not None:
        if stats['hurst'] > 0.55:
            summary.append("Fractal analysis indicates persistent trending behavior.")
        elif stats['hurst'] < 0.45:
            summary.append("Fractal analysis suggests mean-reverting behavior.")
        else:
            summary.append("Fractal analysis shows random walk characteristics.")
    
    return ' '.join(summary)


def _generate_insights(df: pd.DataFrame,
                      segments: pd.DataFrame,
                      transition_matrix: pd.DataFrame,
                      forward_stats: Dict[str, pd.DataFrame]) -> List[str]:
    """Generate key insights from the analysis."""
    insights = []
    
    # Trend insights
    if 'regime' in df.columns:
        up_days = df['regime'].str.contains('Up').sum()
        down_days = df['regime'].str.contains('Down').sum()
        total_days = len(df)
        
        if up_days > down_days * 1.5:
            insights.append(f"• Market spent {up_days/total_days*100:.1f}% of time in uptrend regimes")
        elif down_days > up_days * 1.5:
            insights.append(f"• Market spent {down_days/total_days*100:.1f}% of time in downtrend regimes")
    
    # Volatility insights
    if 'volatility' in df.columns:
        recent_vol = df['volatility'].tail(20).mean()
        historical_vol = df['volatility'].mean()
        
        if recent_vol > historical_vol * 1.2:
            insights.append("• Recent volatility is elevated compared to historical average")
        elif recent_vol < historical_vol * 0.8:
            insights.append("• Recent volatility is subdued compared to historical average")
    
    # Hurst insights
    if 'hurst' in df.columns:
        trending_pct = (df['hurst'] > 0.55).sum() / len(df) * 100
        if trending_pct > 60:
            insights.append(f"• Strong trending behavior observed {trending_pct:.1f}% of the time")
    
    # Regime persistence
    if not transition_matrix.empty:
        avg_persistence = np.diag(transition_matrix).mean()
        if avg_persistence > 0.7:
            insights.append(f"• High regime persistence ({avg_persistence*100:.1f}% average)")
        elif avg_persistence < 0.5:
            insights.append(f"• Low regime persistence suggests choppy conditions")
    
    # Forward returns
    if forward_stats:
        best_regime = None
        best_return = -np.inf
        
        for regime, stats in forward_stats.items():
            if isinstance(stats, pd.DataFrame) and 'fwd_5d' in stats.columns:
                mean_ret = stats['fwd_5d'].mean()
                if mean_ret > best_return:
                    best_return = mean_ret
                    best_regime = regime
        
        if best_regime:
            insights.append(f"• Best 5-day forward returns in {best_regime} regime ({best_return*100:.2f}%)")
    
    return insights if insights else ["• Analysis complete. Review detailed statistics above."]
