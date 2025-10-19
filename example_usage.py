"""
Example usage of the Julia Mandelbrot System library.
Demonstrates complete analysis pipeline for a stock.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import Julia Mandelbrot System modules
from juliams.config import JMSConfig
from juliams.data import DataFetcher
from juliams.features import (
    compute_trend_features,
    compute_volatility_features,
    compute_hurst_features,
    compute_fractal_features
)
from juliams.regimes import RegimeClassifier
from juliams.regimes.fuzzy import FuzzyRegimeClassifier, compute_fuzzy_features
from juliams.analysis import (
    compute_forward_returns,
    analyze_forward_returns_by_regime,
    compute_transition_matrix,
    analyze_regime_segments,
    get_segment_summary
)


def run_complete_analysis(ticker: str = "AAPL", period: str = "2y"):
    """
    Run complete Julia Mandelbrot System analysis on a stock.
    
    Args:
        ticker: Stock ticker symbol
        period: Time period for analysis
    """
    print(f"\n{'='*60}")
    print(f"Julia Mandelbrot System Analysis for {ticker}")
    print(f"{'='*60}\n")
    
    # Step 1: Configuration
    print("Step 1: Setting up configuration...")
    config = JMSConfig()
    config.validate()
    
    # Step 2: Data Fetching
    print(f"Step 2: Fetching data for {ticker} (period: {period})...")
    fetcher = DataFetcher(config)
    df = fetcher.fetch_data(ticker, period=period)
    print(f"  - Fetched {len(df)} days of data")
    print(f"  - Date range: {df.index[0].date()} to {df.index[-1].date()}")
    print(f"  - Current price: ${df['Close'].iloc[-1]:.2f}")
    
    # Step 3: Feature Computation
    print("\nStep 3: Computing features...")
    
    # Trend features
    print("  - Computing trend features...")
    df = compute_trend_features(df, config.to_dict())
    
    # Volatility features
    print("  - Computing volatility features...")
    df = compute_volatility_features(df, config.to_dict())
    
    # Hurst exponent
    print("  - Computing Hurst exponent (this may take a moment)...")
    df = compute_hurst_features(df, config.to_dict())
    
    # Fractal features
    print("  - Computing fractal features...")
    df = compute_fractal_features(df, config.to_dict())
    
    # Step 4: Regime Classification
    print("\nStep 4: Classifying market regimes...")
    
    # Crisp classification
    classifier = RegimeClassifier(
        trend_threshold_up=config.trend_threshold_up,
        trend_threshold_down=config.trend_threshold_down,
        volatility_threshold=config.volatility_percentile
    )
    df = classifier.classify(df)
    
    # Fuzzy classification
    print("  - Computing fuzzy regime memberships...")
    df = compute_fuzzy_features(df, config.to_dict())
    
    # Display current regime
    current_regime = df['regime'].iloc[-1]
    current_regime_name = df['regime_name'].iloc[-1]
    print(f"\n  Current Market Regime: {current_regime_name} ({current_regime})")
    
    if 'fuzzy_confidence' in df.columns:
        confidence = df['fuzzy_confidence'].iloc[-1]
        print(f"  Fuzzy Confidence: {confidence:.1%}")
    
    # Step 5: Forward Returns Analysis
    print("\nStep 5: Analyzing forward returns...")
    df = compute_forward_returns(df, horizons=[5, 10, 20])
    forward_analysis = analyze_forward_returns_by_regime(df)
    
    print("\n  5-Day Forward Returns by Regime:")
    print("  " + "-"*50)
    for regime, stats in forward_analysis.items():
        if regime != 'Overall' and '5d' in stats:
            mean_return = stats['5d']['mean'] * 100
            win_rate = stats['5d']['positive_pct']
            print(f"  {regime:20s}: Mean={mean_return:6.2f}%, Win Rate={win_rate:.1f}%")
    
    # Step 6: Transition Analysis
    print("\nStep 6: Computing regime transitions...")
    transition_matrix = compute_transition_matrix(df)
    
    print("\n  Regime Persistence (self-transition probabilities):")
    for regime in transition_matrix.index:
        persistence = transition_matrix.loc[regime, regime]
        expected_duration = 1 / (1 - persistence) if persistence < 1 else float('inf')
        print(f"  {regime:20s}: {persistence:.1%} (avg duration: {expected_duration:.1f} days)")
    
    # Step 7: Segment Analysis
    print("\nStep 7: Analyzing regime segments...")
    segment_summary = get_segment_summary(df)
    
    if 'statistics' in segment_summary and 'overall' in segment_summary['statistics']:
        overall_stats = segment_summary['statistics']['overall']
        print(f"\n  Total segments: {overall_stats['total_segments']}")
        print(f"  Average segment length: {overall_stats['avg_segment_length']:.1f} days")
        print(f"  Longest segment: {overall_stats['max_segment_length']} days")
    
    # Step 8: Summary Statistics
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    
    # Regime distribution
    regime_counts = df['regime'].value_counts()
    print("\nRegime Distribution:")
    for regime, count in regime_counts.items():
        if regime != 'Unknown':
            pct = count / len(df) * 100
            print(f"  {regime:20s}: {count:4d} days ({pct:5.1f}%)")
    
    # Key indicators
    print("\nCurrent Indicators:")
    print(f"  Trend Strength:    {df['trend_strength'].iloc[-1]:6.3f}")
    print(f"  Volatility:        {df['volatility'].iloc[-1]:6.4f}")
    if 'hurst' in df.columns and not pd.isna(df['hurst'].iloc[-1]):
        print(f"  Hurst Exponent:    {df['hurst'].iloc[-1]:6.3f}")
        
        # Hurst interpretation
        h = df['hurst'].iloc[-1]
        if h > 0.55:
            hurst_interp = "Trending (persistent)"
        elif h < 0.45:
            hurst_interp = "Mean-reverting"
        else:
            hurst_interp = "Random walk"
        print(f"  Hurst Interpretation: {hurst_interp}")
    
    # Fuzzy nowcast
    if 'fuzzy_primary_regime' in df.columns:
        print("\nFuzzy Regime Probabilities (Current):")
        fuzzy_cols = [col for col in df.columns if col.startswith('fuzzy_') and 
                     col not in ['fuzzy_primary_regime', 'fuzzy_confidence', 'fuzzy_entropy']]
        
        memberships = []
        for col in fuzzy_cols:
            regime = col.replace('fuzzy_', '')
            prob = df[col].iloc[-1]
            memberships.append((regime, prob))
        
        # Sort by probability
        memberships.sort(key=lambda x: x[1], reverse=True)
        
        for regime, prob in memberships[:3]:  # Top 3
            print(f"  {regime:20s}: {prob:5.1%}")
    
    return df, segment_summary, transition_matrix


def create_basic_visualizations(df: pd.DataFrame, ticker: str):
    """
    Create basic visualizations of the analysis.
    
    Args:
        df: DataFrame with complete analysis
        ticker: Stock ticker symbol
    """
    # Set style
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")
    
    # Create figure with subplots
    fig, axes = plt.subplots(4, 1, figsize=(14, 12))
    fig.suptitle(f'Julia Mandelbrot System Analysis - {ticker}', fontsize=16, fontweight='bold')
    
    # 1. Price with regime colors
    ax1 = axes[0]
    
    # Define regime colors
    regime_colors = {
        'Up-LowVol': 'green',
        'Up-HighVol': 'lightgreen',
        'Sideways-LowVol': 'orange',
        'Sideways-HighVol': 'darkorange',
        'Down-LowVol': 'lightcoral',
        'Down-HighVol': 'red',
        'Unknown': 'gray'
    }
    
    # Plot price
    ax1.plot(df.index, df['Close'], color='black', linewidth=1, alpha=0.7, label='Close Price')
    
    # Color background by regime
    current_regime = None
    start_idx = 0
    
    for i in range(len(df)):
        regime = df['regime'].iloc[i] if 'regime' in df.columns else 'Unknown'
        
        if regime != current_regime:
            if current_regime is not None and i > start_idx:
                color = regime_colors.get(current_regime, 'gray')
                ax1.axvspan(df.index[start_idx], df.index[i-1], 
                          alpha=0.2, color=color)
            current_regime = regime
            start_idx = i
    
    # Handle last segment
    if current_regime is not None:
        color = regime_colors.get(current_regime, 'gray')
        ax1.axvspan(df.index[start_idx], df.index[-1], 
                  alpha=0.2, color=color)
    
    ax1.set_ylabel('Price ($)', fontsize=10)
    ax1.set_title('Price History with Regime Classification', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='best')
    
    # 2. Trend Strength
    ax2 = axes[1]
    ax2.plot(df.index, df['trend_strength'], color='blue', linewidth=1, label='Trend Strength')
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.axhline(y=0.2, color='green', linestyle='--', linewidth=0.5, alpha=0.5, label='Up Threshold')
    ax2.axhline(y=-0.2, color='red', linestyle='--', linewidth=0.5, alpha=0.5, label='Down Threshold')
    ax2.fill_between(df.index, -0.2, 0.2, alpha=0.1, color='gray', label='Sideways Zone')
    ax2.set_ylabel('Trend Strength', fontsize=10)
    ax2.set_title('Normalized Trend Strength (OLS Slope / Volatility)', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='best', fontsize=8)
    
    # 3. Volatility
    ax3 = axes[2]
    ax3.plot(df.index, df['volatility'], color='purple', linewidth=1, label='Realized Volatility')
    if 'volatility_baseline' in df.columns:
        ax3.plot(df.index, df['volatility_baseline'], color='orange', 
                linewidth=1, linestyle='--', alpha=0.7, label='Baseline (100d MA)')
    ax3.set_ylabel('Volatility', fontsize=10)
    ax3.set_title('Rolling Volatility', fontsize=12)
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc='best', fontsize=8)
    
    # 4. Hurst Exponent
    ax4 = axes[3]
    if 'hurst' in df.columns:
        hurst_data = df['hurst'].dropna()
        
        if len(hurst_data) > 0:
            # Plot the Hurst exponent
            ax4.plot(df.index, df['hurst'], color='brown', linewidth=1, label='Hurst Exponent')
            
            # Dynamic y-axis based on actual data with padding
            data_min = hurst_data.min()
            data_max = hurst_data.max()
            data_range = data_max - data_min
            
            # Add 10% padding above and below
            padding = max(0.05, data_range * 0.1)  # Minimum 0.05 padding
            y_min = max(0.0, data_min - padding)  # Don't go below 0
            y_max = min(1.0, data_max + padding)  # Don't go above 1
            
            # Ensure minimum range for readability
            if (y_max - y_min) < 0.2:
                center = (y_min + y_max) / 2
                y_min = max(0.0, center - 0.1)
                y_max = min(1.0, center + 0.1)
            
            # Always include the theoretical meaningful range (0.4-0.6) if close to data
            if data_min < 0.6 and data_max > 0.4:
                y_min = min(y_min, 0.4)
                y_max = max(y_max, 0.6)
            
            ax4.set_ylim(y_min, y_max)
            
            # Reference lines (only show if they're within the visible range)
            if y_min <= 0.5 <= y_max:
                ax4.axhline(y=0.5, color='black', linestyle='-', linewidth=0.5, label='Random Walk')
            if y_min <= 0.55 <= y_max:
                ax4.axhline(y=0.55, color='green', linestyle='--', linewidth=0.5, alpha=0.5, label='Trending Threshold')
            if y_min <= 0.45 <= y_max:
                ax4.axhline(y=0.45, color='red', linestyle='--', linewidth=0.5, alpha=0.5, label='Mean-Reverting Threshold')
            
            # Color-coded zones (only show if they're within the visible range)
            if y_min <= 0.55 and y_max >= 0.45:
                zone_min = max(y_min, 0.45)
                zone_max = min(y_max, 0.55)
                ax4.fill_between(df.index, zone_min, zone_max, alpha=0.1, color='gray', label='Indeterminate Zone')
            
            # Add current value annotation if available
            if pd.notna(df['hurst'].iloc[-1]):
                current_h = df['hurst'].iloc[-1]
                ax4.annotate(f'Current: {current_h:.3f}', 
                            xy=(df.index[-1], current_h),
                            xytext=(10, 10), textcoords='offset points',
                            bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                            fontsize=8, ha='left')
                            
            # Add interpretation text with dynamic positioning
            interp_text = 'H > 0.55: Persistent\nH < 0.45: Anti-persistent\n0.45 ≤ H ≤ 0.55: Random'
            
            # Position text box based on data distribution
            if data_max < 0.6:
                # Data is in lower range, put text at top
                text_y = 0.98
                text_va = 'top'
            else:
                # Data is in upper range, put text at bottom
                text_y = 0.02
                text_va = 'bottom'
                
            ax4.text(0.02, text_y, interp_text, 
                    transform=ax4.transAxes, fontsize=8, verticalalignment=text_va,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                    
        else:
            # No valid data
            ax4.text(0.5, 0.5, 'Insufficient Data\nfor Hurst Analysis', 
                    transform=ax4.transAxes, ha='center', va='center', fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
            ax4.set_ylim(0.0, 1.0)
    else:
        ax4.text(0.5, 0.5, 'Hurst Exponent\nNot Available', 
                transform=ax4.transAxes, ha='center', va='center', fontsize=12,
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        ax4.set_ylim(0.0, 1.0)
    
    ax4.set_xlabel('Date', fontsize=10)
    ax4.set_title('Hurst Exponent (Fractal Memory Analysis)', fontsize=12)
    ax4.grid(True, alpha=0.3)
    ax4.legend(loc='upper right', fontsize=7)
    
    plt.tight_layout()
    plt.show()
    
    return fig


def main():
    """
    Main function to run the example analysis.
    """
    # Example: Analyze Apple stock
    ticker = "AAPL"
    period = "2y"  # 2 years of data
    
    try:
        # Run complete analysis
        df, segment_summary, transition_matrix = run_complete_analysis(ticker, period)
        
        # Create visualizations
        print("\n" + "="*60)
        print("CREATING VISUALIZATIONS")
        print("="*60)
        fig = create_basic_visualizations(df, ticker)
        
        # Display transition matrix
        print("\nTransition Probability Matrix:")
        print("-"*60)
        print(transition_matrix.round(3))
        
        print("\n" + "="*60)
        print("ANALYSIS COMPLETE")
        print("="*60)
        print(f"\nThe Julia Mandelbrot System has successfully analyzed {ticker}")
        print("Results include:")
        print("  ✓ Market regime classification (6 regimes)")
        print("  ✓ Trend and volatility indicators")
        print("  ✓ Hurst exponent (fractal analysis)")
        print("  ✓ Fuzzy logic probabilities")
        print("  ✓ Forward return distributions")
        print("  ✓ Regime transition analysis")
        print("  ✓ Comprehensive visualizations")
        
        return df, segment_summary, transition_matrix
        
    except Exception as e:
        print(f"\nError during analysis: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None


if __name__ == "__main__":
    # Run the example
    df, segments, transitions = main()
    
    # Optional: Export results
    if df is not None:
        print("\nWould you like to export the results to CSV? (y/n): ", end="")
        # For automated testing, we'll skip the export
        # response = input().lower()
        # if response == 'y':
        #     df.to_csv(f"jms_analysis_{ticker}_{datetime.now().strftime('%Y%m%d')}.csv")
        #     print(f"Results exported to jms_analysis_{ticker}_{datetime.now().strftime('%Y%m%d')}.csv")
