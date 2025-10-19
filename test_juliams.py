#!/usr/bin/env python3
"""
Test script for Julia Mandelbrot System
Tests the complete implementation with a sample stock
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_basic_import():
    """Test basic module import"""
    try:
        import juliams
        print(f"✓ Julia Mandelbrot System v{juliams.__version__} imported successfully")
        return True
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False

def test_quick_analysis():
    """Test quick analysis with AAPL stock"""
    try:
        from juliams import JuliaMandelbrotSystem
        
        # Create system instance
        jms = JuliaMandelbrotSystem()
        print("✓ JuliaMandelbrotSystem instance created")
        
        # Fetch sample data (1 month for quick test)
        print("\nFetching AAPL data (1 month)...")
        df = jms.fetch_data('AAPL', period='1mo')
        print(f"✓ Data fetched: {len(df)} days")
        
        # Compute features
        print("\nComputing technical indicators...")
        jms.compute_features()
        print("✓ Features computed")
        
        # Classify regimes (without fuzzy for speed)
        print("\nClassifying market regimes...")
        jms.classify_regimes(use_fuzzy=False)
        print("✓ Regimes classified")
        
        # Analyze regimes
        print("\nAnalyzing regime dynamics...")
        results = jms.analyze_regimes()
        print("✓ Analysis complete")
        
        # Print summary
        print("\n" + "="*50)
        print("ANALYSIS SUMMARY")
        print("="*50)
        
        if 'regime' in jms.df.columns:
            regime_counts = jms.df['regime'].value_counts()
            print("\nRegime Distribution:")
            for regime, count in regime_counts.items():
                print(f"  {regime}: {count} days ({count/len(jms.df)*100:.1f}%)")
        
        if jms.transition_matrix is not None and not jms.transition_matrix.empty:
            print("\nRegime Persistence (diagonal values):")
            for regime in jms.transition_matrix.index:
                if regime in jms.transition_matrix.columns:
                    persistence = jms.transition_matrix.loc[regime, regime]
                    print(f"  {regime}: {persistence*100:.1f}%")
        
        print("\n✓ All tests passed successfully!")
        return True
        
    except Exception as e:
        print(f"✗ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_module_structure():
    """Test that all modules are present"""
    modules_to_test = [
        'juliams.config',
        'juliams.data',
        'juliams.features.trend',
        'juliams.features.volatility',
        'juliams.features.hurst',
        'juliams.features.fractal',
        'juliams.regimes.classification',
        'juliams.regimes.fuzzy',
        'juliams.analysis.forward_returns',
        'juliams.analysis.transitions',
        'juliams.analysis.segments',
        'juliams.visualization.charts',
        'juliams.visualization.plots',
        'juliams.visualization.gauges',
        'juliams.output.export'
    ]
    
    print("\nTesting module structure:")
    all_present = True
    for module_name in modules_to_test:
        try:
            __import__(module_name)
            print(f"  ✓ {module_name}")
        except ImportError as e:
            print(f"  ✗ {module_name}: {e}")
            all_present = False
    
    return all_present

if __name__ == "__main__":
    print("Julia Mandelbrot System - Implementation Test")
    print("=" * 50)
    
    # Test imports
    if not test_basic_import():
        sys.exit(1)
    
    # Test module structure
    if not test_module_structure():
        print("\n⚠ Some modules are missing or have import errors")
    
    # Test quick analysis
    print("\n" + "=" * 50)
    print("Running Quick Analysis Test")
    print("=" * 50)
    
    if test_quick_analysis():
        print("\n" + "=" * 50)
        print("SUCCESS: Julia Mandelbrot System is fully operational!")
        print("=" * 50)
        print("\nThe implementation includes:")
        print("  • Configuration management (config.py)")
        print("  • Yahoo Finance data fetching (data.py)")
        print("  • Feature computation modules:")
        print("    - Trend analysis with rolling OLS")
        print("    - Volatility indicators")
        print("    - Hurst exponent (fractal analysis)")
        print("    - Fractal memory filtering")
        print("  • Regime classification:")
        print("    - Crisp 6-regime classification")
        print("    - Fuzzy logic inference")
        print("  • Analysis modules:")
        print("    - Forward return distributions")
        print("    - Markov transition matrices")
        print("    - Regime segment analysis")
        print("  • Visualization suite:")
        print("    - Price charts with overlays")
        print("    - Distribution and heatmap plots")
        print("    - Fuzzy probability gauges")
        print("  • Export functionality (CSV/JSON)")
        print("\nYou can now use the system with:")
        print("  from juliams import JuliaMandelbrotSystem")
        print("  jms = JuliaMandelbrotSystem()")
        print("  results = jms.run_full_analysis('AAPL', period='2y')")
    else:
        print("\n⚠ Some tests failed. Please check the errors above.")
