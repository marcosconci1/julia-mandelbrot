"""
Julia Mandelbrot System - A modular Python library for financial time series analysis.

This library implements a comprehensive market regime classification system combining:
- Trend/volatility regime classification (6 regimes)
- Fractal analysis using Hurst exponent
- Fuzzy logic for probabilistic classification
- Forward return analysis and regime transitions
- Rich visualizations and data exports
"""

__version__ = "1.0.0"
__author__ = "Julia Mandelbrot System"

import pandas as pd
import numpy as np
import logging
from typing import Dict, Optional, Tuple, Any
from pathlib import Path

# Set up logging
logger = logging.getLogger(__name__)

# Import configuration
from .config import JMSConfig, DEFAULT_CONFIG

# Import data fetching
from .data import DataFetcher, fetch_data

# Import feature computation
from .features.trend import compute_trend_features
from .features.volatility import compute_volatility_features
from .features.hurst import compute_hurst_features
from .features.fractal import compute_fractal_features

# Import regime classification
from .regimes.classification import RegimeClassifier
from .regimes.fuzzy import FuzzyRegimeClassifier

# Import analysis modules
from .analysis.forward_returns import compute_forward_returns, analyze_forward_returns_by_regime
from .analysis.transitions import compute_transition_matrix
from .analysis.segments import identify_segments, compute_segment_statistics

# Import visualization
try:
    from .visualization.charts import plot_price_with_regimes
    from .visualization.plots import plot_forward_return_distributions, plot_transition_matrix
    from .visualization.gauges import plot_fuzzy_gauge
    VISUALIZATION_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Visualization modules not fully available: {e}")
    VISUALIZATION_AVAILABLE = False

# Import export functionality
try:
    from .output.export import export_full_analysis
    EXPORT_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Export functionality not available: {e}")
    EXPORT_AVAILABLE = False


class JuliaMandelbrotSystem:
    """
    Main API class for Julia Mandelbrot System analysis.
    
    This class orchestrates the complete analysis pipeline from data fetching
    to regime classification, analysis, and visualization.
    """
    
    def __init__(self, config: Optional[JMSConfig] = None):
        """
        Initialize the Julia Mandelbrot System.
        
        Parameters:
        -----------
        config : JMSConfig, optional
            Configuration object. Uses default if not provided.
        """
        self.config = config or DEFAULT_CONFIG
        self.data_fetcher = DataFetcher(self.config)
        self.df = None
        self.segments = None
        self.transition_matrix = None
        self.forward_stats = None
        self.fuzzy_probabilities = None
        
    def fetch_data(self, ticker: str, period: str = '2y', 
                  start: Optional[str] = None, end: Optional[str] = None) -> pd.DataFrame:
        """
        Fetch historical data for analysis.
        
        Parameters:
        -----------
        ticker : str
            Stock ticker symbol
        period : str
            Time period (e.g., '1y', '2y', '5y')
        start : str, optional
            Start date (YYYY-MM-DD)
        end : str, optional
            End date (YYYY-MM-DD)
            
        Returns:
        --------
        pd.DataFrame : Historical price data
        """
        self.ticker = ticker
        self.df = self.data_fetcher.fetch_data(ticker, period=period, start=start, end=end)
        return self.df
    
    def compute_features(self) -> pd.DataFrame:
        """
        Compute all technical indicators and features.
        
        Returns:
        --------
        pd.DataFrame : DataFrame with computed features
        """
        if self.df is None:
            raise ValueError("No data loaded. Call fetch_data() first.")
        
        # Compute trend indicators
        self.df = compute_trend_features(self.df, self.config.to_dict())
        
        # Compute volatility indicators
        self.df = compute_volatility_features(self.df, self.config.to_dict())
        
        # Compute Hurst exponent
        self.df = compute_hurst_features(self.df, self.config.to_dict())
        
        # Compute fractal filter
        self.df = compute_fractal_features(self.df, self.config.to_dict())
        
        return self.df
        
        return self.df
    
    def classify_regimes(self, use_fuzzy: bool = True) -> pd.DataFrame:
        """
        Classify market regimes using crisp or fuzzy logic.
        
        Parameters:
        -----------
        use_fuzzy : bool
            Whether to use fuzzy logic classification
            
        Returns:
        --------
        pd.DataFrame : DataFrame with regime classifications
        """
        if self.df is None:
            raise ValueError("No data loaded. Call fetch_data() first.")
        
        if 'trend_strength' not in self.df.columns:
            self.compute_features()
        
        # Crisp classification
        classifier = RegimeClassifier(
            trend_threshold_up=self.config.trend_threshold_up,
            trend_threshold_down=self.config.trend_threshold_down,
            volatility_threshold=self.config.volatility_percentile
        )
        self.df = classifier.classify(self.df)
        
        # Fuzzy classification if requested
        if use_fuzzy:
            try:
                from .regimes.fuzzy import compute_fuzzy_features
                self.df = compute_fuzzy_features(self.df, self.config.to_dict())
                
                # Store latest fuzzy probabilities
                fuzzy_cols = [col for col in self.df.columns if col.startswith('fuzzy_') and 
                            col not in ['fuzzy_primary_regime', 'fuzzy_confidence', 'fuzzy_entropy']]
                if fuzzy_cols:
                    self.fuzzy_probabilities = {
                        col.replace('fuzzy_', ''): self.df[col].iloc[-1] 
                        for col in fuzzy_cols
                    }
                
            except ImportError:
                print("Warning: scikit-fuzzy not available. Using crisp classification only.")
                use_fuzzy = False
        
        return self.df
    
    def analyze_regimes(self) -> Dict[str, Any]:
        """
        Perform comprehensive regime analysis.
        
        Returns:
        --------
        dict : Analysis results including segments, transitions, and forward returns
        """
        if self.df is None or 'regime' not in self.df.columns:
            raise ValueError("No regime classification found. Call classify_regimes() first.")
        
        # Identify and analyze segments  
        from .analysis.segments import identify_regime_segments, compute_segment_statistics
        segments_list = identify_regime_segments(self.df)
        self.segments = compute_segment_statistics(segments_list)
        
        # Compute transition matrix
        from .analysis.transitions import compute_transition_matrix
        self.transition_matrix = compute_transition_matrix(self.df)
        
        # Compute forward returns
        self.df = compute_forward_returns(
            self.df, 
            horizons=self.config.forward_return_horizons
        )
        
        # Analyze forward returns by regime
        self.forward_stats = analyze_forward_returns_by_regime(self.df)
        
        return {
            'segments': self.segments,
            'transition_matrix': self.transition_matrix,
            'forward_stats': self.forward_stats
        }
    
    def visualize(self, output_dir: Optional[str] = None, show: bool = True) -> Dict[str, Any]:
        """
        Create all visualizations.
        
        Parameters:
        -----------
        output_dir : str, optional
            Directory to save plots
        show : bool
            Whether to display plots
            
        Returns:
        --------
        dict : Dictionary of figure objects
        """
        import matplotlib.pyplot as plt
        
        if self.df is None:
            raise ValueError("No data loaded. Run analysis first.")
        
        figures = {}
        
        if not VISUALIZATION_AVAILABLE:
            logger.warning("Visualization modules not available")
            return figures
        
        # Price chart with regimes
        try:
            fig1 = plot_price_with_regimes(
                self.df, 
                ticker=getattr(self, 'ticker', ''),
                save_path=f"{output_dir}/price_regimes.png" if output_dir else None
            )
            figures['price_regimes'] = fig1
        except Exception as e:
            logger.warning(f"Could not create price chart: {e}")
        
        # Forward return distributions
        if self.forward_stats:
            try:
                fig4 = plot_forward_return_distributions(
                    self.forward_stats,
                    save_path=f"{output_dir}/forward_returns.png" if output_dir else None
                )
                figures['forward_returns'] = fig4
            except Exception as e:
                logger.warning(f"Could not create forward returns chart: {e}")
        
        # Transition matrix
        if self.transition_matrix is not None:
            try:
                fig5 = plot_transition_matrix(
                    self.transition_matrix,
                    save_path=f"{output_dir}/transitions.png" if output_dir else None
                )
                figures['transitions'] = fig5
            except Exception as e:
                logger.warning(f"Could not create transition matrix chart: {e}")
        
        # Fuzzy gauges if available
        if self.fuzzy_probabilities:
            try:
                fig7 = plot_fuzzy_gauge(
                    self.fuzzy_probabilities,
                    save_path=f"{output_dir}/fuzzy_gauge.png" if output_dir else None
                )
                figures['fuzzy_gauge'] = fig7
            except Exception as e:
                logger.warning(f"Could not create fuzzy gauge: {e}")
        
        if show:
            plt.show()
        
        return figures
    
    def export_results(self, output_dir: str, include_fuzzy: bool = True) -> None:
        """
        Export all analysis results to files.
        
        Parameters:
        -----------
        output_dir : str
            Directory to save export files
        include_fuzzy : bool
            Whether to include fuzzy outputs
        """
        if self.df is None:
            raise ValueError("No analysis results to export. Run analysis first.")
        
        if not EXPORT_AVAILABLE:
            logger.warning("Export functionality not available")
            return
        
        try:
            export_full_analysis(
                self.df,
                self.segments if self.segments is not None else pd.DataFrame(),
                self.transition_matrix if self.transition_matrix is not None else pd.DataFrame(),
                self.forward_stats if self.forward_stats else {},
                output_dir,
                ticker=getattr(self, 'ticker', ''),
                include_fuzzy=include_fuzzy
            )
        except Exception as e:
            logger.warning(f"Export failed: {e}")
            # Fallback: save basic CSV
            try:
                from pathlib import Path
                Path(output_dir).mkdir(parents=True, exist_ok=True)
                self.df.to_csv(f"{output_dir}/analysis_results.csv")
                logger.info(f"Basic results exported to {output_dir}/analysis_results.csv")
            except Exception as e2:
                logger.error(f"Fallback export also failed: {e2}")
    
    def run_full_analysis(self, ticker: str, period: str = '2y',
                         use_fuzzy: bool = True,
                         output_dir: Optional[str] = None,
                         show_plots: bool = True) -> Dict[str, Any]:
        """
        Run complete analysis pipeline.
        
        Parameters:
        -----------
        ticker : str
            Stock ticker symbol
        period : str
            Time period for analysis
        use_fuzzy : bool
            Whether to use fuzzy logic
        output_dir : str, optional
            Directory for outputs
        show_plots : bool
            Whether to display plots
            
        Returns:
        --------
        dict : Complete analysis results
        """
        print(f"Starting Julia Mandelbrot System analysis for {ticker}...")
        
        # Fetch data
        print("Fetching data...")
        self.fetch_data(ticker, period)
        
        # Compute features
        print("Computing technical indicators...")
        self.compute_features()
        
        # Classify regimes
        print("Classifying market regimes...")
        self.classify_regimes(use_fuzzy)
        
        # Analyze regimes
        print("Analyzing regime dynamics...")
        analysis_results = self.analyze_regimes()
        
        # Create visualizations
        if show_plots or output_dir:
            print("Creating visualizations...")
            figures = self.visualize(output_dir, show_plots)
        else:
            figures = {}
        
        # Export results if output directory specified
        if output_dir:
            print(f"Exporting results to {output_dir}...")
            self.export_results(output_dir, include_fuzzy=use_fuzzy)
        
        print("Analysis complete!")
        
        return {
            'data': self.df,
            'segments': self.segments,
            'transition_matrix': self.transition_matrix,
            'forward_stats': self.forward_stats,
            'fuzzy_probabilities': self.fuzzy_probabilities,
            'figures': figures
        }


# Convenience function for quick analysis
def analyze_stock(ticker: str, period: str = '2y', 
                 config: Optional[JMSConfig] = None,
                 output_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    Convenience function to run full analysis on a stock.
    
    Parameters:
    -----------
    ticker : str
        Stock ticker symbol
    period : str
        Time period (e.g., '1y', '2y', '5y')
    config : JMSConfig, optional
        Configuration object
    output_dir : str, optional
        Directory to save outputs
        
    Returns:
    --------
    dict : Analysis results
    """
    jms = JuliaMandelbrotSystem(config)
    return jms.run_full_analysis(ticker, period, output_dir=output_dir)


# Export main components
__all__ = [
    # Main class
    'JuliaMandelbrotSystem',
    
    # Convenience function
    'analyze_stock',
    
    # Configuration
    'JMSConfig',
    'DEFAULT_CONFIG',
    
    # Data
    'DataFetcher',
    'fetch_data',
    
    # Features
    'compute_trend_features',
    'compute_volatility_features',
    'compute_hurst_features',
    'compute_fractal_features',
    
    # Classification
    'RegimeClassifier',
    'FuzzyRegimeClassifier',
    
    # Analysis
    'compute_forward_returns',
    'analyze_forward_returns_by_regime',
    'compute_transition_matrix',
    'identify_segments',
    'compute_segment_statistics',
    
    # Visualization
    'plot_price_with_regimes',
    'plot_regime_timeline',
    'plot_technical_overlays',
    'plot_forward_return_distributions',
    'plot_transition_matrix',
    'plot_segment_statistics',
    'plot_fuzzy_gauge',
    'plot_regime_probabilities',
    'create_nowcast_dashboard',
    
    # Export
    'export_daily_regime_csv',
    'export_segment_summary_csv',
    'export_transition_matrix_csv',
    'export_forward_stats_csv',
    'export_fuzzy_nowcast',
    'export_full_analysis',
    'generate_text_report',
    
    # Version
    '__version__'
]
