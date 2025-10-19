"""
Visualization module for Julia Mandelbrot System.

This module provides comprehensive visualization tools for market regime analysis,
including price charts with regime overlays, fuzzy probability gauges, distribution
plots, and transition matrices.
"""

from .charts import (
    plot_price_with_regimes,
    plot_regime_timeline,
    plot_technical_overlays
)

from .plots import (
    plot_forward_return_distributions,
    plot_transition_matrix,
    plot_segment_statistics,
    plot_hurst_timeline
)

from .gauges import (
    plot_fuzzy_gauge,
    plot_regime_probabilities,
    create_nowcast_dashboard
)

__all__ = [
    'plot_price_with_regimes',
    'plot_regime_timeline',
    'plot_technical_overlays',
    'plot_forward_return_distributions',
    'plot_transition_matrix',
    'plot_segment_statistics',
    'plot_hurst_timeline',
    'plot_fuzzy_gauge',
    'plot_regime_probabilities',
    'create_nowcast_dashboard'
]
