"""
Feature computation modules for the Julia Mandelbrot System.
Includes trend, volatility, Hurst exponent, and fractal analysis.
"""

from .trend import (
    compute_trend_strength, 
    compute_rolling_ols_slope,
    compute_trend_features,
    classify_trend_regime
)
from .volatility import (
    compute_volatility, 
    compute_atr, 
    compute_volatility_regime,
    compute_volatility_features
)
from .hurst import (
    compute_rolling_hurst,
    compute_hurst_features,
    classify_hurst_regime
)
from .fractal import (
    compute_fractal_filtered_price,
    compute_fractal_features,
    create_fractal_mask
)

__all__ = [
    'compute_trend_strength',
    'compute_rolling_ols_slope',
    'compute_trend_features',
    'classify_trend_regime',
    'compute_volatility',
    'compute_atr',
    'compute_volatility_regime',
    'compute_volatility_features',
    'compute_rolling_hurst',
    'compute_hurst_features',
    'classify_hurst_regime',
    'compute_fractal_filtered_price',
    'compute_fractal_features',
    'create_fractal_mask'
]
