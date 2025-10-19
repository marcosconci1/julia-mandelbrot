"""
Regime classification modules for the Julia Mandelbrot System.
Includes crisp and fuzzy logic classification systems.
"""

from .classification import (
    RegimeClassifier,
    classify_market_regime,
    get_regime_statistics
)

__all__ = [
    'RegimeClassifier',
    'classify_market_regime',
    'get_regime_statistics'
]
