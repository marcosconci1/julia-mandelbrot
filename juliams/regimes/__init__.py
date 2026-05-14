"""
Regime classification modules for the Julia Mandelbrot System.
Includes crisp and fuzzy logic classification systems.
"""

from .classification import (
    RegimeClassifier,
    classify_market_regime,
    get_regime_statistics
)
from .quality import add_indicator_signal_flags

__all__ = [
    'RegimeClassifier',
    'classify_market_regime',
    'get_regime_statistics',
    'add_indicator_signal_flags',
]
