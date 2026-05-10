"""
Analysis modules for the Julia Mandelbrot System.
Includes forward returns, transitions, and segment analysis.
"""

from .forward_returns import (
    compute_forward_returns,
    analyze_forward_returns_by_regime,
    get_forward_return_statistics
)
from .transitions import (
    compute_transition_matrix,
    analyze_regime_transitions,
    get_transition_statistics
)
from .segments import (
    analyze_regime_segments,
    compute_segment_statistics,
    get_segment_summary
)
from .backtest import (
    DEFAULT_REGIME_EXPOSURE,
    RegimeBacktestResult,
    backtest_regime_strategy,
    build_backtest_narrative,
    regime_exposure,
)

__all__ = [
    'compute_forward_returns',
    'analyze_forward_returns_by_regime',
    'get_forward_return_statistics',
    'compute_transition_matrix',
    'analyze_regime_transitions',
    'get_transition_statistics',
    'analyze_regime_segments',
    'compute_segment_statistics',
    'get_segment_summary',
    'DEFAULT_REGIME_EXPOSURE',
    'RegimeBacktestResult',
    'backtest_regime_strategy',
    'build_backtest_narrative',
    'regime_exposure',
]
