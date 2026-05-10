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
from .diagnostics import (
    DEFAULT_DIAGNOSTIC_EXPOSURE_MAPS,
    backtest_rebound_overlay,
    compare_exposure_maps,
    compare_rebound_overlay,
    compute_rebound_signal,
    compute_regime_forward_diagnostics,
    measure_bullish_turn_lag,
    summarize_group_narratives,
    summarize_variant_comparison_by_group,
)
from .group_rebound import (
    DEFAULT_ASSET_GROUPS,
    DEFAULT_GROUP_OVERLAY_PROFILES,
    ReboundOverlayProfile,
    asset_group_for_symbol,
    compare_group_aware_rebound,
    run_walk_forward_group_rebound,
    summarize_group_rebound_results,
)
from .walk_forward import (
    DEFAULT_WALK_FORWARD_WINDOW_SPECS,
    add_narrative_alignment,
    build_recent_walk_forward_windows,
    compare_group_rebound_overlay,
    evaluate_group_rebound_promotion,
    evaluate_research_grade_rebound_promotion,
    narrative_alignment_score,
    run_walk_forward_diagnostics,
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
    'DEFAULT_DIAGNOSTIC_EXPOSURE_MAPS',
    'backtest_rebound_overlay',
    'compare_exposure_maps',
    'compare_rebound_overlay',
    'compute_rebound_signal',
    'compute_regime_forward_diagnostics',
    'measure_bullish_turn_lag',
    'summarize_group_narratives',
    'summarize_variant_comparison_by_group',
    'DEFAULT_ASSET_GROUPS',
    'DEFAULT_GROUP_OVERLAY_PROFILES',
    'DEFAULT_WALK_FORWARD_WINDOW_SPECS',
    'ReboundOverlayProfile',
    'add_narrative_alignment',
    'asset_group_for_symbol',
    'build_recent_walk_forward_windows',
    'compare_group_aware_rebound',
    'compare_group_rebound_overlay',
    'evaluate_group_rebound_promotion',
    'evaluate_research_grade_rebound_promotion',
    'narrative_alignment_score',
    'run_walk_forward_group_rebound',
    'run_walk_forward_diagnostics',
    'summarize_group_rebound_results',
]
