#!/usr/bin/env python3
"""
Unified command line interface for the Julia Mandelbrot System.

The CLI orchestrates the full workflow: data acquisition (stock or crypto),
feature engineering, regime classification (crisp + optional fuzzy),
forward return analysis, and visual reporting. It consolidates the legacy
example scripts into a single entry point with sensible defaults.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple, cast

import pandas as pd

from juliams.config import JMSConfig
from juliams.data import DataFetcherFactory
from juliams.profiles import (
    apply_indicator_profile,
    available_indicator_profiles,
)
from juliams.data.utils import detect_source_type
from juliams.features import (
    compute_fractal_features,
    compute_hurst_features,
    compute_tail_risk_features,
    compute_trend_features,
    compute_volatility_features,
)
from juliams.regimes import RegimeClassifier
from juliams.regimes.fuzzy import compute_fuzzy_features
from juliams.analysis import (
    analyze_forward_returns_by_regime,
    build_recent_walk_forward_windows,
    compute_forward_returns,
    compute_transition_matrix,
    evaluate_research_grade_rebound_promotion,
    get_segment_summary,
    run_walk_forward_diagnostics,
    summarize_group_rebound_results,
)

DEFAULT_SYMBOL = "AAPL"
DEFAULT_DIAGNOSTIC_SYMBOLS = [
    "SPY",
    "QQQ",
    "IWM",
    "DIA",
    "AAPL",
    "MSFT",
    "NVDA",
    "GOOGL",
    "AMZN",
    "TLT",
    "GLD",
    "BTC-USD",
    "ETH-USD",
]


def _ensure_interactive_matplotlib_backend() -> bool:
    """Switch to an interactive Matplotlib backend when possible."""
    try:
        import matplotlib
        from matplotlib.backends import backend_registry, BackendFilter
    except Exception:
        return False

    backend = matplotlib.get_backend()
    interactive_backends = backend_registry.list_builtin(BackendFilter.INTERACTIVE)
    if backend in interactive_backends:
        return True

    preferred_backends = ("QtAgg", "Qt5Agg", "TkAgg", "GTK3Agg", "WXAgg")
    for candidate in preferred_backends:
        if candidate not in interactive_backends:
            continue
        try:
            matplotlib.use(candidate, force=True)
            return True
        except Exception:
            continue

    return False


def parse_arguments() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Run a complete Julia Mandelbrot System analysis for a given symbol. "
            "The data source (stock vs crypto) is auto-detected by default."
        )
    )
    parser.add_argument(
        "symbol",
        nargs="?",
        help=f"Ticker or trading pair symbol (e.g., AAPL, BTCUSDT). Defaults to {DEFAULT_SYMBOL}.",
    )
    parser.add_argument(
        "--period",
        help="Data period (e.g., 6mo, 1y, 2y). Cannot be combined with --start/--end.",
    )
    parser.add_argument(
        "--start",
        help="Start date in YYYY-MM-DD format. Requires --end or --period.",
    )
    parser.add_argument(
        "--end",
        help="End date in YYYY-MM-DD format. Requires --start or --period.",
    )
    parser.add_argument(
        "--source-type",
        choices=["stock", "crypto"],
        help="Override automatic source detection.",
    )
    parser.add_argument(
        "--interval",
        help="Bar interval to request from the data source, such as 1h, 4h, or 1d.",
    )
    parser.add_argument(
        "--profile",
        choices=tuple(available_indicator_profiles()),
        help="Indicator calibration profile. Sets feature windows, timeframe, and fuzzy calibration.",
    )
    parser.add_argument(
        "--no-fuzzy",
        action="store_true",
        help="Disable fuzzy regime classification.",
    )
    parser.add_argument(
        "--horizons",
        nargs="+",
        type=int,
        help="Forward return horizons to evaluate (space separated list).",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip chart rendering (plots are shown by default).",
    )
    parser.add_argument(
        "--save-plots",
        metavar="DIR",
        help="Directory to save generated plots. Directory will be created if needed.",
    )
    parser.add_argument(
        "--extra-plots",
        action="store_true",
        help="Include extended visualization suite in addition to the legacy dashboard.",
    )
    parser.add_argument(
        "--walk-forward-diagnostics",
        action="store_true",
        help="Run group-aware rebound walk-forward diagnostics instead of single-symbol analysis.",
    )
    parser.add_argument(
        "--diagnostic-symbols",
        nargs="+",
        help="Symbols for --walk-forward-diagnostics. Defaults to the main cross-asset basket.",
    )
    parser.add_argument(
        "--diagnostic-period",
        default="2y",
        help="History period for --walk-forward-diagnostics (default: 2y).",
    )
    parser.add_argument(
        "--diagnostic-csv",
        metavar="PATH",
        help="Optional CSV path for raw walk-forward diagnostic rows.",
    )
    parser.add_argument(
        "--overlay-enabled-groups",
        nargs="+",
        help="Asset groups allowed to use the rebound overlay in diagnostics.",
    )
    parser.add_argument(
        "--rebound-exposure",
        type=float,
        default=0.75,
        help="Rebound overlay exposure for enabled groups (default: 0.75).",
    )
    parser.add_argument(
        "--rebound-fast-window",
        type=int,
        default=5,
        help="Fast momentum window for the rebound overlay (default: 5).",
    )
    parser.add_argument(
        "--rebound-lookback",
        type=int,
        default=20,
        help="Rolling-low lookback for the rebound overlay (default: 20).",
    )
    parser.add_argument(
        "--min-rebound",
        type=float,
        default=0.03,
        help="Minimum recovery from rolling low for rebound overlay (default: 0.03).",
    )
    parser.add_argument(
        "--min-fast-return",
        type=float,
        default=0.02,
        help="Minimum fast-window return for rebound overlay (default: 0.02).",
    )
    parser.add_argument(
        "--no-rebound-fast-ma",
        action="store_true",
        help="Disable the fast moving-average confirmation for rebound overlay diagnostics.",
    )
    parser.add_argument(
        "--no-drawdown-gate",
        action="store_true",
        help="Disable the prior-drawdown gate for rebound overlay diagnostics.",
    )
    parser.add_argument(
        "--drawdown-gate-lookback",
        type=int,
        default=20,
        help="Lookback for requiring a recent prior drawdown before rebound overlay can fire.",
    )
    parser.add_argument(
        "--min-prior-drawdown",
        type=float,
        default=-0.05,
        help="Minimum recent drawdown required before rebound overlay can fire (default: -0.05).",
    )
    parser.add_argument(
        "--max-overlay-holding-period",
        type=int,
        default=5,
        help="Maximum consecutive overlay signal days before forcing an overlay exit (default: 5).",
    )
    parser.add_argument(
        "--overlay-stop-loss",
        type=float,
        default=-0.03,
        help="Overlay stop loss from signal entry price (default: -0.03).",
    )
    parser.add_argument(
        "--transaction-cost-bps",
        type=float,
        default=5.0,
        help="Round-trip friction estimate in basis points for diagnostic backtests (default: 5).",
    )
    parser.add_argument(
        "--no-parameter-stability-check",
        action="store_true",
        help="Skip nearby-parameter promotion checks for faster exploratory diagnostics.",
    )

    # ---- Adaptive regime overlays (opt-in; defaults preserve legacy output) ----
    parser.add_argument(
        "--adaptive-thresholds",
        action="store_true",
        help=(
            "Add a regime_adaptive column from rolling-quantile thresholds on an "
            "EWMA-normalised trend signal. Distribution-driven cutoffs replace the "
            "hardcoded ±0.2 z-score thresholds."
        ),
    )
    parser.add_argument(
        "--adaptive-q-up",
        type=float,
        default=0.70,
        help="Upper quantile for adaptive thresholds (default: 0.70).",
    )
    parser.add_argument(
        "--adaptive-q-down",
        type=float,
        default=0.30,
        help="Lower quantile for adaptive thresholds (default: 0.30).",
    )
    parser.add_argument(
        "--adaptive-floor",
        type=float,
        default=0.10,
        help=(
            "Absolute floor on adaptive thresholds; prevents flat markets from "
            "degenerating to constant trigger frequency (default: 0.10)."
        ),
    )
    parser.add_argument(
        "--adaptive-window",
        type=int,
        default=252,
        help="Rolling window length for adaptive quantiles, in trading days (default: 252).",
    )
    parser.add_argument(
        "--ewma-halflife",
        type=float,
        default=None,
        help=(
            "If set, add a trend_strength_ewma column with this halflife (in days). "
            "Defaults vary by asset class — see juliams.features.ewma_calibration."
        ),
    )
    parser.add_argument(
        "--markov-overlay",
        action="store_true",
        help=(
            "Add markov_prob_high and markov_state columns from a 2-state "
            "Markov-switching variance model fit on log returns."
        ),
    )
    parser.add_argument(
        "--markov-vol-channel",
        type=str,
        default=None,
        help=(
            "yfinance ticker for an implied-volatility index to use as a "
            "second emission channel in the Markov HMM (e.g. ^GVZ for gold, "
            "^VIX for SPY). Falls back to univariate if the fetch fails."
        ),
    )
    parser.add_argument(
        "--markov-auto-vol-channel",
        action="store_true",
        help=(
            "Automatically select the implied-vol ticker for known asset "
            "classes (gold -> ^GVZ, SPY -> ^VIX, oil -> ^OVX, ...). "
            "Overridden by --markov-vol-channel when both are set."
        ),
    )
    parser.add_argument(
        "--bocpd-overlay",
        action="store_true",
        help=(
            "Add bocpd_run_length and bocpd_change_prob columns from Bayesian "
            "online change-point detection."
        ),
    )
    parser.add_argument(
        "--consensus-overlay",
        action="store_true",
        help=(
            "Add a consensus_event boolean column. Auto-enables BOCPD and "
            "Markov overlays. Declares a consensus event only when both "
            "detectors fire within a 5-day window (configurable)."
        ),
    )
    parser.add_argument(
        "--consensus-window-days",
        type=int,
        default=5,
        help="Maximum delay between BOCPD and Markov fires for consensus (default: 5).",
    )
    parser.add_argument(
        "--consensus-cooldown-days",
        type=int,
        default=10,
        help="Cooldown after a consensus event before a new one can fire (default: 10).",
    )
    parser.add_argument(
        "--bocpd-method",
        choices=["standard", "dsm"],
        default="standard",
        help=(
            "BOCPD algorithm: 'standard' (Adams-MacKay 2007 with Gaussian/NIG "
            "conjugate) or 'dsm' (Diffusion Score Matching, Altamirano-Briol-"
            "Knoblauch 2023, more robust to fat-tailed financial returns)."
        ),
    )
    parser.add_argument(
        "--bocpd-expected-run-length",
        type=float,
        default=100.0,
        help="Mean of the BOCPD geometric run-length prior (default: 100).",
    )
    parser.add_argument(
        "--bocpd-omega",
        type=float,
        default=1.0,
        help="DSM-BOCPD robustness weight (default: 1.0). Only used with --bocpd-method dsm.",
    )
    parser.add_argument(
        "--bocpd-robustness-bandwidth",
        type=float,
        default=3.0,
        help=(
            "DSM-BOCPD robustness bandwidth in units of sqrt(varx) "
            "(default: 3.0). Smaller = more aggressive outlier downweighting."
        ),
    )
    parser.add_argument(
        "--min-dwell-days",
        type=int,
        default=1,
        help=(
            "Minimum dwell time for Markov regime labels in days. Suppresses "
            "spurious flips during structural breaks. 1 = no-op (default)."
        ),
    )
    return parser.parse_args()


def resolve_source(symbol: str, explicit: Optional[str]) -> Tuple[str, str]:
    """
    Determine the data source type for the supplied symbol.

    Returns:
        Tuple of (source_type, explanation)
    """
    if explicit:
        return explicit, "explicitly set via --source-type"
    detected = detect_source_type(symbol)
    return detected, "auto-detected from symbol format"


def prepare_request_parameters(
    period: Optional[str],
    start: Optional[str],
    end: Optional[str],
    default_period: str,
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Validate and prepare period/start/end inputs for the fetcher.

    Returns:
        (resolved_period, resolved_start, resolved_end)
    """
    if period and (start or end):
        raise ValueError("Specify either --period or --start/--end, not both.")

    if start and not end:
        raise ValueError("When using --start you must also supply --end (and vice-versa).")
    if end and not start:
        raise ValueError("When using --end you must also supply --start (and vice-versa).")

    if period:
        return period, None, None

    if start or end:
        return None, start, end

    return default_period, None, None


def run_analysis(
    symbol: str,
    period: Optional[str] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
    source_type: Optional[str] = None,
    interval: Optional[str] = None,
    profile: Optional[str] = None,
    use_fuzzy: bool = True,
    horizons: Optional[List[int]] = None,
    *,
    adaptive_thresholds: bool = False,
    adaptive_q_up: float = 0.70,
    adaptive_q_down: float = 0.30,
    adaptive_floor: float = 0.10,
    adaptive_window: int = 252,
    ewma_halflife: Optional[float] = None,
    markov_overlay: bool = False,
    markov_vol_channel: Optional[str] = None,
    markov_auto_vol_channel: bool = False,
    bocpd_overlay: bool = False,
    bocpd_method: str = "standard",
    bocpd_expected_run_length: float = 100.0,
    bocpd_omega: float = 1.0,
    bocpd_robustness_bandwidth: float = 3.0,
    consensus_overlay: bool = False,
    consensus_window_days: int = 5,
    consensus_cooldown_days: int = 10,
    min_dwell_days: int = 1,
) -> Dict[str, object]:
    """
    Execute the full analysis pipeline and return collected artefacts.

    Returns a dictionary containing the enriched dataframe, analysis summaries,
    and metadata for downstream reporting.
    """
    config = JMSConfig()
    if profile is not None:
        apply_indicator_profile(config, profile)
    resolved_source, source_note = resolve_source(symbol, source_type)
    source_config = config.get_source_config(resolved_source)
    resolved_interval = interval or str(source_config.get("timeframe", config.timeframe))

    resolved_period, resolved_start, resolved_end = prepare_request_parameters(
        period, start, end, source_config.get("default_period", config.default_period)
    )

    fetcher = DataFetcherFactory.create(
        symbol=symbol,
        source_type=cast(Literal['stock', 'crypto'], resolved_source),
        config=config,
    )
    df = fetcher.fetch_data(
        symbol,
        period=resolved_period,
        start=resolved_start,
        end=resolved_end,
        interval=resolved_interval,
    )

    # Feature computation
    df = compute_trend_features(df, source_config)
    df = compute_volatility_features(df, source_config)
    df = compute_hurst_features(df, source_config)
    df = compute_fractal_features(df, source_config)
    df = compute_tail_risk_features(df, source_config)

    # Regime classification
    classifier = RegimeClassifier(
        trend_threshold_up=source_config["trend_threshold_up"],
        trend_threshold_down=source_config["trend_threshold_down"],
        volatility_threshold=source_config["volatility_percentile"],
    )
    df = classifier.classify(df)

    if use_fuzzy and source_config.get("use_fuzzy", True):
        df = compute_fuzzy_features(df, source_config)
    else:
        use_fuzzy = False

    # Adaptive overlays (opt-in, additive)
    overlays_used: List[str] = []

    # Consensus needs both bocpd and markov; force-enable up-front so
    # the conditional blocks below all run.
    if consensus_overlay:
        bocpd_overlay = True
        markov_overlay = True

    if adaptive_thresholds:
        from juliams.regimes.overlays import apply_adaptive_threshold_overlay
        df = apply_adaptive_threshold_overlay(
            df,
            window=adaptive_window,
            q_up=adaptive_q_up,
            q_down=adaptive_q_down,
            floor=adaptive_floor,
        )
        overlays_used.append("adaptive_thresholds")

    if ewma_halflife is not None:
        from juliams.regimes.overlays import apply_ewma_overlay
        df = apply_ewma_overlay(df, halflife=float(ewma_halflife))
        overlays_used.append(f"ewma(halflife={ewma_halflife})")

    if markov_overlay:
        from juliams.regimes.overlays import apply_markov_overlay
        vol_ch = markov_vol_channel
        if vol_ch is None and markov_auto_vol_channel:
            from juliams.regimes.vol_tickers import auto_vol_channel
            vol_ch = auto_vol_channel(symbol)
        df = apply_markov_overlay(df, min_dwell=min_dwell_days, vol_channel=vol_ch)
        tag_parts = ["markov"]
        if min_dwell_days > 1:
            tag_parts.append(f"min_dwell={min_dwell_days}")
        if vol_ch:
            tag_parts.append(f"vol={vol_ch}")
        overlays_used.append(
            tag_parts[0] + ("(" + ",".join(tag_parts[1:]) + ")" if len(tag_parts) > 1 else "")
        )

    if bocpd_overlay:
        from juliams.regimes.overlays import apply_bocpd_overlay
        df = apply_bocpd_overlay(
            df,
            expected_run_length=bocpd_expected_run_length,
            method=bocpd_method,
            omega=bocpd_omega,
            robustness_bandwidth=bocpd_robustness_bandwidth,
        )
        tag = (
            f"bocpd-dsm(lambda={bocpd_expected_run_length},omega={bocpd_omega})"
            if bocpd_method == "dsm"
            else f"bocpd(lambda={bocpd_expected_run_length})"
        )
        overlays_used.append(tag)

    if consensus_overlay:
        from juliams.regimes.overlays import apply_consensus_overlay
        df = apply_consensus_overlay(
            df,
            window_days=consensus_window_days,
            cooldown_days=consensus_cooldown_days,
        )
        n_events = int(df["consensus_event"].sum())
        overlays_used.append(f"consensus(events={n_events})")

    # Forward returns & diagnostics
    horizons_to_use = horizons or source_config.get("forward_return_horizons", [5, 10])
    df = compute_forward_returns(df, horizons=horizons_to_use)
    forward_analysis = analyze_forward_returns_by_regime(df)
    transition_matrix = compute_transition_matrix(df)
    segment_summary = get_segment_summary(df)

    return {
        "df": df,
        "source_type": resolved_source,
        "source_note": source_note,
        "config": config,
        "forward_analysis": forward_analysis,
        "transition_matrix": transition_matrix,
        "segment_summary": segment_summary,
        "use_fuzzy": use_fuzzy,
        "horizons": horizons_to_use,
        "period": resolved_period,
        "start": resolved_start,
        "end": resolved_end,
        "interval": resolved_interval,
        "profile": profile,
        "overlays_used": overlays_used,
    }


def _format_percentage(value: float) -> str:
    """Helper for consistent percentage formatting."""
    return f"{value * 100:6.2f}%" if pd.notna(value) else "   n/a"


def _format_percentage_points(value: float) -> str:
    """Format values that are already expressed on a 0-100 percentage scale."""
    return f"{value:6.2f}%" if pd.notna(value) else "   n/a"


def _format_decimal(value: Optional[float], precision: int = 4) -> str:
    """Format numeric values with safe handling for NaN."""
    if value is None or pd.isna(value):
        return "n/a"
    return f"{value:.{precision}f}"


def _build_rebound_stability_variants(
    *,
    rebound_exposure: float,
    rebound_fast_window: int,
    rebound_lookback: int,
    min_rebound: float,
    min_fast_return: float,
    require_prior_drawdown: Optional[bool],
    drawdown_gate_lookback: Optional[int],
    min_prior_drawdown: Optional[float],
    max_overlay_holding_period: Optional[int],
    overlay_stop_loss: Optional[float],
) -> Dict[str, Dict[str, object]]:
    """Build nearby parameter checks for promotion stability."""
    base_gate_lookback = drawdown_gate_lookback or 20
    base_prior_drawdown = min_prior_drawdown if min_prior_drawdown is not None else -0.05
    base_max_hold = max_overlay_holding_period if max_overlay_holding_period is not None else 5
    base_stop_loss = overlay_stop_loss if overlay_stop_loss is not None else -0.03

    return {
        "lower_exposure": {
            "rebound_exposure": max(0.0, min(rebound_exposure, 0.50)),
        },
        "slower_confirmation": {
            "fast_window": max(1, rebound_fast_window + 2),
            "drawdown_lookback": max(1, rebound_lookback + 10),
            "min_rebound": min_rebound + 0.02,
            "min_fast_return": min_fast_return + 0.01,
        },
        "stricter_risk_gate": {
            "require_prior_drawdown": True,
            "drawdown_gate_lookback": base_gate_lookback + 10,
            "min_prior_drawdown": min(base_prior_drawdown, -0.08),
            "max_overlay_holding_period": max(1, base_max_hold - 2),
            "overlay_stop_loss": min(base_stop_loss, -0.04),
        },
    }


def run_walk_forward_diagnostics_report(
    symbols: List[str],
    period: str = "2y",
    overlay_enabled_groups: Optional[List[str]] = None,
    rebound_exposure: float = 0.75,
    rebound_fast_window: int = 5,
    rebound_lookback: int = 20,
    min_rebound: float = 0.03,
    min_fast_return: float = 0.02,
    require_above_fast_ma: bool = True,
    require_prior_drawdown: Optional[bool] = None,
    drawdown_gate_lookback: Optional[int] = None,
    min_prior_drawdown: Optional[float] = None,
    max_overlay_holding_period: Optional[int] = None,
    overlay_stop_loss: Optional[float] = None,
    transaction_cost_bps: float = 5.0,
    run_parameter_stability: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, object], pd.DataFrame]:
    """
    Run the group-aware rebound diagnostic on a basket of prepared assets.
    """
    asset_frames: Dict[str, pd.DataFrame] = {}

    for symbol in symbols:
        artefacts = run_analysis(
            symbol=symbol,
            period=period,
            source_type=None,
            use_fuzzy=False,
            horizons=[5, 10],
        )
        asset_frames[symbol] = cast(pd.DataFrame, artefacts["df"])

    latest_common_end = min(pd.Timestamp(frame.index.max()).normalize() for frame in asset_frames.values())
    windows = build_recent_walk_forward_windows(latest_common_end)
    diagnostic_params: Dict[str, object] = {
        "overlay_enabled_groups": overlay_enabled_groups,
        "rebound_exposure": rebound_exposure,
        "fast_window": rebound_fast_window,
        "drawdown_lookback": rebound_lookback,
        "min_rebound": min_rebound,
        "min_fast_return": min_fast_return,
        "require_above_fast_ma": require_above_fast_ma,
        "require_prior_drawdown": require_prior_drawdown,
        "drawdown_gate_lookback": drawdown_gate_lookback,
        "min_prior_drawdown": min_prior_drawdown,
        "max_overlay_holding_period": max_overlay_holding_period,
        "overlay_stop_loss": overlay_stop_loss,
        "transaction_cost_bps": transaction_cost_bps,
    }
    comparison = run_walk_forward_diagnostics(
        asset_frames,
        windows=windows,
        **diagnostic_params,
    )
    parameter_results: Dict[str, pd.DataFrame] = {}
    if run_parameter_stability:
        for name, overrides in _build_rebound_stability_variants(
            rebound_exposure=rebound_exposure,
            rebound_fast_window=rebound_fast_window,
            rebound_lookback=rebound_lookback,
            min_rebound=min_rebound,
            min_fast_return=min_fast_return,
            require_prior_drawdown=require_prior_drawdown,
            drawdown_gate_lookback=drawdown_gate_lookback,
            min_prior_drawdown=min_prior_drawdown,
            max_overlay_holding_period=max_overlay_holding_period,
            overlay_stop_loss=overlay_stop_loss,
        ).items():
            parameter_results[name] = run_walk_forward_diagnostics(
                asset_frames,
                windows=windows,
                **{**diagnostic_params, **overrides},
            )
    promotion_eligible_groups = tuple(overlay_enabled_groups or ["indices", "mega_cap_tech"])
    promotion = evaluate_research_grade_rebound_promotion(
        comparison,
        eligible_groups=promotion_eligible_groups,
        required_transaction_cost_bps=transaction_cost_bps,
        parameter_results=parameter_results,
        require_parameter_stability=run_parameter_stability,
    )
    summary = summarize_group_rebound_results(comparison)
    return comparison, promotion, summary


def print_walk_forward_diagnostics_report(
    symbols: List[str],
    comparison: pd.DataFrame,
    promotion: Dict[str, object],
    summary: pd.DataFrame,
) -> None:
    """Print a compact walk-forward diagnostic report."""
    print("\n" + "=" * 72)
    print("Group Rebound Walk-Forward Diagnostics")
    print("=" * 72)
    print(f"Assets:        {', '.join(symbols)}")
    print(f"Rows:          {len(comparison):,}")
    print(f"Windows:       {', '.join(str(item) for item in comparison['window'].dropna().unique())}")

    print("\nPromotion Gate")
    print("-" * 72)
    print(f"  Promote:                  {promotion['promote']}")
    print(f"  Eligible improved share:  {_format_percentage(cast(float, promotion['eligible_improved_share']))}")
    print(f"  Protected degraded share: {_format_percentage(cast(float, promotion['protected_degraded_share']))}")
    print(f"  Worst drawdown delta:     {_format_percentage(cast(float, promotion['worst_drawdown_delta']))}")
    if "windows_tested" in promotion:
        print(
            f"  Non-empty windows:        {promotion['windows_tested']}/"
            f"{promotion.get('min_windows', 'n/a')}"
        )
    if "min_transaction_cost_bps" in promotion:
        print(
            f"  Cost check bps:           "
            f"{_format_decimal(cast(float, promotion['min_transaction_cost_bps']), 2)} "
            f"(required {_format_decimal(cast(float, promotion.get('required_transaction_cost_bps', 0.0)), 2)})"
        )
    if "recent_narrative_alignment_score" in promotion:
        print(
            f"  Recent narrative score:   "
            f"{_format_decimal(cast(float, promotion['recent_narrative_alignment_score']), 2)} "
            f"(current {_format_decimal(cast(float, promotion.get('current_recent_narrative_alignment_score', 0.0)), 2)})"
        )
    if "parameter_pass_share" in promotion:
        print(
            f"  Parameter pass share:     "
            f"{_format_percentage(cast(float, promotion['parameter_pass_share']))}"
        )
    reasons = cast(List[str], promotion.get("reasons", []))
    if reasons:
        for reason in reasons:
            print(f"  Blocker:                  {reason}")
    else:
        print("  Blocker:                  none")

    print("\nGroup Candidate Summary")
    print("-" * 72)
    candidate = summary[summary["variant"] == "group_rebound"]
    if candidate.empty:
        print("  No candidate rows available.")
        return

    for _, row in candidate.iterrows():
        label = f"{row.get('window', 'all')} / {row['group']}"
        print(
            f"  {label:28s} "
            f"strategy={_format_percentage(row['mean_strategy_return'])} "
            f"buy_hold={_format_percentage(row['mean_buy_hold_return'])} "
            f"excess={_format_percentage(row['mean_excess_return'])} "
            f"delta={_format_percentage(row['mean_strategy_delta_vs_current'])} "
            f"enabled={int(row['overlay_enabled_count'])}/{int(row['asset_count'])}"
        )


def print_summary(symbol: str, artefacts: Dict[str, object]) -> None:
    """Print human-readable summary of analysis results."""
    df: pd.DataFrame = cast(pd.DataFrame, artefacts["df"])
    source_type: str = cast(str, artefacts["source_type"])
    source_note: str = cast(str, artefacts["source_note"])
    use_fuzzy: bool = cast(bool, artefacts["use_fuzzy"])
    horizons: List[int] = cast(List[int], artefacts["horizons"])
    period: Optional[str] = cast(Optional[str], artefacts["period"])
    start: Optional[str] = cast(Optional[str], artefacts["start"])
    end: Optional[str] = cast(Optional[str], artefacts["end"])
    interval: str = cast(str, artefacts.get("interval", "1d"))
    profile: Optional[str] = cast(Optional[str], artefacts.get("profile"))

    print("\n" + "=" * 72)
    print("Julia Mandelbrot System - Unified Analysis")
    print("=" * 72)
    print(f"Symbol:        {symbol}")
    date_info = f"period={period}" if period else f"start={start} • end={end}"
    print(f"Data range:    {date_info}")
    print(f"Source type:   {source_type} ({source_note})")
    print(f"Interval:      {interval}")
    if profile:
        print(f"Profile:       {profile}")
    print(f"Observations:  {len(df):,}")
    if len(df) > 1:
        print(f"Date span:     {df.index[0].date()} → {df.index[-1].date()}")
    if "Close" in df.columns:
        print(f"Last close:    {df['Close'].iloc[-1]:,.2f}")

    print("\nCurrent Regime")
    print("-" * 72)
    current_regime = df["regime"].iloc[-1]
    current_regime_name = df["regime_name"].iloc[-1]
    print(f"  Regime:       {current_regime_name} ({current_regime})")
    if "trend_strength" in df.columns:
        print(f"  Trend:        {_format_decimal(df['trend_strength'].iloc[-1], 3)}")
    if "volatility" in df.columns:
        print(f"  Volatility:   {_format_decimal(df['volatility'].iloc[-1], 4)}")
    if "hurst" in df.columns and pd.notna(df["hurst"].iloc[-1]):
        print(f"  Hurst:        {_format_decimal(df['hurst'].iloc[-1], 3)}")
    if "hurst_confirmed_regime" in df.columns:
        print(f"  Hurst check:  {df['hurst_confirmed_regime'].iloc[-1]}")
    if "lo_memory_regime" in df.columns:
        print(f"  Lo R/S check: {df['lo_memory_regime'].iloc[-1]}")
    if "survival_regime" in df.columns:
        print(f"  Survival:     {df['survival_regime'].iloc[-1]}")
    if "loss_cvar" in df.columns and pd.notna(df["loss_cvar"].iloc[-1]):
        print(f"  Tail CVaR:    {_format_percentage(df['loss_cvar'].iloc[-1])}")

    if use_fuzzy and "fuzzy_primary_regime" in df.columns:
        print("\nFuzzy Memberships")
        print("-" * 72)
        fuzzy_cols = [
            col
            for col in df.columns
            if col.startswith("fuzzy_")
            and col not in {"fuzzy_primary_regime", "fuzzy_confidence", "fuzzy_entropy"}
        ]
        memberships = sorted(
            ((col.replace("fuzzy_", ""), df[col].iloc[-1]) for col in fuzzy_cols),
            key=lambda item: item[1],
            reverse=True,
        )
        for regime, probability in memberships[:3]:
            print(f"  {regime:20s}: {_format_percentage(probability)}")
        if "fuzzy_confidence" in df.columns:
            print(f"  Confidence:    {_format_percentage(df['fuzzy_confidence'].iloc[-1])}")

    overlays_used = cast(List[str], artefacts.get("overlays_used", []))
    adaptive_columns = {
        "regime_adaptive",
        "trend_strength_ewma",
        "markov_prob_high",
        "markov_state",
        "bocpd_run_length",
        "bocpd_change_prob",
    } & set(df.columns)
    if overlays_used or adaptive_columns:
        print("\nAdaptive Overlays")
        print("-" * 72)
        if overlays_used:
            print(f"  Active:       {', '.join(overlays_used)}")
        if "regime_adaptive" in df.columns:
            print(f"  Adaptive:     {df['regime_adaptive'].iloc[-1]}")
        if "trend_strength_ewma" in df.columns and pd.notna(
            df["trend_strength_ewma"].iloc[-1]
        ):
            print(
                f"  Trend (EWMA): {_format_decimal(df['trend_strength_ewma'].iloc[-1], 3)}"
            )
        if "markov_state" in df.columns:
            state = df["markov_state"].iloc[-1]
            prob = df["markov_prob_high"].iloc[-1] if "markov_prob_high" in df.columns else None
            prob_str = (
                f" (P(high)={_format_percentage(prob)})"
                if prob is not None and pd.notna(prob)
                else ""
            )
            print(f"  Markov:       {state}{prob_str}")
        if "bocpd_run_length" in df.columns and pd.notna(
            df["bocpd_run_length"].iloc[-1]
        ):
            rl = int(df["bocpd_run_length"].iloc[-1])
            cp = df["bocpd_change_prob"].iloc[-1]
            print(
                f"  BOCPD:        run_length={rl}d change_prob={_format_percentage(cp)}"
            )
        if "consensus_event" in df.columns:
            n_events = int(df["consensus_event"].sum())
            event_dates = df.index[df["consensus_event"]].tolist()
            recent = (
                f" most recent={event_dates[-1].date()}" if event_dates else ""
            )
            print(f"  Consensus:    {n_events} event(s){recent}")

    print("\nForward Returns")
    print("-" * 72)
    forward_analysis: Dict[str, Dict[str, Dict[str, float]]] = cast(Dict[str, Dict[str, Dict[str, float]]], artefacts["forward_analysis"])
    for horizon in horizons:
        key = f"{horizon}d"
        overall_stats = forward_analysis.get("Overall", {}).get(key)
        if overall_stats:
            print(
                f"  Horizon {key}: mean={_format_percentage(overall_stats.get('mean', float('nan')))} "
                f"win_rate={_format_percentage_points(overall_stats.get('positive_pct', float('nan')))}"
            )
    print("  By regime:")
    for regime, stats in forward_analysis.items():
        if regime == "Overall":
            continue
        horizon_stats = stats.get(f"{horizons[0]}d")
        if not horizon_stats:
            continue
        print(
            f"    {regime:20s}: mean={_format_percentage(horizon_stats.get('mean', float('nan')))} "
            f"win_rate={_format_percentage_points(horizon_stats.get('positive_pct', float('nan')))}"
        )

    print("\nTransition Matrix")
    print("-" * 72)
    transition_matrix: pd.DataFrame = cast(pd.DataFrame, artefacts["transition_matrix"])
    if transition_matrix is not None and not transition_matrix.empty:
        for regime in transition_matrix.index:
                if regime in transition_matrix.columns:
                    value = transition_matrix.at[regime, regime]
                    if isinstance(value, (int, float)):
                        persistence = float(value)
                        print(f"  {regime:20s}: persistence={_format_percentage(persistence)}")
    else:
        print("  Transition matrix unavailable (insufficient data).")

    print("\nSegment Summary")
    print("-" * 72)
    segment_summary: Dict[str, object] = cast(Dict[str, object], artefacts["segment_summary"])
    statistics_dict = cast(Dict[str, object], segment_summary.get("statistics", {}))
    overall_stats = cast(Dict[str, object], statistics_dict.get("overall", {}))
    if overall_stats:
        print(f"  Total segments:      {overall_stats.get('total_segments', 0)}")
        avg_length = cast(Optional[float], overall_stats.get("avg_segment_length"))
        print(f"  Avg length (days):   {_format_decimal(avg_length, 1)}")
        print(f"  Longest segment:     {overall_stats.get('max_segment_length', 'n/a')}")
    else:
        print("  Segment statistics unavailable.")

    current_segment = segment_summary.get("current_segment")
    if current_segment:
        current_segment_dict = cast(Dict[str, object], current_segment)
        start_date = cast(pd.Timestamp, current_segment_dict['start_date']).date()
        end_date = cast(pd.Timestamp, current_segment_dict['end_date']).date()
        print(
            f"  Current segment:     {current_segment_dict['regime']} | "
            f"length={current_segment_dict['length']} days | "
            f"start={start_date} → end={end_date}"
        )

    print("\nAnalysis complete.\n")


def create_basic_visualizations(df: pd.DataFrame, ticker: str):
    """
    Recreate the legacy 4-panel dashboard (price/regime, trend, volatility, Hurst).

    Returns the matplotlib Figure instance without displaying it.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.style.use("seaborn-v0_8-darkgrid")
    sns.set_palette("husl")

    fig, axes = plt.subplots(4, 1, figsize=(14, 12))
    fig.suptitle(f"Julia Mandelbrot System Analysis - {ticker}", fontsize=16, fontweight="bold")

    # 1. Price with regime shading
    ax1 = axes[0]
    regime_colors = {
        "Up-LowVol": "green",
        "Up-HighVol": "lightgreen",
        "Sideways-LowVol": "orange",
        "Sideways-HighVol": "darkorange",
        "Down-LowVol": "lightcoral",
        "Down-HighVol": "red",
        "Unknown": "gray",
    }

    ax1.plot(df.index, df["Close"], color="black", linewidth=1, alpha=0.7, label="Close Price")

    current_regime = None
    start_idx = 0
    for i in range(len(df)):
        regime = df["regime"].iloc[i] if "regime" in df.columns else "Unknown"
        if regime != current_regime:
            if current_regime is not None and i > start_idx:
                color = regime_colors.get(current_regime, "gray")
                ax1.axvspan(df.index[start_idx], df.index[i - 1], alpha=0.2, color=color)
            current_regime = regime
            start_idx = i
    if current_regime is not None:
        color = regime_colors.get(current_regime, "gray")
        ax1.axvspan(df.index[start_idx], df.index[-1], alpha=0.2, color=color)

    ax1.set_ylabel("Price ($)", fontsize=10)
    ax1.set_title("Price History with Regime Classification", fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="best")

    # 2. Trend strength
    ax2 = axes[1]
    if "trend_strength" in df.columns:
        ax2.plot(df.index, df["trend_strength"], color="blue", linewidth=1, label="Trend Strength")
        ax2.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
        ax2.axhline(y=0.2, color="green", linestyle="--", linewidth=0.5, alpha=0.5, label="Up Threshold")
        ax2.axhline(y=-0.2, color="red", linestyle="--", linewidth=0.5, alpha=0.5, label="Down Threshold")
        ax2.fill_between(df.index, -0.2, 0.2, alpha=0.1, color="gray", label="Sideways Zone")
    else:
        ax2.text(0.5, 0.5, "Trend strength not available", transform=ax2.transAxes, ha="center", va="center")
    ax2.set_ylabel("Trend Strength", fontsize=10)
    ax2.set_title("Normalized Trend Strength (OLS Slope / Volatility)", fontsize=12)
    ax2.grid(True, alpha=0.3)
    handles, _ = ax2.get_legend_handles_labels()
    if handles:
        ax2.legend(loc="best", fontsize=8)

    # 3. Volatility
    ax3 = axes[2]
    if "volatility" in df.columns:
        ax3.plot(df.index, df["volatility"], color="purple", linewidth=1, label="Realized Volatility")
    if "volatility_baseline" in df.columns:
        ax3.plot(
            df.index,
            df["volatility_baseline"],
            color="orange",
            linewidth=1,
            linestyle="--",
            alpha=0.7,
            label="Baseline (100d MA)",
        )
    if "volatility" not in df.columns and "volatility_baseline" not in df.columns:
        ax3.text(0.5, 0.5, "Volatility data not available", transform=ax3.transAxes, ha="center", va="center")
    ax3.set_ylabel("Volatility", fontsize=10)
    ax3.set_title("Rolling Volatility", fontsize=12)
    ax3.grid(True, alpha=0.3)
    handles, _ = ax3.get_legend_handles_labels()
    if handles:
        ax3.legend(loc="best", fontsize=8)

    # 4. Hurst exponent
    ax4 = axes[3]
    if "hurst" in df.columns:
        hurst_data = df["hurst"].dropna()
        if len(hurst_data) > 0:
            ax4.plot(df.index, df["hurst"], color="brown", linewidth=1, label="Hurst Exponent")
            data_min = hurst_data.min()
            data_max = hurst_data.max()
            data_range = data_max - data_min
            padding = max(0.05, data_range * 0.1)
            y_min = max(0.0, data_min - padding)
            y_max = min(1.0, data_max + padding)
            if (y_max - y_min) < 0.2:
                center = (y_min + y_max) / 2
                y_min = max(0.0, center - 0.1)
                y_max = min(1.0, center + 0.1)
            if data_min < 0.6 and data_max > 0.4:
                y_min = min(y_min, 0.4)
                y_max = max(y_max, 0.6)
            ax4.set_ylim(y_min, y_max)
            if y_min <= 0.5 <= y_max:
                ax4.axhline(y=0.5, color="black", linestyle="-", linewidth=0.5, label="Random Walk")
            if y_min <= 0.55 <= y_max:
                ax4.axhline(y=0.55, color="green", linestyle="--", linewidth=0.5, alpha=0.5, label="Trending Threshold")
            if y_min <= 0.45 <= y_max:
                ax4.axhline(y=0.45, color="red", linestyle="--", linewidth=0.5, alpha=0.5, label="Mean-Reverting Threshold")
            if y_min <= 0.55 and y_max >= 0.45:
                zone_min = max(y_min, 0.45)
                zone_max = min(y_max, 0.55)
                ax4.fill_between(df.index, zone_min, zone_max, alpha=0.1, color="gray", label="Indeterminate Zone")
            if pd.notna(df["hurst"].iloc[-1]):
                current_h = df["hurst"].iloc[-1]
                ax4.annotate(
                    f"Current: {current_h:.3f}",
                    xy=(df.index[-1], current_h),
                    xytext=(10, 10),
                    textcoords="offset points",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
                    fontsize=8,
                    ha="left",
                )
            interp_text = "H > 0.55: Persistent\nH < 0.45: Anti-persistent\n0.45 ≤ H ≤ 0.55: Random"
            if data_max < 0.6:
                text_y = 0.98
                text_va = "top"
            else:
                text_y = 0.02
                text_va = "bottom"
            ax4.text(
                0.02,
                text_y,
                interp_text,
                transform=ax4.transAxes,
                fontsize=8,
                verticalalignment=text_va,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
            )
        else:
            ax4.text(
                0.5,
                0.5,
                "Insufficient Data\nfor Hurst Analysis",
                transform=ax4.transAxes,
                ha="center",
                va="center",
                fontsize=10,
                bbox=dict(boxstyle="round", facecolor="lightgray", alpha=0.8),
            )
            ax4.set_ylim(0.0, 1.0)
    else:
        ax4.text(
            0.5,
            0.5,
            "Hurst Exponent\nNot Available",
            transform=ax4.transAxes,
            ha="center",
            va="center",
            fontsize=12,
            bbox=dict(boxstyle="round", facecolor="lightgray", alpha=0.8),
        )
        ax4.set_ylim(0.0, 1.0)
    ax4.set_xlabel("Date", fontsize=10)
    ax4.set_title("Hurst Exponent (Fractal Memory Analysis)", fontsize=12)
    ax4.grid(True, alpha=0.3)
    handles, _ = ax4.get_legend_handles_labels()
    if handles:
        ax4.legend(loc="upper right", fontsize=7)

    fig.tight_layout()
    return fig


def generate_visualizations(
    symbol: str,
    df: pd.DataFrame,
    transition_matrix: pd.DataFrame,
    save_dir: Optional[str] = None,
    show: bool = True,
    include_extra: bool = False,
) -> None:
    """
    Produce visualizations similar to the legacy examples.

    Charts are displayed by default and can optionally be saved to disk.
    """
    if show and not _ensure_interactive_matplotlib_backend():
        print(
            "Warning: Matplotlib is using a non-interactive backend. "
            "Install/enable a GUI backend (Qt/Tk/GTK) to display plots with plt.show()."
        )
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure

    figures: List[Tuple[str, Figure]] = []

    try:
        figures.append(("legacy_dashboard", create_basic_visualizations(df, symbol)))
    except Exception as exc:
        print(f"Warning: failed to generate legacy dashboard ({exc}).")

    if include_extra:
        try:
            from juliams.visualization import (
                plot_price_with_regimes,
                plot_regime_timeline,
                plot_technical_overlays,
            )
            from juliams.visualization.plots import (
                plot_transition_matrix as plot_transition_heatmap,
            )
        except Exception as exc: 
            print(f"Warning: unable to import extra visualization modules ({exc}).")
        else:
            try:
                figures.append(("price_regimes", plot_price_with_regimes(df, ticker=symbol)))
            except Exception as exc:
                print(f"Warning: failed to generate price/regime chart ({exc}).")

            try:
                figures.append(("technical_overlays", plot_technical_overlays(df, ticker=symbol)))
            except Exception as exc:
                print(f"Warning: failed to generate technical overlay chart ({exc}).")

            try:
                figures.append(("regime_timeline", plot_regime_timeline(df)))
            except Exception as exc:
                print(f"Warning: failed to generate regime timeline ({exc}).")

            if transition_matrix is not None and not transition_matrix.empty:
                try:
                    figures.append(("transition_matrix", plot_transition_heatmap(transition_matrix)))
                except Exception as exc:
                    print(f"Warning: failed to generate transition matrix heatmap ({exc}).")

    if not figures:
        return

    if save_dir:
        output_dir = Path(save_dir).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        for name, fig in figures:
            file_path = output_dir / f"{symbol.lower()}_{name}.png"
            try:
                fig.savefig(file_path, dpi=300, bbox_inches="tight")
                print(f"Saved plot: {file_path}")
            except Exception as exc:
                print(f"Warning: failed to save {file_path} ({exc}).")

    if show:
        try:
            plt.show()
        except Exception as exc:
            print(f"Warning: unable to display plots ({exc}). Use --save-plots to export images.")
        finally:
            for _, fig in figures:
                plt.close(fig)
    else:
        for _, fig in figures:
            plt.close(fig)


def main() -> None:
    """Entry point for the CLI."""
    args = parse_arguments()
    if args.walk_forward_diagnostics:
        symbols = args.diagnostic_symbols or DEFAULT_DIAGNOSTIC_SYMBOLS
        try:
            comparison, promotion, summary = run_walk_forward_diagnostics_report(
                symbols=symbols,
                period=args.diagnostic_period,
                overlay_enabled_groups=args.overlay_enabled_groups,
                rebound_exposure=args.rebound_exposure,
                rebound_fast_window=args.rebound_fast_window,
                rebound_lookback=args.rebound_lookback,
                min_rebound=args.min_rebound,
                min_fast_return=args.min_fast_return,
                require_above_fast_ma=not args.no_rebound_fast_ma,
                require_prior_drawdown=not args.no_drawdown_gate,
                drawdown_gate_lookback=args.drawdown_gate_lookback,
                min_prior_drawdown=args.min_prior_drawdown,
                max_overlay_holding_period=args.max_overlay_holding_period,
                overlay_stop_loss=args.overlay_stop_loss,
                transaction_cost_bps=args.transaction_cost_bps,
                run_parameter_stability=not args.no_parameter_stability_check,
            )
        except Exception as exc:
            print(f"Error: {exc}")
            raise SystemExit(1) from exc

        print_walk_forward_diagnostics_report(symbols, comparison, promotion, summary)
        if args.diagnostic_csv:
            output_path = Path(args.diagnostic_csv).expanduser().resolve()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            comparison.to_csv(output_path, index=False)
            print(f"\nSaved diagnostic rows: {output_path}")
        return

    symbol = args.symbol or DEFAULT_SYMBOL
    if args.symbol is None:
        print(f"No symbol supplied; defaulting to {DEFAULT_SYMBOL}.")
    try:
        artefacts = run_analysis(
            symbol=symbol,
            period=args.period,
            start=args.start,
            end=args.end,
            source_type=args.source_type,
            interval=args.interval,
            profile=args.profile,
            use_fuzzy=not args.no_fuzzy,
            horizons=args.horizons,
            adaptive_thresholds=args.adaptive_thresholds,
            adaptive_q_up=args.adaptive_q_up,
            adaptive_q_down=args.adaptive_q_down,
            adaptive_floor=args.adaptive_floor,
            adaptive_window=args.adaptive_window,
            ewma_halflife=args.ewma_halflife,
            markov_overlay=args.markov_overlay,
            markov_vol_channel=args.markov_vol_channel,
            markov_auto_vol_channel=args.markov_auto_vol_channel,
            bocpd_overlay=args.bocpd_overlay,
            bocpd_method=args.bocpd_method,
            bocpd_expected_run_length=args.bocpd_expected_run_length,
            bocpd_omega=args.bocpd_omega,
            bocpd_robustness_bandwidth=args.bocpd_robustness_bandwidth,
            consensus_overlay=args.consensus_overlay,
            consensus_window_days=args.consensus_window_days,
            consensus_cooldown_days=args.consensus_cooldown_days,
            min_dwell_days=args.min_dwell_days,
        )
    except Exception as exc:
        print(f"Error: {exc}")
        raise SystemExit(1) from exc

    print_summary(symbol, artefacts)

    should_generate_plots = not args.no_plot or bool(args.save_plots)
    if should_generate_plots:
        generate_visualizations(
            symbol,
            cast(pd.DataFrame, artefacts["df"]),
            cast(pd.DataFrame, artefacts["transition_matrix"]),
            save_dir=args.save_plots,
            show=not args.no_plot,
            include_extra=args.extra_plots,
        )


if __name__ == "__main__":
    main()
