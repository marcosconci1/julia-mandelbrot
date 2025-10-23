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
from juliams.data.utils import detect_source_type
from juliams.features import (
    compute_fractal_features,
    compute_hurst_features,
    compute_trend_features,
    compute_volatility_features,
)
from juliams.regimes import RegimeClassifier
from juliams.regimes.fuzzy import compute_fuzzy_features
from juliams.analysis import (
    analyze_forward_returns_by_regime,
    compute_forward_returns,
    compute_transition_matrix,
    get_segment_summary,
)

DEFAULT_SYMBOL = "AAPL"


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
    use_fuzzy: bool = True,
    horizons: Optional[List[int]] = None,
) -> Dict[str, object]:
    """
    Execute the full analysis pipeline and return collected artefacts.

    Returns a dictionary containing the enriched dataframe, analysis summaries,
    and metadata for downstream reporting.
    """
    config = JMSConfig()
    resolved_source, source_note = resolve_source(symbol, source_type)
    source_config = config.get_source_config(resolved_source)

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
    )

    # Feature computation
    df = compute_trend_features(df, source_config)
    df = compute_volatility_features(df, source_config)
    df = compute_hurst_features(df, source_config)
    df = compute_fractal_features(df, source_config)

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
    }


def _format_percentage(value: float) -> str:
    """Helper for consistent percentage formatting."""
    return f"{value * 100:6.2f}%" if pd.notna(value) else "   n/a"


def _format_decimal(value: Optional[float], precision: int = 4) -> str:
    """Format numeric values with safe handling for NaN."""
    if value is None or pd.isna(value):
        return "n/a"
    return f"{value:.{precision}f}"


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

    print("\n" + "=" * 72)
    print("Julia Mandelbrot System - Unified Analysis")
    print("=" * 72)
    print(f"Symbol:        {symbol}")
    date_info = f"period={period}" if period else f"start={start} • end={end}"
    print(f"Data range:    {date_info}")
    print(f"Source type:   {source_type} ({source_note})")
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

    print("\nForward Returns")
    print("-" * 72)
    forward_analysis: Dict[str, Dict[str, Dict[str, float]]] = cast(Dict[str, Dict[str, Dict[str, float]]], artefacts["forward_analysis"])
    for horizon in horizons:
        key = f"{horizon}d"
        overall_stats = forward_analysis.get("Overall", {}).get(key)
        if overall_stats:
            print(
                f"  Horizon {key}: mean={_format_percentage(overall_stats.get('mean', float('nan')))} "
                f"win_rate={_format_percentage(overall_stats.get('positive_pct', float('nan')))}"
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
            f"win_rate={_format_percentage(horizon_stats.get('positive_pct', float('nan')))}"
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
            use_fuzzy=not args.no_fuzzy,
            horizons=args.horizons,
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
