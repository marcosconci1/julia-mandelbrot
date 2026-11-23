"""Command-line entry point for the package scaffold and validation harness."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import tempfile
from typing import Any

from . import __version__
from .data_adapters import BinanceAdapter, YahooFinanceAdapter
from .diagnose import build_bundle
from .doctor import run_doctor
from .journal import log_event
from .report import write_report
from .validation import run_validation


def _sanitize_for_json(value: Any) -> Any:
    """Recursively convert NaN/Inf floats to None for strict JSON serialization."""
    if isinstance(value, dict):
        return {str(k): _sanitize_for_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize_for_json(v) for v in value]
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return value
    return value


def _fetch_prices(source: str, ticker: str, start: str, end: str):
    adapter_cls = BinanceAdapter if source == "binance" else YahooFinanceAdapter
    adapter = adapter_cls()
    try:
        return adapter.get_prices(ticker, start, end)
    except OSError:
        fallback_cache = Path(tempfile.gettempdir()) / "julia-mandelbrot-cache"
        return adapter_cls(cache_dir=fallback_cache).get_prices(ticker, start, end)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="julia-mandelbrot",
        description="Research package for market-state indicator columns.",
        epilog="Use 'julia-mandelbrot validate --help' for validation options.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--version", action="version", version=f"julia-mandelbrot {__version__}"
    )

    subparsers = parser.add_subparsers(dest="command")
    validate_parser = subparsers.add_parser(
        "validate",
        help="Run the validation harness and print JSON verdicts.",
    )
    validate_parser.add_argument(
        "--source",
        choices=("binance", "yahoo"),
        required=True,
        help="Price data source adapter ('binance' or 'yahoo').",
    )
    validate_parser.add_argument(
        "--ticker",
        required=True,
        help="Instrument symbol or ticker (e.g. BTCUSDT or SPY).",
    )
    validate_parser.add_argument(
        "--start",
        required=True,
        help="Start date in YYYY-MM-DD format.",
    )
    validate_parser.add_argument(
        "--end",
        required=True,
        help="End date in YYYY-MM-DD format.",
    )

    report_parser = subparsers.add_parser(
        "report",
        help="Render the self-contained HTML regime-indicator report.",
    )
    report_parser.add_argument(
        "--source",
        choices=("binance", "yahoo"),
        required=True,
        help="Price data source adapter ('binance' or 'yahoo').",
    )
    report_parser.add_argument(
        "--ticker",
        required=True,
        help="Instrument symbol or ticker (e.g. BTCUSDT or SPY).",
    )
    report_parser.add_argument(
        "--start",
        required=True,
        help="Start date in YYYY-MM-DD format.",
    )
    report_parser.add_argument(
        "--end",
        required=True,
        help="End date in YYYY-MM-DD format.",
    )
    report_parser.add_argument(
        "--out",
        default="./julia-report.html",
        help="Output HTML file path (default: ./julia-report.html).",
    )
    report_parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip running the validation suite.",
    )

    subparsers.add_parser(
        "doctor",
        help="Run per-item local environment and dependency checks.",
    )
    subparsers.add_parser(
        "diagnose",
        help="Print a redacted diagnostic bundle for bug reports.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    """Parse CLI arguments and run the requested command or print version and usage."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "doctor":
        exit_code = run_doctor()
        if exit_code != 0:
            raise SystemExit(exit_code)
        return

    if args.command == "diagnose":
        print(build_bundle())
        return

    if args.command == "validate":
        prices = _fetch_prices(args.source, args.ticker, args.start, args.end)
        verdicts = run_validation(prices)
        print(json.dumps(_sanitize_for_json(verdicts), indent=2, sort_keys=True))
        return

    if args.command == "report":
        prices = _fetch_prices(args.source, args.ticker, args.start, args.end)
        prices.data.attrs["ticker"] = args.ticker.upper()
        verdicts = None if args.no_validate else run_validation(prices)
        out_path = write_report(prices, args.out, validated=verdicts, ticker=args.ticker)
        log_event(
            "report",
            ticker=args.ticker,
            rows=len(prices.data),
            out=Path(out_path).name,
        )
        print(out_path)
        return

    print(f"julia-mandelbrot {__version__}")
    parser.print_help()


if __name__ == "__main__":
    main()
