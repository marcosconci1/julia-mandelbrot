"""Offline synthetic tests for the HTML report renderer and CLI report subcommand."""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pandas as pd

import julia_mandelbrot.cli as cli_mod
from julia_mandelbrot.cells import cell_frame
from julia_mandelbrot.data_adapters import PriceSeries
from julia_mandelbrot.report import build_report, write_report
from julia_mandelbrot.transitions import CANONICAL_CELLS
from julia_mandelbrot.validation import run_validation


def _synthetic_price_series(length: int = 900, seed: int = 42, source: str = "binance") -> PriceSeries:
    rng = np.random.default_rng(seed)
    t = np.arange(length, dtype=float)
    drift = 0.002 * np.sin(2.0 * np.pi * t / 110.0)
    sigma = 0.015 * np.exp(0.35 * np.sin(2.0 * np.pi * t / 70.0))
    log_returns = drift + rng.normal(0.0, sigma)
    close = 25000.0 * np.exp(np.cumsum(log_returns))
    index = pd.date_range("2021-01-01", periods=length, freq="D", tz="UTC")
    frame = pd.DataFrame(
        {
            "Open": close * 0.998,
            "High": close * 1.01,
            "Low": close * 0.99,
            "Close": close,
            "Volume": np.full(length, 1500.0),
        },
        index=index,
    )
    return PriceSeries(frame, source)


def test_build_report_contains_panels_current_cell_p_next_and_no_none_leaks():
    prices = _synthetic_price_series(length=900, seed=42)
    c_frame = cell_frame(prices)
    current_cell = str(c_frame.loc[c_frame["valid"], "cell"].iloc[-1])

    html_no_val = build_report(prices)

    # Required section and panel titles
    for title in (
        "PRICE /",
        "TREND STRGTH OLS/σ",
        "VOL PCTL",
        "MEMB TREND",
        "MEMB VOL",
        "ENTROPY",
        "HURST LOGVOL",
        "TAIL",
        "REGIME STATE",
        "TRANSITION MODEL",
        "IN-CELL HISTORY",
        "VALIDATION",
        "LEGEND",
    ):
        assert title in html_no_val

    # Stats ribbon headers
    for ribbon_key in (
        "LAST",
        "CELL",
        "DAYS IN CELL",
        "P(STAY)",
        "TREND σ-NORM",
        "VOL PCTL",
        "H LOGVOL",
        "CVAR 95%",
        "MAX DD",
    ):
        assert ribbon_key in html_no_val

    # Active cell label and all 6 canonical transition p_next entries
    assert current_cell.upper() in html_no_val
    for cell_name in CANONICAL_CELLS:
        assert cell_name in html_no_val
        assert f"p_next_{cell_name}" in html_no_val

    # Without `validated`, shows 'RUN VALIDATE' placeholders
    assert "RUN VALIDATE" in html_no_val
    assert "SCORE · NOT PROB" in html_no_val

    # No 'None' or 'nan' leaks and strictly self-contained (no external URLs or file:// font paths)
    assert "None" not in html_no_val
    assert "http://" not in html_no_val
    assert "https://" not in html_no_val
    assert "file://" not in html_no_val

    # With `validated` dict provided, renders validation numbers instead of placeholders
    verdicts = run_validation(prices)
    html_with_val = build_report(prices, validated=verdicts)
    assert "RUN VALIDATE" not in html_with_val
    assert "SCORE · NOT PROB" in html_with_val
    assert "None" not in html_with_val
    assert "http://" not in html_with_val
    assert "https://" not in html_with_val
    assert "file://" not in html_with_val


def test_write_report_writes_self_contained_html_file(tmp_path):
    prices = _synthetic_price_series(length=600, seed=7)
    target = tmp_path / "nested" / "jm-report.html"
    out = write_report(prices, target)

    assert out == target
    assert target.is_file()
    content = target.read_text(encoding="utf-8")
    assert content.startswith("<!DOCTYPE html>")
    assert "<svg" in content
    assert "None" not in content
    assert "http://" not in content
    assert "https://" not in content


def test_cli_help_includes_report_and_report_subcommand_writes_output(tmp_path, monkeypatch, capsys):
    proc = subprocess.run(
        [sys.executable, "-m", "julia_mandelbrot.cli", "--help"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "report" in proc.stdout

    proc_sub = subprocess.run(
        [sys.executable, "-m", "julia_mandelbrot.cli", "report", "--help"],
        capture_output=True,
        text=True,
        check=True,
    )
    for flag in ("--source", "--ticker", "--start", "--end", "--out", "--no-validate"):
        assert flag in proc_sub.stdout

    # Offline end-to-end execution of `julia-mandelbrot report`
    mock_prices = _synthetic_price_series(length=650, seed=19)
    monkeypatch.setattr(cli_mod, "_fetch_prices", lambda source, ticker, start, end: mock_prices)

    out_file = tmp_path / "cli-report.html"
    cli_mod.main(
        [
            "report",
            "--source",
            "binance",
            "--ticker",
            "BTCUSDT",
            "--start",
            "2021-01-01",
            "--end",
            "2022-10-01",
            "--out",
            str(out_file),
        ]
    )
    captured = capsys.readouterr()
    assert str(out_file) in captured.out.strip()
    assert out_file.is_file()
    rendered = out_file.read_text(encoding="utf-8")
    assert "RUN VALIDATE" not in rendered
    assert "None" not in rendered
