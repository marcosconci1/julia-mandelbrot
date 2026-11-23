"""Offline tests for the diagnostics trio: journal, doctor, diagnose, and CLI wiring."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from julia_mandelbrot import __version__
import julia_mandelbrot.cli as cli_mod
from julia_mandelbrot.data_adapters import PriceSeries
import julia_mandelbrot.diagnose as diagnose_mod
import julia_mandelbrot.doctor as doctor_mod
import julia_mandelbrot.journal as journal_mod


def _synthetic_price_series(length: int = 650, seed: int = 19, source: str = "binance") -> PriceSeries:
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


def test_journal_write_read_and_best_effort_never_raises(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))

    journal_mod.log_event("report", ticker="BTCUSDT", rows=650, out="julia-report.html")

    jpath = journal_mod.get_journal_path()
    assert jpath == tmp_path / "julia-mandelbrot" / "journal.jsonl"
    assert jpath.is_file()

    events = journal_mod.read_events()
    assert len(events) == 1
    ev = events[0]
    assert ev["kind"] == "report"
    assert ev["ticker"] == "BTCUSDT"
    assert ev["rows"] == 650
    assert ev["out"] == "julia-report.html"
    parsed_ts = datetime.fromisoformat(ev["ts"])
    assert parsed_ts.tzinfo is not None

    # Best-effort: pointing XDG_CACHE_HOME at a regular file must never raise
    blocker = tmp_path / "blocker_file"
    blocker.write_text("not a directory", encoding="utf-8")
    monkeypatch.setenv("XDG_CACHE_HOME", str(blocker))
    journal_mod.log_event("should_not_raise", detail="ignored")


def test_journal_retention_drops_events_older_than_30_days(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    jpath = journal_mod.get_journal_path()
    jpath.parent.mkdir(parents=True, exist_ok=True)

    now = datetime.now(timezone.utc)
    old_45d = (now - timedelta(days=45)).isoformat()
    old_31d = (now - timedelta(days=31)).isoformat()
    recent_10d = (now - timedelta(days=10)).isoformat()

    seed_lines = [
        json.dumps({"ts": old_45d, "kind": "stale_45d", "seq": 1}) + "\n",
        json.dumps({"ts": old_31d, "kind": "stale_31d", "seq": 2}) + "\n",
        json.dumps({"ts": recent_10d, "kind": "recent_10d", "seq": 3}) + "\n",
    ]
    jpath.write_text("".join(seed_lines), encoding="utf-8")

    # Opening the journal via log_event prunes entries older than 30 days
    journal_mod.log_event("fresh_now", seq=4)

    events = journal_mod.read_events()
    kinds = [e["kind"] for e in events]
    assert "stale_45d" not in kinds
    assert "stale_31d" not in kinds
    assert kinds == ["recent_10d", "fresh_now"]


def test_journal_retention_caps_at_1_mib_and_drops_oldest_half(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    jpath = journal_mod.get_journal_path()
    jpath.parent.mkdir(parents=True, exist_ok=True)

    now_iso = datetime.now(timezone.utc).isoformat()
    padding = "x" * 950
    total_seed = 1200
    seed_lines = [
        json.dumps({"ts": now_iso, "kind": "bulk", "seq": idx, "pad": padding}) + "\n"
        for idx in range(total_seed)
    ]
    jpath.write_text("".join(seed_lines), encoding="utf-8")
    assert jpath.stat().st_size > journal_mod.MAX_JOURNAL_BYTES

    journal_mod.log_event("post_cap", seq=total_seed)

    assert jpath.stat().st_size <= journal_mod.MAX_JOURNAL_BYTES
    events = journal_mod.read_events()
    assert len(events) == (total_seed // 2) + 1
    assert events[0]["seq"] == total_seed // 2
    assert events[-1]["kind"] == "post_cap"
    assert events[-1]["seq"] == total_seed


def test_journal_redaction_keeps_ts_and_kind_only(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))

    for idx in range(25):
        journal_mod.log_event(
            f"event_{idx}",
            ticker="SECRET_TICKER_XYZ",
            out="/home/secretuser/report.html",
            api_key="EXAMPLE-KEY-VALUE",
        )

    redacted = journal_mod.read_redacted_events(limit=20)
    assert len(redacted) == 20
    assert redacted[0]["kind"] == "event_5"
    assert redacted[-1]["kind"] == "event_24"
    for item in redacted:
        assert set(item.keys()) == {"ts", "kind"}
        assert "SECRET_TICKER_XYZ" not in json.dumps(item)
        assert "secretuser" not in json.dumps(item)
        assert "EXAMPLE-KEY-VALUE" not in json.dumps(item)


def test_doctor_checks_pass_on_sane_env_and_fail_injection(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    monkeypatch.setattr(
        doctor_mod,
        "_probe_binance",
        lambda timeout=2.0: (True, "api.binance.com reachable (offline mock)"),
    )

    results = doctor_mod.run_checks()
    names = [r.name for r in results]
    assert names == [
        "python_version",
        "pandas_numpy",
        "subcommands",
        "config_defaults",
        "cache_dir",
        "binance_reachability",
        "journal_writable",
    ]
    assert all(r.ok for r in results)

    exit_code = doctor_mod.run_doctor()
    assert exit_code == 0
    out = capsys.readouterr().out
    for r in results:
        assert f"PASS {r.name} — " in out

    # Optional binance reachability failure must NOT fail the doctor
    monkeypatch.setattr(
        doctor_mod,
        "_probe_binance",
        lambda timeout=2.0: (False, "api.binance.com unreachable (offline test)"),
    )
    opt_exit = doctor_mod.run_doctor()
    opt_out = capsys.readouterr().out
    assert opt_exit == 0
    assert "FAIL binance_reachability — OPTIONAL:" in opt_out

    # Required check failure injection via monkeypatch must fail the doctor (exit 1)
    monkeypatch.setattr(
        doctor_mod,
        "_check_config_defaults",
        lambda: doctor_mod.CheckResult(
            name="config_defaults",
            ok=False,
            detail="injected config failure",
        ),
    )
    fail_exit = doctor_mod.run_doctor()
    fail_out = capsys.readouterr().out
    assert fail_exit == 1
    assert "FAIL config_defaults — injected config failure" in fail_out

    with pytest.raises(SystemExit) as exc_info:
        cli_mod.main(["doctor"])
    assert exc_info.value.code == 1
    capsys.readouterr()


def test_diagnose_bundle_contains_doctor_results_and_zero_home_paths(tmp_path, monkeypatch):
    # Place cache dir under actual Path.home() path string conceptually to verify redaction
    fake_home_cache = tmp_path / "home" / "testuser" / ".cache"
    monkeypatch.setenv("XDG_CACHE_HOME", str(fake_home_cache))
    monkeypatch.setenv("SECRET_ENV_SENTINEL", "DO_NOT_LEAK_ENV_VALUE_98765")
    monkeypatch.setattr(
        doctor_mod,
        "_probe_binance",
        lambda timeout=2.0: (True, "api.binance.com reachable (offline mock)"),
    )

    journal_mod.log_event(
        "report",
        ticker="SPY",
        rows=756,
        out="spy-report.html",
        secret_path=str(Path.home() / "private_dir" / "file.txt"),
    )

    bundle = diagnose_mod.build_bundle()

    # Package version, Python version, doctor results, and config parameters present
    assert f"package_version: {__version__}" in bundle
    assert "python_version:" in bundle
    assert "platform:" in bundle
    assert "[doctor]" in bundle
    assert "PASS python_version — " in bundle
    assert "PASS pandas_numpy — " in bundle
    assert "PASS subcommands — " in bundle
    assert "PASS config_defaults — " in bundle
    assert "PASS cache_dir — " in bundle
    assert "PASS binance_reachability — OPTIONAL:" in bundle
    assert "PASS journal_writable — " in bundle
    assert "[config]" in bundle
    assert "trend_window: 63" in bundle
    assert "transition_window: 756" in bundle
    assert "tail_quantile: 0.95" in bundle

    # Journal event includes ts and kind only, with field values redacted
    assert '"kind":"report"' in bundle
    assert "spy-report.html" not in bundle
    assert "private_dir" not in bundle

    # Zero absolute home paths and zero env var values
    home_str = str(Path.home())
    assert home_str not in bundle
    assert "/home/" not in bundle
    assert "/Users/" not in bundle
    assert "DO_NOT_LEAK_ENV_VALUE_98765" not in bundle


def test_cli_subprocess_help_and_report_logs_single_event(tmp_path, monkeypatch, capsys):
    for cmd in (
        [sys.executable, "-m", "julia_mandelbrot.cli", "--help"],
        [sys.executable, "-m", "julia_mandelbrot.cli", "doctor", "--help"],
        [sys.executable, "-m", "julia_mandelbrot.cli", "diagnose", "--help"],
    ):
        proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
        assert "usage:" in proc.stdout

    root_help = subprocess.run(
        [sys.executable, "-m", "julia_mandelbrot.cli", "--help"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert "doctor" in root_help
    assert "diagnose" in root_help

    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    monkeypatch.setattr(
        doctor_mod,
        "_probe_binance",
        lambda timeout=2.0: (True, "api.binance.com reachable (offline mock)"),
    )

    cli_mod.main(["doctor"])
    doctor_stdout = capsys.readouterr().out
    assert "PASS python_version — " in doctor_stdout

    cli_mod.main(["diagnose"])
    diagnose_stdout = capsys.readouterr().out
    assert "=== julia-mandelbrot diagnostic bundle ===" in diagnose_stdout
    assert "/home/" not in diagnose_stdout

    # Verify `cli.report` logs one report completion event after successful write
    mock_prices = _synthetic_price_series(length=650, seed=19)
    monkeypatch.setattr(cli_mod, "_fetch_prices", lambda source, ticker, start, end: mock_prices)
    out_file = tmp_path / "nested_dir" / "my-report.html"
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
    capsys.readouterr()

    events = journal_mod.read_events()
    report_events = [e for e in events if e.get("kind") == "report"]
    assert len(report_events) == 1
    rep = report_events[0]
    assert rep["ticker"] == "BTCUSDT"
    assert rep["rows"] == 650
    assert rep["out"] == "my-report.html"
