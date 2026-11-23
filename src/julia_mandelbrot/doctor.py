"""Environment health checks for the ``julia-mandelbrot doctor`` command.

Runs per-item pass/fail checks over the local Python runtime, dependencies,
subcommands, configuration defaults, cache directory, optional Binance API
reachability, and journal writability.

Privacy guarantee:
Check output and journal entries stay on the local machine and never leave
except by the user's own hand.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib
from pathlib import Path
import sys
import tempfile
from typing import Iterator
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .config import IndicatorConfig
from .journal import get_cache_dir, get_journal_path, log_event, redact_path

_BINANCE_PING_URL = "https://api.binance.com/api/v3/ping"


@dataclass(frozen=True, slots=True)
class CheckResult:
    """Outcome of a single ``doctor`` check item."""

    name: str
    ok: bool
    detail: str
    optional: bool = False

    def __iter__(self) -> Iterator[ str | bool ]:
        """Allow unpacking as ``name, ok, detail``."""
        yield self.name
        yield self.ok
        yield self.detail


def _check_python_version() -> CheckResult:
    info = sys.version_info
    version_str = f"{info.major}.{info.minor}.{info.micro}"
    ok = (info.major, info.minor) >= (3, 10)
    detail = (
        f"Python {version_str} (>= 3.10)"
        if ok
        else f"Python {version_str} is below required 3.10"
    )
    return CheckResult(name="python_version", ok=ok, detail=detail)


def _check_pandas_numpy() -> CheckResult:
    try:
        np_mod = importlib.import_module("numpy")
        pd_mod = importlib.import_module("pandas")
        np_ver = getattr(np_mod, "__version__", "unknown")
        pd_ver = getattr(pd_mod, "__version__", "unknown")
        return CheckResult(
            name="pandas_numpy",
            ok=True,
            detail=f"numpy {np_ver}, pandas {pd_ver} importable",
        )
    except Exception as exc:
        return CheckResult(
            name="pandas_numpy",
            ok=False,
            detail=f"import failed ({type(exc).__name__}: {exc})",
        )


def _check_subcommands() -> CheckResult:
    try:
        report_mod = importlib.import_module("julia_mandelbrot.report")
        val_mod = importlib.import_module("julia_mandelbrot.validation")
        cli_mod = importlib.import_module("julia_mandelbrot.cli")
        if not callable(getattr(report_mod, "write_report", None)):
            raise ImportError("write_report is not callable")
        if not callable(getattr(val_mod, "run_validation", None)):
            raise ImportError("run_validation is not callable")
        if not callable(getattr(cli_mod, "_build_parser", None)):
            raise ImportError("_build_parser is not callable")
        return CheckResult(
            name="subcommands",
            ok=True,
            detail="report and validate subcommands importable",
        )
    except Exception as exc:
        return CheckResult(
            name="subcommands",
            ok=False,
            detail=f"subcommand import failed ({type(exc).__name__}: {exc})",
        )


def _check_config_defaults() -> CheckResult:
    try:
        cfg = IndicatorConfig()
        return CheckResult(
            name="config_defaults",
            ok=True,
            detail=(
                f"IndicatorConfig() loaded "
                f"(trend_window={cfg.trend_window}, percentile_window={cfg.percentile_window})"
            ),
        )
    except Exception as exc:
        return CheckResult(
            name="config_defaults",
            ok=False,
            detail=f"IndicatorConfig() failed ({type(exc).__name__}: {exc})",
        )


def _check_cache_dir() -> CheckResult:
    try:
        cdir = Path(get_cache_dir())
        cdir.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=cdir,
            prefix=".doctor-probe-",
            delete=True,
        ) as probe:
            probe.write("ok")
            probe.flush()
        return CheckResult(
            name="cache_dir",
            ok=True,
            detail=f"cache directory writable ({redact_path(cdir)})",
        )
    except Exception as exc:
        return CheckResult(
            name="cache_dir",
            ok=False,
            detail=f"cache directory not writable ({type(exc).__name__})",
        )


def _probe_binance(timeout: float = 2.0) -> tuple[bool, str]:
    request = Request(_BINANCE_PING_URL, headers={"User-Agent": "julia-mandelbrot/0.1"})
    try:
        with urlopen(request, timeout=timeout) as response:
            status = getattr(response, "status", 200)
            return True, f"api.binance.com reachable (HTTP {status})"
    except HTTPError as exc:
        return False, f"api.binance.com returned HTTP {exc.code}"
    except (URLError, TimeoutError, OSError) as exc:
        return False, f"api.binance.com unreachable ({type(exc).__name__})"


def _check_binance_reachability() -> CheckResult:
    ok, msg = _probe_binance()
    return CheckResult(
        name="binance_reachability",
        ok=ok,
        detail=f"OPTIONAL: {msg}",
        optional=True,
    )


def _check_journal_writable() -> CheckResult:
    try:
        jpath = Path(get_journal_path())
        jpath.parent.mkdir(parents=True, exist_ok=True)
        with jpath.open("a", encoding="utf-8") as fh:
            fh.flush()
        return CheckResult(
            name="journal_writable",
            ok=True,
            detail=f"journal file writable ({redact_path(jpath)})",
        )
    except Exception as exc:
        return CheckResult(
            name="journal_writable",
            ok=False,
            detail=f"journal file not writable ({type(exc).__name__})",
        )


def run_checks() -> list[CheckResult]:
    """Execute all environment diagnostic checks and return their results."""
    return [
        _check_python_version(),
        _check_pandas_numpy(),
        _check_subcommands(),
        _check_config_defaults(),
        _check_cache_dir(),
        _check_binance_reachability(),
        _check_journal_writable(),
    ]


def format_check(result: CheckResult) -> str:
    """Format a check result as ``PASS name — detail`` or ``FAIL name — detail``."""
    status = "PASS" if result.ok else "FAIL"
    return f"{status} {result.name} — {result.detail}"


def checks_passed(results: list[CheckResult]) -> bool:
    """Return True iff all non-optional checks passed."""
    return all(result.ok for result in results if not result.optional)


def run_doctor() -> int:
    """Run all checks, print one line per check, and return exit code (0 or 1)."""
    results = run_checks()
    for item in results:
        print(format_check(item))
    overall_ok = checks_passed(results)
    log_event(
        "doctor",
        ok=overall_ok,
        passed=sum(1 for r in results if r.ok),
        total=len(results),
    )
    return 0 if overall_ok else 1
