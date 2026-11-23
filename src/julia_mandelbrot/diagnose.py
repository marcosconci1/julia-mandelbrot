"""Redacted diagnostic bundle builder for ``julia-mandelbrot diagnose``.

Builds a human-readable, paste-safe plain-text report containing:
- Package version, Python version, and platform summary
- Per-item ``doctor`` check results
- ``IndicatorConfig`` default parameter names and values
- The last 20 journal events restricted to ``ts`` and ``kind`` only (all event
  field values are redacted)

Privacy guarantee:
The journal and diagnostic bundle never leave the machine except by the user's
own hand. The bundle omits environment variable values, network payloads, and
all filesystem paths beyond the ``~``-redacted cache directory.
"""

from __future__ import annotations

from dataclasses import fields as dataclass_fields
import json
import platform

from . import __version__
from .config import IndicatorConfig
from .doctor import CheckResult, format_check, run_checks
from .journal import get_cache_dir, read_redacted_events, redact_path


def build_bundle(checks: list[CheckResult] | None = None) -> str:
    """Build and return a redacted plain-text diagnostic bundle for bug reports.

    The returned text contains package version, Python version, platform,
    doctor results, IndicatorConfig parameter names and values, and the last
    20 journal events (``ts`` and ``kind`` only, with all other field values
    redacted). It contains zero absolute home paths, zero environment variable
    values, and zero network payloads, and never leaves the machine except by
    the user's own hand.
    """
    doctor_results = checks if checks is not None else run_checks()
    config = IndicatorConfig()
    recent_events = read_redacted_events(limit=20)
    platform_summary = f"{platform.system()} {platform.release()} ({platform.machine()})"

    lines: list[str] = [
        "=== julia-mandelbrot diagnostic bundle ===",
        "note: local-only bundle; never leaves this machine except by the user's own hand.",
        f"package_version: {__version__}",
        f"python_version: {platform.python_version()}",
        f"platform: {platform_summary}",
        f"cache_dir: {redact_path(get_cache_dir())}",
        "",
        "[doctor]",
    ]

    for item in doctor_results:
        lines.append(format_check(item))

    lines.append("")
    lines.append("[config]")
    for field_info in dataclass_fields(config):
        value = getattr(config, field_info.name)
        lines.append(f"{field_info.name}: {value!r}")

    lines.append("")
    lines.append("[journal (last 20 events, ts+kind only)]")
    if recent_events:
        for ev in recent_events:
            redacted_entry = {"ts": ev["ts"], "kind": ev["kind"]}
            lines.append(json.dumps(redacted_entry, separators=(",", ":")))
    else:
        lines.append("(no journal events)")

    return redact_path("\n".join(lines))
