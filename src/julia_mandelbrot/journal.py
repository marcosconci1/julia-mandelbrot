"""Append-only local JSONL event journal with age and size retention.

The journal is stored at ``(XDG_CACHE_HOME or ~/.cache)/julia-mandelbrot/journal.jsonl``.

Retention policy:
1. Age retention: on open, any JSONL lines with a UTC ``ts`` older than 30 days
   (``RETENTION_DAYS = 30``) are dropped.
2. Size cap: on open (and when appending), the file is capped at 1 MiB
   (``MAX_JOURNAL_BYTES = 1_048_576`` bytes); whenever the file exceeds 1 MiB,
   the oldest half of lines is dropped.

Privacy guarantee:
The journal and diagnostic bundle are strictly local and never leave the machine
except by the user's own hand (no telemetry, no background network uploads).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any

RETENTION_DAYS: int = 30
MAX_JOURNAL_BYTES: int = 1024 * 1024  # 1 MiB
MAX_BYTES: int = MAX_JOURNAL_BYTES
CACHE_DIR_NAME: str = "julia-mandelbrot"
JOURNAL_FILENAME: str = "journal.jsonl"


def get_cache_dir() -> Path:
    """Return the platform cache directory for julia-mandelbrot.

    Uses ``$XDG_CACHE_HOME/julia-mandelbrot`` when ``XDG_CACHE_HOME`` is set,
    otherwise ``~/.cache/julia-mandelbrot`` (with a fallback to the system
    temporary directory only when running inside a read-only home mount).
    """
    xdg = os.environ.get("XDG_CACHE_HOME")
    if xdg:
        return Path(xdg) / CACHE_DIR_NAME
    home = Path.home()
    default_root = home / ".cache"
    target = default_root / CACHE_DIR_NAME
    if os.access(target, os.W_OK) or (not target.exists() and os.access(default_root, os.W_OK)):
        return target
    if not os.access(default_root, os.W_OK) and not os.access(home, os.W_OK):
        return Path(tempfile.gettempdir()) / CACHE_DIR_NAME
    return target


cache_dir = get_cache_dir


def get_journal_path(base_dir: str | Path | None = None) -> Path:
    """Return the path to ``journal.jsonl`` inside the cache directory."""
    if base_dir is not None:
        candidate = Path(base_dir)
        if candidate.suffix == ".jsonl":
            return candidate
        return candidate / JOURNAL_FILENAME
    return get_cache_dir() / JOURNAL_FILENAME


journal_path = get_journal_path


def redact_path(path: str | Path) -> str:
    """Format a path with any absolute home directory prefix replaced by ``~``."""
    text = str(path)
    try:
        home = str(Path.home())
    except Exception:
        home = ""
    if home and home != "/":
        text = text.replace(home, "~")
    text = re.sub(r"/(?:home|Users)/[^/\s\"']+", "~", text)
    text = re.sub(r"[A-Za-z]:\\Users\\[^\\\s\"']+", "~", text)
    return text


def _parse_ts(value: Any) -> datetime | None:
    """Parse an ISO-8601 timestamp string into a UTC-aware datetime."""
    if not isinstance(value, str) or not value.strip():
        return None
    normalized = value.strip()
    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def apply_retention(
    lines: list[str],
    *,
    now: datetime | None = None,
    max_age_days: int = RETENTION_DAYS,
    max_bytes: int = MAX_JOURNAL_BYTES,
) -> list[str]:
    """Filter journal lines by 30-day age retention and 1 MiB size cap.

    Drops lines older than ``max_age_days`` days relative to ``now`` (UTC), and
    whenever total UTF-8 byte size exceeds ``max_bytes``, drops the oldest half
    of remaining lines.
    """
    reference_now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    cutoff = reference_now - timedelta(days=max_age_days)

    retained: list[str] = []
    for raw in lines:
        stripped = raw.strip()
        if not stripped:
            continue
        try:
            payload = json.loads(stripped)
        except (ValueError, TypeError):
            continue
        if not isinstance(payload, dict):
            continue
        ts_dt = _parse_ts(payload.get("ts"))
        if ts_dt is None or ts_dt < cutoff:
            continue
        retained.append(stripped + "\n")

    def _total_bytes(items: list[str]) -> int:
        return sum(len(item.encode("utf-8")) for item in items)

    while _total_bytes(retained) > max_bytes and len(retained) > 1:
        retained = retained[len(retained) // 2 :]

    if len(retained) == 1 and _total_bytes(retained) > max_bytes:
        retained = []

    return retained


def log_event(kind: str, **fields: Any) -> None:
    """Append one event line ``{"ts": <UTC ISO>, "kind": kind, ...fields}`` to the journal.

    On open, drops lines older than 30 days and caps the journal file at 1 MiB
    by dropping the oldest half of lines when exceeded. Flushes after writing.
    Best-effort: never raises exceptions to callers.
    The journal never leaves the machine except by the user's own hand.
    """
    try:
        override_path = fields.pop("_journal_path", None)
        target = get_journal_path(override_path)
        target.parent.mkdir(parents=True, exist_ok=True)

        now_dt = datetime.now(timezone.utc)
        explicit_ts = fields.pop("ts", None)
        if explicit_ts is None:
            ts_str = now_dt.isoformat()
        elif isinstance(explicit_ts, datetime):
            ts_str = explicit_ts.astimezone(timezone.utc).isoformat()
        else:
            ts_str = str(explicit_ts)

        record: dict[str, Any] = {"ts": ts_str, "kind": str(kind)}
        record.update(fields)
        new_line = json.dumps(record, default=str, separators=(",", ":")) + "\n"

        existing_lines: list[str] = []
        if target.is_file():
            try:
                existing_lines = target.read_text(encoding="utf-8", errors="replace").splitlines(
                    keepends=True
                )
            except OSError:
                existing_lines = []

        retained_existing = apply_retention(
            existing_lines,
            now=now_dt,
            max_age_days=RETENTION_DAYS,
            max_bytes=MAX_JOURNAL_BYTES,
        )

        # Check if the new line is itself within the 30-day retention window
        combined = apply_retention(
            retained_existing + [new_line],
            now=now_dt,
            max_age_days=RETENTION_DAYS,
            max_bytes=MAX_JOURNAL_BYTES,
        )

        needs_rewrite = (
            retained_existing != existing_lines
            or combined != retained_existing + [new_line]
        )

        if needs_rewrite:
            with target.open("w", encoding="utf-8") as fh:
                fh.writelines(combined)
                fh.flush()
        else:
            with target.open("a", encoding="utf-8") as fh:
                fh.write(new_line)
                fh.flush()
    except Exception:
        return


def read_events(
    path: str | Path | None = None,
    *,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Read parsed JSON event records from the journal file (best-effort)."""
    try:
        target = get_journal_path(path)
        if not target.is_file():
            return []
        raw_lines = target.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)
        retained = apply_retention(raw_lines)
        events: list[dict[str, Any]] = []
        for line in retained:
            try:
                obj = json.loads(line)
            except (ValueError, TypeError):
                continue
            if isinstance(obj, dict):
                events.append(obj)
        if limit is not None and limit >= 0:
            return events[-limit:] if limit > 0 else []
        return events
    except Exception:
        return []


def read_redacted_events(
    path: str | Path | None = None,
    *,
    limit: int = 20,
) -> list[dict[str, str]]:
    """Return the last ``limit`` journal events containing ONLY ``ts`` and ``kind``.

    All additional event field values are redacted/omitted so the resulting records
    can be safely included in a public bug-report diagnostic bundle.
    """
    events = read_events(path, limit=limit)
    return [
        {
            "ts": str(event.get("ts", "")),
            "kind": str(event.get("kind", "")),
        }
        for event in events
    ]
