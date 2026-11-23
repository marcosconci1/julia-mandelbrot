"""Public daily price data adapters and a small file based cache."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd

_COLUMNS = ("Open", "High", "Low", "Close", "Volume")
_YAHOO_ROOT = "https://query1.finance.yahoo.com/v8/finance/chart/"
_BINANCE_URL = "https://api.binance.com/api/v3/klines"
_BINANCE_PAGE_SIZE = 500
_BINANCE_MAX_REQUESTS = 50
_DAY_MS = 86_400_000
_TRADING_DAYS_PER_CALENDAR_DAY = 252.0 / 365.25
_YAHOO_MIN_COVERAGE_RATIO = 0.90


class DataAdapterError(ValueError):
    """Raised when a provider response cannot be read as daily prices."""


@dataclass(frozen=True, slots=True)
class PriceSeries:
    """Daily OHLCV data with a UTC date index and provider identity."""

    data: pd.DataFrame
    source: str

    def __post_init__(self) -> None:
        frame = self.data.copy()
        if not isinstance(frame.index, pd.DatetimeIndex):
            raise DataAdapterError("PriceSeries index must be a DatetimeIndex")
        if frame.index.tz is None:
            frame.index = frame.index.tz_localize("UTC")
        else:
            frame.index = frame.index.tz_convert("UTC")
        missing = set(_COLUMNS).difference(frame.columns)
        if missing:
            raise DataAdapterError(f"PriceSeries is missing columns: {', '.join(sorted(missing))}")
        frame = frame.loc[:, _COLUMNS].sort_index()
        frame.index.name = "Date"
        object.__setattr__(self, "data", frame)

    @property
    def frame(self) -> pd.DataFrame:
        """Alias for callers that prefer an explicit frame name."""
        return self.data


def _default_cache_dir() -> Path:
    root = os.environ.get("XDG_CACHE_HOME")
    return Path(root) / "julia-mandelbrot" if root else Path.home() / ".cache" / "julia-mandelbrot"


class _CachedAdapter:
    source = ""

    def __init__(self, cache_dir: str | Path | None = None, max_age_days: float = 1.0):
        if max_age_days < 0:
            raise ValueError("max_age_days must be non-negative")
        self.cache_dir = Path(cache_dir) if cache_dir is not None else _default_cache_dir()
        self.max_age_days = float(max_age_days)

    def _cache_path(
        self,
        ticker: str,
        interval: str,
        start: date | datetime | str = "",
        end: date | datetime | str = "",
    ) -> Path:
        identity = f"{self.source}\0{ticker.upper()}\0{interval}\0{start}\0{end}"
        key = hashlib.sha256(identity.encode()).hexdigest()
        return self.cache_dir / f"{key}.csv"

    def get_prices(
        self,
        ticker: str,
        start: date | datetime | str,
        end: date | datetime | str,
        *,
        interval: str = "1d",
        max_age_days: float | None = None,
        refresh: bool = False,
    ) -> PriceSeries:
        """Return cached prices when fresh, otherwise fetch and atomically cache.

        Cache-key semantics:
        Each cache file is keyed by the SHA-256 digest of
        ``f"{source}\\0{ticker.upper()}\\0{interval}\\0{start}\\0{end}"``.
        The cache key includes the requested ``start`` and ``end`` date window
        alongside ``source``, ``ticker``, and ``interval`` so that requests for
        different date windows cache separately.
        """
        path = self._cache_path(ticker, interval, start, end)
        age_limit = self.max_age_days if max_age_days is None else max_age_days
        if age_limit < 0:
            raise ValueError("max_age_days must be non-negative")
        if not refresh and path.is_file() and time.time() - path.stat().st_mtime <= age_limit * 86400:
            try:
                cached = pd.read_csv(path, index_col="Date", parse_dates=["Date"])
                return PriceSeries(cached, self.source)
            except (OSError, ValueError, KeyError, pd.errors.ParserError):
                # A damaged cache is treated as a miss and replaced on success.
                pass

        result = self.fetch(ticker, start, end, interval=interval)
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_name: str | None = None
        try:
            with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="", dir=path.parent, delete=False) as tmp:
                temp_name = tmp.name
                result.data.to_csv(tmp, index_label="Date", date_format="%Y-%m-%dT%H:%M:%S%z")
                tmp.flush()
                os.fsync(tmp.fileno())
            os.replace(temp_name, path)
        finally:
            if temp_name and os.path.exists(temp_name):
                os.unlink(temp_name)
        return result

    def fetch(self, ticker: str, start: date | datetime | str, end: date | datetime | str, *, interval: str = "1d") -> PriceSeries:
        raise NotImplementedError


def _epoch(value: date | datetime | str) -> int:
    if isinstance(value, str):
        value = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if isinstance(value, date) and not isinstance(value, datetime):
        value = datetime.combine(value, datetime.min.time(), tzinfo=timezone.utc)
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return int(value.timestamp())


def _http_json(url: str) -> Any:
    request = Request(url, headers={"User-Agent": "julia-mandelbrot/0.1"})
    for attempt in range(2):
        try:
            with urlopen(request, timeout=20) as response:
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            if exc.code not in (408, 429, 500, 502, 503, 504) or attempt == 1:
                raise DataAdapterError(f"Provider HTTP error {exc.code}") from exc
        except (URLError, TimeoutError) as exc:
            if attempt == 1:
                raise DataAdapterError(f"Provider request failed: {exc.reason if isinstance(exc, URLError) else exc}") from exc
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise DataAdapterError("Provider returned invalid JSON") from exc
        time.sleep(0.25 * (attempt + 1))
    raise DataAdapterError("Provider request failed")


class YahooFinanceAdapter(_CachedAdapter):
    """Daily equities and FX from Yahoo Finance's public chart endpoint."""

    source = "yahoo"

    def fetch(self, ticker: str, start: date | datetime | str, end: date | datetime | str, *, interval: str = "1d") -> PriceSeries:
        """Fetch daily OHLCV bars from Yahoo Finance for ``[start, end]``.

        Yahoo's v8 chart endpoint returns the full requested range in a single
        response without pagination, so this method verifies that the returned
        timestamps cover at least 90% of the requested trading days and fails
        loudly with :class:`DataAdapterError` if the response is truncated.
        """
        start_epoch = _epoch(start)
        end_epoch = _epoch(end)
        params = urlencode({"interval": interval, "period1": start_epoch, "period2": end_epoch})
        payload = _http_json(f"{_YAHOO_ROOT}{ticker}?{params}")
        try:
            result = payload["chart"]["result"][0]
            timestamps = result["timestamp"]
            quote = result["indicators"]["quote"][0]
            if not timestamps:
                raise DataAdapterError(f"Yahoo returned no price rows for {ticker}")
            frame = pd.DataFrame({name: quote[key] for name, key in zip(_COLUMNS, ("open", "high", "low", "close", "volume"))}, index=pd.to_datetime(timestamps, unit="s", utc=True))
            frame = frame.dropna(subset=["Open", "High", "Low", "Close"])
        except DataAdapterError:
            raise
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            description = payload.get("chart", {}).get("error") if isinstance(payload, dict) else None
            raise DataAdapterError(f"Malformed Yahoo response for {ticker}: {description or exc}") from exc
        if frame.empty:
            raise DataAdapterError(f"Yahoo response contains no usable OHLC rows for {ticker}")

        requested_days = max(0.0, (end_epoch - start_epoch) / 86400.0)
        requested_trading_days = int(requested_days * _TRADING_DAYS_PER_CALENDAR_DAY)
        span_days = max(0.0, (max(timestamps) - min(timestamps)) / 86400.0) + 1.0
        covered_trading_days = min(float(len(frame)), span_days * _TRADING_DAYS_PER_CALENDAR_DAY)
        if (
            requested_trading_days > 0
            and covered_trading_days < _YAHOO_MIN_COVERAGE_RATIO * requested_trading_days
            and len(timestamps) < requested_trading_days
        ):
            raise DataAdapterError(
                f"Yahoo returned incomplete date coverage for {ticker}: "
                f"{len(timestamps)} timestamps covering {covered_trading_days:.1f} "
                f"of {requested_trading_days} requested trading days (<90%)"
            )
        return PriceSeries(frame, self.source)


class BinanceAdapter(_CachedAdapter):
    """Daily cryptocurrency klines from Binance's public API."""

    source = "binance"
    page_size = _BINANCE_PAGE_SIZE
    max_requests = _BINANCE_MAX_REQUESTS

    def fetch(self, ticker: str, start: date | datetime | str, end: date | datetime | str, *, interval: str = "1d") -> PriceSeries:
        """Fetch and paginate daily klines from Binance for ``[start, end]``.

        Pagination behavior:
        Binance's ``/api/v3/klines`` endpoint caps each response at ``page_size``
        (500) rows, so this method paginates klines across long date ranges to
        return complete history. It loops requests by advancing ``startTime`` to
        the last received open timestamp plus one day (``86_400_000`` ms) until
        the API returns fewer rows than ``page_size`` (500) or ``endTime`` is
        covered. Total requests per ``fetch`` call are capped at
        ``max_requests`` (50, covering ~68 years of daily bars); exceeding the
        cap raises :class:`DataAdapterError`.
        """
        start_ms = _epoch(start) * 1000
        end_ms = _epoch(end) * 1000
        next_start_ms = start_ms
        rows: list[list[float]] = []
        dates: list[pd.Timestamp] = []

        for _ in range(self.max_requests):
            params = urlencode(
                {
                    "symbol": ticker.upper(),
                    "interval": interval,
                    "startTime": next_start_ms,
                    "endTime": end_ms,
                }
            )
            payload = _http_json(f"{_BINANCE_URL}?{params}")
            try:
                if not isinstance(payload, list) or (not payload and not rows):
                    raise DataAdapterError(f"Binance returned no price rows for {ticker}")
                if not payload:
                    break
                last_open_ms = next_start_ms
                for item in payload:
                    if not isinstance(item, list) or len(item) < 6:
                        raise DataAdapterError(f"Malformed Binance kline for {ticker}")
                    last_open_ms = int(item[0])
                    dates.append(pd.to_datetime(item[0], unit="ms", utc=True))
                    rows.append([float(item[i]) for i in (1, 2, 3, 4, 5)])
            except DataAdapterError:
                raise
            except (TypeError, ValueError, IndexError) as exc:
                raise DataAdapterError(f"Malformed Binance response for {ticker}: {exc}") from exc

            if len(payload) < self.page_size:
                break
            next_start_ms = last_open_ms + _DAY_MS
            if next_start_ms > end_ms:
                break
        else:
            raise DataAdapterError(
                f"Binance pagination exceeded {self.max_requests} requests for {ticker}"
            )

        frame = pd.DataFrame(rows, columns=_COLUMNS, index=pd.DatetimeIndex(dates))
        return PriceSeries(frame, self.source)
