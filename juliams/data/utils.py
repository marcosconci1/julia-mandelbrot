"""
Utility helpers for data sources.
"""

from __future__ import annotations

import re
import time
from collections import deque
from functools import wraps
from threading import Lock
from typing import Callable, Literal
import logging

logger = logging.getLogger(__name__)

_STABLECOIN_SUFFIXES = ("USDT", "BUSD")
_CRYPTO_TICKERS = {
    "ADA",
    "ARB",
    "ATOM",
    "AVAX",
    "BCH",
    "BNB",
    "BTC",
    "DOGE",
    "DOT",
    "ETH",
    "ETC",
    "FIL",
    "ICP",
    "LINK",
    "LTC",
    "MATIC",
    "NEAR",
    "OP",
    "SOL",
    "TRX",
    "XLM",
    "XMR",
    "XRP",
}

def detect_source_type(symbol: str) -> Literal["stock", "crypto"]:
    """
    Detect the appropriate data source based on symbol format.

    Args:
        symbol: Trading symbol to analyse

    Returns:
        'stock' for Yahoo Finance symbols (equities, indices, forex, futures)
        or 'crypto' for Binance trading pairs

    Examples:
        >>> detect_source_type("AAPL")
        'stock'
        >>> detect_source_type("BTCUSDT")
        'crypto'
        >>> detect_source_type("BRL=X")
        'stock'  # Yahoo Finance handles forex pairs
    """
    symbol_upper = symbol.upper().strip()

    # Stablecoin/fiat quote suffixes (exact matches)
    for suffix in _STABLECOIN_SUFFIXES:
        if symbol_upper.endswith(suffix):
            return "crypto"

    # Fiat USD pairs (e.g., BTCUSD) only when the base is a known crypto ticker
    if symbol_upper.endswith("USD") and not symbol_upper.endswith(_STABLECOIN_SUFFIXES):
        base = symbol_upper[:-3]
        if base in _CRYPTO_TICKERS:
            return "crypto"

    # Crypto-to-crypto pairs (e.g., ETHBTC, BNBBTC) using deterministic splits
    for base in _CRYPTO_TICKERS:
        if symbol_upper.startswith(base):
            quote = symbol_upper[len(base):]
            if quote in _CRYPTO_TICKERS:
                return "crypto"

    return "stock"

def validate_stock_symbol(symbol: str) -> bool:
    """
    Validate stock ticker format.

    Supports:
        - Stocks: AAPL, BRK.B, TSM
        - Indices: ^GSPC, ^DJI
        - Forex: EURUSD=X, BRL=X (Yahoo Finance)
        - Futures: GC=F, ES=F (Yahoo Finance)
    """
    sanitized = symbol.upper().strip()
    pattern = r"^[\^]?[A-Z0-9\.\-]{1,10}(=[XF])?$"
    return bool(re.fullmatch(pattern, sanitized))

def validate_crypto_symbol(symbol: str) -> bool:
    pattern = r"^[A-Z]{2,10}(USDT|BUSD|BTC|ETH|BNB|USD)$"
    return bool(re.match(pattern, symbol.upper()))

class RateLimiter:
    """Simple sliding-window rate limiter."""

    def __init__(self, max_calls: int, period: float):
        self.max_calls = max_calls
        self.period = period
        self.calls = deque()
        self.lock = Lock()

    def __call__(self, func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            with self.lock:
                now = time.time()
                while self.calls and self.calls[0] < now - self.period:
                    self.calls.popleft()
                if len(self.calls) >= self.max_calls:
                    sleep_for = self.period - (now - self.calls[0])
                    if sleep_for > 0:
                        logger.debug("Rate limit reached, sleeping for %.2fs", sleep_for)
                        time.sleep(sleep_for)
                        now = time.time()
                        while self.calls and self.calls[0] < now - self.period:
                            self.calls.popleft()
                self.calls.append(time.time())

            return func(*args, **kwargs)

        return wrapper

    def reset(self):
        with self.lock:
            self.calls.clear()


def retry_on_failure(max_retries: int = 3, backoff: float = 2.0, exceptions: tuple = (Exception,)):
    """Retry decorator with exponential backoff."""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as exc:  
                    last_exception = exc
                    if attempt < max_retries:
                        sleep_time = backoff ** attempt
                        logger.warning(
                            "%s failed (attempt %d/%d): %s. Retrying in %.1fs...",
                            func.__name__,
                            attempt + 1,
                            max_retries + 1,
                            exc,
                            sleep_time,
                        )
                        time.sleep(sleep_time)
            raise last_exception

        return wrapper

    return decorator


def yfinance_rate_limit(func: Callable):
    limiter = RateLimiter(max_calls=30, period=60)
    return limiter(func)


def binance_rate_limit(weight: int = 1):
    limiter = RateLimiter(max_calls=1200, period=60)

    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for _ in range(weight):
                limiter.calls.append(0)  
            return func(*args, **kwargs)

        return wrapper

    return decorator
