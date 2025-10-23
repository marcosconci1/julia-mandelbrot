"""
Utility helpers for data sources.
"""

from __future__ import annotations

import re
import time
from collections import deque
from functools import wraps
from threading import Lock
from typing import Callable, Literal, Optional
import logging

logger = logging.getLogger(__name__)

def detect_source_type(symbol: str) -> Literal["stock", "crypto"]:
    """
    Auto-detect data source type from symbol format.

    Args:
        symbol: Symbol/ticker to analyse

    Returns:
        'crypto' if matches crypto patterns, 'stock' otherwise
    """
    crypto_patterns = [
        r"^[A-Z]{2,10}USDT$",  
        r"^[A-Z]{2,10}BUSD$", 
        r"^[A-Z]{2,10}BTC$",   
        r"^[A-Z]{2,10}ETH$",   
        r"^[A-Z]{2,10}BNB$",   
        r"^[A-Z]{2,10}USD$",   
    ]
    symbol_upper = symbol.upper()
    for pattern in crypto_patterns:
        if re.match(pattern, symbol_upper):
            return "crypto"
    return "stock"

# Accept Yahoo Finance FX tickers like USDBRL=X, EURUSD=X
_YF_FX_PATTERN = re.compile(r"^[A-Z]{3}[A-Z]{3}=X$")

def validate_stock_symbol(symbol: str) -> bool:
    s = symbol.upper().strip()
    # Allow Yahoo FX pairs
    if _YF_FX_PATTERN.fullmatch(s):
        return True
    # Allow standard stock symbols
    pattern = r"^[A-Z]{1,5}([.-][A-Z0-9]{1,3})?$"
    return bool(re.match(pattern, s))

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
