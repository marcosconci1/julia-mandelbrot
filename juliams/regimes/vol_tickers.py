"""
Symbol to implied-volatility-index mapping for the multivariate Markov
overlay.

The multivariate HMM in :func:`juliams.regimes.markov.fit_multivariate_markov_regime`
fuses returns with an implied-vol channel. The right vol channel
depends on the underlying asset:

- Gold (GC=F, GLD, IAU, XAUUSD style symbols) is best paired with
  CBOE GVZ (^GVZ).
- US equity indices and broad ETFs (SPY, IVV, VOO, ^GSPC, QQQ, IWM)
  pair with CBOE VIX (^VIX).
- Crude oil (CL=F, USO) pairs with CBOE OVX (^OVX).
- Most FX, single-name equities, and crypto have no liquid daily
  implied-vol index and the function returns None, signalling to the
  caller that the univariate path should be used.

The mapping below is intentionally conservative: only well-known
liquid IV indices with a long history. Adding more entries is cheap
but the auto-fetch can still fail (network, ticker change, rate
limit), and the consumer must handle that gracefully.
"""

from __future__ import annotations

from typing import Optional


VOL_INDEX_BY_PATTERN: dict[str, str] = {
    # Gold
    "GC=F": "^GVZ",
    "GLD": "^GVZ",
    "IAU": "^GVZ",
    "XAUUSD": "^GVZ",
    "XAUUSD=X": "^GVZ",
    # US equity broad indices and ETFs
    "SPY": "^VIX",
    "IVV": "^VIX",
    "VOO": "^VIX",
    "^GSPC": "^VIX",
    "ES=F": "^VIX",
    "QQQ": "^VXN",
    "^IXIC": "^VXN",
    "NQ=F": "^VXN",
    "IWM": "^RVX",
    "^RUT": "^RVX",
    "DIA": "^VXD",
    "^DJI": "^VXD",
    # Crude oil
    "CL=F": "^OVX",
    "USO": "^OVX",
    "BZ=F": "^OVX",
}


def auto_vol_channel(symbol: str) -> Optional[str]:
    """Return the canonical implied-vol ticker for the given symbol, or
    None when the symbol has no well-known liquid IV index.

    Matching is exact and case-insensitive against keys in
    :data:`VOL_INDEX_BY_PATTERN`. We deliberately do not do fuzzy or
    substring matching: a single-name equity like AAPL has no IV
    *index* even though it has an options market.
    """
    if not symbol:
        return None
    key = symbol.strip().upper()
    return VOL_INDEX_BY_PATTERN.get(key)
