"""
Per-asset-class defaults for the BOCPD geometric run-length prior.

The expected run length λ controls how strongly BOCPD biases toward
long regimes (large λ) versus short regimes (small λ). A single value
across asset classes leads to over-smoothing on volatile commodities
and crypto and under-smoothing on slow-moving FX.

The defaults below come from practitioner consensus reported in the
Cambridge tradition (Turner 2009 onwards) and from recent applied
write-ups: equity indices typically tune to 250-500 daily observations,
FX to 100-200, commodities to 75-150, crypto to 50-100. We pick
mid-range values; users should still tune on a held-out window when
serious money is at stake.
"""

from __future__ import annotations


DEFAULT_BOCPD_RUN_LENGTH_BY_ASSET: dict[str, float] = {
    "equity": 250.0,
    "fx": 150.0,
    "commodity": 100.0,
    "crypto": 75.0,
}


def default_bocpd_expected_run_length(asset_class: str) -> float:
    """Look up a sensible expected_run_length for a given asset class."""
    key = asset_class.lower().strip()
    if key not in DEFAULT_BOCPD_RUN_LENGTH_BY_ASSET:
        raise ValueError(
            f"Unknown asset_class={asset_class!r}; "
            f"choose from {sorted(DEFAULT_BOCPD_RUN_LENGTH_BY_ASSET)}"
        )
    return DEFAULT_BOCPD_RUN_LENGTH_BY_ASSET[key]
