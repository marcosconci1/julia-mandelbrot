"""
Indicator calibration profiles.

Profiles set feature windows and calibration inputs for a market/timeframe.
They do not contain trade rules.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping


@dataclass(frozen=True)
class IndicatorProfile:
    name: str
    asset_class: str
    timeframe: str
    trend_window: int
    volatility_window: int
    volatility_percentile_lookback: int
    volatility_baseline_window: int
    hurst_window: int
    tail_risk_window: int
    fuzzy_volatility_source: str = "volatility_percentile"

    def to_overrides(self) -> Dict[str, object]:
        return {
            "timeframe": self.timeframe,
            "trend_window": self.trend_window,
            "volatility_window": self.volatility_window,
            "volatility_percentile_lookback": self.volatility_percentile_lookback,
            "volatility_baseline_window": self.volatility_baseline_window,
            "hurst_window": self.hurst_window,
            "tail_risk_window": self.tail_risk_window,
            "fuzzy_volatility_source": self.fuzzy_volatility_source,
        }


INDICATOR_PROFILES: Dict[str, IndicatorProfile] = {
    "equity_index_h1": IndicatorProfile(
        name="equity_index_h1",
        asset_class="equity_index",
        timeframe="1h",
        trend_window=48,
        volatility_window=48,
        volatility_percentile_lookback=500,
        volatility_baseline_window=250,
        hurst_window=240,
        tail_risk_window=500,
    ),
    "equity_index_h4": IndicatorProfile(
        name="equity_index_h4",
        asset_class="equity_index",
        timeframe="4h",
        trend_window=30,
        volatility_window=30,
        volatility_percentile_lookback=250,
        volatility_baseline_window=125,
        hurst_window=180,
        tail_risk_window=250,
    ),
    "gold_h4": IndicatorProfile(
        name="gold_h4",
        asset_class="gold",
        timeframe="4h",
        trend_window=30,
        volatility_window=30,
        volatility_percentile_lookback=250,
        volatility_baseline_window=125,
        hurst_window=180,
        tail_risk_window=250,
    ),
    "crypto_h4": IndicatorProfile(
        name="crypto_h4",
        asset_class="crypto",
        timeframe="4h",
        trend_window=36,
        volatility_window=36,
        volatility_percentile_lookback=360,
        volatility_baseline_window=180,
        hurst_window=240,
        tail_risk_window=360,
    ),
    "fx_daily": IndicatorProfile(
        name="fx_daily",
        asset_class="fx",
        timeframe="1d",
        trend_window=20,
        volatility_window=20,
        volatility_percentile_lookback=252,
        volatility_baseline_window=100,
        hurst_window=100,
        tail_risk_window=252,
    ),
}


def available_indicator_profiles() -> Iterable[str]:
    return tuple(sorted(INDICATOR_PROFILES))


def get_indicator_profile(name: str) -> IndicatorProfile:
    try:
        return INDICATOR_PROFILES[name]
    except KeyError as exc:
        available = ", ".join(available_indicator_profiles())
        raise ValueError(f"Unknown indicator profile '{name}'. Available profiles: {available}") from exc


def apply_indicator_profile(config: object, profile: str | IndicatorProfile) -> object:
    selected = get_indicator_profile(profile) if isinstance(profile, str) else profile
    overrides = selected.to_overrides()
    _apply_overrides(config, overrides)

    for source_name in ("stock", "crypto"):
        source_cfg = getattr(config, source_name, None)
        if source_cfg is not None:
            _apply_overrides(source_cfg, overrides)

    return config


def _apply_overrides(target: object, overrides: Mapping[str, object]) -> None:
    for key, value in overrides.items():
        if hasattr(target, key):
            setattr(target, key, value)
