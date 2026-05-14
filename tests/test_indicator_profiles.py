from juliams import JMSConfig
from juliams.profiles import (
    apply_indicator_profile,
    available_indicator_profiles,
    get_indicator_profile,
)


def test_known_profiles_are_available():
    assert "equity_index_h4" in available_indicator_profiles()
    assert "fx_daily" in available_indicator_profiles()


def test_profile_applies_indicator_overrides_to_config_and_sources():
    cfg = JMSConfig()

    apply_indicator_profile(cfg, "equity_index_h4")

    assert cfg.timeframe == "4h"
    assert cfg.trend_window == 30
    assert cfg.volatility_percentile_lookback == 250
    assert cfg.fuzzy_volatility_source == "volatility_percentile"
    assert cfg.stock.timeframe == "4h"
    assert cfg.stock.volatility_percentile_lookback == 250
    assert cfg.crypto.timeframe == "4h"


def test_profile_to_overrides_contains_no_strategy_fields():
    profile = get_indicator_profile("gold_h4")

    overrides = profile.to_overrides()

    forbidden = {"entry_rule", "exit_rule", "horizon", "position", "exposure"}
    assert not (forbidden & set(overrides))
