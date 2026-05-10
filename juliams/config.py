"""
Configuration management for the Julia Mandelbrot System.
All key parameters are configurable here for easy tuning and experimentation.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple


@dataclass
class StockConfig:
    trend_window: int = 20
    volatility_window: int = 20
    volatility_method: str = "std"
    volatility_annualize: bool = False
    hurst_window: int = 100
    volatility_baseline_window: int = 100
    trend_threshold_up: float = 0.2
    trend_threshold_down: float = -0.2
    volatility_percentile: float = 0.67
    hurst_threshold: float = 0.55
    hurst_indeterminate_range: Tuple[float, float] = (0.45, 0.55)
    hurst_confidence_level: float = 0.95
    lo_modified_rs_lags: Optional[int] = None
    tail_risk_window: int = 252
    tail_var_alpha: float = 0.95
    tail_cvar_alpha: float = 0.95
    hill_tail_fraction: float = 0.10
    hill_min_tail_observations: int = 5
    fragile_cvar: float = 0.04
    stressed_cvar: float = 0.02
    fragile_tail_index: float = 3.0
    stressed_tail_index: float = 4.0
    fragile_drawdown: float = -0.15
    stressed_drawdown: float = -0.07
    default_period: str = "2y"


@dataclass
class CryptoConfig:
    trend_window: int = 14
    volatility_window: int = 14
    volatility_method: str = "std"
    volatility_annualize: bool = False
    hurst_window: int = 75
    volatility_baseline_window: int = 75
    trend_threshold_up: float = 0.2
    trend_threshold_down: float = -0.2
    volatility_percentile: float = 0.75
    hurst_threshold: float = 0.55
    hurst_indeterminate_range: Tuple[float, float] = (0.45, 0.55)
    hurst_confidence_level: float = 0.95
    lo_modified_rs_lags: Optional[int] = None
    tail_risk_window: int = 120
    tail_var_alpha: float = 0.95
    tail_cvar_alpha: float = 0.95
    hill_tail_fraction: float = 0.10
    hill_min_tail_observations: int = 5
    fragile_cvar: float = 0.07
    stressed_cvar: float = 0.04
    fragile_tail_index: float = 2.5
    stressed_tail_index: float = 3.5
    fragile_drawdown: float = -0.25
    stressed_drawdown: float = -0.12
    default_period: str = "1y"


@dataclass
class JMSConfig:
    stock: Optional[StockConfig] = None
    crypto: Optional[CryptoConfig] = None

    trend_window: int = 20
    volatility_window: int = 20
    volatility_method: str = "std"
    volatility_annualize: bool = False
    hurst_window: int = 100
    volatility_baseline_window: int = 100
    trend_threshold_up: float = 0.2
    trend_threshold_down: float = -0.2
    volatility_percentile: float = 0.67
    hurst_threshold: float = 0.55
    hurst_indeterminate_range: Tuple[float, float] = (0.45, 0.55)
    hurst_confidence_level: float = 0.95
    lo_modified_rs_lags: Optional[int] = None
    tail_risk_window: int = 252
    tail_var_alpha: float = 0.95
    tail_cvar_alpha: float = 0.95
    hill_tail_fraction: float = 0.10
    hill_min_tail_observations: int = 5
    fragile_cvar: float = 0.04
    stressed_cvar: float = 0.02
    fragile_tail_index: float = 3.0
    stressed_tail_index: float = 4.0
    fragile_drawdown: float = -0.15
    stressed_drawdown: float = -0.07

    use_fuzzy: bool = True
    fuzzy_trend_range: Tuple[float, float] = (-3.0, 3.0)
    fuzzy_vol_range: Tuple[float, float] = (0.0, 0.5)

    # Adaptive (data-driven) regime overlays. All default to off so that
    # existing callers see identical behaviour.
    adaptive_thresholds: bool = False
    adaptive_threshold_window: int = 252
    adaptive_q_up: float = 0.70
    adaptive_q_down: float = 0.30
    adaptive_floor: float = 0.10
    ewma_halflife: Optional[float] = None
    markov_overlay: bool = False
    bocpd_overlay: bool = False
    bocpd_expected_run_length: float = 100.0
    min_dwell_days: int = 1

    forward_return_horizons: list = field(default_factory=lambda: [5, 10])

    regime_colors: Dict[str, str] = field(default_factory=lambda: {
        "Up-LowVol": "#2E7D32",
        "Up-HighVol": "#66BB6A",
        "Sideways-LowVol": "#FFA726",
        "Sideways-HighVol": "#FF9800",
        "Down-LowVol": "#EF5350",
        "Down-HighVol": "#C62828",
    })

    regime_names: Dict[str, str] = field(default_factory=lambda: {
        "Up-LowVol": "Bull Quiet",
        "Up-HighVol": "Bull Volatile",
        "Sideways-LowVol": "Sideways Quiet",
        "Sideways-HighVol": "Sideways Volatile",
        "Down-LowVol": "Bear Quiet",
        "Down-HighVol": "Bear Volatile",
    })

    sma_periods: list = field(default_factory=lambda: [50, 200])
    default_period: str = "2y"
    fill_missing_data: str = "ffill"
    float_precision: int = 4
    date_format: str = "%Y-%m-%d"
    use_parallel: bool = False
    cache_data: bool = True
    timezone: str = "UTC"

    def __post_init__(self):
        if self.stock is None:
            self.stock = StockConfig()
        if self.crypto is None:
            self.crypto = CryptoConfig()

    def get_source_config(self, source_type: str) -> Dict:
        if source_type == "stock":
            cfg = self.stock
        elif source_type == "crypto":
            cfg = self.crypto
        else:
            return self.to_dict()

        source_dict = {
            "trend_window": cfg.trend_window,
            "volatility_window": cfg.volatility_window,
            "volatility_method": cfg.volatility_method,
            "volatility_annualize": cfg.volatility_annualize,
            "hurst_window": cfg.hurst_window,
            "volatility_baseline_window": cfg.volatility_baseline_window,
            "trend_threshold_up": cfg.trend_threshold_up,
            "trend_threshold_down": cfg.trend_threshold_down,
            "volatility_percentile": cfg.volatility_percentile,
            "hurst_threshold": cfg.hurst_threshold,
            "hurst_indeterminate_range": cfg.hurst_indeterminate_range,
            "hurst_confidence_level": cfg.hurst_confidence_level,
            "lo_modified_rs_lags": cfg.lo_modified_rs_lags,
            "tail_risk_window": cfg.tail_risk_window,
            "tail_var_alpha": cfg.tail_var_alpha,
            "tail_cvar_alpha": cfg.tail_cvar_alpha,
            "hill_tail_fraction": cfg.hill_tail_fraction,
            "hill_min_tail_observations": cfg.hill_min_tail_observations,
            "fragile_cvar": cfg.fragile_cvar,
            "stressed_cvar": cfg.stressed_cvar,
            "fragile_tail_index": cfg.fragile_tail_index,
            "stressed_tail_index": cfg.stressed_tail_index,
            "fragile_drawdown": cfg.fragile_drawdown,
            "stressed_drawdown": cfg.stressed_drawdown,
            "default_period": cfg.default_period,
        }
        source_dict.update({
            "use_fuzzy": self.use_fuzzy,
            "fuzzy_trend_range": self.fuzzy_trend_range,
            "fuzzy_vol_range": self.fuzzy_vol_range,
            "forward_return_horizons": self.forward_return_horizons,
            "regime_colors": self.regime_colors,
            "regime_names": self.regime_names,
            "sma_periods": self.sma_periods,
            "fill_missing_data": self.fill_missing_data,
            "float_precision": self.float_precision,
            "date_format": self.date_format,
            "use_parallel": self.use_parallel,
            "cache_data": self.cache_data,
            "timezone": self.timezone,
        })
        return source_dict

    def to_dict(self) -> dict:
        return {
            "trend_window": self.trend_window,
            "volatility_window": self.volatility_window,
            "volatility_method": self.volatility_method,
            "volatility_annualize": self.volatility_annualize,
            "hurst_window": self.hurst_window,
            "volatility_baseline_window": self.volatility_baseline_window,
            "trend_threshold_up": self.trend_threshold_up,
            "trend_threshold_down": self.trend_threshold_down,
            "volatility_percentile": self.volatility_percentile,
            "hurst_threshold": self.hurst_threshold,
            "hurst_indeterminate_range": self.hurst_indeterminate_range,
            "hurst_confidence_level": self.hurst_confidence_level,
            "lo_modified_rs_lags": self.lo_modified_rs_lags,
            "tail_risk_window": self.tail_risk_window,
            "tail_var_alpha": self.tail_var_alpha,
            "tail_cvar_alpha": self.tail_cvar_alpha,
            "hill_tail_fraction": self.hill_tail_fraction,
            "hill_min_tail_observations": self.hill_min_tail_observations,
            "fragile_cvar": self.fragile_cvar,
            "stressed_cvar": self.stressed_cvar,
            "fragile_tail_index": self.fragile_tail_index,
            "stressed_tail_index": self.stressed_tail_index,
            "fragile_drawdown": self.fragile_drawdown,
            "stressed_drawdown": self.stressed_drawdown,
            "use_fuzzy": self.use_fuzzy,
            "fuzzy_trend_range": self.fuzzy_trend_range,
            "fuzzy_vol_range": self.fuzzy_vol_range,
            "adaptive_thresholds": self.adaptive_thresholds,
            "adaptive_threshold_window": self.adaptive_threshold_window,
            "adaptive_q_up": self.adaptive_q_up,
            "adaptive_q_down": self.adaptive_q_down,
            "adaptive_floor": self.adaptive_floor,
            "ewma_halflife": self.ewma_halflife,
            "markov_overlay": self.markov_overlay,
            "bocpd_overlay": self.bocpd_overlay,
            "bocpd_expected_run_length": self.bocpd_expected_run_length,
            "min_dwell_days": self.min_dwell_days,
            "forward_return_horizons": self.forward_return_horizons,
            "regime_colors": self.regime_colors,
            "regime_names": self.regime_names,
            "sma_periods": self.sma_periods,
            "default_period": self.default_period,
            "fill_missing_data": self.fill_missing_data,
            "float_precision": self.float_precision,
            "date_format": self.date_format,
            "use_parallel": self.use_parallel,
            "cache_data": self.cache_data,
            "timezone": self.timezone,
        }

    @classmethod
    def from_dict(cls, config_dict: dict) -> "JMSConfig":
        return cls(**config_dict)

    def validate(self) -> bool:
        def positive(value: int, name: str):
            if value <= 0:
                raise ValueError(f"{name} must be positive")

        positive(self.trend_window, "trend_window")
        positive(self.volatility_window, "volatility_window")
        positive(self.hurst_window, "hurst_window")

        if self.trend_threshold_up <= 0:
            raise ValueError("trend_threshold_up must be positive")
        if self.trend_threshold_down >= 0:
            raise ValueError("trend_threshold_down must be negative")
        if not 0 < self.volatility_percentile < 1:
            raise ValueError("volatility_percentile must be between 0 and 1")
        if not 0 < self.hurst_threshold < 1:
            raise ValueError("hurst_threshold must be between 0 and 1")
        if self.volatility_method not in {"std", "ewm", "parkinson", "realized"}:
            raise ValueError("volatility_method must be one of: std, ewm, parkinson, realized")
        if self.hurst_confidence_level not in {0.90, 0.95, 0.99}:
            raise ValueError("hurst_confidence_level must be one of: 0.90, 0.95, 0.99")
        if self.lo_modified_rs_lags is not None and self.lo_modified_rs_lags < 0:
            raise ValueError("lo_modified_rs_lags must be non-negative")
        if self.tail_risk_window <= 0:
            raise ValueError("tail_risk_window must be positive")
        if not 0 < self.tail_var_alpha < 1:
            raise ValueError("tail_var_alpha must be between 0 and 1")
        if not 0 < self.tail_cvar_alpha < 1:
            raise ValueError("tail_cvar_alpha must be between 0 and 1")
        if not 0 < self.hill_tail_fraction < 1:
            raise ValueError("hill_tail_fraction must be between 0 and 1")
        if self.hill_min_tail_observations <= 1:
            raise ValueError("hill_min_tail_observations must be greater than 1")
        if self.fragile_cvar <= 0 or self.stressed_cvar <= 0:
            raise ValueError("CVaR thresholds must be positive")
        if self.fragile_tail_index <= 0 or self.stressed_tail_index <= 0:
            raise ValueError("tail-index thresholds must be positive")
        if self.fragile_drawdown >= 0 or self.stressed_drawdown >= 0:
            raise ValueError("drawdown thresholds must be negative")
        if self.fuzzy_trend_range[0] >= self.fuzzy_trend_range[1]:
            raise ValueError("Invalid fuzzy_trend_range")
        if self.fuzzy_vol_range[0] >= self.fuzzy_vol_range[1]:
            raise ValueError("Invalid fuzzy_vol_range")
        return True


DEFAULT_CONFIG = JMSConfig()
