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
    hurst_window: int = 100
    volatility_baseline_window: int = 100
    trend_threshold_up: float = 0.2
    trend_threshold_down: float = -0.2
    volatility_percentile: float = 0.67
    hurst_threshold: float = 0.55
    hurst_indeterminate_range: Tuple[float, float] = (0.45, 0.55)
    default_period: str = "2y"


@dataclass
class CryptoConfig:
    trend_window: int = 14
    volatility_window: int = 14
    hurst_window: int = 75
    volatility_baseline_window: int = 75
    trend_threshold_up: float = 0.2
    trend_threshold_down: float = -0.2
    volatility_percentile: float = 0.75
    hurst_threshold: float = 0.55
    hurst_indeterminate_range: Tuple[float, float] = (0.45, 0.55)
    default_period: str = "1y"


@dataclass
class JMSConfig:
    stock: Optional[StockConfig] = None
    crypto: Optional[CryptoConfig] = None

    trend_window: int = 20
    volatility_window: int = 20
    hurst_window: int = 100
    volatility_baseline_window: int = 100
    trend_threshold_up: float = 0.2
    trend_threshold_down: float = -0.2
    volatility_percentile: float = 0.67
    hurst_threshold: float = 0.55
    hurst_indeterminate_range: Tuple[float, float] = (0.45, 0.55)

    use_fuzzy: bool = True
    fuzzy_trend_range: Tuple[float, float] = (-3.0, 3.0)
    fuzzy_vol_range: Tuple[float, float] = (0.0, 0.5)

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
            "hurst_window": cfg.hurst_window,
            "volatility_baseline_window": cfg.volatility_baseline_window,
            "trend_threshold_up": cfg.trend_threshold_up,
            "trend_threshold_down": cfg.trend_threshold_down,
            "volatility_percentile": cfg.volatility_percentile,
            "hurst_threshold": cfg.hurst_threshold,
            "hurst_indeterminate_range": cfg.hurst_indeterminate_range,
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
            "hurst_window": self.hurst_window,
            "volatility_baseline_window": self.volatility_baseline_window,
            "trend_threshold_up": self.trend_threshold_up,
            "trend_threshold_down": self.trend_threshold_down,
            "volatility_percentile": self.volatility_percentile,
            "hurst_threshold": self.hurst_threshold,
            "hurst_indeterminate_range": self.hurst_indeterminate_range,
            "use_fuzzy": self.use_fuzzy,
            "fuzzy_trend_range": self.fuzzy_trend_range,
            "fuzzy_vol_range": self.fuzzy_vol_range,
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
        if self.fuzzy_trend_range[0] >= self.fuzzy_trend_range[1]:
            raise ValueError("Invalid fuzzy_trend_range")
        if self.fuzzy_vol_range[0] >= self.fuzzy_vol_range[1]:
            raise ValueError("Invalid fuzzy_vol_range")
        return True


DEFAULT_CONFIG = JMSConfig()
