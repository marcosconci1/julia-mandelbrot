"""
Configuration management for the Julia Mandelbrot System.
All key parameters are configurable here for easy tuning and experimentation.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple


@dataclass
class JMSConfig:
    """Configuration class for Julia Mandelbrot System parameters."""
    
    # Window sizes for rolling calculations
    trend_window: int = 20  # Days for rolling OLS trend
    volatility_window: int = 20  # Days for rolling volatility
    hurst_window: int = 100  # Days for Hurst exponent
    volatility_baseline_window: int = 100  # Days for adaptive volatility baseline
    
    # Threshold parameters
    trend_threshold_up: float = 0.2  # Threshold for significant uptrend
    trend_threshold_down: float = -0.2  # Threshold for significant downtrend
    volatility_percentile: float = 0.67  # Percentile for high/low volatility split
    hurst_threshold: float = 0.55  # Threshold for fractal memory filter
    hurst_indeterminate_range: Tuple[float, float] = (0.45, 0.55)  # Random walk range
    
    # Fuzzy logic parameters
    use_fuzzy: bool = True  # Enable fuzzy logic classification
    fuzzy_trend_range: Tuple[float, float] = (-3.0, 3.0)  # Range for trend strength
    fuzzy_vol_range: Tuple[float, float] = (0.0, 0.5)  # Range for volatility
    
    # Forward return horizons (in days)
    forward_return_horizons: list = field(default_factory=lambda: [5, 10])
    
    # Visualization settings
    regime_colors: Dict[str, str] = field(default_factory=lambda: {
        "Up-LowVol": "#2E7D32",  # Dark green - Bull Quiet
        "Up-HighVol": "#66BB6A",  # Light green - Bull Volatile
        "Sideways-LowVol": "#FFA726",  # Orange - Sideways Quiet
        "Sideways-HighVol": "#FF9800",  # Dark orange - Sideways Volatile
        "Down-LowVol": "#EF5350",  # Light red - Bear Quiet
        "Down-HighVol": "#C62828",  # Dark red - Bear Volatile
    })
    
    regime_names: Dict[str, str] = field(default_factory=lambda: {
        "Up-LowVol": "Bull Quiet",
        "Up-HighVol": "Bull Volatile",
        "Sideways-LowVol": "Sideways Quiet",
        "Sideways-HighVol": "Sideways Volatile",
        "Down-LowVol": "Bear Quiet",
        "Down-HighVol": "Bear Volatile",
    })
    
    # Technical indicators
    sma_periods: list = field(default_factory=lambda: [50, 200])  # SMA periods for overlays
    
    # Data settings
    default_period: str = "2y"  # Default data period if no dates specified
    fill_missing_data: str = "ffill"  # Method to handle missing data: 'ffill', 'drop', 'interpolate'
    
    # Export settings
    float_precision: int = 4  # Decimal places for float exports
    date_format: str = "%Y-%m-%d"  # Date format for exports
    
    # Performance settings
    use_parallel: bool = False  # Enable parallel processing for multiple tickers
    cache_data: bool = True  # Cache fetched data to avoid repeated API calls
    
    def to_dict(self) -> dict:
        """Convert configuration to dictionary."""
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
        }
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> "JMSConfig":
        """Create configuration from dictionary."""
        return cls(**config_dict)
    
    def validate(self) -> bool:
        """Validate configuration parameters."""
        # Check window sizes
        if self.trend_window <= 0 or self.volatility_window <= 0 or self.hurst_window <= 0:
            raise ValueError("Window sizes must be positive integers")
        
        # Check thresholds
        if self.trend_threshold_up <= 0:
            raise ValueError("Uptrend threshold must be positive")
        if self.trend_threshold_down >= 0:
            raise ValueError("Downtrend threshold must be negative")
        
        # Check percentiles
        if not 0 < self.volatility_percentile < 1:
            raise ValueError("Volatility percentile must be between 0 and 1")
        
        # Check Hurst range
        if not 0 < self.hurst_threshold < 1:
            raise ValueError("Hurst threshold must be between 0 and 1")
        
        # Check fuzzy ranges
        if self.fuzzy_trend_range[0] >= self.fuzzy_trend_range[1]:
            raise ValueError("Invalid fuzzy trend range")
        if self.fuzzy_vol_range[0] >= self.fuzzy_vol_range[1]:
            raise ValueError("Invalid fuzzy volatility range")
        
        return True


# Default configuration instance
DEFAULT_CONFIG = JMSConfig()
