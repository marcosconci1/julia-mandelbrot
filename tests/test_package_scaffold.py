"""Smoke tests for the package scaffold."""

import subprocess
import sys

from julia_mandelbrot import IndicatorConfig, __version__


def test_package_imports() -> None:
    assert __version__ == "0.1.0"
    assert IndicatorConfig


def test_configuration_defaults_match_specification() -> None:
    config = IndicatorConfig()

    assert config.trend_window == 63
    assert config.volatility_window == 63
    assert config.percentile_window == 252
    assert config.trend_lower_quantile == 0.30
    assert config.trend_upper_quantile == 0.70
    assert config.volatility_quantile == 0.67
    assert config.membership_trend_lower_quantile == 0.30
    assert config.membership_trend_upper_quantile == 0.70
    assert config.membership_volatility_quantile == 0.67
    assert config.transition_window == 756
    assert config.transition_count_floor == 30
    assert (config.variogram_min_lag, config.variogram_max_lag) == (1, 10)
    assert config.roughvol_realized_volatility_window == 20
    assert config.roughvol_window == 252
    assert config.tail_window == 252
    assert config.tail_quantile == 0.95


def test_module_cli_prints_version_and_help() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "julia_mandelbrot"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert f"julia-mandelbrot {__version__}" in result.stdout
    assert "usage:" in result.stdout
