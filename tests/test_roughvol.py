"""Offline synthetic tests for realized log-volatility roughness by variogram (spec section 6.4)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import julia_mandelbrot.roughvol as roughvol_mod
from julia_mandelbrot.config import IndicatorConfig
from julia_mandelbrot.data_adapters import PriceSeries
from julia_mandelbrot.roughvol import (
    realized_log_vol,
    rolling_roughness,
    roughness_frame,
    variogram_hurst,
)
from julia_mandelbrot.trend_vol import realized_volatility


def _series(values, name: str = "logvol") -> pd.Series:
    index = pd.date_range("2020-01-01", periods=len(values), freq="B", tz="UTC")
    return pd.Series(values, index=index, name=name, dtype=float)


def _prices(length: int, seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    # Time-varying volatility so realized_log_vol varies smoothly over time.
    t = np.arange(length, dtype=float)
    sigma_t = 0.01 * np.exp(0.4 * np.sin(2.0 * np.pi * t / 60.0) + 0.1 * rng.standard_normal(length))
    log_returns = rng.normal(0.0003, sigma_t)
    close = 100.0 * np.exp(np.cumsum(log_returns))
    index = pd.date_range("2020-01-01", periods=length, freq="B", tz="UTC")
    return pd.Series(close, index=index, name="Close", dtype=float)


def _simulate_fbm_hosking(n: int, hurst: float, seed: int = 12345) -> np.ndarray:
    """Simulate an exact fractional Brownian motion path of length ``n`` via the Hosking (1984) recursion.

    Fractional Gaussian noise (fGn) increments ``X_i = B_H(i+1) - B_H(i)`` have
    autocovariance ``gamma(k) = 0.5 * (|k+1|^{2H} - 2|k|^{2H} + |k-1|^{2H})``.
    Hosking's method applies the Levinson-Durbin recursion to draw each ``X_m``
    from its exact conditional Gaussian law given ``(X_0, ..., X_{m-1})`` using
    only NumPy, and returns ``B_H(t) = sum_{i=0}^{t-1} X_i`` whose theoretical
    semi-variogram satisfies ``0.5 * E[(B_H(t+k) - B_H(t))^2] = 0.5 * k^{2H}``.
    """
    rng = np.random.default_rng(seed)
    k = np.arange(n, dtype=float)
    cov = 0.5 * (
        np.abs(k + 1.0) ** (2.0 * hurst)
        - 2.0 * np.abs(k) ** (2.0 * hurst)
        + np.abs(k - 1.0) ** (2.0 * hurst)
    )

    fgn = np.empty(n, dtype=float)
    fgn[0] = rng.standard_normal()

    phi = np.zeros(n, dtype=float)
    cond_var = 1.0

    for m in range(1, n):
        # Reflection coefficient kappa_m from Toeplitz Levinson-Durbin recursion.
        prev_phi = phi[: m - 1]
        kappa = (cov[m] - float(np.dot(prev_phi, cov[m - 1 : 0 : -1]))) / cond_var
        if m > 1:
            phi[: m - 1] = prev_phi - kappa * prev_phi[::-1]
        phi[m - 1] = kappa
        cond_var *= 1.0 - kappa * kappa

        cond_mean = float(np.dot(phi[:m], fgn[m - 1 :: -1]))
        fgn[m] = cond_mean + np.sqrt(max(cond_var, 0.0)) * rng.standard_normal()

    return np.cumsum(fgn)


@pytest.mark.parametrize("target_h,seed", [(0.20, 101), (0.75, 202)])
def test_known_hurst_recovered_on_synthetic_fbm_via_hosking(target_h: float, seed: int):
    # Simulate fBm via Hosking (1984) without external dependencies.
    # Finite-sample variogram estimation on a single path has sampling noise,
    # so we assert recovery within +/- 0.15 of the true Hurst exponent.
    path = _simulate_fbm_hosking(n=1200, hurst=target_h, seed=seed)
    h_est, r2 = variogram_hurst(path, min_lag=1, max_lag=10)

    assert np.isfinite(h_est)
    assert np.isfinite(r2)
    assert abs(h_est - target_h) < 0.15
    assert r2 > 0.90

    # Rolling roughness over 252-bar windows of the same fBm path also centers near target_h.
    rolling = rolling_roughness(_series(path), window=252, min_lag=1, max_lag=10)
    valid = rolling.dropna()
    assert not valid.empty
    assert abs(float(valid["hurst_logvol"].median()) - target_h) < 0.15
    assert float(valid["hurst_r2"].median()) > 0.85


def test_white_noise_increments_random_walk_gives_h_near_half_and_iid_level_near_zero():
    rng = np.random.default_rng(2026)

    # 1. logvol with white-noise increments (level of a Gaussian random walk):
    # x_t = sum_{s <= t} eps_s with i.i.d. eps_s ~ N(0, sigma^2).
    # Then E[(x_{t+k} - x_t)^2] = k * sigma^2 = sigma^2 * k^{2H} with H = 0.5.
    rw_level = np.cumsum(rng.normal(0.0, 0.05, size=2000))
    h_rw, r2_rw = variogram_hurst(rw_level, min_lag=1, max_lag=10)
    assert abs(h_rw - 0.5) < 0.10
    assert r2_rw > 0.95

    rw_rolling = rolling_roughness(_series(rw_level), window=252, min_lag=1, max_lag=10).dropna()
    assert abs(float(rw_rolling["hurst_logvol"].mean()) - 0.5) < 0.10

    # 2. Stationary i.i.d. white noise in level (x_t = eps_t):
    # E[(x_{t+k} - x_t)^2] = 2 * sigma^2 * k^0 for all k >= 1, so expected H = 0.0.
    iid_level = rng.normal(-3.0, 0.2, size=2000)
    h_iid, _ = variogram_hurst(iid_level, min_lag=1, max_lag=10)
    assert abs(h_iid - 0.0) < 0.10


def test_r2_sanity_clean_power_law_and_pathological_constant_windows():
    # Clean deterministic linear ramp x_t = a * t + b has exact power-law variogram
    # gamma(k) = 0.5 * a^2 * k^2 -> H = 1.0 and R^2 = 1.0 (> 0.9).
    ramp = np.linspace(1.0, 50.0, 252)
    h_ramp, r2_ramp = variogram_hurst(ramp, min_lag=1, max_lag=10)
    assert h_ramp == pytest.approx(1.0, rel=1e-12)
    assert r2_ramp == pytest.approx(1.0, rel=1e-12)
    assert r2_ramp > 0.9

    # Pathological constant window: gamma(k) = 0 for all lags -> log(0) undefined -> (NaN, NaN).
    constant_win = np.full(252, -4.2)
    h_const, r2_const = variogram_hurst(constant_win, min_lag=1, max_lag=10)
    assert np.isnan(h_const)
    assert np.isnan(r2_const)

    # Insufficient data (len <= max_lag) or NaN inside the window -> (NaN, NaN).
    assert all(np.isnan(v) for v in variogram_hurst(np.arange(10, dtype=float), min_lag=1, max_lag=10))
    with_nan = ramp.copy()
    with_nan[10] = np.nan
    assert all(np.isnan(v) for v in variogram_hurst(with_nan, min_lag=1, max_lag=10))

    # In rolling_roughness, a constant window produces NaN for both columns while a ramp window succeeds.
    mixed = _series(np.concatenate([np.full(30, -3.0), np.linspace(-3.0, 2.0, 30)]))
    rolled = rolling_roughness(mixed, window=25, min_lag=1, max_lag=10)
    # Rows 0..23 have insufficient data (< 25 observations) -> NaN.
    assert rolled.iloc[:24].isna().all().all()
    # Rows 24..29 have a full 25-bar constant window -> failed fit -> NaN for both columns.
    assert rolled.iloc[24:30].isna().all().all()
    # By the final row (pure linear ramp window), H == 1.0 and R^2 == 1.0 (> 0.9).
    assert rolled["hurst_logvol"].iloc[-1] == pytest.approx(1.0, rel=1e-12)
    assert rolled["hurst_r2"].iloc[-1] == pytest.approx(1.0, rel=1e-12)


def test_no_lookahead_perturbing_last_value_leaves_prior_rows_unchanged():
    # 1. Test directly on rolling_roughness(logvol).
    rng = np.random.default_rng(77)
    logvol = _series(np.cumsum(rng.normal(0.0, 0.05, size=80)) - 3.5)
    base_rough = rolling_roughness(logvol, window=30, min_lag=1, max_lag=10)

    perturbed_logvol = logvol.copy()
    perturbed_logvol.iloc[-1] = logvol.iloc[-1] + 5.0
    pert_rough = rolling_roughness(perturbed_logvol, window=30, min_lag=1, max_lag=10)

    pd.testing.assert_frame_equal(base_rough.iloc[:-1], pert_rough.iloc[:-1])
    assert base_rough["hurst_logvol"].iloc[-1] != pert_rough["hurst_logvol"].iloc[-1]

    # 2. Test end-to-end on roughness_frame(prices) and realized_log_vol(prices).
    prices = _prices(90, seed=88)
    base_frame = roughness_frame(prices, rv_window=10, window=30, min_lag=1, max_lag=10)

    perturbed_prices = prices.copy()
    perturbed_prices.iloc[-1] = prices.iloc[-1] * 2.5
    pert_frame = roughness_frame(perturbed_prices, rv_window=10, window=30, min_lag=1, max_lag=10)

    pd.testing.assert_frame_equal(base_frame.iloc[:-1], pert_frame.iloc[:-1])
    assert base_frame["realized_log_vol"].iloc[-1] != pert_frame["realized_log_vol"].iloc[-1]
    assert base_frame["hurst_logvol"].iloc[-1] != pert_frame["hurst_logvol"].iloc[-1]

    # 3. Appending a future observation leaves all historical rows identical.
    extended_prices = _prices(91, seed=88)
    extended_prices.iloc[:90] = prices.to_numpy()
    ext_frame = roughness_frame(extended_prices, rv_window=10, window=30, min_lag=1, max_lag=10)
    pd.testing.assert_frame_equal(base_frame, ext_frame.iloc[:-1])


def test_alignment_warm_up_equivalence_and_config_defaults():
    cfg = IndicatorConfig()
    assert cfg.roughvol_realized_volatility_window == 20
    assert cfg.roughvol_window == 252
    assert cfg.variogram_min_lag == 1
    assert cfg.variogram_max_lag == 10

    prices = _prices(310, seed=42)
    frame = roughness_frame(prices)

    # Index alignment and expected columns.
    assert frame.index.equals(prices.index)
    assert list(frame.columns) == ["realized_log_vol", "hurst_logvol", "hurst_r2"]

    # realized_log_vol equals natural log of trend_vol.realized_volatility(prices, 20).
    expected_logvol = np.log(realized_volatility(prices, cfg.roughvol_realized_volatility_window)).rename(
        "realized_log_vol"
    )
    pd.testing.assert_series_equal(frame["realized_log_vol"], expected_logvol)

    # Warm-up boundary: first return is at index 1, so 20-return RV is first valid at index 20.
    # Rolling 252-bar roughness over realized_log_vol is first valid at index 20 + 252 - 1 = 271.
    first_valid_rv = cfg.roughvol_realized_volatility_window
    first_valid_hurst = first_valid_rv + cfg.roughvol_window - 1
    assert frame["realized_log_vol"].iloc[:first_valid_rv].isna().all()
    assert frame["realized_log_vol"].iloc[first_valid_rv:].notna().all()
    assert frame[["hurst_logvol", "hurst_r2"]].iloc[:first_valid_hurst].isna().all().all()
    assert frame[["hurst_logvol", "hurst_r2"]].iloc[first_valid_hurst:].notna().all().all()

    # Vectorized rolling_roughness matches per-window variogram_hurst to machine precision.
    for row_idx in (first_valid_hurst, first_valid_hurst + 10, len(prices) - 1):
        win_slice = frame["realized_log_vol"].iloc[row_idx - cfg.roughvol_window + 1 : row_idx + 1]
        h_expected, r2_expected = variogram_hurst(win_slice)
        assert frame["hurst_logvol"].iloc[row_idx] == pytest.approx(h_expected, rel=1e-12)
        assert frame["hurst_r2"].iloc[row_idx] == pytest.approx(r2_expected, rel=1e-12)

    # Also accepts a PriceSeries container.
    ohlcv = pd.DataFrame(
        {
            "Open": prices,
            "High": prices * 1.01,
            "Low": prices * 0.99,
            "Close": prices,
            "Volume": np.full(len(prices), 1000.0),
        },
        index=prices.index,
    )
    ps = PriceSeries(ohlcv, "synthetic")
    ps_frame = roughness_frame(ps)
    assert ps_frame.index.equals(ps.data.index)
    pd.testing.assert_frame_equal(roughness_frame(ps.data["Close"]), ps_frame)


def test_docstrings_cite_spec_6_4_fixed_lags_cont_das_caveat_and_no_lookahead():
    for obj in (
        roughvol_mod,
        realized_log_vol,
        variogram_hurst,
        rolling_roughness,
        roughness_frame,
    ):
        doc = obj.__doc__ or ""
        assert "no-lookahead" in doc.lower() or "no lookahead" in doc.lower()
        assert "1..10" in doc or ("1" in doc and "10" in doc)
        assert "Cont" in doc and "Das" in doc
        assert "observed" in doc.lower()
        assert "latent" in doc.lower()
        assert "lookahead" in doc.lower() or "through t" in doc.lower() or "through ``t``" in doc.lower()
