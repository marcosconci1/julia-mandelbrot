"""Offline synthetic unit tests for the specification section 7 validation harness."""

from __future__ import annotations

import json
import math
import time

import numpy as np
import pandas as pd
import pytest

import julia_mandelbrot.cli as cli_mod
import julia_mandelbrot.validation as validation_mod
from julia_mandelbrot.data_adapters import PriceSeries
from julia_mandelbrot.validation import (
    DEFAULT_RELIABILITY_BINS,
    LOG_LOSS_EPS,
    MAX_MISCALIBRATION,
    ROUGHNESS_MAX_HURST,
    ROUGHNESS_MIN_FINITE_RATIO,
    ROUGHNESS_MIN_HURST,
    ROUGHNESS_MIN_MEDIAN_R2,
    WALK_FORWARD_HORIZON,
    WALK_FORWARD_STEP,
    WALK_FORWARD_TRAIN_BARS,
    reliability_diagram,
    roughness_sanity,
    run_validation,
    transition_log_loss,
    walk_forward,
)


def _series(values, name: str = "value", dtype=float) -> pd.Series:
    index = pd.date_range("2020-01-01", periods=len(values), freq="B", tz="UTC")
    return pd.Series(values, index=index, name=name, dtype=dtype)


def _synthetic_prices(length: int, seed: int = 42) -> pd.Series:
    rng = np.random.default_rng(seed)
    t = np.arange(length, dtype=float)
    # Regime-switching drift and smooth time-varying volatility
    drift = 0.002 * np.sin(2.0 * np.pi * t / 120.0)
    sigma = 0.012 * np.exp(0.35 * np.sin(2.0 * np.pi * t / 75.0))
    log_returns = drift + rng.normal(0.0, sigma)
    close = 100.0 * np.exp(np.cumsum(log_returns))
    index = pd.date_range("2019-01-01", periods=length, freq="B", tz="UTC")
    return pd.Series(close, index=index, name="Close", dtype=float)


def test_reliability_diagram_perfect_calibration_and_skewed_miscalibration():
    """Perfect calibration gives ~0 miscalibration; skewed outcomes give exact positive miscalibration."""
    # 1. Construct a 10-bin dataset (100 observations, 10 per decile) where in bin k (k=0..9)
    # every score is p_k = (k + 0.5) / 10 = 0.05, 0.15, ..., 0.95, and wait: for exact
    # realized frequency matching p_k in each bin of 20 observations:
    # In bin k (20 obs), p_k = (k + 1) / 20 * 2 = (k + 1) / 10 for k=0..9 (0.1, 0.2, ..., 1.0),
    # and exactly 2*(k + 1) of the 20 outcomes are 1.0!
    scores_list: list[float] = []
    outcomes_perfect: list[float] = []
    outcomes_skewed: list[float] = []

    for k in range(10):
        p_k = (k + 1) / 10.0  # 0.1, 0.2, ..., 1.0
        n_ones = 2 * (k + 1)  # out of 20 observations -> exact frequency = p_k
        # Add tiny jitter (< 1e-9) within bin so qcut forms 10 clean deciles
        for i in range(20):
            scores_list.append(p_k + (i - 9.5) * 1e-10)
            outcomes_perfect.append(1.0 if i < n_ones else 0.0)
            # Skewed case: realized frequency is always 0.0 (complete overprediction)
            outcomes_skewed.append(0.0)

    score_ser = _series(scores_list, name="score")
    perfect_ser = _series(outcomes_perfect, name="outcome")
    skewed_ser = _series(outcomes_skewed, name="outcome")

    diag_perfect = reliability_diagram(score_ser, perfect_ser, bins=10)
    assert len(diag_perfect) == 10
    assert list(diag_perfect.columns) == [
        "bin",
        "count",
        "mean_score",
        "frequency",
        "abs_error",
        "miscalibration",
    ]
    assert (diag_perfect["count"] == 20).all()
    np.testing.assert_allclose(
        diag_perfect["mean_score"],
        diag_perfect["frequency"],
        atol=1e-9,
    )
    assert diag_perfect.attrs["miscalibration"] == pytest.approx(0.0, abs=1e-9)
    assert diag_perfect["miscalibration"].iloc[0] == pytest.approx(0.0, abs=1e-9)

    # 2. Skewed case: outcomes are all 0.0 while mean_score in bin k is (k + 1) / 10.
    # Weighted mean |mean_score - 0| = mean(0.1, 0.2, ..., 1.0) = 0.55.
    diag_skewed = reliability_diagram(score_ser, skewed_ser, bins=10)
    assert len(diag_skewed) == 10
    assert diag_skewed.attrs["miscalibration"] == pytest.approx(0.55, abs=1e-8)
    assert diag_skewed["miscalibration"].iloc[0] == pytest.approx(0.55, abs=1e-8)
    assert diag_skewed.attrs["miscalibration"] > 0.50

    # 3. Validation guards: index mismatch and invalid bins raise ValueError.
    other_index_outcome = pd.Series(outcomes_perfect, index=pd.RangeIndex(len(outcomes_perfect)))
    with pytest.raises(ValueError, match="index"):
        reliability_diagram(score_ser, other_index_outcome, bins=10)
    with pytest.raises(ValueError, match="bins"):
        reliability_diagram(score_ser, perfect_ser, bins=0)


def test_transition_log_loss_markov_chain_vs_random_and_zero_prob_clamping():
    """transition_log_loss beats baselines on a structured Markov chain, not on i.i.d. noise, and clamps zero probabilities."""
    # 1. Constructed deterministic-tendency cyclic Markov chain on {"A", "B", "C"}:
    # A -> B (prob 0.90), A -> C (prob 0.10)
    # B -> C (prob 0.90), B -> A (prob 0.10)
    # C -> A (prob 0.90), C -> B (prob 0.10)
    # Self-transitions have probability 0, so persistence is wrong 100% of the time,
    # unconditional is ~1/3 per state (log-loss ~ log(3) = 1.0986), while the
    # transition matrix learns the 0.90/0.10 outgoing distribution (log-loss ~ 0.325).
    rng = np.random.default_rng(123)
    states = ("A", "B", "C")
    next_map = {
        "A": ("B", "C"),
        "B": ("C", "A"),
        "C": ("A", "B"),
    }
    chain: list[str] = ["A"]
    for _ in range(450):
        cur = chain[-1]
        primary, secondary = next_map[cur]
        nxt = primary if rng.random() < 0.90 else secondary
        chain.append(nxt)

    cells_markov = _series(chain, name="cell", dtype=object)
    res_markov = transition_log_loss(cells_markov, window=150, floor=10)

    assert res_markov["n_evaluated"] > 200
    assert res_markov["transition_log_loss"] < res_markov["unconditional_log_loss"]
    assert res_markov["transition_log_loss"] < res_markov["persistence_log_loss"]
    assert res_markov["beats_persistence"] is True
    assert res_markov["beats_unconditional"] is True
    assert res_markov["pass"] is True

    # Because self-transitions never occur in `chain`, persistence predicts 0.0 on
    # every step, which is clamped to LOG_LOSS_EPS (1e-12) -> -log(1e-12) ~ 27.631.
    expected_clamped_loss = -math.log(LOG_LOSS_EPS)
    assert res_markov["persistence_log_loss"] == pytest.approx(expected_clamped_loss, rel=1e-9)

    # 2. Constructed i.i.d. uniform random sequence over {"A", "B"}:
    # No conditional Markov dependency beyond unconditional frequencies, and
    # finite-sample transition estimates have slightly higher variance than global
    # unconditional counts, while on a pure persistent sequence ("A", "A", ...),
    # persistence has 0.0 log-loss and cannot be beaten.
    pure_persistent = _series(["A"] * 200, name="cell", dtype=object)
    res_pers = transition_log_loss(pure_persistent, window=50, floor=10)
    assert res_pers["persistence_log_loss"] == pytest.approx(0.0, abs=1e-12)
    assert res_pers["beats_persistence"] is False
    assert res_pers["pass"] is False

    # Uniform i.i.d. 4-state noise with small window has noisy transition estimates
    # that suffer higher log-loss than the pooled unconditional baseline.
    rng_iid = np.random.default_rng(999)
    iid_cells = _series(
        rng_iid.choice(["S0", "S1", "S2", "S3"], size=300).tolist(),
        name="cell",
        dtype=object,
    )
    res_iid = transition_log_loss(iid_cells, window=25, floor=3)
    assert res_iid["n_evaluated"] > 100
    assert res_iid["transition_log_loss"] >= res_iid["unconditional_log_loss"]
    assert res_iid["beats_unconditional"] is False
    assert res_iid["pass"] is False


def test_roughness_sanity_pass_and_fail_branches():
    """roughness_sanity passes on well-behaved power-law variogram series and fails on each broken condition."""
    # 1. PASS branch: Gaussian random walk in log-vol has theoretical H = 0.5 in [0, 1]
    # and high variogram R^2 > 0.90 across 252-bar windows.
    rng = np.random.default_rng(2024)
    clean_logvol = _series(np.cumsum(rng.normal(0.0, 0.03, size=600)) - 4.0, name="logvol")
    res_pass = roughness_sanity(clean_logvol, window=150)

    assert res_pass["finite_pass"] is True
    assert res_pass["r2_pass"] is True
    assert res_pass["hurst_band_pass"] is True
    assert res_pass["pass"] is True
    assert res_pass["finite_ratio"] >= ROUGHNESS_MIN_FINITE_RATIO
    assert res_pass["median_r2"] >= ROUGHNESS_MIN_MEDIAN_R2
    assert ROUGHNESS_MIN_HURST <= res_pass["median_hurst"] <= ROUGHNESS_MAX_HURST

    # 2. FAIL branch A (finite_pass = False): constant series produces zero variogram -> NaN fits.
    const_logvol = _series(np.full(300, -3.5), name="logvol")
    res_const = roughness_sanity(const_logvol, window=100)
    assert res_const["n_finite"] == 0
    assert res_const["finite_pass"] is False
    assert res_const["pass"] is False

    # 3. FAIL branch B (r2_pass = False): deterministic periodic cosine with period 5
    # makes gamma(k) oscillate wildly across lags 1..10 (high at odd multiples of 2.5,
    # near zero at multiples of 5), destroying log-log linearity (R^2 << 0.8).
    t = np.arange(400, dtype=float)
    osc_logvol = _series(np.cos(2.0 * np.pi * t / 4.5), name="logvol")
    res_low_r2 = roughness_sanity(osc_logvol, window=100)
    assert res_low_r2["finite_pass"] is True
    assert res_low_r2["median_r2"] < 0.20
    assert res_low_r2["r2_pass"] is False
    assert res_low_r2["pass"] is False

    # 4. FAIL branch C (hurst_band_pass = False): periodic signal with period ~10.2
    # makes lag-10 differences nearly vanish relative to lag-1 differences, yielding
    # a negative log-log variogram slope (H < 0.0, outside [0, 1]).
    negative_h_logvol = _series(
        np.cos(2.0 * np.pi * np.arange(300, dtype=float) / 10.2),
        name="logvol",
    )
    res_bad_h = roughness_sanity(negative_h_logvol, window=100)
    assert res_bad_h["finite_pass"] is True
    assert res_bad_h["median_hurst"] < ROUGHNESS_MIN_HURST
    assert res_bad_h["hurst_band_pass"] is False
    assert res_bad_h["pass"] is False


def test_walk_forward_no_leakage_guard_and_purged_horizon():
    """Perturbing post-horizon evaluation bars or purged bars does not change train fit, and purge gap is enforced."""
    prices = _synthetic_prices(320, seed=77)
    train_bars = 120
    horizon = 7
    step = 40

    base_wf = walk_forward(
        prices,
        train_bars=train_bars,
        horizon=horizon,
        step=step,
        window=10,
        percentile_window=20,
    )
    assert not base_wf.empty
    assert len(base_wf) >= 4

    # 1. Verify the purged horizon is strictly applied on every fold:
    # train slice is [0, train_end), purged gap is [train_end, eval_start),
    # and eval_start - train_end == horizon.
    assert (base_wf["purge_start"] == base_wf["train_end"]).all()
    assert (base_wf["purge_end"] == base_wf["eval_start"]).all()
    assert (base_wf["eval_start"] - base_wf["train_end"] == horizon).all()
    assert (base_wf["purged_bars"] == horizon).all()

    # 2. No-leakage guard on fold 0:
    # Fold 0 trains on indices [0, 120), purges [120, 127), evaluates on [127, 167).
    # Perturb post-horizon evaluation bars (>= 127) drastically; fold 0's training
    # fit (train_q30, train_q70, train_q67) must remain bitwise identical!
    perturbed_post_horizon = prices.copy()
    perturbed_post_horizon.iloc[train_bars + horizon :] *= np.linspace(
        0.2, 5.0, len(prices) - (train_bars + horizon)
    )
    wf_pert_eval = walk_forward(
        perturbed_post_horizon,
        train_bars=train_bars,
        horizon=horizon,
        step=step,
        window=10,
        percentile_window=20,
    )

    for col in ("train_q30", "train_q70", "train_q67"):
        assert base_wf.loc[0, col] == wf_pert_eval.loc[0, col]

    # 3. Also perturb the purged horizon bars [120, 127) themselves; fold 0's
    # training fit on [0, 120) must STILL remain bitwise identical!
    perturbed_purge = prices.copy()
    perturbed_purge.iloc[train_bars : train_bars + horizon] *= 10.0
    wf_pert_purge = walk_forward(
        perturbed_purge,
        train_bars=train_bars,
        horizon=horizon,
        step=step,
        window=10,
        percentile_window=20,
    )
    for col in ("train_q30", "train_q70", "train_q67"):
        assert base_wf.loc[0, col] == wf_pert_purge.loc[0, col]

    # 4. Performance sanity on ~1500 bars with full IndicatorConfig defaults.
    prices_1500 = _synthetic_prices(1500, seed=101)
    t0 = time.perf_counter()
    wf_1500 = walk_forward(
        prices_1500,
        train_bars=WALK_FORWARD_TRAIN_BARS,
        horizon=WALK_FORWARD_HORIZON,
        step=WALK_FORWARD_STEP,
    )
    elapsed = time.perf_counter() - t0
    assert not wf_1500.empty
    assert elapsed < 2.0
    assert (wf_1500["eval_start"] - wf_1500["train_end"] == WALK_FORWARD_HORIZON).all()


def test_run_validation_verdict_structure_and_membership_score_semantics():
    """run_validation returns all 4 section-7 verdicts with pass booleans and preserves membership as a score."""
    prices = _synthetic_prices(1250, seed=2026)
    verdicts = run_validation(prices)

    assert set(verdicts.keys()) == {
        "membership",
        "transition",
        "roughness",
        "walk_forward",
    }
    for key, entry in verdicts.items():
        assert isinstance(entry, dict), f"{key} verdict must be a dict"
        assert "pass" in entry, f"{key} verdict must contain 'pass'"
        assert isinstance(entry["pass"], bool), f"{key}['pass'] must be a bool"

    # Membership semantics: even on pass (or fail), status is always 'score', never 'probability'.
    mem = verdicts["membership"]
    assert mem["status"] == "score"
    assert "score" in mem["note"].lower()
    assert "not probability" in mem["note"].lower()
    assert mem["max_miscalibration_threshold"] == MAX_MISCALIBRATION

    # Also test PriceSeries container input.
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
    ps_verdicts = run_validation(ps)
    assert set(ps_verdicts.keys()) == set(verdicts.keys())
    assert ps_verdicts["membership"]["status"] == "score"

    # Docstrings must state that calibration only licenses further study, not probability status.
    for obj in (validation_mod, reliability_diagram, run_validation):
        doc = (obj.__doc__ or "").lower()
        assert "score" in doc
        assert "further study" in doc
        assert "probability" in doc


def test_cli_validate_subcommand_outputs_json_offline(monkeypatch, capsys):
    """CLI `validate` subcommand runs offline with mocked adapter and prints valid JSON verdicts."""
    prices = _synthetic_prices(600, seed=55)
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
    mock_series = PriceSeries(ohlcv, "binance")
    called_args: dict[str, str] = {}

    def fake_fetch_prices(source: str, ticker: str, start: str, end: str) -> PriceSeries:
        called_args["source"] = source
        called_args["ticker"] = ticker
        called_args["start"] = start
        called_args["end"] = end
        return mock_series

    monkeypatch.setattr(cli_mod, "_fetch_prices", fake_fetch_prices)

    cli_mod.main(
        [
            "validate",
            "--source",
            "binance",
            "--ticker",
            "BTCUSDT",
            "--start",
            "2019-01-01",
            "--end",
            "2026-11-22",
        ]
    )
    captured = capsys.readouterr()
    assert called_args == {
        "source": "binance",
        "ticker": "BTCUSDT",
        "start": "2019-01-01",
        "end": "2026-11-22",
    }

    parsed = json.loads(captured.out)
    assert set(parsed.keys()) == {
        "membership",
        "transition",
        "roughness",
        "walk_forward",
    }
    for section in parsed.values():
        assert isinstance(section["pass"], bool)
    assert parsed["membership"]["status"] == "score"
