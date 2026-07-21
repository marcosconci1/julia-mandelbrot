# Julia Mandelbrot System

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

A Python library for market regime indicators. It combines trend/volatility analysis, fractal (Hurst) analysis, tail-risk diagnostics, fuzzy memberships, and change-point overlays into a regime nowcast pipeline.

This repo does not define trading entries, exits, position sizing, or risk limits. It produces indicator columns that another system can consume.

<img width="941" height="990" alt="Figure_1" src="https://github.com/user-attachments/assets/8e445176-3e86-401c-80cc-671094fe3597" />

## Overview

The Julia Mandelbrot System provides a modular framework for analyzing financial markets through multiple lenses:

- **Six Market Regimes**: Classification into Bull Quiet/Volatile, Bear Quiet/Volatile, and Sideways Quiet/Volatile states
- **Fractal Analysis**: Hurst exponent computation to identify persistent trends vs mean-reverting behavior
- **Tail-Risk Survival Layer**: CVaR, VaR, Hill tail-index, kurtosis, drawdown, and survival-state diagnostics
- **Fuzzy Logic**: Probabilistic regime classification, with raw or percentile-based volatility inputs
- **Adaptive Overlays**: Optional data driven thresholds, EWMA trend, Markov switching variance, and Bayesian change point detection that can run alongside the legacy classifier
- **Indicator Readiness**: Warmup and valid-signal columns so startup rows are easy to filter
- **Forward Returns Analysis**: Research diagnostics for checking how regimes behaved historically
- **Markov Transitions**: Regime transition probability matrices and persistence analysis
- **Rich Visualizations**: Comprehensive charts showing price, indicators, and regime evolution (legacy dashboard + optional extended plots)
- **Unified CLI**: Single entry point that auto-detects the correct data source (Yahoo Finance for equities, Binance for crypto)

## Features

### Core Capabilities

1. **Market Regime Classification**
   - Six distinct regimes based on trend direction (Up/Down/Sideways) × volatility level (High/Low)
   - Crisp classification with configurable thresholds
   - Fuzzy logic for probabilistic assessment

2. **Technical Indicators**
   - Normalized trend strength (OLS slope / volatility)
   - Rolling realized volatility
   - Average True Range (ATR)
   - Volatility percentile rankings

3. **Fractal Analysis**
   - Rolling Hurst exponent calculation
   - Fractal memory filtering
   - Persistence vs mean-reversion identification

4. **Statistical Analysis**
   - Forward return distributions by regime
   - Survival-aware rebound overlay guard that blocks added exposure during fragile tail-risk states
   - Markov transition matrices
   - Regime segment analysis
   - Expected regime durations

5. **Data Management**
   - Yahoo Finance (equities) and Binance (crypto) integrations
   - Automatic caching with rate limiting and retry logic
   - Symbol validation and source auto-detection

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/julia-mandelbrot.git
cd julia-mandelbrot

# (Optional) create a virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

```bash
# Unified CLI with auto source detection (defaults to AAPL if omitted)
python main.py

# Analyse a crypto pair, generate extra plots, and export them
python main.py BTCUSDT --period 1y --extra-plots --save-plots output/charts --no-plot

# Analyse forex pairs (handled by Yahoo Finance)
python main.py BRL=X --period 2y
python main.py EURUSD=X --period 1y

# Analyse futures contracts
python main.py GC=F --period 6mo  # Gold futures
python main.py ES=F --period 3mo  # S&P 500 E-mini futures

# Use a bar interval and an indicator calibration profile
python main.py GC=F --period 6mo --profile gold_h4 --interval 4h
```

## Supported Instruments

The system supports multiple asset classes through automatic source detection:

- **Equities**: AAPL, MSFT, GOOGL, BRK.B
- **Indices**: ^GSPC (S&P 500), ^DJI (Dow Jones), ^IXIC (NASDAQ)
- **Cryptocurrencies**: BTCUSDT, ETHUSDT (via Binance)
- **Forex**: EURUSD=X, GBPUSD=X, BRL=X (via Yahoo Finance)
- **Futures**: GC=F (Gold), ES=F (E-mini S&P), CL=F (Crude Oil)

Forex pairs with strong external drivers, such as `BRL=X`, can be labeled by the indicator, but should be calibrated and interpreted on their own terms. Do not read an equity-index regime profile into a currency pair without retuning windows and thresholds.

Programmatic usage (for notebooks / pipelines):

```python
from juliams.config import JMSConfig
from juliams.data import DataFetcherFactory
from juliams.features import compute_trend_features, compute_volatility_features, compute_hurst_features, compute_tail_risk_features
from juliams.profiles import apply_indicator_profile
from juliams.regimes import RegimeClassifier, add_indicator_signal_flags

config = JMSConfig()
apply_indicator_profile(config, "equity_index_h4")
source_cfg = config.get_source_config("stock")

fetcher = DataFetcherFactory.create(symbol="AAPL", config=config)
df = fetcher.fetch_data("AAPL", period=source_cfg["default_period"], interval=source_cfg["timeframe"])

df = compute_trend_features(df, source_cfg)
df = compute_volatility_features(df, source_cfg)
df = compute_hurst_features(df, source_cfg)
df = compute_tail_risk_features(df, source_cfg)

clf = RegimeClassifier(
    trend_threshold_up=source_cfg["trend_threshold_up"],
    trend_threshold_down=source_cfg["trend_threshold_down"],
    volatility_threshold=source_cfg["volatility_percentile"],
)
df = clf.classify(df)
df = add_indicator_signal_flags(df, source_cfg)
print(f"Current regime: {df['regime_name'].iloc[-1]}")
print(f"Ready: {df['valid_signal'].iloc[-1]}")
```

## Detailed Usage

### Running Complete Analysis

```bash
python main.py AAPL --period 2y
```

This will:
1. Auto-detect the correct data source (Yahoo Finance for AAPL)
2. Fetch 2 years of price history (cached when possible)
3. Compute trend, volatility, Hurst, fractal, and tail-risk survival features
4. Classify regimes (crisp + optional fuzzy)
5. Add warmup and valid-signal flags
6. Analyse forward returns and regime transitions
7. Render the legacy 4-panel dashboard (plus optional extended plots)

### Indicator Profiles

Profiles are calibration presets. They set bar interval, feature windows, volatility percentile lookback, tail-risk window, and fuzzy volatility input. They do not contain trade rules.

Available profiles:

| Profile | Timeframe | Notes |
|---------|-----------|-------|
| `equity_index_h1` | 1h | Intraday index calibration |
| `equity_index_h4` | 4h | Larger intraday index calibration |
| `gold_h4` | 4h | Gold/futures-style calibration |
| `crypto_h4` | 4h | 24/7 crypto calibration |
| `fx_daily` | 1d | Daily FX calibration |

Example:

```bash
python main.py AAPL --profile equity_index_h4
```

Use `--interval` when you want to override the profile interval:

```bash
python main.py AAPL --profile equity_index_h4 --interval 1h
```

### Readiness Columns

The classifier adds four columns that help consumers decide whether a row is usable:

| Column | Meaning |
|--------|---------|
| `calibration_observations` | Number of bars seen up to this row |
| `warmup_bars` | Largest configured lookback used by the indicator |
| `warmup_complete` | True once enough bars have accumulated |
| `valid_signal` | True when warmup is complete and the core regime is not `Unknown` |

The rolling calculations are right-aligned and intended for live-style use after the current bar is known. Consumers that trade on these columns should still apply their own signal lag and execution assumptions.

### Adaptive Overlays

The legacy classifier uses fixed thresholds that can drift out of calibration when market structure changes. Opt in to one or more adaptive overlays to add columns alongside the existing `regime` labels.

```bash
python main.py PLTR \
    --adaptive-thresholds \
    --markov-overlay \
    --bocpd-overlay \
    --min-dwell-days 5 \
    --ewma-halflife 25
```

Each flag adds columns rather than replacing them:

| Flag | Adds | Notes |
|------|------|-------|
| `--adaptive-thresholds` | `regime_adaptive` | Rolling quantile cutoffs on an EWMA z score (Hurst, Ooi, Pedersen 2017; Wang & Lin 2020) |
| `--ewma-halflife H` | `trend_strength_ewma` | EWMA std denominator with halflife H (Caporin & Lillo 2023) |
| `--markov-overlay` | `markov_prob_high`, `markov_state` | Two-state switching variance via `statsmodels.MarkovRegression` |
| `--bocpd-overlay` | `bocpd_run_length`, `bocpd_change_prob` | Bayesian online change-point detection (Adams & MacKay 2007; Tsaknaki, Lillo, Mazzarisi 2024) |
| `--bocpd-method dsm` | (same as above) | Robust DSM-BOCPD (Altamirano, Briol, Knoblauch, ICML 2023). More reliable on heavy tailed assets like gold and crypto |
| `--markov-vol-channel TICKER` | (extends `markov_*` columns) | Add an implied volatility channel to the Markov fit (e.g. `^GVZ` for gold, `^VIX` for SPY) |
| `--markov-auto-vol-channel` | (extends `markov_*` columns) | Auto select the vol ticker for known asset classes |
| `--consensus-overlay` | `consensus_event` | Boolean events that fire only when DSM BOCPD and Markov both flag a change within a 5 day window |
| `--min-dwell-days N` | (post-processes `markov_state`) | Suppresses runs shorter than N days |

The same flags map to keyword arguments on `JuliaMandelbrotSystem.classify_regimes` and `JMSConfig` for programmatic use. For walk-forward calibration with strict no-future-leakage guarantees, see `docs/no_future_leakage.md`.

#### Gold case study

On `GC=F` 2024 to 2026, the standard BOCPD reports `run_length=502` (no change point detected) despite the worst gold rout since 1983 in early February 2026. Switching to `--bocpd-method dsm` correctly triggers a change-point spike near 1.0 around the shock window and shortens average run length to around two weeks.

### Custom Configuration

```python
from juliams.config import JMSConfig

config = JMSConfig(
    trend_window=30,
    volatility_window=20,
    volatility_percentile_lookback=250,
    hurst_window=120,
    trend_threshold_up=0.3,
    use_fuzzy=True,
    fuzzy_volatility_source="volatility_percentile",
)

crypto_cfg = config.get_source_config("crypto")
crypto_cfg["trend_window"] = 12
crypto_cfg["volatility_percentile"] = 0.8
```

### Analyzing Multiple Stocks

```python
from juliams.data import DataFetcherFactory

fetcher = DataFetcherFactory.create(source_type="stock")
tickers = ["AAPL", "MSFT", "GOOGL"]

datasets = fetcher.fetch_multiple(tickers, period="1y")

for ticker, df in datasets.items():
    # Run analysis pipeline
    ...
```

## Module Structure

```
juliams/
├── config.py              # Global + source-specific configuration
├── profiles.py            # Indicator calibration profiles
├── data/
│   ├── base.py            # Abstract data-source interface
│   ├── stock.py           # Yahoo Finance implementation
│   ├── crypto.py          # Binance implementation
│   └── utils.py           # Detection, validation, rate limiting
├── features/
│   ├── trend.py           # Trend analysis
│   ├── volatility.py      # Volatility indicators
│   ├── hurst.py           # Hurst exponent
│   ├── fractal.py         # Fractal filtering
│   └── tail.py            # Tail-risk survival diagnostics
├── regimes/
│   ├── classification.py  # Crisp regime classification
│   ├── fuzzy.py           # Fuzzy logic system
│   └── quality.py         # Warmup and valid-signal flags
├── analysis/
│   ├── forward_returns.py # Forward return analysis
│   ├── transitions.py     # Markov transitions
│   └── segments.py        # Segment analysis
├── visualization/         # Plotting functions
└── output/                # Export utilities
```

## Market Regimes

The system classifies markets into six regimes:

| Regime | Trend | Volatility | Characteristics |
|--------|-------|------------|-----------------|
| Bull Quiet | Up | Low | Steady uptrend, low risk |
| Bull Volatile | Up | High | Uptrend with high volatility |
| Sideways Quiet | Sideways | Low | Range-bound, low volatility |
| Sideways Volatile | Sideways | High | Choppy, directionless |
| Bear Quiet | Down | Low | Steady downtrend |
| Bear Volatile | Down | High | Crash-like conditions |

## Key Indicators

### Trend Strength
- Normalized OLS slope of log prices
- Divided by volatility for dimensionless measure
- Similar to t-statistic or Sharpe ratio of trend

### Hurst Exponent
- H > 0.55: Trending (persistent)
- H < 0.45: Mean-reverting
- H ≈ 0.5: Random walk

### Tail-Risk Survival
- `compute_tail_risk_features` adds loss-side VaR, CVaR / expected shortfall, Hill left-tail index, excess kurtosis, rolling max drawdown, `survival_score`, and `survival_regime`.
- `survival_regime` flags fragile states where rebound overlays should not add exposure merely because price momentum improved.
- Group-aware rebound diagnostics now report `survival_gate_share` and keep prior behavior for DataFrames that do not include survival columns.

### Fuzzy Membership
- Probabilistic regime assessment
- Degrees of membership in each regime
- Entropy measure for classification uncertainty

## Configuration Parameters

Key parameters in `JMSConfig`:

- `stock` / `crypto`: Source-specific dataclasses with window sizes, thresholds, and default periods
- `trend_window`: Legacy default rolling window for trend calculation (used when source-specific overrides absent)
- `volatility_window`: Legacy default rolling window for volatility
- `hurst_window`: Legacy default window for Hurst exponent
- `trend_threshold_up/down`: Global thresholds for trend classification
- `volatility_percentile`: Global percentile for high/low volatility split
- `hurst_threshold`: Threshold for fractal memory filtering
- `tail_risk_window`, `tail_cvar_alpha`, `hill_tail_fraction`, and survival thresholds: control non-Gaussian tail-risk diagnostics
- `use_fuzzy`: Enable fuzzy logic classification (default: True)

Use `config.get_source_config('stock')` or `'crypto'` to obtain a ready-to-use dictionary for each data source.

## Output Formats

The system can export results in multiple formats:

1. **Daily Classifications** (CSV)
   - Date, price, indicators, regime labels
   - Fuzzy membership probabilities

2. **Segment Summaries** (CSV)
   - Contiguous regime periods
   - Duration, returns, average indicators

3. **Transition Matrix** (CSV/JSON)
   - Regime transition probabilities
   - Expected durations

4. **Forward Return Statistics** (CSV)
   - Mean/median returns by regime
   - Risk metrics (Sharpe, Sortino)

5. **Fuzzy Nowcast** (JSON/Text)
   - Current regime probabilities
   - Human-readable interpretation

## Theoretical Background

The Julia Mandelbrot System is inspired by:

1. **Van Tharp's Market Classification**: Six market types based on trend and volatility
2. **Mandelbrot's Fractal Markets**: Hurst exponent for long-term memory
3. **Fuzzy Set Theory**: Probabilistic classification for uncertainty handling
4. **Markov Processes**: Regime transition modeling

## Performance Considerations

- **Hurst Calculation**: Computationally intensive, consider using larger step sizes
- **Caching**: Enabled by default to minimize API calls
- **Parallel Processing**: Available for multi-ticker analysis
- **Memory Usage**: ~100MB for 2 years of daily data per ticker

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request


## Acknowledgments

- Benoit Mandelbrot for fractal market theory
- Van Tharp for market regime classification framework
- Yahoo Finance for data provision
- Open source community for supporting libraries

## Contact

For questions or support, please open an issue on GitHub.

---

**Disclaimer**: This library is for educational and research purposes only. Not financial advice. Use at your own risk in trading or investment decisions.
