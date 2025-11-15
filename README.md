# Julia Mandelbrot System

A Python library for financial time series analysis that implements a comprehensive market regime classification system combining trend/volatility analysis, fractal (Hurst) analysis, and fuzzy logic for probabilistic market state assessment.

<img width="941" height="990" alt="Figure_1" src="https://github.com/user-attachments/assets/8e445176-3e86-401c-80cc-671094fe3597" />

## Overview

The Julia Mandelbrot System provides a modular framework for analyzing financial markets through multiple lenses:

- **Six Market Regimes**: Classification into Bull Quiet/Volatile, Bear Quiet/Volatile, and Sideways Quiet/Volatile states
- **Fractal Analysis**: Hurst exponent computation to identify persistent trends vs mean-reverting behavior
- **Fuzzy Logic**: Probabilistic regime classification providing degrees of membership rather than binary states
- **Forward Returns Analysis**: Statistical analysis of future returns conditioned on current regime
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
```

## Supported Instruments

The system supports multiple asset classes through automatic source detection:

- **Equities**: AAPL, MSFT, GOOGL, BRK.B
- **Indices**: ^GSPC (S&P 500), ^DJI (Dow Jones), ^IXIC (NASDAQ)
- **Cryptocurrencies**: BTCUSDT, ETHUSDT (via Binance)
- **Forex**: EURUSD=X, GBPUSD=X, BRL=X (via Yahoo Finance)
- **Futures**: GC=F (Gold), ES=F (E-mini S&P), CL=F (Crude Oil)

Programmatic usage (for notebooks / pipelines):

```python
from juliams.config import JMSConfig
from juliams.data import DataFetcherFactory
from juliams.features import compute_trend_features, compute_volatility_features, compute_hurst_features
from juliams.regimes import RegimeClassifier

config = JMSConfig()
source_cfg = config.get_source_config("stock")

fetcher = DataFetcherFactory.create(symbol="AAPL", config=config)
df = fetcher.fetch_data("AAPL", period=source_cfg["default_period"])

df = compute_trend_features(df, source_cfg)
df = compute_volatility_features(df, source_cfg)
df = compute_hurst_features(df, source_cfg)

clf = RegimeClassifier(
    trend_threshold_up=source_cfg["trend_threshold_up"],
    trend_threshold_down=source_cfg["trend_threshold_down"],
    volatility_threshold=source_cfg["volatility_percentile"],
)
df = clf.classify(df)
print(f"Current regime: {df['regime_name'].iloc[-1]}")
```

## Detailed Usage

### Running Complete Analysis

```bash
python main.py AAPL --period 2y
```

This will:
1. Auto-detect the correct data source (Yahoo Finance for AAPL)
2. Fetch 2 years of price history (cached when possible)
3. Compute trend, volatility, Hurst, and fractal features
4. Classify regimes (crisp + optional fuzzy)
5. Analyse forward returns and regime transitions
6. Render the legacy 4-panel dashboard (plus optional extended plots)

### Custom Configuration

```python
from juliams.config import JMSConfig

config = JMSConfig(
    trend_window=30,
    volatility_window=20,
    hurst_window=120,
    trend_threshold_up=0.3,
    use_fuzzy=True,
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
├── data/
│   ├── base.py            # Abstract data-source interface
│   ├── stock.py           # Yahoo Finance implementation
│   ├── crypto.py          # Binance implementation
│   └── utils.py           # Detection, validation, rate limiting
├── features/
│   ├── trend.py           # Trend analysis
│   ├── volatility.py      # Volatility indicators
│   ├── hurst.py           # Hurst exponent
│   └── fractal.py         # Fractal filtering
├── regimes/
│   ├── classification.py  # Crisp regime classification
│   └── fuzzy.py           # Fuzzy logic system
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

