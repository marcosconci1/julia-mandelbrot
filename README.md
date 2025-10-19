# Julia Mandelbrot System

A Python library for financial time series analysis that implements a comprehensive market regime classification system combining trend/volatility analysis, fractal (Hurst) analysis, and fuzzy logic for probabilistic market state assessment.

## Overview

The Julia Mandelbrot System provides a modular framework for analyzing financial markets through multiple lenses:

- **Six Market Regimes**: Classification into Bull Quiet/Volatile, Bear Quiet/Volatile, and Sideways Quiet/Volatile states
- **Fractal Analysis**: Hurst exponent computation to identify persistent trends vs mean-reverting behavior
- **Fuzzy Logic**: Probabilistic regime classification providing degrees of membership rather than binary states
- **Forward Returns Analysis**: Statistical analysis of future returns conditioned on current regime
- **Markov Transitions**: Regime transition probability matrices and persistence analysis
- **Rich Visualizations**: Comprehensive charts showing price, indicators, and regime evolution

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
   - Yahoo Finance integration
   - Automatic caching
   - Missing data handling

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Install from Source

```bash
# Clone the repository
git clone https://github.com/yourusername/julia-mandelbrot.git
cd julia-mandelbrot

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

Core requirements:
- `pandas>=1.3.0`
- `numpy>=1.21.0`
- `yfinance>=0.2.0`
- `scipy>=1.7.0`
- `statsmodels>=0.13.0`
- `nolds>=0.5.0` (for Hurst exponent)
- `scikit-fuzzy>=0.4.2` (for fuzzy logic)
- `matplotlib>=3.4.0`
- `seaborn>=0.11.0`

## Quick Start

```python
from juliams.config import JMSConfig
from juliams.data import DataFetcher
from juliams.features import (
    compute_trend_features,
    compute_volatility_features,
    compute_hurst_features
)
from juliams.regimes import RegimeClassifier

# 1. Setup configuration
config = JMSConfig()

# 2. Fetch data
fetcher = DataFetcher(config)
df = fetcher.fetch_data("AAPL", period="2y")

# 3. Compute features
df = compute_trend_features(df, config.to_dict())
df = compute_volatility_features(df, config.to_dict())
df = compute_hurst_features(df, config.to_dict())

# 4. Classify regimes
classifier = RegimeClassifier()
df = classifier.classify(df)

# 5. View current regime
print(f"Current regime: {df['regime_name'].iloc[-1]}")
```

## Detailed Usage

### Running Complete Analysis

```python
python example_usage.py
```

This will:
1. Fetch 2 years of Apple (AAPL) stock data
2. Compute all technical indicators
3. Classify market regimes
4. Analyze forward returns
5. Generate visualizations
6. Display comprehensive statistics

### Custom Configuration

```python
from juliams.config import JMSConfig

# Create custom configuration
config = JMSConfig(
    trend_window=30,  # 30-day trend window
    volatility_window=20,  # 20-day volatility
    hurst_window=100,  # 100-day Hurst
    trend_threshold_up=0.3,  # Higher threshold for uptrend
    use_fuzzy=True  # Enable fuzzy logic
)
```

### Analyzing Multiple Stocks

```python
from juliams.data import DataFetcher

fetcher = DataFetcher()
tickers = ["AAPL", "MSFT", "GOOGL"]

# Fetch data for multiple tickers
data = fetcher.fetch_multiple(tickers, period="1y")

# Analyze each stock
for ticker, df in data.items():
    # Run analysis pipeline
    # ...
```

## Module Structure

```
juliams/
├── config.py              # Configuration management
├── data.py                # Data fetching (Yahoo Finance)
├── features/
│   ├── trend.py          # Trend analysis
│   ├── volatility.py     # Volatility indicators
│   ├── hurst.py          # Hurst exponent
│   └── fractal.py        # Fractal filtering
├── regimes/
│   ├── classification.py # Crisp regime classification
│   └── fuzzy.py          # Fuzzy logic system
├── analysis/
│   ├── forward_returns.py # Forward return analysis
│   ├── transitions.py     # Markov transitions
│   └── segments.py        # Segment analysis
├── visualization/         # Plotting functions
└── output/               # Export utilities
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

- `trend_window`: Rolling window for trend calculation (default: 20)
- `volatility_window`: Rolling window for volatility (default: 20)
- `hurst_window`: Window for Hurst exponent (default: 100)
- `trend_threshold_up/down`: Thresholds for trend classification (default: ±0.2)
- `volatility_percentile`: Percentile for high/low volatility split (default: 0.67)
- `hurst_threshold`: Threshold for fractal memory (default: 0.55)
- `use_fuzzy`: Enable fuzzy logic classification (default: True)

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

## Troubleshooting

### Common Issues

1. **ImportError for nolds/scikit-fuzzy**
   ```bash
   pip install nolds scikit-fuzzy
   ```

2. **Yahoo Finance connection errors**
   - Check internet connection
   - Verify ticker symbol validity
   - Clear cache if data seems stale

3. **Insufficient data for Hurst**
   - Requires minimum 100 data points
   - Adjust `hurst_window` parameter

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






