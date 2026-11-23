# Architecture

## 1. Data layer

`YahooFinanceAdapter` and `BinanceAdapter` fetch daily OHLCV into a validated `PriceSeries` (UTC index, complete OHLCV columns, provider tag). Binance paginates klines; Yahoo responses are checked to span the requested range. A file cache keyed by source, ticker, interval, and date window sits in front, with mtime staleness and a refresh override. Cache writes are atomic: temp file, fsync, rename.

## 2. State columns

`trend_vol.py` produces the dimensionless trend score and realized volatility. `cells.py` bands them with expanding quantile thresholds and marks row validity. `membership.py` adds logistic scores and entropy. Every threshold at row t is estimated through t-1, which is the load bearing rule of the layer: **the row never sets its own standard**.

## 3. Dynamics

`transitions.py` counts cell changes in a 756 session window, always among transitions that completed before the current bar. Odds come out only above a 30 transition floor, pooled history fills the gap, and below the pooled floor the output is an explicit empty value. **An absent estimate is a feature, not a gap**: thin evidence posing as odds is the failure mode the floor exists to prevent.

## 4. Descriptors

`roughvol.py` estimates the variogram Hurst exponent of realized log volatility over fixed lags 1 to 10 and reports the fit quality next to it. `tail.py` computes unconditional expected shortfall and drawdown. Both ignore the regime cell on purpose: **descriptors describe the asset, not the state**, so the report can show them side by side without circular reasoning.

## 5. Proof and rendering

`validation.py` runs reliability diagrams, transition log loss against persistence and unconditional baselines, roughness sanity, and a purged walk forward. A failed check demotes the column in the output. `report.py` renders every column as one self contained HTML page: inline SVG, system monospace stack, terminal grammar, no external resources. `doctor` and `diagnose` close the loop with environment checks and a redacted local bundle over the event journal.

It never phones home. It never reads anything but the market endpoints it names and the local cache it writes. It never estimates with data from the future.
