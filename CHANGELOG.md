# Changelog

## 0.1.0 (2026-11-22)

First release.

* Daily price adapters for Yahoo Finance and Binance with a file cache keyed by source, ticker, interval, and date window.
* State columns: trend strength, realized volatility and its percentile, six regime cells, membership scores with entropy.
* Transition model: causal counts, next cell odds with a count floor and pooled fallback, state duration, in cell mean and shortfall.
* Roughness: variogram Hurst estimate of realized log volatility with fit quality.
* Tail: unconditional expected shortfall and drawdown.
* `report`: one screen HTML terminal page per ticker.
* `validate`: reliability diagrams, transition log loss against baselines, roughness sanity, purged walk forward.
* `doctor` and `diagnose`: environment checks and a redacted diagnostics bundle over a local event journal.
