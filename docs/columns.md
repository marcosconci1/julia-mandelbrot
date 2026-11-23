# The columns

Every column is a function of daily OHLCV observed through the current bar. Classification thresholds are estimated through the previous bar only. Defaults live in `IndicatorConfig` (`src/julia_mandelbrot/config.py`).

| Column | Meaning | Estimation |
| --- | --- | --- |
| `trend_strength` | OLS slope of log price over the window, divided by the dispersion of the same sessions. Dimensionless, comparable across assets. | 63 sessions |
| `realized_volatility`, `realized_volatility_percentile` | Standard deviation of daily log returns and its percentile rank inside a trailing window. | 63 sessions; percentile over 252 sessions |
| `q30_threshold`, `q70_threshold`, `q67_threshold` | Expanding quantile boundaries for trend and volatility states. | Through the previous bar |
| `trend_state`, `vol_state`, `cell`, `valid` | Trend banded low, mid, high; volatility banded low, high; the pair forms one of six cells. `valid` marks rows whose windows are all full. | Thresholds through the previous bar |
| `trend_m`, `vol_m`, `entropy` | Logistic membership scores around the thresholds and the entropy of the pair. Scores express position, not probability. | Thresholds through the previous bar |
| Transition counts, `p_next_<cell>`, `pooled` | Outgoing next cell odds conditioned on the current cell. | 756 session window, 30 transition floor; pooled history below the floor; nothing below the pooled floor |
| `state_duration` | Consecutive sessions in the active cell. | Through the current bar |
| `cond_mean`, `cond_cvar` | Mean daily return and left tail shortfall observed in the current cell. | Prior in cell observations, 30 observation floor |
| `hurst_logvol`, `hurst_r2` | Variogram Hurst estimate of realized log volatility over fixed lags 1 to 10, with the fit quality. Describes the observed series. | 252 session window |
| `cvar`, `drawdown` | Unconditional expected shortfall at the 95 percent level and trailing max drawdown. | 252 session window |

Two properties hold across the set. Causality: no column at row t reads t+1, and no row sets its own threshold. Orthogonality: `cvar` and `drawdown` describe the asset independent of the regime cell, and the transition model conditions on the cell, not on the tail.
