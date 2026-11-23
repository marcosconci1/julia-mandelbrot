# Context

## The situation

Daily price data is free, and the first tool every analyst builds is a regime read: is this a trending market, is volatility high, has something changed. Most of that read happens by eye, against a chart, once a day.

## The complication

The eye fails on schedule. Trend reads flip on one candle. Volatility feels high or low only relative to memory, and memory is short. Thresholds picked once (a 0 slope, a fixed volatility level) stop meaning anything when the asset or the year changes. And the indicators that do exist tend to hand back an opinion on direction, when the question was about state.

## The resolution

julia-mandelbrot splits the state read into two dimensions that scale with the asset: a dimensionless trend score and a volatility percentile ranked inside the asset's own history. Six cells come out of the pair. Boundaries are rolling quantiles through the previous bar, so the standard moves with history and never with the day being classified. A transition model turns past cell changes into odds for the next one, behind an evidence floor. The whole state lands on one screen by `report`, and `validate` keeps the columns honest against baselines.

## What this system refuses to do

* No entry, exit, or position sizing. The columns feed such decisions; the tool does not make them.
* No direction opinion. The state is trend x volatility, not buy or sell.
* No probability language without calibration. Membership scores are scores.
* No threshold you cannot see. Every boundary is an output column.
* No silent estimation. A column below its evidence floor reports nothing rather than a guess.
