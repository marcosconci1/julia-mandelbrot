# julia-mandelbrot wiki

> A market regime indicator for daily price series. It classifies the current trend and volatility state into one of six cells, estimates the odds of the next cell, and renders the whole picture as a one screen terminal report.

Judging a market's state by eye fails in a specific way: the answer changes with the last candle, and a regime shift is only certain once it has finished. Tools that promise a direction deliver an opinion, while the useful question, what kind of day is this, goes unanswered.

julia-mandelbrot reads daily OHLCV from Yahoo Finance or Binance, computes a dimensionless trend score and a volatility percentile, and bands them into six cells whose boundaries come from expanding history through the previous bar. A transition model built on 756 sessions of cell changes converts the current cell into next cell odds behind a count floor of 30. A roughness estimate, an entropy read, and unconditional tail descriptors complete the picture, and the `report` command draws all of it as one dense page.

The result is an auditable state read. Every threshold is visible, every odds figure carries its evidence count, and the `validate` command scores the columns against persistence and unconditional baselines so the parts that stop earning their place are visible in numbers.

## Map

| If you want to | Read |
|---|---|
| Why the indicator exists, and what it refuses to do | [Context](Context) |
| How it works, and why it is built this way | [Architecture](Architecture) |
| Install and first run | [Installation](https://github.com/marcosconci1/julia-mandelbrot/blob/main/docs/install.md) |
| What each column means | [The columns](https://github.com/marcosconci1/julia-mandelbrot/blob/main/docs/columns.md) |
| How columns are scored and demoted | [Validation](https://github.com/marcosconci1/julia-mandelbrot/blob/main/docs/validation.md) |
