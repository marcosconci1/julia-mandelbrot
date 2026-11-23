# Installation

This page covers a clean install: requirements, setup, and the first run.

## Requirements

* Python 3.10 or newer
* [uv](https://docs.astral.sh/uv/) for dependency management
* Outbound HTTPS to query1.finance.yahoo.com (Yahoo source) or api.binance.com (Binance source)

## Install

```bash
git clone https://github.com/marcosconci1/julia-mandelbrot.git
cd julia-mandelbrot
uv sync
```

`uv sync` installs the package, the runtime dependencies (numpy, pandas), and the dev group (pytest) into `.venv`.

## First run

Run the environment check:

```bash
uv run julia-mandelbrot doctor
```

Every non optional check should read PASS. The Binance reachability check is optional and never fails the doctor on its own.

Then render a first report:

```bash
uv run julia-mandelbrot report --source binance --ticker BTCUSDT \
    --start 2024-01-01 --end 2025-01-01 --out report.html
```

Open `report.html` in a browser. The page is self contained: inline SVG charts, system monospace fonts, no external resources.

## Requirements recap

* Python 3.10+
* uv
* Outbound HTTPS to the chosen data source
