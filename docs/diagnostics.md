# Diagnostics

Three local commands answer environment and history questions without opening a debugger.

## doctor

```bash
uv run julia-mandelbrot doctor
```

Per item pass or fail checks: Python version, numpy and pandas imports, subcommand imports, config defaults, cache directory writability, journal writability, and an optional Binance reachability probe that never fails the run on its own. Exit code is 0 when every non optional check passes.

## The journal

`journal.jsonl` lives in the platform cache directory (`~/.cache/julia-mandelbrot/` on Linux) and records command lifecycle events, one JSON object per line. The file holds 30 days of events and stays under 1 MiB by dropping the oldest half when it grows past that. It never leaves the machine except by your own hand.

## diagnose

```bash
uv run julia-mandelbrot diagnose
```

Prints a redacted bundle for bug reports: package version, Python version, platform, doctor results, config values, and the last 20 journal events reduced to timestamp and kind. Home paths are scrubbed. Nothing else is included.
