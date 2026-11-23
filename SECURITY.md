# Security

Report vulnerabilities privately: open a GitHub security advisory on this repository rather than a public issue.

## What the tool touches

* Outbound HTTPS to the data source you name (Yahoo Finance or Binance public endpoints).
* Local reads and writes inside its own cache directory (`~/.cache/julia-mandelbrot/` on Linux): the price cache and the event journal.
* Nothing else. No telemetry, no config files outside the cache directory, no credentials.

The `diagnose` bundle is redacted by construction: config values, doctor results, and journal event kinds with timestamps. Review it before pasting it anywhere.
