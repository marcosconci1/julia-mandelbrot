# Validation

`uv run julia-mandelbrot validate --source <source> --ticker <ticker> --start <date> --end <date>` runs the checks and prints verdicts as JSON.

## What it computes

* Reliability diagrams: membership scores are binned into quantile groups and compared against realized frequencies. The output is a miscalibration number per dimension.
* Transition log loss: the next cell odds are scored on transitions arriving after the estimation window, against two baselines: persistence (the current cell repeats) and the unconditional cell frequencies.
* Roughness sanity: the Hurst estimate must be finite, in a sane band, and its fit quality above a threshold on most windows.
* Walk forward: an expanding window protocol with a purged horizon between estimation and evaluation, scored as one step ahead hit rate against persistence.

## Demotion

A failed check demotes a column rather than deleting it. Membership scores that calibrate poorly stay scores and never read as probabilities. A transition model that does not beat persistence out of sample does not enter the contract. The Hurst estimate remains a descriptor of the observed series either way.
