# Contributing

Small project, small rules.

## Setup

```bash
git clone https://github.com/marcosconci1/julia-mandelbrot.git
cd julia-mandelbrot
uv sync
uv run julia-mandelbrot doctor
```

## Before you open a PR

* `uv run pytest` green.
* One topic per branch, named `feat/<topic>` or `fix/<topic>`.
* Conventional Commits, one coherent change per commit.
* New behavior carries a test. Causal estimation code carries a no lookahead test.
* Public claims match the code. The docs are read with the source next to them.

## Reporting a problem

Open an issue with your environment (`julia-mandelbrot doctor` output), the command you ran, and `julia-mandelbrot diagnose` pasted in.
