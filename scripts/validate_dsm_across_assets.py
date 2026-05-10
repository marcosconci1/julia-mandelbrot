"""Multi-asset comparison of standard BOCPD vs DSM-BOCPD.

Run on a basket of assets that span different volatility regimes and
tail behaviours. Reports the count of run-length resets each detector
flags. The goal is to verify that DSM-BOCPD detects real regime breaks
on volatile assets (gold, oil, crypto) without over-detecting on
quieter assets (long-duration bonds).

Usage
-----
    python scripts/validate_dsm_across_assets.py

Outputs a markdown-style table to stdout. Requires network access
(yfinance). Non-deterministic (data evolves over time) so this is a
diagnostic script, not a unit test.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
import yfinance as yf

from juliams.regimes.bocpd import detect_change_points_bocpd
from juliams.regimes.dsm_bocpd import detect_change_points_dsm_bocpd


# Asset basket spans: equity (SPY, IEF, TLT), commodities (GC=F, CL=F),
# crypto (BTC-USD), FX (EURUSD=X). Curated to cover quiet (TLT, IEF)
# through volatile (BTC-USD).
ASSETS: list[str] = [
    "SPY",
    "GC=F",
    "BTC-USD",
    "EURUSD=X",
    "CL=F",
    "TLT",
    "IEF",
]


@dataclass
class DetectionResult:
    asset: str
    n_obs: int
    sigma: float
    max_abs_sigma: float
    std_resets: int
    dsm_resets: int
    std_last_rl: int
    dsm_last_rl: int


def count_resets(map_run_length: pd.Series, drop_threshold: int = 30) -> int:
    """Count days where MAP run length drops by >= drop_threshold from
    the previous day. This is a proxy for change-point detections."""
    rl = map_run_length.dropna().values
    drops = (rl[:-1] - rl[1:]) >= drop_threshold
    return int(drops.sum())


def fetch_returns(ticker: str, period: str = "2y") -> pd.Series:
    raw = yf.download(ticker, period=period, progress=False, auto_adjust=False)
    if raw is None or len(raw) == 0:
        raise RuntimeError(f"no data for {ticker}")
    close = raw["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    return np.log(close).diff().dropna()


def evaluate(asset: str, expected_run_length: float = 100.0) -> DetectionResult:
    ret = fetch_returns(asset)
    varx = float(ret.var())
    sigma = float(ret.std())
    max_abs_sigma = float(ret.abs().max() / sigma) if sigma > 0 else 0.0

    std = detect_change_points_bocpd(ret, expected_run_length=expected_run_length)
    dsm = detect_change_points_dsm_bocpd(
        ret,
        expected_run_length=expected_run_length,
        varx=varx,
        omega=1.0,
        robustness_bandwidth=3.0,
    )
    return DetectionResult(
        asset=asset,
        n_obs=len(ret),
        sigma=sigma,
        max_abs_sigma=max_abs_sigma,
        std_resets=count_resets(std.map_run_length),
        dsm_resets=count_resets(dsm.map_run_length),
        std_last_rl=int(std.map_run_length.iloc[-1]),
        dsm_last_rl=int(dsm.map_run_length.iloc[-1]),
    )


def main() -> int:
    print("# DSM-BOCPD vs standard BOCPD: multi-asset comparison\n")
    print(
        "| Asset | N | sigma | max |x|/sigma | std resets | dsm resets | "
        "std last rl | dsm last rl |"
    )
    print(
        "|---|---:|---:|---:|---:|---:|---:|---:|"
    )
    results = []
    for ticker in ASSETS:
        try:
            r = evaluate(ticker)
        except Exception as exc:
            print(f"| {ticker} | ERROR | | | | | | {exc} |")
            continue
        results.append(r)
        print(
            f"| {r.asset} | {r.n_obs} | {r.sigma:.4f} | {r.max_abs_sigma:.2f} | "
            f"{r.std_resets} | {r.dsm_resets} | {r.std_last_rl} | {r.dsm_last_rl} |"
        )

    print("\n## Summary\n")
    if not results:
        print("No results.")
        return 1
    std_total = sum(r.std_resets for r in results)
    dsm_total = sum(r.dsm_resets for r in results)
    print(f"- Total resets, standard: {std_total}")
    print(f"- Total resets, DSM:      {dsm_total}")
    over_detect = [r for r in results if r.dsm_resets > 5 * max(r.std_resets, 1)]
    if over_detect:
        print(
            "\n**Possible over-detection on these assets** "
            "(DSM resets > 5x standard):"
        )
        for r in over_detect:
            print(f"  - {r.asset}: std={r.std_resets} dsm={r.dsm_resets}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
