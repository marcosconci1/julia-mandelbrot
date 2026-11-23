"""Self-contained Bloomberg-terminal style HTML report renderer for the regime indicator."""

from __future__ import annotations

import html
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .cells import cell_frame
from .config import IndicatorConfig
from .data_adapters import PriceSeries
from .membership import membership_frame
from .roughvol import roughness_frame
from .tail import tail_frame
from .transitions import (
    CANONICAL_CELLS,
    conditional_stats,
    next_cell_probabilities,
    state_duration,
)
from .trend_vol import _close

FONT_STACK = "'Geist Mono',ui-monospace,Menlo,Consolas,monospace"
SVG_FONT_STACK = "Geist Mono, ui-monospace, Menlo, Consolas, monospace"
PLOT_WINDOW_BARS = 252

_TREND_STRIP_COLORS: dict[str, str] = {
    "low": "#5a5a5a",
    "mid": "#ff9f1a",
    "high": "#ff5252",
}
_VOL_STRIP_COLORS: dict[str, str] = {
    "low": "#3a3a3a",
    "high": "#4caf50",
}
_CELL_BAR_COLORS: dict[str, str] = {
    "low|low": "#4caf50",
    "low|high": "#81c784",
    "mid|low": "#ff9f1a",
    "mid|high": "#ffb74d",
    "high|low": "#ff5252",
    "high|high": "#e57373",
}


def _last_finite(series: pd.Series, default: float = 0.0) -> float:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return float(default)
    val = float(numeric.iloc[-1])
    return val if math.isfinite(val) else float(default)


def _format_price(value: float) -> str:
    if not math.isfinite(value):
        return "0"
    if abs(value) >= 1000.0:
        return f"{value:,.0f}"
    return f"{value:,.2f}"


def _rle_strip_rects(
    states: pd.Series,
    y: int,
    height: int,
    color_map: dict[str, str],
    default_color: str,
) -> str:
    values = [str(v) if pd.notna(v) else "" for v in states.tolist()]
    n = len(values)
    if n == 0:
        return ""
    denom = max(n - 1, 1)
    rects: list[str] = []
    start = 0
    for idx in range(1, n + 1):
        if idx == n or values[idx] != values[start]:
            end = idx if idx < n else n - 1
            if end <= start:
                end = min(start + 1, denom)
            x = 100.0 * start / denom
            width = 100.0 * (end - start) / denom + 0.15
            fill = color_map.get(values[start], default_color)
            rects.append(
                f'<rect x="{x:.2f}" y="{y}" width="{width:.2f}" height="{height}" fill="{fill}"/>'
            )
            start = idx
    return "".join(rects)


def _price_panel_svg(
    close: pd.Series,
    trend_states: pd.Series,
    vol_states: pd.Series,
) -> str:
    vals = pd.to_numeric(close, errors="coerce").ffill().bfill().fillna(1.0).to_numpy(dtype=float)
    n = len(vals)
    denom = max(n - 1, 1)
    lo = float(np.min(vals)) if n > 0 else 0.0
    hi = float(np.max(vals)) if n > 0 else 1.0
    span = hi - lo if hi > lo else 1.0

    coords: list[tuple[float, float]] = []
    for i, val in enumerate(vals):
        x = 100.0 * i / denom
        y = 87.5 - ((float(val) - lo) / span) * 75.0
        coords.append((x, y))

    if not coords:
        coords = [(0.0, 50.0), (100.0, 50.0)]

    path_d = "M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in coords)
    last_x, last_y = coords[-1]
    last_label = _format_price(float(vals[-1]) if n > 0 else 0.0)

    grid = "".join(
        f'<line x1="0" y1="{gy:.1f}" x2="100" y2="{gy:.1f}" stroke="#2a2a2a" stroke-width="0.5" vector-effect="non-scaling-stroke"/>'
        for gy in (24.8, 41.6, 58.4, 75.2)
    )
    trend_rects = _rle_strip_rects(trend_states, y=98, height=5, color_map=_TREND_STRIP_COLORS, default_color="#5a5a5a")
    vol_rects = _rle_strip_rects(vol_states, y=105, height=5, color_map=_VOL_STRIP_COLORS, default_color="#3a3a3a")

    return (
        '<svg width="100%" height="118" viewBox="0 0 100 118" preserveAspectRatio="none" style="display:block">'
        f"{grid}"
        f'<path d="{path_d}" fill="none" stroke="#ffffff" stroke-width="1.4"/>'
        f'<circle cx="{last_x:.2f}" cy="{last_y:.2f}" r="0.7" fill="#ff9f1a"/>'
        f'<text x="99.0" y="{last_y - 1.5:.1f}" font-size="4.2" fill="#ff9f1a" font-family="{SVG_FONT_STACK}">{last_label}</text>'
        f"{trend_rects}{vol_rects}"
        "</svg>"
    )


def _sparkline_svg(
    series: pd.Series,
    color: str,
    *,
    ref_value: float | None = None,
    ref_color: str = "#ff9f1a",
) -> str:
    clean = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
    last_val = _last_finite(clean, default=0.0)
    vals = clean.ffill().bfill().fillna(last_val).to_numpy(dtype=float)
    n = len(vals)
    denom = max(n - 1, 1)

    lo = float(np.min(vals)) if n > 0 else 0.0
    hi = float(np.max(vals)) if n > 0 else 1.0
    if ref_value is not None and math.isfinite(ref_value):
        lo = min(lo, float(ref_value))
        hi = max(hi, float(ref_value))
    span = hi - lo if hi > lo else 1.0

    def _map_y(v: float) -> float:
        return 41.4 - ((v - lo) / span) * 32.8

    coords = [(100.0 * i / denom, _map_y(float(v))) for i, v in enumerate(vals)]
    if not coords:
        coords = [(0.0, 25.0), (100.0, 25.0)]

    ref_line = ""
    if ref_value is not None and math.isfinite(ref_value):
        ry = _map_y(float(ref_value))
        ref_line = (
            f'<line x1="0" y1="{ry:.1f}" x2="100" y2="{ry:.1f}" '
            f'stroke="{ref_color}" stroke-width="0.8" stroke-dasharray="3 2" opacity=".9"/>'
        )

    path_d = "M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in coords)
    _, last_y = coords[-1]
    label_text = f"{last_val:.3g}"

    return (
        '<svg width="100%" height="52" viewBox="0 0 100 52" preserveAspectRatio="none" style="display:block">'
        f"{ref_line}"
        f'<path d="{path_d}" fill="none" stroke="{color}" stroke-width="1.2"/>'
        f'<circle cx="99.0" cy="{last_y:.1f}" r="1.8" fill="{color}"/>'
        f'<text x="97.0" y="{last_y - 4.0:.1f}" text-anchor="end" font-size="10" fill="#e8e8e8" font-family="{SVG_FONT_STACK}">{label_text}</text>'
        "</svg>"
    )


def _panel(title: str, right: str, body: str, height_px: int) -> str:
    return (
        f'<div style="border:1px solid #2a2a2a;height:{height_px}px;display:flex;flex-direction:column;overflow:hidden">'
        '<div style="display:flex;justify-content:space-between;background:#111;padding:3px 8px;border-bottom:1px solid #2a2a2a">'
        f'<span style="color:#ff9f1a;font-size:11px;letter-spacing:.08em">{title}</span>'
        f'<span style="color:#8a8a8a;font-size:10.5px">{right}</span></div>'
        f'<div style="flex:1;padding:6px 8px">{body}</div></div>'
    )


def _kv_row(key: str, value: str, value_color: str = "#e8e8e8") -> str:
    return (
        '<div style="display:flex;justify-content:space-between">'
        f'<span style="color:#8a8a8a">{key}</span>'
        f'<span style="color:{value_color}">{value}</span></div>'
    )


def build_report(
    prices: PriceSeries | pd.Series,
    validated: dict[str, Any] | None = None,
    *,
    ticker: str | None = None,
) -> str:
    """Compute the full indicator chain on all history and render the self-contained HTML report."""
    cfg = IndicatorConfig()
    close = _close(prices)
    n_obs = len(close)

    if isinstance(prices, PriceSeries):
        source_label = html.escape((prices.source or "DATA").upper())
        raw_ticker = (
            ticker
            or getattr(prices, "ticker", None)
            or prices.data.attrs.get("ticker")
            or ("BTCUSDT" if prices.source.lower() == "binance" else "SPY")
        )
    else:
        source_label = "SYNTHETIC"
        raw_ticker = ticker or close.attrs.get("ticker") or "BTCUSDT"
    ticker_label = html.escape(str(raw_ticker).upper())

    default_warmup = cfg.volatility_window + cfg.percentile_window
    if n_obs > default_warmup + 5:
        cell_win = cfg.trend_window
        pct_win = cfg.percentile_window
    else:
        cell_win = max(2, min(cfg.trend_window, n_obs // 6))
        pct_win = max(2, min(cfg.percentile_window, n_obs // 3))

    c_frame = cell_frame(close, window=cell_win, percentile_window=pct_win)
    m_frame = membership_frame(close, window=cell_win, percentile_window=pct_win, config=cfg)

    cells = c_frame["cell"]
    probs = next_cell_probabilities(
        cells,
        window=cfg.transition_window,
        floor=cfg.transition_count_floor,
    )
    prob_cols = [col for col in probs.columns if col.startswith("p_next_")]
    if not prob_cols or probs[prob_cols].dropna(how="all").empty:
        probs = next_cell_probabilities(cells, window=max(5, n_obs // 2), floor=1)
        prob_cols = [col for col in probs.columns if col.startswith("p_next_")]

    durations = state_duration(cells)
    returns = close.pct_change()
    cond = conditional_stats(cells, returns, floor=cfg.transition_count_floor)
    if cond["cond_mean"].dropna().empty:
        cond = conditional_stats(cells, returns, floor=1)

    if n_obs > cfg.roughvol_realized_volatility_window + cfg.roughvol_window + 5:
        rv_win = cfg.roughvol_realized_volatility_window
        rough_win = cfg.roughvol_window
    else:
        rv_win = max(2, min(cfg.roughvol_realized_volatility_window, n_obs // 10))
        rough_win = max(12, min(cfg.roughvol_window, n_obs // 2))
    r_frame = roughness_frame(close, rv_window=rv_win, window=rough_win, config=cfg)

    tail_win = cfg.tail_window if n_obs >= cfg.tail_window else max(2, n_obs // 2)
    t_frame = tail_frame(close, window=tail_win, quantile=cfg.tail_quantile, config=cfg)

    full = pd.concat(
        [
            close.rename("Close"),
            m_frame,
            probs,
            durations,
            cond,
            r_frame[["realized_log_vol", "hurst_logvol", "hurst_r2"]],
            t_frame[["cvar", "drawdown"]],
        ],
        axis=1,
    )

    valid_rows = full.loc[full["valid"].fillna(False)]
    last_row = valid_rows.iloc[-1] if not valid_rows.empty else full.iloc[-1]

    current_cell = str(last_row["cell"]) if pd.notna(last_row.get("cell")) else "mid|low"
    trend_state_str = (
        str(last_row["trend_state"])
        if pd.notna(last_row.get("trend_state"))
        else current_cell.split("|")[0]
    )
    vol_state_str = (
        str(last_row["vol_state"])
        if pd.notna(last_row.get("vol_state"))
        else current_cell.split("|")[-1]
    )

    last_close = _last_finite(full["Close"], default=100.0)
    last_duration = max(1, int(round(_last_finite(full["state_duration"], default=1.0))))
    last_trend = _last_finite(full["trend_strength"], default=0.0)
    last_vol_pctl = _last_finite(full["realized_volatility_percentile"], default=0.5)
    last_trend_m = _last_finite(full["trend_m"], default=0.5)
    last_vol_m = _last_finite(full["vol_m"], default=0.5)
    last_entropy = _last_finite(full["entropy"], default=0.5)
    last_hurst = _last_finite(full["hurst_logvol"], default=0.5)
    last_cvar = _last_finite(full["cvar"], default=-0.02)
    last_dd = _last_finite(full["drawdown"], default=-0.10)
    last_cond_mean = _last_finite(full["cond_mean"], default=0.0)
    last_cond_cvar = _last_finite(full["cond_cvar"], default=last_cvar)
    last_q30 = _last_finite(full["q30_threshold"], default=0.0)
    last_q67 = _last_finite(full["q67_threshold"], default=0.67)

    p_map: dict[str, float] = {}
    for cell_name in CANONICAL_CELLS:
        col_name = f"p_next_{cell_name}"
        val = float("nan")
        if col_name in full.columns:
            raw_v = last_row.get(col_name, float("nan"))
            if pd.notna(raw_v) and math.isfinite(float(raw_v)):
                val = float(raw_v)
            else:
                val = _last_finite(full[col_name], default=float("nan"))
        p_map[cell_name] = val

    if not any(math.isfinite(v) for v in p_map.values()):
        for cell_name in CANONICAL_CELLS:
            p_map[cell_name] = 1.0 if cell_name == current_cell else 0.0
    else:
        for cell_name in CANONICAL_CELLS:
            if not math.isfinite(p_map[cell_name]):
                p_map[cell_name] = 0.0

    p_stay = p_map.get(current_cell, 0.0)
    is_pooled = bool(last_row.get("pooled", False)) if pd.notna(last_row.get("pooled")) else False

    n_valid = int(full["valid"].fillna(False).sum())
    start_str = full.index[0].strftime("%Y-%m-%d")
    end_str = full.index[-1].strftime("%Y-%m-%d")

    plot_df = full.tail(PLOT_WINDOW_BARS)

    # Stats ribbon
    ribbon_items = [
        ("LAST", _format_price(last_close), "#ffffff", False),
        ("CELL", current_cell.upper(), "#ff9f1a", True),
        ("DAYS IN CELL", f"{last_duration}", "#e8e8e8", False),
        ("P(STAY)", f"{p_stay:.2f}", "#4caf50", True),
        ("TREND σ-NORM", f"{last_trend:.2f}", "#e8e8e8", False),
        ("VOL PCTL", f"{last_vol_pctl * 100.0:.0f}%", "#e8e8e8", False),
        ("H LOGVOL", f"{last_hurst:.2f}", "#e8e8e8", False),
        ("CVAR 95%", f"{last_cvar * 100.0:.1f}%", "#ff5252", False),
        ("MAX DD", f"{last_dd * 100.0:.1f}%", "#ff5252", False),
    ]
    ribbon_html = "".join(
        f'<div style="flex:1;border-right:1px solid #2a2a2a;padding:6px 12px;{"background:#0d0d0d" if hi else ""}">'
        f'<div style="font-size:10px;color:#8a8a8a;letter-spacing:.1em">{label}</div>'
        f'<div style="font-size:19px;font-family:{FONT_STACK};color:{color};margin-top:2px">{val}</div></div>'
        for label, val, color, hi in ribbon_items
    )

    # Left column charts
    price_panel = _panel(
        f"PRICE / {ticker_label} DAILY — REGIME STRIP: TREND (GRAY-AMBER-RED) · VOL (DIM-GREEN)",
        f"{PLOT_WINDOW_BARS}D VIEW",
        _price_panel_svg(
            plot_df["Close"],
            plot_df["trend_state"].fillna(trend_state_str),
            plot_df["vol_state"].fillna(vol_state_str),
        ),
        330,
    )

    spark_trend = _panel(
        f"TREND STRGTH OLS/σ {cfg.trend_window}D — Q30 T-1",
        f"NOW {last_trend:.2f}",
        _sparkline_svg(plot_df["trend_strength"], "#ff9f1a", ref_value=last_q30, ref_color="#ff9f1a"),
        84,
    )
    spark_vol = _panel(
        f"VOL PCTL {cfg.percentile_window}D — Q67 T-1",
        f"NOW {last_vol_pctl * 100.0:.0f}%",
        _sparkline_svg(
            plot_df["realized_volatility_percentile"],
            "#ff9f1a",
            ref_value=last_q67,
            ref_color="#ff9f1a",
        ),
        84,
    )
    spark_memb_trend = _panel(
        "MEMB TREND",
        f"{last_trend_m:.2f}",
        _sparkline_svg(plot_df["trend_m"], "#ffffff"),
        84,
    )
    spark_memb_vol = _panel(
        "MEMB VOL",
        f"{last_vol_m:.2f}",
        _sparkline_svg(plot_df["vol_m"], "#ffffff"),
        84,
    )
    spark_entropy = _panel(
        "ENTROPY",
        f"{last_entropy:.2f}",
        _sparkline_svg(plot_df["entropy"], "#8a8a8a"),
        84,
    )
    spark_hurst = _panel(
        f"HURST LOGVOL {cfg.roughvol_window}D LAGS{cfg.variogram_min_lag}-{cfg.variogram_max_lag} — REF 0.50",
        f"NOW {last_hurst:.2f}",
        _sparkline_svg(plot_df["hurst_logvol"], "#ffffff", ref_value=0.50, ref_color="#8a8a8a"),
        84,
    )
    tail_body = (
        '<div style="display:flex;gap:4px">'
        f'{_sparkline_svg(plot_df["cvar"], "#ff9f1a")}'
        f'{_sparkline_svg(plot_df["drawdown"], "#ff5252")}'
        "</div>"
    )
    spark_tail = _panel(
        f"TAIL {cfg.tail_window}D — CVAR95 / MAXDD",
        f"COND MEAN {last_cond_mean * 100.0:.2f}% · COND CVAR {last_cond_cvar * 100.0:.2f}%",
        tail_body,
        84,
    )

    # Right rail panels
    regime_state_body = (
        '<div style="font-size:11.5px;line-height:1.75">'
        + _kv_row("CELL", current_cell.upper(), "#ff9f1a")
        + _kv_row("TREND / VOL", f"{trend_state_str.upper()} / {vol_state_str.upper()}", "#e8e8e8")
        + _kv_row("DURATION", f"{last_duration} SESSIONS", "#e8e8e8")
        + _kv_row("VALID ROWS", f"{n_valid} / {n_obs}", "#8a8a8a")
        + _kv_row("WINDOW", "T THRU T · THR THRU T-1", "#8a8a8a")
        + "</div>"
    )
    regime_state_panel = _panel("REGIME STATE", "", regime_state_body, 148)

    sorted_probs = sorted(p_map.items(), key=lambda kv: (-kv[1], CANONICAL_CELLS.index(kv[0])))
    trans_rows: list[str] = []
    for cell_name, prob_val in sorted_probs:
        n_blocks = int(round(prob_val * 20.0))
        bar_str = ("█" * n_blocks) if n_blocks > 0 else '<span style="color:#8a8a8a">·</span>'
        label_color = "white" if cell_name == current_cell else "#8a8a8a"
        bar_color = _CELL_BAR_COLORS.get(cell_name, "#8a8a8a")
        trans_rows.append(
            f'<div data-col="p_next_{cell_name}" style="display:flex;align-items:center;gap:6px;font-size:11px;line-height:1.65">\n'
            f'<span style="width:62px;color:{label_color}">{cell_name}</span>\n'
            f'<span style="color:{bar_color};letter-spacing:-.5px">{bar_str}</span>\n'
            f'<span style="color:#e8e8e8">{prob_val:.2f}</span></div>'
        )
    pool_flag = "YES" if is_pooled else "NO"
    transition_body = (
        f'<div style="margin-top:2px">{"".join(trans_rows)}</div>'
        f'<div style="margin-top:6px;font-size:10px;color:#8a8a8a">'
        f"W{cfg.transition_window} · FLOOR{cfg.transition_count_floor} · POOL {pool_flag}<br>"
        "SRC: CAUSAL COUNTS U+1≤T-1</div>"
    )
    transition_panel = _panel("TRANSITION MODEL", "", transition_body, 190)

    in_cell_body = (
        '<div style="font-size:11.5px;line-height:1.75">'
        + _kv_row("COND MEAN", f"{last_cond_mean * 100.0:.2f}%/D", "#e8e8e8")
        + _kv_row("COND CVAR 5%", f"{last_cond_cvar * 100.0:.2f}%", "#ff5252")
        + _kv_row("EST FROM", "U ≤ T-1 IN CELL", "#8a8a8a")
        + "</div>"
    )
    in_cell_panel = _panel("IN-CELL HISTORY", "", in_cell_body, 96)

    if validated is not None:
        trans_v = validated.get("transition", {})
        rough_v = validated.get("roughness", {})

        trans_ll = float(trans_v.get("transition_log_loss", trans_v.get("transition", float("nan"))))
        pers_ll = float(trans_v.get("persistence_log_loss", trans_v.get("persistence", float("nan"))))
        uncond_ll = float(trans_v.get("unconditional_log_loss", trans_v.get("unconditional", float("nan"))))
        trans_pass = bool(trans_v.get("pass", False))

        trans_str = f"{trans_ll:.3f}" if math.isfinite(trans_ll) else "N/A"
        pers_str = f"{pers_ll:.3f}" if math.isfinite(pers_ll) else "N/A"
        uncond_str = f"{uncond_ll:.3f}" if math.isfinite(uncond_ll) else "N/A"
        trans_color = "#4caf50" if trans_pass else "#e8e8e8"

        rough_pass = bool(rough_v.get("pass", False))
        med_h = float(rough_v.get("median_hurst", float("nan")))
        med_r2 = float(rough_v.get("median_r2", float("nan")))
        if math.isfinite(med_h) and math.isfinite(med_r2):
            h_fmt = f"{med_h:.2f}".lstrip("0") if 0.0 < med_h < 1.0 else f"{med_h:.2f}"
            rough_str = f"{'PASS' if rough_pass else 'FAIL'} (H {h_fmt} R² {med_r2:.2f})"
        else:
            rough_str = "PASS" if rough_pass else "FAIL"
    else:
        trans_str = "RUN VALIDATE"
        pers_str = "RUN VALIDATE"
        uncond_str = "RUN VALIDATE"
        rough_str = "RUN VALIDATE"
        trans_color = "#8a8a8a"

    validation_body = (
        '<div style="font-size:11.5px;line-height:1.75">'
        + _kv_row("TRANS LOGLOSS", trans_str, trans_color)
        + _kv_row("PERSISTENCE", pers_str, "#8a8a8a")
        + _kv_row("UNCONDITIONAL", uncond_str, "#8a8a8a")
        + _kv_row("ROUGHNESS", rough_str, "#8a8a8a")
        + _kv_row("MEMBERSHIP", "SCORE · NOT PROB", "#ff9f1a")
        + "</div>"
    )
    validation_panel = _panel("VALIDATION", "", validation_body, 178)

    return (
        '<!DOCTYPE html><html><head><meta charset="utf-8"><style>\n'
        f"*{{box-sizing:border-box;margin:0}} body{{width:1440px;background:#000000;color:#e8e8e8;font-family:{FONT_STACK};padding:10px 14px}}\n"
        ".pnl{border:1px solid #2a2a2a}\n"
        "</style></head><body>\n"
        '<div style="display:flex;justify-content:space-between;align-items:center;padding:2px 0 8px">\n'
        f'<div><span style="color:#ff9f1a">JM</span> &gt; {ticker_label} · {source_label} DAILY · {start_str}→{end_str} · {n_obs} BARS · {n_valid} VALID</div>\n'
        f'<div style="color:#8a8a8a">AS OF {end_str} · RUN JM REPORT</div></div>\n\n'
        f'<div style="display:flex;border:1px solid #2a2a2a;margin-bottom:6px">{ribbon_html}</div>\n\n'
        '<div style="display:grid;grid-template-columns:2fr 1fr;gap:6px">\n'
        "<div>\n"
        f"{price_panel}\n"
        '<div style="height:6px"></div>\n'
        '<div style="display:grid;grid-template-columns:1fr 1fr;gap:6px">\n'
        f"{spark_trend}\n"
        f"{spark_vol}\n"
        "</div>\n"
        '<div style="height:6px"></div>\n'
        '<div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:6px">\n'
        f"{spark_memb_trend}\n"
        f"{spark_memb_vol}\n"
        f"{spark_entropy}\n"
        "</div>\n"
        '<div style="height:6px"></div>\n'
        '<div style="display:grid;grid-template-columns:1fr 1fr;gap:6px">\n'
        f"{spark_hurst}\n"
        f"{spark_tail}\n"
        "</div>\n"
        "</div>\n"
        '<div style="display:flex;flex-direction:column;gap:6px">\n'
        f"{regime_state_panel}\n"
        f"{transition_panel}\n"
        f"{in_cell_panel}\n"
        f"{validation_panel}\n"
        "</div></div>\n\n"
        '<div style="display:flex;justify-content:space-between;margin-top:6px;padding:4px 8px;border:1px solid #2a2a2a;font-size:10.5px;color:#8a8a8a">\n'
        "<span>LEGEND · TREND STRIP: GRAY LOW / AMBER MID / RED HIGH · VOL STRIP: DIM LOW / GREEN HIGH · DASHED = T-1 THRESHOLD</span>\n"
        f'<span style="color:#ff9f1a">DATA {source_label} {ticker_label} · ESTIMATED THRU T · NO LOOKAHEAD</span></div>\n'
        "</body></html>"
    )


def write_report(
    prices: PriceSeries | pd.Series,
    path: str | Path,
    validated: dict[str, Any] | None = None,
    *,
    ticker: str | None = None,
) -> Path:
    """Render the HTML report for ``prices`` and write it to ``path``."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    html_content = build_report(prices, validated=validated, ticker=ticker)
    out_path.write_text(html_content, encoding="utf-8")
    return out_path
