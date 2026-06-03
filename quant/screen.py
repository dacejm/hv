"""Screener / sizing layer — built ONLY from validated components.

Direction (cross-sectional, sector-neutral): rank on ROIC, the STRONGEST validated picker
(candidates.py gauntlet: sector-neutral t3.81 @63d, OOS-stable). growth_accel was the old
primary but a clean full-universe re-measure put it at t~2.4 (modest, doesn't clear t3) --
the recorded t5.3 didn't reproduce -- so it's demoted to context. The equal-weight roic+
growth_accel blend was MEASURED and DILUTES (check_blend.py), so we rank on roic ALONE and
carry growth_accel as context, not a blend.

Sizing/risk: the validated GEX vol-regime conditioner (gex.regime_state) scales gross
exposure -- full in suppressive (pos-GEX), reduced in amplifying (neg-GEX).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import data, gex, roc

RANK_SIGNAL = "roic"          # the strongest validated DIRECTION signal (quarterly/63d horizon)


def _z_in_sector(df: pd.DataFrame, col: str, by) -> pd.Series:
    g = df.groupby(by)[col]
    return (df[col] - g.transform("mean")) / g.transform("std")


def composite(panel: pd.DataFrame, feat: str = RANK_SIGNAL) -> pd.DataFrame:
    """Add the sector-neutral z-score of the ranking signal to a feature panel."""
    panel = panel.copy()
    q = panel["date"].dt.to_period("Q")
    panel["composite"] = _z_in_sector(panel, feat, [q, panel["sector"]])
    return panel


def live_screen(asof: pd.Timestamp, universe: list[str], top: int = 25) -> dict:
    """Point-in-time ranked screen + the GEX regime sizing context."""
    asof = pd.Timestamp(asof)
    sect = data.sectors()
    rows = []
    for sym in universe:
        try:
            rc = roc.roic(sym, asof=asof)                    # PRIMARY (validated, strongest)
        except FileNotFoundError:
            continue
        if rc.empty:
            continue
        try:
            rev = roc.revenue_roc(sym, asof=asof).dropna(subset=["growth_accel"])
            ga = rev.iloc[-1]["growth_accel"] if not rev.empty else np.nan
        except FileNotFoundError:
            ga = np.nan
        rows.append({"symbol": sym, "sector": sect.get(sym, "Unknown"),
                     "roic": rc.iloc[-1]["roic"], "growth_accel": ga})   # growth_accel = context
    df = pd.DataFrame(rows).dropna(subset=[RANK_SIGNAL])
    df = df[df["sector"] != "Unknown"]
    counts = df["sector"].value_counts()
    df = df[df["sector"].isin(counts[counts >= 5].index)]  # need peers to neutralize
    if df.empty:
        return {}
    # outlier-robust sector-neutral score: within-sector percentile rank, centered
    df["composite"] = df.groupby("sector")[RANK_SIGNAL].rank(pct=True) - 0.5
    df = df.dropna(subset=["composite"]).sort_values("composite", ascending=False)

    regime = gex.regime_state("SPY", asof=asof)
    cols = ["symbol", "sector", "composite", "roic", "growth_accel"]
    return {"asof": asof, "regime": regime,
            "longs": df.head(top)[cols].reset_index(drop=True),
            "shorts": df.tail(top)[cols].reset_index(drop=True)}
