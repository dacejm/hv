"""Screener / sizing layer — built ONLY from validated components.

Direction (cross-sectional, sector-neutral): a composite of the two signals that
cleared Mat's t>=3 bar -- growth_accel (his 2nd derivative) and revision (analyst
momentum) -- plus fcf_yield as a weak, right-signed blend member. Combining
independent low-IC signals raises IR (Grinold's Fundamental Law).

Sizing/risk: the validated GEX vol-regime conditioner (gex.regime_state) scales
gross exposure -- full in suppressive (pos-GEX), reduced in amplifying (neg-GEX).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import data, gex, roc

# growth_accel is the single strongest validated signal (ICIR 0.92, t5.3). A naive
# equal-weight blend with revision measurably DILUTED it (ICIR -> 0.58), so we rank on
# growth_accel and carry revision as context. (Future: IC-weighted blend on the subset
# of names where both signals exist -- their cross-sectional corr is only ~0.18.)
RANK_SIGNAL = "growth_accel"


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
            rev = roc.revenue_roc(sym, asof=asof)
        except FileNotFoundError:
            continue
        rev = rev.dropna(subset=["growth_accel"])
        if rev.empty:
            continue
        rm = roc.revision_momentum(sym, asof=asof)
        revis = rm[rm["published"] <= asof]["revision"].tail(4).mean() if not rm.empty else np.nan
        rows.append({"symbol": sym, "sector": sect.get(sym, "Unknown"),
                     "growth_accel": rev.iloc[-1]["growth_accel"], "revision": revis})
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
    cols = ["symbol", "sector", "composite", "revision"]
    return {"asof": asof, "regime": regime,
            "longs": df.head(top)[cols].reset_index(drop=True),
            "shorts": df.tail(top)[cols].reset_index(drop=True)}
