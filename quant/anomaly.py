"""General anomaly detection — one reusable z-vs-trailing-history detector for ANY stream.

Factors out what was being re-coded per signal (PCR-z, RND-z, GEX percentile, ...). Plug
in any series (options skew/IV/PCR, realized vol, volume, estimate revisions, macro
surprises, GEX, ...) and get the same red/blue/star convention:

  z      = (value - trailing mean) / trailing std   (window EXCLUDES today -> point-in-time)
  red    = z >= 0  (excess density / above normal)
  blue   = z < 0   (deficit / below normal)
  star   = |z| >= k  (anomaly, default 1.8 sigma)
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def zscore(s, window: int = 60) -> pd.Series:
    s = pd.Series(s).astype(float)
    m = s.rolling(window).mean().shift(1)        # trailing, excludes current point
    sd = s.rolling(window).std().shift(1)
    return (s - m) / sd


def detect(s, window: int = 60, k: float = 1.8) -> pd.DataFrame:
    z = zscore(s, window)
    df = pd.DataFrame({"value": pd.Series(s).astype(float).values, "z": z.values}, index=pd.Series(s).index)
    df["color"] = np.where(df["z"] >= 0, "red", "blue")     # red = excess, blue = deficit
    df["star"] = df["z"].abs() >= k
    return df


def latest(s, window: int = 60, k: float = 1.8):
    df = detect(s, window, k).dropna(subset=["z"])
    if df.empty:
        return None
    r = df.iloc[-1]
    return {"z": round(float(r["z"]), 2), "color": r["color"], "star": bool(r["star"])}


def scan(series_map: dict, window: int = 60, k: float = 1.8) -> pd.DataFrame:
    """Current anomaly reading across many named streams -> which are flagged now."""
    rows = []
    for name, s in series_map.items():
        info = latest(s, window, k)
        if info:
            rows.append({"stream": name, **info})
    return pd.DataFrame(rows).sort_values("z", key=lambda c: c.abs(), ascending=False) if rows else pd.DataFrame()
