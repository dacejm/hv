"""Node mapping — the price levels that act as magnets / walls.

Two families (not just "ceiling"):
  OI/gamma nodes (need open interest -> index ETFs SPY/QQQ/IWM):
    ceiling     = call wall   (largest call-OI strike above spot)   -> resistance
    floor       = put wall    (largest put-OI strike below spot)    -> support
    pin         = max gamma*OI strike near spot                     -> expiry magnet
    zero_gamma  = strike where net dealer gamma flips sign          -> regime line
  Volume-profile nodes (just OHLCV -> ANY ticker):
    poc / hvn   = highest-volume price levels                       -> magnet
    lvn         = low-volume gaps                                   -> fast-through
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.parquet as pq
from scipy.stats import norm

from . import data

IDX = Path(__file__).resolve().parent.parent / "data" / "QQQ_SPY_IWM"


def oi_nodes(symbol: str = "SPY", asof: pd.Timestamp | None = None, dte_max: int = 60) -> dict:
    """Ceiling / floor / pin / zero-gamma from the latest option chain (needs OI)."""
    path = IDX / f"{symbol}_options.parquet"
    dts = pq.read_table(path, columns=["date"])["date"]
    maxd = pc.max(dts).as_py() if asof is None else \
        pc.max(pc.filter(dts, pc.less_equal(dts, pd.Timestamp(asof)))).as_py()
    tbl = pq.read_table(path, columns=["date", "type", "strike", "gamma", "open_interest",
                                       "expiration", "implied_volatility"],
                        filters=[("date", "=", maxd)]).to_pandas()
    und = pd.read_parquet(IDX / f"{symbol}_underlying.parquet", columns=["date", "close"])
    und["date"] = pd.to_datetime(und["date"]).dt.normalize()
    spot = float(und[und["date"] <= pd.Timestamp(maxd)]["close"].iloc[-1])

    tbl["dte"] = (pd.to_datetime(tbl["expiration"]) - pd.Timestamp(maxd)).dt.days
    d = tbl[(tbl["dte"] >= 0) & (tbl["dte"] <= dte_max)
            & (tbl["strike"] >= 0.5 * spot) & (tbl["strike"] <= 1.5 * spot)].copy()  # drop stale far OI
    d["call"] = d["type"].astype(str).str.lower().str.startswith("c")
    calls, puts = d[d["call"]], d[~d["call"]]

    call_oi = calls.groupby("strike")["open_interest"].sum()
    put_oi = puts.groupby("strike")["open_interest"].sum()
    above = call_oi[call_oi.index > spot]
    below = put_oi[put_oi.index < spot]
    ceiling = float(above.idxmax()) if not above.empty else None
    floor = float(below.idxmax()) if not below.empty else None

    d["goi"] = d["gamma"].fillna(0) * d["open_interest"].fillna(0)
    pin = float(d.groupby("strike")["goi"].sum().abs().idxmax())

    # zero-gamma flip (proper): reprice each contract's gamma across a spot grid, sum net
    # dealer gamma (call +, put -), find the spot level nearest current where it crosses zero.
    sign = np.where(d["call"], 1.0, -1.0) * d["open_interest"].fillna(0).to_numpy()
    K = d["strike"].to_numpy()
    iv = d["implied_volatility"].clip(0.01, 3).fillna(0.3).to_numpy()
    T = np.clip(d["dte"].to_numpy() / 365.0, 1 / 365, None)
    grid = np.linspace(0.85 * spot, 1.15 * spot, 61)
    net = np.array([
        ((norm.pdf((np.log(S / K) + 0.5 * iv**2 * T) / (iv * np.sqrt(T)))
          / (S * iv * np.sqrt(T))) * sign * S * S).sum()
        for S in grid])
    sc = np.where(np.sign(net[:-1]) != np.sign(net[1:]))[0]
    flip = float(min(grid[sc], key=lambda k: abs(k - spot))) if len(sc) else None

    return {"symbol": symbol, "date": str(pd.Timestamp(maxd).date()), "spot": round(spot, 2),
            "ceiling": ceiling, "floor": floor, "pin": pin, "zero_gamma": flip}


def volume_nodes(symbol: str, lookback: int = 120, bins: int = 40) -> dict:
    """POC / high- & low-volume nodes from the price-by-volume profile (any ticker)."""
    px = data.ohlcv(symbol).tail(lookback)
    if px.empty:
        return {}
    tp = (px["high"] + px["low"] + px["close"]) / 3
    hist, edges = np.histogram(tp, bins=bins, weights=px["volume"])
    centers = (edges[:-1] + edges[1:]) / 2
    poc = float(centers[hist.argmax()])
    order = hist.argsort()
    hvn = sorted(round(float(centers[i]), 2) for i in order[-3:])      # 3 biggest magnets
    lvn = sorted(round(float(centers[i]), 2) for i in order[:3] if hist[i] > 0)  # thin gaps
    return {"symbol": symbol, "price": round(float(px["close"].iloc[-1]), 2),
            "poc": round(poc, 2), "hvn": hvn, "lvn": lvn}
