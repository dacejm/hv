"""Cached, vectorized feature store -> makes strategy tests run in SECONDS instead of minutes.

The slowness was: every run re-read ~3,300x3 parquet files and looped name-by-name, rebalance-by-
rebalance in Python. Here we pay that cost ONCE to build matrices (date x symbol), cache them, and
then every signal/backtest is a vectorized pandas/numpy op over ~110 rebalance rows.

  python featcache.py build      # one-time (~minutes); writes data/_featcache/*.parquet
  then: from featcache import load, momentum, vol, run_signal   # instant
"""
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from quant import data, roc

CACHE = data.DATA / "_featcache"
REBAL = 21
START = "2016-01-01"


def build():
    CACHE.mkdir(exist_ok=True)
    sect = data.sectors()
    inc = {p.stem for p in (data.EARN / "income_statement").glob("*.parquet")}
    universe = sorted(inc & set(sect.index))
    print(f"building cache for {len(universe)} symbols...")

    closes, dvols, roic_known = {}, {}, {}
    for i, s in enumerate(universe):
        try:
            c = data.adj_close(s)                      # SPLIT-ADJUSTED (critical fix)
            o = data.ohlcv(s).set_index("date")
        except Exception:
            continue
        ret = c.pct_change()
        c = c[ret.abs() <= 0.5]                        # catch residual data errors (splits handled)
        closes[s] = c
        vol = o["volume"].reindex(c.index)
        dvols[s] = (c * vol)                           # dollar volume = robust, fully-covered size proxy
        try:
            r = roc.roic(s)
            if not r.empty:
                rr = r.copy(); rr["known_on"] = pd.to_datetime(rr["period_end"]) + data.DEFAULT_REPORT_LAG
                roic_known[s] = rr.set_index("known_on")["roic"]
        except Exception:
            pass
        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{len(universe)}")

    C = pd.DataFrame(closes).sort_index()
    C = C[C.index >= pd.Timestamp(START)]
    spy = data.benchmark()
    cal = spy.index[(spy.index >= C.index[0]) & (spy.index <= C.index[-1])]
    C = C.reindex(cal)
    grid = cal[::REBAL]

    # roic / shares onto the daily calendar (ffill from knowledge date), then we sample on the grid
    def to_grid(known: dict):
        m = pd.DataFrame({s: v[~v.index.duplicated()] for s, v in known.items()})
        return m.sort_index().reindex(cal, method="ffill")
    roic_g = to_grid(roic_known).reindex(grid)
    # size = trailing-63d mean dollar volume on the grid (fully covered large/liquid proxy)
    DV = pd.DataFrame(dvols).sort_index().reindex(cal)
    mcap_g = DV.rolling(63, min_periods=20).mean().reindex(grid)

    C.to_parquet(CACHE / "close.parquet")
    roic_g.to_parquet(CACHE / "roic_grid.parquet")
    mcap_g.to_parquet(CACHE / "mcap_grid.parquet")
    pd.Series(sect).to_frame("sector").to_parquet(CACHE / "sectors.parquet")
    pd.Series(grid).to_frame("d").to_parquet(CACHE / "grid.parquet")
    print(f"cached: close {C.shape}, roic_grid {roic_g.shape}, grid {len(grid)} rebalances -> {CACHE}")


def load():
    C = pd.read_parquet(CACHE / "close.parquet")
    roic_g = pd.read_parquet(CACHE / "roic_grid.parquet")
    mcap_g = pd.read_parquet(CACHE / "mcap_grid.parquet")
    grid = pd.read_parquet(CACHE / "grid.parquet")["d"]
    grid = pd.DatetimeIndex(grid.values)
    sect = pd.read_parquet(CACHE / "sectors.parquet")["sector"]
    return C, roic_g, mcap_g, grid, sect


def momentum(C, grid, skip=21, look=252):
    pos = C.index.get_indexer(grid)
    rows = {}
    for d, p in zip(grid, pos):
        rows[d] = (C.iloc[p - skip] / C.iloc[p - look] - 1) if p - look >= 0 else pd.Series(np.nan, index=C.columns)
    return pd.DataFrame(rows).T


def vol(C, grid, look=126):
    r = C.pct_change()
    pos = C.index.get_indexer(grid)
    rows = {d: r.iloc[max(p - look, 0):p].std() * np.sqrt(252) for d, p in zip(grid, pos)}
    return pd.DataFrame(rows).T


def run_signal(C, grid, sig, mcap_g, top_mcap=500, quantile=0.20, weight="cap", cost_bps=10):
    """Vectorized long-only backtest: build a forward-filled weight matrix, then daily return =
    (weights[t-1] * returns[t]).sum. Covers EVERY trading day (no dict-accumulation gaps)."""
    rets = C.pct_change()
    W = pd.DataFrame(0.0, index=C.index, columns=C.columns)
    turn = pd.Series(0.0, index=C.index)
    prev = set()
    for i in range(len(grid) - 1):
        D = grid[i]
        mc = mcap_g.loc[D].dropna()
        univ = mc.nlargest(top_mcap).index
        srow = sig.loc[D, univ].dropna()
        if len(srow) < 50:
            continue
        sel = srow.nlargest(max(int(len(srow) * quantile), 10)).index
        w = (mc[sel] / mc[sel].sum()) if weight == "cap" else pd.Series(1.0 / len(sel), index=sel)
        seg = C.index[(C.index > D) & (C.index <= grid[i + 1])]
        W.loc[seg, sel] = np.tile(w.reindex(sel).values, (len(seg), 1))
        if len(seg):
            turn[seg[0]] = len(set(sel) ^ prev) * cost_bps / 1e4 / max(len(sel), 1)
        prev = set(sel)
    port = (W.shift(1) * rets).sum(axis=1) - turn
    return port.loc[port.ne(0).idxmax():]              # trim leading zeros before first position


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "build":
        build()
    else:
        print(__doc__)
