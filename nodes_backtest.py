"""Measure whether the OI nodes actually behave as claimed (SPY, full history).

  pin     -> magnet : does price drift TOWARD the pin? (regression of fwd return on pin-gap;
                      positive slope = magnet; convergence rate > 50% = price closes the gap)
  ceiling -> resistance : is fwd return weak when spot sits just below the call wall?
  floor   -> support    : is fwd return strong when spot sits just above the put wall?
"""
import sys
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, ".")
from quant import nodes

IDX = nodes.IDX
HOLD = 5


def main():
    opt = pq.read_table(IDX / "SPY_options.parquet",
                        columns=["date", "type", "strike", "gamma", "open_interest", "expiration"]).to_pandas()
    opt["date"] = pd.to_datetime(opt["date"]).dt.normalize()
    opt["dte"] = (pd.to_datetime(opt["expiration"]) - opt["date"]).dt.days
    opt["call"] = opt["type"].astype(str).str.lower().str.startswith("c")

    und = pd.read_parquet(IDX / "SPY_underlying.parquet", columns=["date", "close"])
    und["date"] = pd.to_datetime(und["date"]).dt.normalize()
    und = und.sort_values("date").reset_index(drop=True)
    und["fwd"] = und["close"].shift(-HOLD) / und["close"] - 1
    spotmap = und.set_index("date")["close"]

    recs = []
    for d, g in opt.groupby("date"):
        spot = spotmap.get(d)
        if spot is None or np.isnan(spot):
            continue
        n = g[(g.dte >= 0) & (g.dte <= 60) & (g.strike >= 0.5 * spot) & (g.strike <= 1.5 * spot)]
        if n.empty:
            continue
        co = n[n.call].groupby("strike")["open_interest"].sum()
        po = n[~n.call].groupby("strike")["open_interest"].sum()
        above, below = co[co.index > spot], po[po.index < spot]
        goi = (n.gamma.fillna(0) * n.open_interest.fillna(0)).groupby(n.strike).sum().abs()
        recs.append({"date": d, "spot": spot,
                     "pin": goi.idxmax() if not goi.empty else np.nan,
                     "ceiling": above.idxmax() if not above.empty else np.nan,
                     "floor": below.idxmax() if not below.empty else np.nan})
    nd = pd.DataFrame(recs).merge(und[["date", "fwd"]], on="date").dropna(subset=["fwd"])
    print(f"node-days: {len(nd)} | {nd.date.min().date()} -> {nd.date.max().date()}")
    base = nd["fwd"].mean()
    print(f"baseline mean fwd {HOLD}d: {base:+.4f}\n")

    # PIN as magnet
    s = nd.dropna(subset=["pin"]).copy()
    s["gap"] = (s["pin"] - s["spot"]) / s["spot"]
    s = s[s["gap"].abs() < 0.1]                       # ignore degenerate far pins
    slope = np.polyfit(s["gap"], s["fwd"], 1)[0]
    spot5 = s["spot"] * (1 + s["fwd"])
    conv = (abs(spot5 - s["pin"]) < abs(s["spot"] - s["pin"])).mean()
    print(f"PIN magnet: n={len(s)}  slope(fwd~gap)={slope:+.3f} (>0 = pulls toward pin)  "
          f"corr={s['gap'].corr(s['fwd']):+.3f}  convergence={conv:.1%}")

    # CEILING resistance: spot just below the call wall
    nd["cg"] = (nd["ceiling"] - nd["spot"]) / nd["spot"]
    nc = nd[(nd.cg > 0) & (nd.cg < 0.01)]
    print(f"CEILING (<1% below call wall): n={len(nc)}  mean fwd={nc['fwd'].mean():+.4f}  vs base {base:+.4f}")

    # FLOOR support: spot just above the put wall
    nd["fg"] = (nd["spot"] - nd["floor"]) / nd["spot"]
    nf = nd[(nd.fg > 0) & (nd.fg < 0.01)]
    print(f"FLOOR (<1% above put wall):    n={len(nf)}  mean fwd={nf['fwd'].mean():+.4f}  vs base {base:+.4f}")


if __name__ == "__main__":
    main()
