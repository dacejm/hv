"""Options-vs-stock / VRP — rebuilt to match Mat's volatility book (Book 05).

FIXES from cross-check:
  - VRP = IV - RV (book sign; positive on avg = seller premium). Was RV-IV before.
  - vol-regime per book sec.5.1 (IV level x term-structure shape) -> sell/buy/reduce.
  - term structure added (near vs far ATM IV on SPY = contango/backwardation).

Decision: SELL vol / use STOCK when IV rich (high VRP, contango); BUY options (long
convexity) when IV cheap vs its history (book 1.5: "IV below future RV in low-vol regimes
just before a spike"). Size to the VRP level (book 5.2).
"""
import sys
import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.parquet as pq

sys.path.insert(0, ".")
from quant import rnd as R

IDX = R.IDX
SQ = np.sqrt(252)


def term_structure(symbol="SPY", asof=None) -> dict:
    """Near (20-40d) vs far (80-120d) ATM IV on the index -> contango/backwardation, plus the
    25-delta risk reversal (RR25), the 3rd factor of the validated surface signal (vol-surface
    Bk05 9.1: IVP + term-slope + RR25). RR25 = IV(25d call) - IV(25d put): the equity skew is
    structurally negative (puts richer); a steeper-than-usual RR25 means put protection is
    expensive -> finance it (put spreads/ratios) rather than buy outright."""
    path = IDX / f"{symbol}_options.parquet"
    dts = pq.read_table(path, columns=["date"])["date"]
    maxd = pc.max(dts).as_py() if asof is None else \
        pc.max(pc.filter(dts, pc.less_equal(dts, pd.Timestamp(asof)))).as_py()
    d = pq.read_table(path, columns=["date", "type", "strike", "expiration",
                                     "implied_volatility", "delta"],
                      filters=[("date", "=", maxd)]).to_pandas()
    und = pd.read_parquet(IDX / f"{symbol}_underlying.parquet", columns=["date", "close"])
    und["date"] = pd.to_datetime(und["date"]).dt.normalize()
    spot = float(und[und["date"] <= pd.Timestamp(maxd)]["close"].iloc[-1])
    d["dte"] = (pd.to_datetime(d["expiration"]) - pd.Timestamp(maxd)).dt.days
    d["is_call"] = d["type"].astype(str).str.lower().str.startswith("c")
    calls = d[d["is_call"] & (d["implied_volatility"] > 0)]

    def atm(lo, hi):
        w = calls[(calls["dte"] >= lo) & (calls["dte"] <= hi)]
        return float(w.iloc[(w["strike"] - spot).abs().argmin()]["implied_volatility"]) if not w.empty else np.nan
    near, far = atm(20, 45), atm(80, 120)
    slope = far - near

    # RR25 on the near tenor: IV at the 25-delta call vs the 25-delta put (delta col is signed)
    nr = d[(d["dte"] >= 20) & (d["dte"] <= 45) & (d["implied_volatility"] > 0) & d["delta"].notna()]
    rr25 = np.nan
    if not nr.empty:
        cc = nr[nr["is_call"]]
        pp = nr[~nr["is_call"]]
        if not cc.empty and not pp.empty:
            iv_c = float(cc.iloc[(cc["delta"] - 0.25).abs().argmin()]["implied_volatility"])
            iv_p = float(pp.iloc[(pp["delta"] + 0.25).abs().argmin()]["implied_volatility"])
            rr25 = iv_c - iv_p
    return {"near_iv": round(near, 3), "far_iv": round(far, 3), "slope": round(slope, 3),
            "shape": "contango" if slope > 0.005 else "backwardation" if slope < -0.005 else "flat",
            "rr25": round(rr25, 4) if pd.notna(rr25) else None,
            "skew": ("steep put skew" if pd.notna(rr25) and rr25 < -0.02
                     else "flat/inverted skew" if pd.notna(rr25) and rr25 > 0
                     else "normal skew" if pd.notna(rr25) else None)}


def vol_regime(near_iv, slope) -> str:
    """Book sec.5.1 regime table (near ATM IV as VIX proxy + term-structure shape)."""
    inv = slope < 0
    if near_iv < 0.15 and not inv:
        return "LOW-VOL CONTANGO (sell vol)"
    if near_iv < 0.25 and not inv:
        return "ELEVATED CONTANGO (sell condors/calendars)"
    if near_iv < 0.35 or (inv and near_iv < 0.35):
        return "TRANSITIONAL (reduce shorts, buy wings)"
    return "CRISIS BACKWARDATION (close shorts, hold long gamma)"


def measure(symbol="SPY", hold=21):
    opt = pq.read_table(IDX / f"{symbol}_options.parquet",
                        columns=["date", "type", "strike", "expiration", "implied_volatility"]).to_pandas()
    opt["date"] = pd.to_datetime(opt["date"]).dt.normalize()
    opt = opt[(opt["type"].astype(str).str.lower().str.startswith("c")) & (opt["implied_volatility"] > 0)]
    dates = np.sort(opt["date"].unique())[::5]
    opt = opt[opt["date"].isin(dates)].copy()
    opt["dte"] = (pd.to_datetime(opt["expiration"]) - opt["date"]).dt.days
    und = pd.read_parquet(IDX / f"{symbol}_underlying.parquet", columns=["date", "close"])
    und["date"] = pd.to_datetime(und["date"]).dt.normalize(); und = und.sort_values("date").reset_index(drop=True)
    r = und["close"].pct_change()
    und["rv_fwd"] = [r.iloc[i + 1:i + 1 + hold].std() * SQ if i + 1 + hold <= len(r) else np.nan for i in range(len(r))]
    spotmap, rvmap = und.set_index("date")["close"], und.set_index("date")["rv_fwd"]

    rows = []
    for dte_, g in opt.groupby("date"):
        spot = spotmap.get(dte_)
        if spot is None or np.isnan(spot):
            continue
        g = g[(g["dte"] >= 20) & (g["dte"] <= 40)]
        if g.empty:
            continue
        rows.append({"date": dte_, "atm_iv": float(g.iloc[(g["strike"] - spot).abs().argmin()]["implied_volatility"]),
                     "rv_fwd": rvmap.get(dte_)})
    s = pd.DataFrame(rows).dropna()
    s["iv_rank"] = s["atm_iv"].rolling(52).apply(lambda w: (w < w[-1]).mean(), raw=True)
    s["vrp"] = s["atm_iv"] - s["rv_fwd"]                        # IV - RV (book sign)
    s = s.dropna(subset=["iv_rank", "vrp"])
    s["bucket"] = pd.cut(s["iv_rank"], [-.01, .33, .66, 1.01], labels=["LOW iv (cheap)", "MID", "HIGH iv (rich)"])
    print(f"{symbol} VRP = IV - RV by IV-rank ({s['date'].min().date()}->{s['date'].max().date()}, n={len(s)}):\n")
    print(s.groupby("bucket", observed=True)["vrp"].agg(n="size", mean_VRP="mean",
          pct_IV_over_RV=lambda x: (x > 0).mean()).round(4).to_string())
    print(f"\ncorr(iv_rank, VRP) = {s['iv_rank'].corr(s['vrp']):+.3f}  "
          "(+ = rich IV -> sell vol/use stock; cheap IV -> negative VRP -> BUY options, book 1.5)")
    ts = term_structure(symbol)
    print(f"\nlatest term structure: near {ts['near_iv']} / far {ts['far_iv']} -> {ts['shape']}  "
          f"=> regime: {vol_regime(ts['near_iv'], ts['slope'])}")
    print(f"RR25 (3rd factor): {ts['rr25']}  -> {ts['skew']}  "
          f"(steep put skew => finance puts via spreads/ratios, don't buy outright)")


if __name__ == "__main__":
    measure(sys.argv[1] if len(sys.argv) > 1 else "SPY")
