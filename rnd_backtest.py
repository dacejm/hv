"""Measure the RND anomaly: compute the risk-neutral density across SPY history, derive
tail-mass features, z-score each vs its OWN trailing history, and test whether those
deviations predict forward returns.

  left_tail  = RN prob mass below 0.95*spot (downside priced)
  right_tail = RN prob mass above 1.05*spot (upside priced)
  rn_skew    = right_tail - left_tail
Each z-scored vs trailing 52 (weekly) obs -> the time-varying anomaly. Then rank-IC vs
forward 5d & 21d returns, with a non-overlapping deflation.
"""
import sys
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, ".")
from quant import rnd as R

IDX = R.IDX


def main():
    opt = pq.read_table(IDX / "SPY_options.parquet",
                        columns=["date", "type", "strike", "expiration", "implied_volatility"]).to_pandas()
    opt["date"] = pd.to_datetime(opt["date"]).dt.normalize()
    opt = opt[(opt["type"].astype(str).str.lower().str.startswith("c")) & (opt["implied_volatility"] > 0)]
    dates = np.sort(opt["date"].unique())[::5]                      # weekly sample
    opt = opt[opt["date"].isin(dates)].copy()
    opt["dte"] = (pd.to_datetime(opt["expiration"]) - opt["date"]).dt.days

    und = pd.read_parquet(IDX / "SPY_underlying.parquet", columns=["date", "close"])
    und["date"] = pd.to_datetime(und["date"]).dt.normalize()
    und = und.sort_values("date").reset_index(drop=True)
    spotmap = und.set_index("date")["close"]

    recs = []
    for d, g in opt.groupby("date"):
        spot = spotmap.get(d)
        if spot is None or np.isnan(spot):
            continue
        g = g[(g["dte"] >= 10) & (g["dte"] <= 45)]
        if g.empty:
            continue
        exp = g.iloc[(g["dte"] - 30).abs().argmin()]["expiration"]
        c = g[(g["expiration"] == exp) & (g["strike"] >= 0.7 * spot) & (g["strike"] <= 1.3 * spot)].sort_values("strike")
        if len(c) < 8:
            continue
        T = float(c["dte"].iloc[0]) / 365.0
        K = np.linspace(c["strike"].min(), c["strike"].max(), 150)
        iv = np.interp(K, c["strike"], c["implied_volatility"])
        C = R._bs_call(spot, K, T, iv)
        d2 = np.clip(np.exp(R.R * T) * np.gradient(np.gradient(C, K), K), 0, None)
        area = np.sum((d2[:-1] + d2[1:]) / 2 * np.diff(K))
        if area <= 0:
            continue
        pdf = d2 / area
        cdf = np.concatenate([[0], np.cumsum((pdf[:-1] + pdf[1:]) / 2 * np.diff(K))])
        left = float(np.interp(0.95 * spot, K, cdf))
        right = 1 - float(np.interp(1.05 * spot, K, cdf))
        recs.append({"date": d, "left_tail": left, "right_tail": right, "rn_skew": right - left})

    rt = pd.DataFrame(recs).sort_values("date").reset_index(drop=True)
    for col in ("left_tail", "right_tail", "rn_skew"):
        m = rt[col].rolling(52).mean().shift(1)
        s = rt[col].rolling(52).std().shift(1)
        rt[col + "_z"] = (rt[col] - m) / s

    for hold in (5, 21):
        und[f"fwd{hold}"] = und["close"].shift(-hold) / und["close"] - 1
    m = rt.merge(und[["date", "fwd5", "fwd21"]], on="date")
    print(f"RND history: {len(rt)} weekly obs, {rt['date'].min().date()}->{rt['date'].max().date()}\n")
    for col in ("left_tail_z", "right_tail_z", "rn_skew_z"):
        for hold in (5, 21):
            s = m.dropna(subset=[col, f"fwd{hold}"])
            ic = s[col].corr(s[f"fwd{hold}"], method="spearman")
            print(f"{col:14} vs fwd{hold:2}d: n={len(s):4} rank-IC={ic:+.3f}  t~{ic*np.sqrt(len(s)):+.1f}")


if __name__ == "__main__":
    main()
