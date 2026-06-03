"""Lane test: for each index-level component, measure its predictive content against
BOTH forward return (DIRECTION lane) and forward realized vol (SIZING/VOL lane), so each
lands in its correct lane empirically. Weekly-sampled (reduces overlap) on SPY history.

Cross-sectional components (growth_accel direction, PCR/RND) were measured separately and
are cited in the scorecard for completeness.
"""
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from quant import data, gex, regime

SQ = np.sqrt(252)


def main():
    spy = data.benchmark()
    ret = spy.pct_change()
    feat = pd.DataFrame(index=spy.index)
    feat["momentum"] = 0.2 * (spy / spy.shift(5) - 1) + 0.3 * (spy / spy.shift(20) - 1) + 0.5 * (spy / spy.shift(60) - 1)
    feat["realized_vol"] = ret.rolling(21).std() * SQ
    g = gex.net_gex("SPY").set_index("date")["gex"]
    feat["GEX"] = g.reindex(spy.index)
    rates = pd.read_parquet(regime.RATES, columns=["date", "2_year", "10_year"])
    rates["date"] = pd.to_datetime(rates["date"])
    feat["yield_curve"] = (rates.set_index("date")["10_year"] - rates.set_index("date")["2_year"]).reindex(spy.index, method="ffill")

    r = ret.values
    fwd_ret5 = spy.shift(-5) / spy - 1
    fwd_ret21 = spy.shift(-21) / spy - 1
    fwd_vol21 = pd.Series([np.std(r[i + 1:i + 22]) * SQ if i + 22 <= len(r) else np.nan
                           for i in range(len(r))], index=spy.index)

    df = feat.assign(fwd_ret5=fwd_ret5, fwd_ret21=fwd_ret21, fwd_vol21=fwd_vol21).iloc[::5].dropna()
    print(f"lane test: {len(df)} weekly obs, {df.index.min().date()}->{df.index.max().date()}\n")
    print(f"{'component':14} {'dirIC_5d':>9} {'dirIC_21d':>10} {'volIC_21d':>10}   lane")
    for c in ["momentum", "GEX", "yield_curve", "realized_vol"]:
        d5 = df[c].corr(df["fwd_ret5"], method="spearman")
        d21 = df[c].corr(df["fwd_ret21"], method="spearman")
        v21 = df[c].corr(df["fwd_vol21"], method="spearman")
        lane = ("SIZING/VOL" if abs(v21) > 0.15 and abs(v21) > 2 * max(abs(d5), abs(d21))
                else "DIRECTION" if max(abs(d5), abs(d21)) > 0.05 else "CONTEXT")
        print(f"{c:14} {d5:+9.3f} {d21:+10.3f} {v21:+10.3f}   {lane}")

    print("\nseparately measured (cross-sectional / heavy):")
    print("  roic           -> DIRECTION  (sector-neutral t3.81 @63d, OOS-stable -- STRONGEST picker)")
    print("  growth_accel   -> DIRECTION  (MODEST: IC ~0.018, t~2.4 @21d/t2.7 @63d; recorded t5.3 didn't reproduce)")
    print("  pcr_vol_z      -> CONTEXT/weak-dir (IC ~0.07 but t2.1 non-overlap)")
    print("  rnd right-tail -> CONTEXT/weak-dir (IC ~0.09 @5d)")
    print("  nodes          -> CONTEXT (no daily edge)")
    print("  options/IV-rank-> EXPRESSION (VRP: long vol pays when IV cheap)")


if __name__ == "__main__":
    main()
