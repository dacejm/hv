"""Does MACRO x (the regime sector-tilt) actually improve the momentum screen, or is it
noise dressed as macro? Compare cross-sectional rank-IC of momentum WITHOUT the tilt vs
WITH it (mom x MACRO x), across history. If tilted IC isn't higher, regime stays a
context-only conditioner, not part of the score (the macro-as-signal guard).
"""
import glob, os, random, sys
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from quant import data, regime

random.seed(7)


def main():
    rates = pd.read_parquet(regime.RATES, columns=["date", "2_year", "10_year"])
    rates["date"] = pd.to_datetime(rates["date"])
    rates = rates.sort_values("date").set_index("date")
    spy = data.benchmark()
    curve = (rates["10_year"] - rates["2_year"]).reindex(spy.index, method="ffill")
    growth = spy / spy.shift(126) - 1
    phase = pd.Series(np.where(growth <= -0.02, "CONTRACTION",
                      np.where(curve > 1.0, "EARLY_EXPANSION",
                      np.where(curve > 0.2, "MID_EXPANSION", "LATE_EXPANSION"))), index=spy.index)

    grid = spy.index[::5]
    grid = grid[(grid >= "2010-01-01") & (grid <= spy.index[-1] - pd.Timedelta(days=35))]

    sect = data.sectors()
    uni = sorted({os.path.basename(p)[:-8] for p in glob.glob("data/stocks/parquet/ohlcv/*.parquet")} & set(sect.index))
    uni = random.sample(uni, 600)

    parts = []
    for sym in uni:
        px = data.ohlcv(sym).set_index("date")["close"]
        if len(px) < 130:
            continue
        blend = 0.2 * (px / px.shift(5) - 1) + 0.3 * (px / px.shift(20) - 1) + 0.5 * (px / px.shift(60) - 1)
        fwd = (px.shift(-21) / px - 1).clip(-0.5, 0.5)                  # winsorize split artifacts
        sub = pd.DataFrame({"mom": blend, "fwd": fwd}).reindex(grid).dropna()
        sub["symbol"], sub["sector"] = sym, sect.get(sym, "Unknown")
        parts.append(sub.reset_index(names="date"))
    P = pd.concat(parts)
    P["phase"] = P["date"].map(phase)
    P["tilt"] = [regime.sector_tilt(ph).get(sec, 1.0) for ph, sec in zip(P["phase"], P["sector"])]
    P["mom"] = P["mom"].clip(-1, 1)

    ics_u, ics_t = [], []
    for d, g in P.groupby("date"):
        if len(g) < 20 or g["mom"].std() == 0:
            continue
        z = (g["mom"] - g["mom"].mean()) / g["mom"].std()
        ics_u.append(z.corr(g["fwd"], method="spearman"))
        ics_t.append((z * g["tilt"]).corr(g["fwd"], method="spearman"))
    u, t = pd.Series(ics_u).dropna(), pd.Series(ics_t).dropna()
    print(f"cross-sections: {len(u)} ({P['date'].min().date()}->{P['date'].max().date()}), ~{P['symbol'].nunique()} names")
    print(f"momentum ALONE        : mean IC {u.mean():+.4f}  t {u.mean()/u.std()*np.sqrt(len(u)):+.1f}")
    print(f"momentum x MACRO tilt : mean IC {t.mean():+.4f}  t {t.mean()/t.std()*np.sqrt(len(t)):+.1f}")
    print(f"delta from MACRO x    : {t.mean()-u.mean():+.4f}  -> {'HELPS' if t.mean()>u.mean() else 'does NOT help'}")


if __name__ == "__main__":
    main()
