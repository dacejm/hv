"""Does the COMBINATION validate? Test whether GEX-sizing the growth_accel long/short
basket improves risk-adjusted return vs the unsized basket.

  direction: quarterly sector-neutral growth_accel L/S basket (top vs bottom quintile),
             63d-forward excess return per rebalance.
  sizing:    GEX sign at rebalance -> gross 0.5 in amplifying (neg-GEX) else 1.0.
Compare unsized vs GEX-sized: annualized return, vol, Sharpe, max drawdown.
"""
import glob, os, random, sys
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from quant import backtest, gex

random.seed(7)


def _stats(x):
    x = x.dropna()
    ann, vol = x.mean() * 4, x.std() * np.sqrt(4)
    cum = (1 + x).cumprod()
    dd = (cum / cum.cummax() - 1).min()
    return ann, vol, (ann / vol if vol else np.nan), dd


def main():
    inc = {os.path.basename(p)[:-8] for p in glob.glob("data/earnings/parquet/income_statement/*.parquet")}
    ohl = {os.path.basename(p)[:-8] for p in glob.glob("data/stocks/parquet/ohlcv/*.parquet")}
    sample = random.sample(sorted(inc & ohl), 800)
    panel = backtest.replay(sample)
    g = gex.net_gex("SPY").set_index("date")["gex"].sort_index()

    panel["q"] = panel["date"].dt.to_period("Q")
    rows = []
    for q, grp in panel.groupby("q"):
        sub = grp.dropna(subset=["growth_accel", "excess_63"])
        if len(sub) < 20:
            continue
        sub = sub.assign(ga=sub["growth_accel"] - sub.groupby("sector")["growth_accel"].transform("mean"))
        sub = sub.sort_values("ga")
        k = max(len(sub) // 5, 3)
        ls = sub.tail(k)["excess_63"].mean() - sub.head(k)["excess_63"].mean()
        qd = sub["date"].min()
        prior = g[g.index <= qd]
        neg = bool(prior.iloc[-1] < 0) if len(prior) else False
        rows.append({"date": qd, "ls": ls, "neg_gex": neg})
    B = pd.DataFrame(rows).dropna()
    B["sized"] = B["ls"] * np.where(B["neg_gex"], 0.5, 1.0)

    print(f"quarterly rebalances: {len(B)} ({B['date'].min().date()}->{B['date'].max().date()}), "
          f"amplifying quarters: {int(B['neg_gex'].sum())}\n")
    print(f"{'':14}{'ann_ret':>9}{'ann_vol':>9}{'Sharpe':>8}{'maxDD':>9}")
    for name, col in [("UNSIZED", "ls"), ("GEX-SIZED", "sized")]:
        a, v, s, dd = _stats(B[col])
        print(f"{name:14}{a:+9.3f}{v:9.3f}{s:+8.2f}{dd:+9.3f}")


if __name__ == "__main__":
    main()
