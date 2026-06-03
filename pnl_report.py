"""P&L report for the final book (60% QM + 40% trend, vol-targeted ~15%) vs QM-only and SPY:
dollar growth from $100k, year-by-year returns, drawdown, and the equity-curve chart."""
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from quant import data

START_CAP = 100_000
TARGET = 0.15
COST = 10 / 1e4


def vol_target(r, tgt=TARGET):
    lev = (tgt / (r.rolling(63).std() * np.sqrt(252))).clip(0.3, 3.0).shift(1).fillna(1.0)
    return (r * lev - lev.diff().abs().fillna(0) * COST).dropna()


def stats(r, label):
    eq = (1 + r).cumprod(); yrs = len(r) / 252
    cagr = eq.iloc[-1] ** (1 / yrs) - 1
    dd = eq / eq.cummax() - 1
    val = START_CAP * eq.iloc[-1]
    print(f"{label:26} ${val:>12,.0f} | total {eq.iloc[-1]-1:+.0%} | CAGR {cagr:+.1%} | "
          f"vol {r.std()*np.sqrt(252):.0%} | Sharpe {r.mean()/r.std()*np.sqrt(252):.2f} | "
          f"maxDD {dd.min():.0%} (${START_CAP*(eq.cummax()-eq).max():,.0f})")
    return eq


df = pd.read_csv("strategy_equity.csv", index_col=0, parse_dates=True)
book = vol_target(df["qm60_trend40"])
qm = vol_target(df["qm"])
spy = data.benchmark().pct_change().reindex(df.index).dropna()
spy = spy[(spy.index >= book.index[0]) & (spy.index <= book.index[-1])]

print(f"=== P&L: ${START_CAP:,} invested {book.index[0].date()} -> {book.index[-1].date()}, net of costs ===\n")
eqb = stats(book, "THE BOOK (QM+trend 15%)")
eqq = stats(qm, "QM-only (15%)")
eqs = stats(spy, "SPY buy-hold")

print("\n--- calendar-year total return ---")
yr = pd.DataFrame({"BOOK": book, "QM": qm, "SPY": spy})
ann = yr.groupby(yr.index.year).apply(lambda g: (1 + g).prod() - 1)
print((ann * 100).round(1).to_string())
print(f"\nbest year: BOOK {ann['BOOK'].max():+.0%} | worst year: BOOK {ann['BOOK'].min():+.0%} "
      f"(SPY worst {ann['SPY'].min():+.0%})")
print(f"years BOOK beat SPY: {(ann['BOOK'] > ann['SPY']).sum()}/{len(ann)}")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(11, 6))
    (START_CAP * eqb).plot(ax=ax, label="Book: QM+trend (vol-tgt 15%)", lw=2, color="#1f77b4")
    (START_CAP * eqq).plot(ax=ax, label="QM-only (vol-tgt 15%)", lw=1.3, color="#2ca02c", alpha=0.8)
    (START_CAP * eqs).plot(ax=ax, label="SPY buy-hold", lw=1.3, color="#888", alpha=0.8)
    ax.set_yscale("log"); ax.set_title("Growth of $100,000 (net of costs)")
    ax.set_ylabel("portfolio value ($, log)"); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig("pnl_curve.png", dpi=110)
    print("\nchart -> pnl_curve.png")
except Exception as e:
    print("chart skipped:", e)
