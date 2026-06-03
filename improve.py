"""Iterate the TREND sleeve (reuses saved QM equity returns). Test adding CRYPTO (BTC/ETH, free
Binance data) as trend assets -- inverse-vol scaled so they don't dominate. Beat the current book
(QM/trend vol-tgt 15%, Sharpe ~0.97) in BOTH OOS halves at equal risk."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from quant import data

COST = 10 / 1e4
TARGET = 0.15
CRYPTO = Path("data/crypto")


def _load(sym):
    try:
        return data.ohlcv(sym).set_index("date")["close"]
    except FileNotFoundError:
        f = CRYPTO / f"{sym}.parquet"
        if f.exists():
            s = pd.read_parquet(f); s["date"] = pd.to_datetime(s["date"]); return s.set_index("date")["close"]
    return None


def trend_sleeve(syms, lookbacks=(252,), start=None, end=None):
    series = {s: _load(s) for s in syms}
    series = {k: v for k, v in series.items() if v is not None and len(v) > max(lookbacks) + 60}
    cal = pd.DatetimeIndex(sorted(set().union(*[s.index for s in series.values()])))
    if start: cal = cal[cal >= pd.Timestamp(start)]
    if end: cal = cal[cal <= pd.Timestamp(end)]
    daily = pd.Series(0.0, index=cal)
    n = len(series)
    for e, s in series.items():
        px = s.reindex(cal).ffill(); r = px.pct_change()
        sig = sum(np.sign(px.shift(21) / px.shift(lb) - 1) for lb in lookbacks) / len(lookbacks)
        scale = (0.10 / (r.rolling(63).std() * np.sqrt(252))).clip(0, 3)
        daily = daily.add((sig.shift(1) * scale.shift(1) * r).fillna(0) / n, fill_value=0)
    return daily


def vt(r, tgt=TARGET):
    lev = (tgt / (r.rolling(63).std() * np.sqrt(252))).clip(0.3, 3.0).shift(1).fillna(1.0)
    return (r * lev - lev.diff().abs().fillna(0) * COST).dropna()


def metr(r):
    r = r.dropna(); eq = (1 + r).cumprod()
    return (r.mean()/r.std()*np.sqrt(252), eq.iloc[-1]**(252/len(r))-1, (eq/eq.cummax()-1).min())


def oos(r):
    mid = r.index[len(r)//2]; return metr(r[r.index < mid])[0], metr(r[r.index >= mid])[0]


if __name__ == "__main__":
    df = pd.read_csv("strategy_equity.csv", index_col=0, parse_dates=True)
    qm = df["qm"]; s0, e0 = qm.index[0], qm.index[-1]
    etfs = ["SPY", "QQQ", "TLT", "IEF", "GLD", "SLV", "DBC", "UUP", "HYG", "EEM", "VNQ", "XLE"]

    sleeves = {
        "ETF-only":        trend_sleeve(etfs, (252,), s0, e0),
        "ETF+BTC":         trend_sleeve(etfs + ["BTCUSDT"], (252,), s0, e0),
        "ETF+BTC+ETH":     trend_sleeve(etfs + ["BTCUSDT", "ETHUSDT"], (252,), s0, e0),
    }
    print("standalone trend sleeves:")
    for n, t in sleeves.items():
        sr, cg, dd = metr(t); print(f"  {n:14} Sharpe {sr:.2f} CAGR {cg:+.1%} maxDD {dd:.0%}")

    cur = vt(0.50 * qm + 0.50 * sleeves["ETF-only"].reindex(qm.index).fillna(0))
    csr, cc, cd = metr(cur); o1, o2 = oos(cur)
    print(f"\nCURRENT-ish (QM50/ETF-only) @15%: Sharpe {csr:.2f} CAGR {cc:+.1%} maxDD {cd:.0%} | OOS {o1:.2f}/{o2:.2f}\n")

    best = None
    for n, t in sleeves.items():
        t = t.reindex(qm.index).fillna(0)
        for w in (0.50, 0.60, 0.65):
            r = vt(w * qm + (1 - w) * t); sr, cg, dd = metr(r); a1, a2 = oos(r)
            beats = sr > csr and a1 >= o1 - 0.02 and a2 >= o2 - 0.02 and (a1 > o1 or a2 > o2)
            print(f"QM{int(w*100)}/{n:12} Sharpe {sr:.2f} CAGR {cg:+.1%} maxDD {dd:.0%} | OOS {a1:.2f}/{a2:.2f}{'  <== BETTER' if beats else ''}")
            if beats and (best is None or sr > best[1]):
                best = (f"QM{int(w*100)}/{n}", sr, cg, dd)
    print(f"\nbest: {best if best else 'none beat current'}")