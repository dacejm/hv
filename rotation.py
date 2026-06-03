"""Dual-momentum asset-class ROTATION (Antonacci/Keller style) -- a strategy that aims to beat the
market WITHOUT holding SPY. Rotate monthly into the best-trending assets across a broad NO-SPY universe
(QQQ/IWM/intl/sectors/bonds/metals/commodities/crypto); use ABSOLUTE momentum to flee to T-bills (BIL)
when trends are negative. Split-adjusted prices; compared to SPY, out-of-sample.

  python rotation.py
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, ".")
from quant import data

CASH = "BIL"
COST = 10 / 1e4
START = "2012-01-01"
# offensive universe -- deliberately EXCLUDES SPY
OFFENSE = ["QQQ", "IWM", "EFA", "EEM", "VGK", "EWJ", "XLK", "XLE", "XLF", "XLV", "XLY", "XLP",
           "XLI", "XLU", "XLB", "XLC", "TLT", "IEF", "LQD", "HYG", "GLD", "SLV", "DBC", "USO",
           "VNQ", "BTCUSDT", "ETHUSDT"]


def _load(sym):
    try:
        return data.adj_close(sym)
    except Exception:
        f = Path("data/crypto") / f"{sym}.parquet"
        if f.exists():
            s = pd.read_parquet(f); s["date"] = pd.to_datetime(s["date"]); return s.set_index("date")["close"]
    return None


def price_matrix(syms):
    d = {s: _load(s) for s in syms}
    d = {k: v for k, v in d.items() if v is not None}
    return pd.DataFrame(d).sort_index()


def backtest(K=5, lookbacks=(21, 63, 126, 252), rebal=21, weight="equal", target_vol=None,
             universe=None, crypto_cap=None):
    offense = universe if universe is not None else OFFENSE
    P = price_matrix(offense + [CASH])
    spy = data.benchmark()
    cal = P.index[(P.index >= pd.Timestamp(START)) & (P.index.isin(spy.index))]
    P = P.reindex(cal).ffill()
    rets = P.pct_change()
    vol = rets.rolling(63).std() * np.sqrt(252)
    grid = cal[::rebal]
    mom = sum((P / P.shift(lb) - 1) for lb in lookbacks) / len(lookbacks)

    W = pd.DataFrame(0.0, index=cal, columns=P.columns)
    prev = {}
    turn = pd.Series(0.0, index=cal)
    for i in range(len(grid) - 1):
        D = grid[i]
        m = mom.loc[D, offense].dropna()
        if len(m) < 8:
            continue
        cash_m = mom.loc[D, CASH] if pd.notna(mom.loc[D, CASH]) else 0.0
        top = list(m.nlargest(K).index)
        slot = [t if m[t] > cash_m else CASH for t in top]      # absolute filter -> cash
        w = pd.Series(0.0, index=P.columns)
        if weight == "invvol":
            raw = {s: 1.0 / max(vol.loc[D, s], 0.05) for s in slot}
            tot = sum(raw.values())
            for s in slot:
                w[s] += raw[s] / tot
        else:
            for s in slot:
                w[s] += 1.0 / K
        if crypto_cap is not None:                              # cap crypto, spill to cash
            for cc in ("BTCUSDT", "ETHUSDT"):
                if w[cc] > crypto_cap:
                    w[CASH] += w[cc] - crypto_cap; w[cc] = crypto_cap
        seg = cal[(cal > D) & (cal <= grid[i + 1])]
        W.loc[seg] = w.values
        if len(seg):
            turn[seg[0]] = sum(abs(w.get(k, 0) - prev.get(k, 0)) for k in P.columns) * COST
        prev = w.to_dict()
    port = (W.shift(1) * rets).sum(axis=1) - turn
    port = port.loc[port.ne(0).idxmax():]
    if target_vol:
        lev = (target_vol / (port.rolling(63).std() * np.sqrt(252))).clip(0.3, 2.5).shift(1).fillna(1.0)
        port = port * lev - lev.diff().abs().fillna(0) * COST
    spyr = spy.pct_change().reindex(port.index)
    return port, spyr, W


def stats(r, label):
    r = r.dropna(); e = (1 + r).cumprod(); sh = r.mean() / r.std() * np.sqrt(252)
    dd = (e / e.cummax() - 1).min(); cagr = e.iloc[-1] ** (252 / len(r)) - 1
    mid = r.index[len(r) // 2]
    o1 = r[r.index < mid].mean() / r[r.index < mid].std() * np.sqrt(252)
    o2 = r[r.index >= mid].mean() / r[r.index >= mid].std() * np.sqrt(252)
    print(f"{label:26} Sharpe {sh:.2f} | CAGR {cagr:+.1%} | maxDD {dd:.0%} | Calmar {cagr/abs(dd):.2f} | OOS {o1:.2f}/{o2:.2f}")
    return sh


def main():
    print("=== dual-momentum rotation (NO SPY), split-adjusted, net of costs ===")
    port, spyr, _ = backtest(K=5)
    stats(spyr, "SPY buy-hold (benchmark)")
    stats(port, "rotation top5 (full universe)")
    nocrypto = [s for s in OFFENSE if s not in ("BTCUSDT", "ETHUSDT")]
    stats(backtest(K=5, universe=nocrypto)[0], "rotation top5 NO-crypto")
    stats(backtest(K=5, crypto_cap=0.10)[0], "rotation top5 crypto-cap10%")
    stats(backtest(K=5, crypto_cap=0.20)[0], "rotation top5 crypto-cap20%")


if __name__ == "__main__":
    main()
