"""Improve the EQUITY sleeve (the main return lever). Precompute the panel once, test signal variants:
  QM      : mom12 + roic                 (current)
  RAM     : mom12/vol + roic             (risk-adjusted momentum -- higher Sharpe, fewer crashes)
  MH      : (mom6 + mom12)/2 + roic       (multi-horizon)
  RAM-MH  : (mom6/vol + mom12/vol)/2 + roic
Each: long top 20% of top-500 mcap, cap-weighted; then 50/50 with the (ETF+BTC) trend sleeve, vol-tgt
15%. Beat the current book (Sharpe 1.02, OOS 1.13/0.90) in BOTH OOS halves to win.
"""
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from quant import data
from book_backtest import _precompute, _mcap_at
import improve

REBAL, TOP_MCAP, QUANTILE, COST, TARGET = 21, 500, 0.20, 10/1e4, 0.15


def panel(universe, sectors):
    roic_ts, rets, levels, shares = _precompute(universe)
    syms = list(roic_ts); spy = data.benchmark()
    cal = spy.index[spy.index >= pd.Timestamp("2016-01-01")]; rebal = cal[::REBAL]
    print(f"universe {len(syms)} | building panel (mom6/mom12/vol/roic)...")
    out = []
    for i, D in enumerate(rebal[:-1]):
        rows = []
        for s in syms:
            if sectors.get(s, "Unknown") == "Unknown":
                continue
            lv = levels[s]; lv = lv[lv.index <= D]
            if len(lv) < 260:
                continue
            mom12 = lv.iloc[-21] / lv.iloc[-252] - 1
            mom6 = lv.iloc[-21] / lv.iloc[-126] - 1
            vol = float(lv.pct_change().iloc[-126:].std() * np.sqrt(252))
            kt = roic_ts[s][roic_ts[s]["known_on"] <= D]
            ro = float(kt.iloc[-1]["roic"]) if not kt.empty else np.nan
            rows.append((s, mom12, mom6, vol, ro, _mcap_at(levels, shares, s, D)))
        df = pd.DataFrame(rows, columns=["sym", "mom12", "mom6", "vol", "roic", "mcap"]).dropna()
        df = df[df["vol"] > 0]
        if len(df) >= 50:
            out.append((D, rebal[i+1], df.nlargest(TOP_MCAP, "mcap")))
    return out, rets, cal, spy


def eq_returns(panels, rets, cal, signal):
    daily, prev = {}, set()
    for D, nxt, df in panels:
        df = df.copy()
        if signal == "QM":
            df["sig"] = df["mom12"].rank(pct=True) + df["roic"].rank(pct=True)
        elif signal == "RAM":
            df["sig"] = (df["mom12"]/df["vol"]).rank(pct=True) + df["roic"].rank(pct=True)
        elif signal == "MH":
            df["sig"] = (df["mom6"].rank(pct=True)+df["mom12"].rank(pct=True))/2 + df["roic"].rank(pct=True)
        else:  # RAM-MH
            df["sig"] = ((df["mom6"]/df["vol"]).rank(pct=True)+(df["mom12"]/df["vol"]).rank(pct=True))/2 + df["roic"].rank(pct=True)
        sel = df.nlargest(max(int(len(df)*QUANTILE), 10), "sig")
        w = sel.set_index("sym")["mcap"]; w = w/w.sum()
        hold = set(sel["sym"]); c = len(hold ^ prev)*COST/max(len(hold),1); prev = hold
        for dt in cal[(cal > D) & (cal <= nxt)]:
            daily[dt] = np.nansum([w.get(s,0)*rets[s].get(dt,np.nan) if np.isfinite(rets[s].get(dt,np.nan)) else 0 for s in hold])
        win = cal[(cal > D) & (cal <= nxt)]
        if len(win): daily[win[0]] -= c
    return pd.Series(daily).sort_index()


def vt(r, tgt=TARGET):
    lev = (tgt/(r.rolling(63).std()*np.sqrt(252))).clip(0.3,3.0).shift(1).fillna(1.0)
    return (r*lev - lev.diff().abs().fillna(0)*COST).dropna()


def metr(r):
    r = r.dropna(); eq = (1+r).cumprod()
    return r.mean()/r.std()*np.sqrt(252), eq.iloc[-1]**(252/len(r))-1, (eq/eq.cummax()-1).min()


def oos(r):
    m = r.index[len(r)//2]; return metr(r[r.index < m])[0], metr(r[r.index >= m])[0]


def main():
    sect = data.sectors()
    inc = {p.stem for p in (data.EARN/"income_statement").glob("*.parquet")}
    universe = sorted(inc & set(sect.index))
    panels, rets, cal, spy = panel(universe, sect)
    eqs = {s: eq_returns(panels, rets, cal, s) for s in ["QM", "RAM", "MH", "RAM-MH"]}
    idx = eqs["QM"].index
    trend = improve.trend_sleeve(["SPY","QQQ","TLT","IEF","GLD","SLV","DBC","UUP","HYG","EEM","VNQ","XLE","BTCUSDT"],
                                 (252,), idx[0], idx[-1]).reindex(idx).fillna(0)
    cur = vt(0.5*eqs["QM"] + 0.5*trend); csr,_,_ = metr(cur); o1,o2 = oos(cur)
    print(f"\nCURRENT BOOK (QM + BTC-trend) @15%: Sharpe {csr:.2f} | OOS {o1:.2f}/{o2:.2f}\n")
    for s, e in eqs.items():
        solo = metr(e); r = vt(0.5*e.reindex(idx).fillna(0) + 0.5*trend); sr,cg,dd = metr(r); a1,a2 = oos(r)
        beats = sr > csr and a1 >= o1-0.02 and a2 >= o2-0.02 and (a1>o1 or a2>o2)
        print(f"{s:7} equity-solo Sh {solo[0]:.2f} | BOOK Sharpe {sr:.2f} CAGR {cg:+.1%} maxDD {dd:.0%} | OOS {a1:.2f}/{a2:.2f}{'  <== BETTER' if beats else ''}")


if __name__ == "__main__":
    main()
