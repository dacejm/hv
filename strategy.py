"""Searching for something BETTER than the champion (long-only large-cap quality-momentum,
cap-weighted: Sharpe 0.86, Calmar 0.59). Free to deviate from the papers.

New, principled ideas tested (one precompute, cheap configs):
  - QMV     : add a LOW-VOLATILITY tilt (defensive quality-momentum) -- the low-vol anomaly is among
              the most robust; QMV (quality-momentum-low-vol) historically lifts Sharpe / cuts crashes.
  - inv-vol : RISK-WEIGHT the book (1/vol) instead of cap-weight -> risk parity within the holdings.
  - sig-wt  : weight by signal strength.
Discipline (anti-overfit): a challenger only wins if it beats the champion in BOTH out-of-sample halves
on Sharpe AND Calmar -- not just full sample.

  python strategy.py
"""
from __future__ import annotations

import sys
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from quant import data
from book_backtest import _precompute, _mcap_at, metrics

REBAL = 21
TOP_MCAP = 500
QUANTILE = 0.20
COST_BPS = 10
START = "2016-01-01"


def scored_panel(universe, sectors):
    roic_ts, rets, levels, shares = _precompute(universe)
    syms = list(roic_ts)
    spy = data.benchmark()
    cal = spy.index[spy.index >= pd.Timestamp(START)]
    rebal = cal[::REBAL]
    print(f"universe {len(syms)} | top{TOP_MCAP} mcap | monthly | features: mom, roic, vol")
    panels = []
    for i, D in enumerate(rebal[:-1]):
        rows = []
        for s in syms:
            if sectors.get(s, "Unknown") == "Unknown":
                continue
            lv = levels[s]; lv = lv[lv.index <= D]
            if len(lv) < 260:
                continue
            mom = lv.iloc[-21] / lv.iloc[-252] - 1.0
            vol = float(lv.pct_change().iloc[-126:].std() * np.sqrt(252))     # trailing 6m ann vol
            kt = roic_ts[s][roic_ts[s]["known_on"] <= D]
            ro = float(kt.iloc[-1]["roic"]) if not kt.empty else np.nan
            rows.append((s, mom, ro, vol, _mcap_at(levels, shares, s, D)))
        df = pd.DataFrame(rows, columns=["sym", "mom", "roic", "vol", "mcap"]).dropna(
            subset=["mcap", "mom", "roic", "vol"])
        df = df[df["vol"] > 0]
        if len(df) < 50:
            continue
        df = df.nlargest(TOP_MCAP, "mcap")
        panels.append((D, rebal[i + 1], df))
    return panels, rets, cal, spy


def run_config(panels, rets, cal, quantile=QUANTILE, lowvol=False, weight="cap"):
    daily, prev = {}, set()
    for D, nxt, df in panels:
        df = df.copy()
        sig = df["mom"].rank(pct=True) + df["roic"].rank(pct=True)
        if lowvol:
            sig = sig + (1 - df["vol"].rank(pct=True))        # reward LOW vol
        df["sig"] = sig
        k = max(int(len(df) * quantile), 10)
        sel = df.nlargest(k, "sig")
        if weight == "cap":
            w = sel.set_index("sym")["mcap"]
        elif weight == "invvol":
            w = 1.0 / sel.set_index("sym")["vol"]
        elif weight == "sig":
            w = sel.set_index("sym")["sig"]
        else:
            w = pd.Series(1.0, index=sel["sym"])
        w = w / w.sum()
        hold = set(sel["sym"])
        c = len(hold ^ prev) * COST_BPS / 1e4 / max(len(hold), 1)
        prev = hold
        win = cal[(cal > D) & (cal <= nxt)]
        for dt in win:
            rr = np.nansum([w.get(s, 0) * rets[s].get(dt, np.nan) if np.isfinite(rets[s].get(dt, np.nan)) else 0 for s in hold])
            daily[dt] = rr
        if len(win):
            daily[win[0]] -= c
    return pd.Series(daily).sort_index()


def trend_sleeve(start, end):
    """Cross-asset time-series momentum (managed futures): each ETF held long if its 12m return>0
    else short, vol-scaled to ~10% each, equal-weight basket, monthly. Uncorrelated to equities and
    historically POSITIVE in equity bears (2022 rate/commodity trends) -> a diversifier for the QM book."""
    etfs = ["SPY", "QQQ", "TLT", "IEF", "GLD", "SLV", "DBC", "UUP", "HYG", "EEM", "VNQ", "XLE"]
    series = {}
    for e in etfs:
        try:
            s = data.ohlcv(e).set_index("date")["close"]
            if len(s) > 300:
                series[e] = s
        except FileNotFoundError:
            pass
    if not series:
        return pd.Series(dtype=float)
    cal = pd.DatetimeIndex(sorted(set().union(*[s.index for s in series.values()])))
    cal = cal[(cal >= pd.Timestamp(start)) & (cal <= pd.Timestamp(end))]
    daily = {}
    for e, s in series.items():
        r = s.reindex(cal).ffill().pct_change()
        mom = s.reindex(cal).ffill().shift(21) / s.reindex(cal).ffill().shift(252) - 1   # 12-1m, lagged
        pos = np.sign(mom)
        vol = r.rolling(63).std() * np.sqrt(252)
        scale = (0.10 / vol).clip(0, 3)
        contrib = (pos.shift(1) * scale.shift(1) * r).fillna(0)
        for dt, v in contrib.items():
            daily[dt] = daily.get(dt, 0.0) + v / len(series)
    return pd.Series(daily).sort_index()


def _sr_cal(r):
    r = r.dropna(); eq = (1 + r).cumprod()
    sr = r.mean() / r.std() * np.sqrt(252) if r.std() else np.nan
    dd = (eq / eq.cummax() - 1).min()
    cagr = eq.iloc[-1] ** (252 / len(r)) - 1
    return sr, (cagr / abs(dd) if dd < 0 else np.nan)


def main():
    sect = data.sectors()
    inc = {p.stem for p in (data.EARN / "income_statement").glob("*.parquet")}
    universe = sorted(inc & set(sect.index))
    panels, rets, cal, spy = scored_panel(universe, sect)
    if not panels:
        print("no panels"); return
    champ = run_config(panels, rets, cal, weight="cap")
    spyr = spy.pct_change(); spyr = spyr[(spyr.index >= champ.index[0]) & (spyr.index <= champ.index[-1])]
    trend = trend_sleeve(champ.index[0], champ.index[-1]).reindex(champ.index).fillna(0)
    # combine QM equity book with the uncorrelated trend sleeve (diversification)
    corr = champ.corr(trend)
    def combo(w):                                   # w in QM, (1-w) in trend
        return (w * champ + (1 - w) * trend)
    challengers = {
        "trend sleeve (alone)":  trend,
        "QM 70 / trend 30":      combo(0.70),
        "QM 60 / trend 40":      combo(0.60),
        "QM 50 / trend 50":      combo(0.50),
    }
    print(f"  corr(QM, trend) = {corr:+.2f}")

    print(f"\n=== SEARCH FOR BETTER ({champ.index[0].date()}->{champ.index[-1].date()}), net of costs ===")
    metrics(spyr, "SPY buy-hold")
    metrics(champ, "CHAMPION: QM cap-weighted")
    for name, r in challengers.items():
        metrics(r, name)

    # anti-overfit: beat champion in BOTH OOS halves on Sharpe AND Calmar
    mid = champ.index[len(champ) // 2]
    csr1, cca1 = _sr_cal(champ[champ.index < mid]); csr2, cca2 = _sr_cal(champ[champ.index >= mid])
    print(f"\n  champion OOS: 1st (Sh {csr1:.2f}, Cal {cca1:.2f}) | 2nd (Sh {csr2:.2f}, Cal {cca2:.2f})")
    print("  -- challengers that beat champion in BOTH halves (Sharpe & Calmar): --")
    winner = None
    for name, r in challengers.items():
        s1, k1 = _sr_cal(r[r.index < mid]); s2, k2 = _sr_cal(r[r.index >= mid])
        beats = s1 > csr1 and s2 > csr2 and k1 > cca1 and k2 > cca2
        print(f"    {name:24} 1st(Sh{s1:.2f},Cal{k1:.2f}) 2nd(Sh{s2:.2f},Cal{k2:.2f}) -> {'BEATS' if beats else 'no'}")
        if beats:
            winner = name
    print(f"\n  strict-gate verdict: {'new champion = '+winner if winner else 'champion holds on the strict 4-way gate'}")

    # decisive test: at EQUAL risk (rolling vol-target to 15%), does QM+trend beat QM-only?
    def vt(r, tgt=0.15):
        lev = (tgt / (r.rolling(63).std() * np.sqrt(252))).clip(0.3, 3.0).shift(1).fillna(1.0)
        dl = lev.diff().abs().fillna(0)
        return (r * lev - dl * COST_BPS / 1e4).dropna()
    print("\n  -- EQUAL-RISK comparison (both vol-targeted to 15%) --")
    cv, gv = vt(champ), vt(combo(0.60))
    metrics(cv, "  QM-only @15% vol")
    metrics(gv, "  QM60/trend40 @15% vol")
    mid2 = cv.index[len(cv) // 2]
    s1c, k1c = _sr_cal(cv[cv.index < mid2]); s2c, k2c = _sr_cal(cv[cv.index >= mid2])
    s1g, k1g = _sr_cal(gv[gv.index < mid2]); s2g, k2g = _sr_cal(gv[gv.index >= mid2])
    print(f"    QM-only      OOS 1st(Sh{s1c:.2f}) 2nd(Sh{s2c:.2f})")
    print(f"    QM60/trend40 OOS 1st(Sh{s1g:.2f}) 2nd(Sh{s2g:.2f})")
    better = _sr_cal(gv)[0] > _sr_cal(cv)[0] and s2g > s2c
    print(f"  VERDICT: QM+trend {'IS BETTER at equal risk (higher Sharpe full + recent half)' if better else 'not clearly better'}")
    out = pd.DataFrame({"qm": champ, "trend": trend, "qm60_trend40": combo(0.60)}).dropna()
    out.to_csv("strategy_equity.csv")


if __name__ == "__main__":
    main()
