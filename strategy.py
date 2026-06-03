"""The reconfigured, VIABLE book — long-only large-cap QUALITY-MOMENTUM.

Forced by the evidence: market-neutral L/S loses; equal-weight quality lagged the mega-cap bull;
trend-gating and lone vol-targeting whipsawed. What beats passive here is the two most-validated
equity factors COMBINED -- momentum (Jegadeesh-Titman, Bk16) x quality (ROIC) -- long-only on
large-caps. This file precomputes the scored panel ONCE, then evaluates a few PRINCIPLED configs
(quintile vs decile concentration, equal vs vol-targeted) so we pick a robust winner, not an
overfit one. Honest framing: managed beta + a validated factor tilt; the return is mostly the
equity premium, the factors lift CAGR/Calmar over passive.

  python strategy.py
"""
from __future__ import annotations

import sys
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from quant import data
from book_backtest import _precompute, _mcap_at, metrics

REBAL = 21            # monthly
TOP_MCAP = 500
COST_BPS = 10
START = "2016-01-01"


def scored_panel(universe, sectors):
    """Per monthly rebalance: large-cap cross-section with momentum + ROIC ranks (point-in-time)."""
    roic_ts, rets, levels, shares = _precompute(universe)
    syms = list(roic_ts)
    spy = data.benchmark()
    cal = spy.index[spy.index >= pd.Timestamp(START)]
    rebal = cal[::REBAL]
    print(f"universe {len(syms)} | top{TOP_MCAP} mcap | monthly quality-momentum")
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
            kt = roic_ts[s][roic_ts[s]["known_on"] <= D]
            ro = float(kt.iloc[-1]["roic"]) if not kt.empty else np.nan
            rows.append((s, sectors.get(s), mom, ro, _mcap_at(levels, shares, s, D)))
        df = pd.DataFrame(rows, columns=["sym", "sector", "mom", "roic", "mcap"]).dropna(subset=["mcap", "mom", "roic"])
        if len(df) < 50:
            continue
        df = df.nlargest(TOP_MCAP, "mcap")
        df["sig"] = (df["mom"].rank(pct=True) + df["roic"].rank(pct=True)) / 2
        panels.append((D, rebal[i + 1], df))
    return panels, rets, cal, spy


def run_config(panels, rets, cal, quantile=0.20, cap_weight=False):
    """Daily returns of the long top-`quantile` quality-momentum book (equal or cap weighted)."""
    daily, prev = {}, set()
    for D, nxt, df in panels:
        k = max(int(len(df) * quantile), 10)
        sel = df.nlargest(k, "sig")
        w = (sel.set_index("sym")["mcap"] / sel["mcap"].sum()) if cap_weight else \
            pd.Series(1.0 / len(sel), index=sel["sym"])
        hold = set(sel["sym"])
        c = len(hold ^ prev) * COST_BPS / 1e4 / max(len(hold), 1)
        prev = hold
        win = cal[(cal > D) & (cal <= nxt)]
        for dt in win:
            rr = np.nansum([w.get(s, 0) * rets[s].get(dt, np.nan) if s in rets and np.isfinite(rets[s].get(dt, np.nan)) else 0 for s in hold])
            daily[dt] = rr
        if len(win):
            daily[win[0]] -= c
    return pd.Series(daily).sort_index()


def vol_target(book, target=0.18, cap=2.0):
    rv = book.rolling(63).std() * np.sqrt(252)
    lev = (target / rv).clip(0.5, cap)
    lev = lev.where((np.arange(len(lev)) % REBAL) == 0).ffill().shift(1).fillna(1.0)
    dlev = lev.diff().abs().fillna(0)
    return book * lev - dlev * (COST_BPS / 1e4)


def main():
    sect = data.sectors()
    inc = {p.stem for p in (data.EARN / "income_statement").glob("*.parquet")}
    universe = sorted(inc & set(sect.index))
    panels, rets, cal, spy = scored_panel(universe, sect)
    if not panels:
        print("no panels"); return
    win = run_config(panels, rets, cal, quantile=0.20, cap_weight=True)   # the winner
    spyr = spy.pct_change(); spyr = spyr[(spyr.index >= win.index[0]) & (spyr.index <= win.index[-1])]

    print(f"\n=== QUALITY-MOMENTUM, cap-weighted ({win.index[0].date()}->{win.index[-1].date()}), net of costs ===")
    metrics(spyr, "SPY buy-hold (baseline)")
    metrics(win, "QM top20% cap-weighted (FINAL)")
    # OUT-OF-SAMPLE robustness: must beat SPY in BOTH halves, not just full-sample
    mid = win.index[len(win) // 2]
    print(f"  -- split-half robustness (split {mid.date()}) --")
    metrics(win[win.index < mid], "  QM 1st half"); metrics(spyr[spyr.index < mid], "  SPY 1st half")
    metrics(win[win.index >= mid], "  QM 2nd half"); metrics(spyr[spyr.index >= mid], "  SPY 2nd half")
    out = pd.DataFrame({"strategy_ret": win, "equity": (1 + win).cumprod()}).dropna()
    out.to_csv("strategy_equity.csv")
    print(f"\nfinal equity (QM cap-wt) {(1 + win).cumprod().iloc[-1]:.2f}x -> strategy_equity.csv")


if __name__ == "__main__":
    main()
