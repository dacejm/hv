"""ACTUAL P&L backtest of the validated book — not IC, the strategy itself.

Strategy (the live book's logic, point-in-time):
  - DIRECTION: ROIC, sector-neutralized (within-sector percentile rank). Long the top quintile,
    short the bottom quintile -> dollar-neutral long/short. Quarterly rebalance (ROIC's 63d horizon).
  - Each ROIC value is only used after its filing knowledge date (period_end + report lag) -> no lookahead.
  - SIZING: report the raw dollar-neutral book AND a vol-targeted variant (scale to 10% ann vol on
    trailing realized). Costs charged on turnover each rebalance.
Outputs: equity curve + CAGR / ann vol / Sharpe / maxDD / Calmar / hit-rate, gross and net of costs,
with SPY for reference (the book is market-neutral, so its return is the alpha, not beta).

  python book_backtest.py            # writes book_backtest.out + equity csv
"""
from __future__ import annotations

import sys
from collections import defaultdict

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from quant import data

REBAL = 63                  # trading days (~quarter) — ROIC's validated horizon
N_QUINTILE = 0.20           # long top 20% / short bottom 20% of the scored universe
MIN_SECTOR = 5              # need peers to sector-neutralize
TARGET_VOL = 0.10
COST_BPS = 10               # per side, per name, per rebalance (mid/small-cap realistic)
START = "2016-01-01"


def _shares(sym):
    """Point-in-time shares-outstanding series (period_end+lag -> known) for market cap."""
    try:
        e = pd.read_parquet(data.EARN / "balance_sheet_equity" / f"{sym}.parquet")
    except Exception:
        return None
    e = e[e["period"] == "Quarter"].copy()
    e["known_on"] = pd.to_datetime(e["date"]) + data.DEFAULT_REPORT_LAG
    e["sh"] = pd.to_numeric(e["shares_outstanding"], errors="coerce")
    e = e.dropna(subset=["sh"]).sort_values("known_on")
    return e[["known_on", "sh"]] if not e.empty else None


def _precompute(universe):
    """Per symbol: ROIC time series, daily close + return series, close levels, shares (for mcap)."""
    roic_ts, closes, levels, shares = {}, {}, {}, {}
    from quant import roc
    for sym in universe:
        try:
            r = roc.roic(sym)
            px = data.ohlcv(sym)
        except Exception:
            continue
        if r.empty or px.empty:
            continue
        r = r.copy()
        r["known_on"] = pd.to_datetime(r["period_end"]) + data.DEFAULT_REPORT_LAG
        roic_ts[sym] = r[["known_on", "roic"]].sort_values("known_on")
        s = px.set_index("date")["close"]
        ret = s.pct_change()
        ret[ret.abs() > 0.5] = np.nan          # drop split-artifact days
        closes[sym] = ret
        levels[sym] = s
        sh = _shares(sym)
        if sh is not None:
            shares[sym] = sh
    return roic_ts, closes, levels, shares


def _mcap_at(levels, shares, sym, D):
    """Market cap = latest-known shares * price at/just-before D (point-in-time). None if missing."""
    if sym not in shares or sym not in levels:
        return None
    sh = shares[sym]; sh = sh[sh["known_on"] <= D]
    px = levels[sym]; px = px[px.index <= D]
    if sh.empty or px.empty:
        return None
    return float(sh.iloc[-1]["sh"]) * float(px.iloc[-1])


def backtest(universe, sectors, top_mcap=None):
    roic_ts, rets, levels, shares = _precompute(universe)
    syms = list(roic_ts)
    print(f"universe with ROIC + prices: {len(syms)}"
          + (f" | restricting to top {top_mcap} by market cap each rebalance" if top_mcap else ""))
    spy = data.benchmark()
    cal = spy.index[spy.index >= pd.Timestamp(START)]
    rebal_dates = cal[::REBAL]

    daily, long_daily, short_daily, uni_daily = {}, {}, {}, {}   # book + leg decomposition
    prev_holds = set()
    turnover_cost = defaultdict(float)
    for i, D in enumerate(rebal_dates[:-1]):
        # cross-section of latest-known ROIC at D
        rows = []
        for s in syms:
            rt = roic_ts[s]
            known = rt[rt["known_on"] <= D]
            if known.empty:
                continue
            sec = sectors.get(s, "Unknown")
            if sec == "Unknown":
                continue
            rows.append((s, sec, float(known.iloc[-1]["roic"])))
        if len(rows) < 50:
            continue
        df = pd.DataFrame(rows, columns=["sym", "sector", "roic"])
        if top_mcap:                              # large-cap restriction (point-in-time market cap)
            df["mcap"] = [_mcap_at(levels, shares, s, D) for s in df["sym"]]
            df = df.dropna(subset=["mcap"]).nlargest(top_mcap, "mcap")
        vc = df["sector"].value_counts()
        df = df[df["sector"].isin(vc[vc >= MIN_SECTOR].index)]
        # sector-neutral score = within-sector percentile rank, centered
        df["score"] = df.groupby("sector")["roic"].rank(pct=True) - 0.5
        df = df.sort_values("score")
        k = max(int(len(df) * N_QUINTILE), 5)
        shorts, longs = set(df.head(k)["sym"]), set(df.tail(k)["sym"])

        # cost on the names that changed since last rebalance
        holds = longs | shorts
        turnover_cost[D] = len(holds ^ prev_holds) * COST_BPS / 1e4 / max(len(holds), 1)
        prev_holds = holds

        # daily dollar-neutral return over [D, next rebal]: mean(long) - mean(short)
        end = rebal_dates[i + 1]
        win = cal[(cal > D) & (cal <= end)]
        scored = set(df["sym"])
        for dt in win:
            lr = np.nanmean([rets[s].get(dt, np.nan) for s in longs])
            sr = np.nanmean([rets[s].get(dt, np.nan) for s in shorts])
            ur = np.nanmean([rets[s].get(dt, np.nan) for s in scored])    # universe (all scored names)
            if np.isfinite(lr) and np.isfinite(sr):
                daily[dt] = (lr - sr) / 2.0           # /2 so gross exposure = 1 (0.5 long + 0.5 short)
            if np.isfinite(lr):
                long_daily[dt] = lr
            if np.isfinite(sr):
                short_daily[dt] = sr
            if np.isfinite(ur):
                uni_daily[dt] = ur
        # charge the rebalance cost on the first day of the window
        if len(win):
            daily[win[0]] = daily.get(win[0], 0.0) - turnover_cost[D]

    legs = {"long_q": pd.Series(long_daily).sort_index(),
            "short_q": pd.Series(short_daily).sort_index(),
            "universe": pd.Series(uni_daily).sort_index()}
    return pd.Series(daily).sort_index(), spy, legs


def metrics(r, label):
    r = r.dropna()
    if len(r) < 20:
        return None
    eq = (1 + r).cumprod()
    yrs = len(r) / 252
    cagr = eq.iloc[-1] ** (1 / yrs) - 1
    vol = r.std() * np.sqrt(252)
    sharpe = r.mean() / r.std() * np.sqrt(252) if r.std() else np.nan
    dd = (eq / eq.cummax() - 1).min()
    calmar = cagr / abs(dd) if dd < 0 else np.nan
    hit = (r > 0).mean()
    print(f"{label:24} CAGR {cagr:+.2%} | vol {vol:.1%} | Sharpe {sharpe:.2f} | "
          f"maxDD {dd:.1%} | Calmar {calmar:.2f} | daily hit {hit:.0%}")
    return eq


def main(top_mcap=None):
    sect = data.sectors()
    inc = {p.stem for p in (data.EARN / "income_statement").glob("*.parquet")}
    universe = sorted(inc & set(sect.index))
    r, spy, legs = backtest(universe, sect, top_mcap=top_mcap)
    if r.empty:
        print("no returns"); return
    tag = f"LARGE-CAP top{top_mcap}" if top_mcap else "FULL universe"
    print(f"\n=== BOOK BACKTEST [{tag}] (ROIC L/S, dollar-neutral, quarterly, {r.index[0].date()}->{r.index[-1].date()}) ===")
    eq_gross = metrics(r, "gross (pre-cost incl.)")        # costs already in r; this is net
    # vol-targeted variant: scale by target / trailing 63d realized vol
    tv = TARGET_VOL / (r.rolling(63).std() * np.sqrt(252)).shift(1)
    rv = (r * tv.clip(upper=3)).dropna()
    metrics(rv, "vol-targeted (10%)")
    spyr = spy.pct_change()
    spyr = spyr[(spyr.index >= r.index[0]) & (spyr.index <= r.index[-1])]
    metrics(spyr, "SPY (reference)")
    eq = metrics(r, "book (net of costs)")
    print("\n  -- leg decomposition (where is the P&L?) --")
    metrics(legs["long_q"], "long quintile (raw)")
    metrics(legs["short_q"], "short quintile (raw)")
    metrics(legs["universe"], "universe mean (raw)")
    metrics((legs["long_q"] - legs["universe"]).dropna(), "long alpha (long - universe)")
    metrics((legs["universe"] - legs["short_q"]).dropna(), "short alpha (universe - short)")
    out = pd.DataFrame({"book_ret": r, "book_equity": (1 + r).cumprod()})
    out.to_csv("book_equity.csv")
    print(f"\nequity curve -> book_equity.csv | final equity {(1+r).cumprod().iloc[-1]:.2f}x | "
          f"cost drag total {sum(1 for _ in r):d} days")
    print("note: market-NEUTRAL book -> its return IS the alpha; SPY shown only for context. "
          "Costs at 10bps/side/rebalance; survivorship mitigated (delisted names kept if in universe).")


if __name__ == "__main__":
    main(top_mcap=int(sys.argv[1]) if len(sys.argv) > 1 else None)
