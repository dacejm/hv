"""The book — reconfigured to the VALIDATED, VIABLE strategy.

History: the dollar-neutral ROIC long/short book LOST (Sharpe -0.30; the short leg was toxic --
shorting low-ROIC junk got squeezed). The full P&L sweep (book_backtest.py / strategy.py) showed the
only thing that beats passive net of costs, OUT OF SAMPLE, is a long-only large-cap QUALITY-MOMENTUM
book, cap-weighted:
    full sample   CAGR 17.7% vs SPY 13.5% | Sharpe 0.86 vs 0.78 | maxDD -30% vs -34% | Calmar 0.59 vs 0.40
    1st half      Sharpe 0.97 vs 0.85   |   2nd half  Sharpe 0.75 vs 0.71   (wins both halves)

So the live book is:
  DIRECTION : composite rank of 12-1m MOMENTUM + ROIC (the two most-validated equity factors), long
              the top quintile of the top-500 by market cap. No short leg.
  SIZING    : CAP-WEIGHTED (lean into the larger quality-momentum names -- what actually drove returns).
Honest framing: this is ENHANCED BETA -- most of the return is the equity risk premium; the QM tilt +
cap-weighting lift risk-adjusted return over passive, robustly. Monthly rebalance, large-cap = viable
to trade. Risk: higher vol than SPY (~22%) and concentration in large-cap winners (mega-cap reversal risk).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import data, roc

TOP_MCAP = 500
QUANTILE = 0.20
MOM_SKIP, MOM_LOOK = 21, 252        # 12-1 month momentum


def _shares_latest(sym, asof):
    try:
        e = pd.read_parquet(data.EARN / "balance_sheet_equity" / f"{sym}.parquet")
    except Exception:
        return np.nan
    e = e[e["period"] == "Quarter"].copy()
    e["known_on"] = pd.to_datetime(e["date"]) + data.DEFAULT_REPORT_LAG
    e = e[e["known_on"] <= asof]
    sh = pd.to_numeric(e["shares_outstanding"], errors="coerce").dropna()
    return float(sh.iloc[-1]) if len(sh) else np.nan


def build(universe, asof=None, top_mcap=TOP_MCAP, quantile=QUANTILE) -> dict:
    """Live long-only cap-weighted quality-momentum large-cap book (point-in-time)."""
    asof = pd.Timestamp(asof) if asof else pd.Timestamp.today()
    rows = []
    for sym in universe:
        try:
            px = data.ohlcv(sym)
        except FileNotFoundError:
            continue
        s = px[px["date"] <= asof].set_index("date")["close"]
        if len(s) < MOM_LOOK + 5:
            continue
        mom = float(s.iloc[-MOM_SKIP] / s.iloc[-MOM_LOOK] - 1.0)
        try:
            rc = roc.roic(sym, asof=asof)
        except Exception:
            rc = None
        if rc is None or rc.empty:
            continue
        sh = _shares_latest(sym, asof)
        if not np.isfinite(sh):
            continue
        rows.append({"symbol": sym, "mom": mom, "roic": float(rc.iloc[-1]["roic"]),
                     "mcap": sh * float(s.iloc[-1]), "price": float(s.iloc[-1])})
    df = pd.DataFrame(rows).dropna(subset=["mom", "roic", "mcap"])
    if df.empty:
        return {}
    df = df.nlargest(top_mcap, "mcap")                       # large-cap universe
    df["sig"] = (df["mom"].rank(pct=True) + df["roic"].rank(pct=True)) / 2
    k = max(int(len(df) * quantile), 10)
    book = df.nlargest(k, "sig").copy()
    book["weight"] = (book["mcap"] / book["mcap"].sum()).round(4)   # cap-weighted, long-only
    book["side"] = "LONG"
    book = book.sort_values("weight", ascending=False)
    return {"asof": asof.date().isoformat(), "n_universe": len(df), "n_holdings": len(book),
            "strategy": "long-only large-cap quality-momentum, cap-weighted (monthly)",
            "book": book[["side", "symbol", "weight", "mom", "roic", "mcap"]].reset_index(drop=True)}
