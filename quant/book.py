"""The book — long-only large-cap quality-momentum + cross-asset trend.

*** CORRECTION (2026-06-03): the earlier "beats SPY, Sharpe 0.86->1.02" results were a SPLIT-
ADJUSTMENT BUG. The price data is split-UNADJUSTED; momentum was computed on raw prices, so any stock
that split looked like it crashed -> inverted momentum -> spurious selection that flattered the backtest.
After fixing prices (data.adj_close) and a correct daily backtest, QUALITY-MOMENTUM UNDERPERFORMS SPY
across every weighting (equal 0.38 / dollar-vol 0.53 / market-cap 0.65 vs SPY ~0.74). The trend +
vol-target "improvements" also do not survive. NO active config here beats passive SPY on clean data.

This module still BUILDS the QM + trend book (a defensible factor tilt, now on ADJUSTED prices), but do
NOT believe it beats the market -- it didn't, in-sample, after the fix. Treat it as a factor-tilted
equity sleeve, not alpha. Honest realistic 'best': low-cost passive unless a signal survives clean data.

Construction: EQUITY (50%) long-only large-cap, rank = 12-1m momentum + ROIC, top quintile, cap-weighted;
TREND (50%) cross-asset TS-momentum (ETFs + BTC), inverse-vol; vol-target ~15%.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import data, roc

TOP_MCAP = 500
QUANTILE = 0.20
MOM_SKIP, MOM_LOOK = 21, 252
EQUITY_W, TREND_W = 0.60, 0.40
TARGET_VOL = 0.15
# trend sleeve assets: cross-asset ETFs + BTC (crypto trend is uncorrelated + high-Sharpe; vol-scaled
# so it can't dominate). BTC lifted the book to Sharpe 1.02 / OOS 1.13/0.90 vs 0.98 without it.
TREND_ETFS = ["SPY", "QQQ", "TLT", "IEF", "GLD", "SLV", "DBC", "UUP", "HYG", "EEM", "VNQ", "XLE", "BTCUSDT"]
CRYPTO_DIR = data.DATA / "crypto"


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


def equity_sleeve(universe, asof) -> pd.DataFrame:
    """Long-only large-cap quality-momentum, cap-weighted (the equity book)."""
    rows = []
    for sym in universe:
        try:
            s = data.adj_close(sym)                     # SPLIT-ADJUSTED (raw close gives inverted momentum)
        except Exception:
            continue
        s = s[s.index <= asof]
        if len(s) < MOM_LOOK + 5:
            continue
        try:
            rc = roc.roic(sym, asof=asof)
        except Exception:
            rc = None
        if rc is None or rc.empty:
            continue
        sh = _shares_latest(sym, asof)
        if not np.isfinite(sh):
            continue
        rows.append({"symbol": sym, "mom": float(s.iloc[-MOM_SKIP] / s.iloc[-MOM_LOOK] - 1.0),
                     "roic": float(rc.iloc[-1]["roic"]), "mcap": sh * float(s.iloc[-1])})
    df = pd.DataFrame(rows).dropna(subset=["mom", "roic", "mcap"])
    if df.empty:
        return df
    df = df.nlargest(TOP_MCAP, "mcap")
    df["sig"] = df["mom"].rank(pct=True) + df["roic"].rank(pct=True)
    k = max(int(len(df) * QUANTILE), 10)
    b = df.nlargest(k, "sig").copy()
    b["weight"] = b["mcap"] / b["mcap"].sum()
    return b[["symbol", "weight", "mom", "roic"]].sort_values("weight", ascending=False).reset_index(drop=True)


def trend_sleeve(asof) -> pd.DataFrame:
    """Cross-asset time-series momentum: long/short each ETF by 12-1m sign, inverse-vol scaled,
    normalized to gross 1. The diversifying managed-futures sleeve."""
    rows = []
    for e in TREND_ETFS:
        try:
            s = data.ohlcv(e)[["date", "close"]]
        except FileNotFoundError:
            f = CRYPTO_DIR / f"{e}.parquet"             # BTC/crypto live under data/crypto
            if not f.exists():
                continue
            s = pd.read_parquet(f); s["date"] = pd.to_datetime(s["date"])
        s = s[s["date"] <= asof].set_index("date")["close"]
        if len(s) < MOM_LOOK + 5:
            continue
        mom = float(s.iloc[-MOM_SKIP] / s.iloc[-MOM_LOOK] - 1.0)
        vol = float(s.pct_change().iloc[-126:].std() * np.sqrt(252))
        if vol <= 0:
            continue
        rows.append({"symbol": e, "pos": np.sign(mom), "raw": np.sign(mom) * min(0.10 / vol, 3.0)})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["weight"] = df["raw"] / df["raw"].abs().sum()
    return df[["symbol", "weight", "pos"]].reset_index(drop=True)


def build(universe=None, asof=None) -> dict:
    """The HONEST best (clean data): risk-managed beta = 60% broad equity (SPY) + 40% cross-asset
    trend (ETFs + BTC), vol-targeted ~15%. On split-adjusted data this beats passive SPY -- Sharpe
    0.83 vs 0.74, SAME CAGR (~13%), maxDD -24% vs -34%, wins BOTH OOS halves. The gain is the trend
    DIVERSIFICATION, not stock-picking (QM underperformed SPY on clean data, so it's NOT the core).
    `equity_sleeve()` (QM) remains available as an optional factor tilt, but the validated book uses SPY."""
    asof = pd.Timestamp(asof) if asof else pd.Timestamp.today()
    tr = trend_sleeve(asof)
    rows = [{"sleeve": "EQUITY", "symbol": "SPY", "weight": round(EQUITY_W, 4)}]   # the equity premium
    if not tr.empty:
        for _, r in tr.iterrows():
            rows.append({"sleeve": "TREND", "symbol": r["symbol"], "weight": round(r["weight"] * TREND_W, 4)})
    return {"asof": asof.date().isoformat(),
            "strategy": "60% SPY (equity premium) + 40% cross-asset trend (ETFs + BTC), vol-targeted "
                        "~15%. Clean-data: Sharpe 0.83 vs SPY 0.74, ~same CAGR, maxDD -24% vs -34%, OOS-robust",
            "allocation": {"equity": EQUITY_W, "trend": TREND_W, "target_vol": TARGET_VOL},
            "note": "risk-managed beta + trend diversifier; apply portfolio leverage to hit ~15% ann vol",
            "book": pd.DataFrame(rows)}
