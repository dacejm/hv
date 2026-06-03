"""Screeners replicating Mat's format: momentum, macro-adjusted by sector via regime.

Columns (per his screenshots):
  5d/20d/60d %  trailing returns        RS/SPY = stock 60d ret / SPY 60d ret
  VOL x        volume vs 20d avg        MOM    = cross-sectional z of a momentum blend
  MACRO x      regime sector multiplier SCORE  = MOM x MACRO x
  FIT          IN CYCLE if sector favored in regime (MACRO x >= 1.10) else NEUTRAL

Cap bands: small 0.3-2b, mid 2-10b (separate screens). The MACRO x tilt table below is the
MID_EXPANSION regime read off Mat's screenshots; the regime CLASSIFIER itself is the separate
'regime' plan item (here it's a fixed default) and whether MACRO x actually helps must be MEASURED.
"""
from __future__ import annotations

import calendar
import glob
import json
from collections import defaultdict
from datetime import date

import numpy as np
import pandas as pd

from . import data, regime

OPT_CHAIN = data.DATA / "options data" / "parquet" / "option_chain"
SUMM = data.DATA.parent / "research" / "summaries"

BANDS = {"SMALL ($300M-$2B)": (3e8, 2e9), "MID ($2B-$10B)": (2e9, 1e10)}

# regime -> sector multiplier (MID_EXPANSION read from Mat's screenshots; others approx)
TILTS = {"MID_EXPANSION": {"Technology": 1.30, "Industrials": 1.20, "Basic Materials": 1.10,
                           "Consumer Cyclical": 1.10, "Communication Services": 1.05, "Financial": 1.00,
                           "Real Estate": 0.95, "Utilities": 0.95, "Energy": 0.90, "Consumer Defensive": 0.90,
                           "Healthcare": 0.85, "Unknown": 1.00}}


def _features(symbol, asof, spy_r60):
    try:
        px = data.ohlcv(symbol)
    except FileNotFoundError:
        return None
    if asof is not None:
        px = px[px["date"] <= asof]
    if len(px) < 61:
        return None
    c = px["close"].to_numpy(); v = px["volume"].to_numpy()
    dr = np.diff(c[-61:]) / c[-61:-1]                              # ohlcv is unadjusted -> a split
    if np.nanmax(np.abs(dr)) > 0.5:                               # shows as a >50% 1-day jump: skip artifact
        return None
    r5, r20, r60 = c[-1] / c[-6] - 1, c[-1] / c[-21] - 1, c[-1] / c[-61] - 1
    vol_x = v[-1] / np.mean(v[-20:]) if np.mean(v[-20:]) > 0 else np.nan
    rs = (r60 / spy_r60) if spy_r60 not in (0, np.nan) else np.nan
    bse = _shares(symbol)
    mc = c[-1] * bse if bse else None
    return {"symbol": symbol, "r5": r5, "r20": r20, "r60": r60, "vol_x": vol_x,
            "rs_spy": rs, "mktcap": mc, "price": round(float(c[-1]), 2)}


def _shares(symbol):
    try:
        b = pd.read_parquet(data.EARN / "balance_sheet_equity" / f"{symbol}.parquet")
        b = b[b["period"] == "Quarter"].sort_values("date")
        return float(b.iloc[-1]["shares_outstanding"])
    except Exception:
        return None


def run(universe, asof=None, phase=None, top=15) -> dict:
    asof = pd.Timestamp(asof) if asof else None
    spy = data.benchmark()
    if asof is not None:
        spy = spy[spy.index <= asof]
    spy_r60 = float(spy.iloc[-1] / spy.iloc[-61] - 1)
    sect = data.sectors()
    rinfo = regime.classify(asof)                          # dynamic cycle regime
    ph = phase or rinfo["phase"]
    tilt = regime.sector_tilt(ph)

    rows = [f for s in universe if (f := _features(s, asof, spy_r60)) and f["mktcap"]]
    df = pd.DataFrame(rows)
    df["sector"] = df["symbol"].map(lambda s: sect.get(s, "Unknown"))

    out = {}
    for band, (lo, hi) in BANDS.items():
        b = df[(df["mktcap"] >= lo) & (df["mktcap"] < hi)].dropna(subset=["r5", "r20", "r60"]).copy()
        if b.empty:
            out[band] = b; continue
        blend = 0.2 * b["r5"] + 0.3 * b["r20"] + 0.5 * b["r60"]            # momentum blend
        b["mom"] = ((blend - blend.mean()) / blend.std()).round(3)         # cross-sectional z
        b["macro_x"] = b["sector"].map(lambda s: tilt.get(s, 1.0))
        # SCORE = momentum only. MACRO x / FIT kept as CONTEXT columns (regime label), NOT in the
        # score: measured (regime_macro_test) to add nothing to returns -> regime != directional signal.
        b["score"] = b["mom"].round(3)
        b["fit"] = np.where(b["macro_x"] >= 1.10, "IN CYCLE", "NEUTRAL")
        b["mktcap_b"] = (b["mktcap"] / 1e9).round(2)
        for col in ("r5", "r20", "r60"):
            b[col] = (b[col] * 100).round(1)
        b["rs_spy"] = b["rs_spy"].round(2); b["vol_x"] = b["vol_x"].round(2)
        out[band] = b.sort_values("score", ascending=False).head(top)[
            ["symbol", "sector", "r5", "r20", "r60", "rs_spy", "vol_x", "mom", "macro_x", "score", "fit"]]
    out["_regime"] = {"phase": ph, **{k: rinfo[k] for k in ("risk", "confidence")}}
    return out


def next_opex(asof=None) -> date:
    """Next monthly options expiry (3rd Friday)."""
    d = (pd.Timestamp(asof) if asof else pd.Timestamp.today()).date()
    def third_friday(y, m):
        return [x for x in calendar.Calendar().itermonthdates(y, m)
                if x.month == m and x.weekday() == 4][2]
    o = third_friday(d.year, d.month)
    if o < d:
        m, y = (d.month % 12 + 1), d.year + (1 if d.month == 12 else 0)
        o = third_friday(y, m)
    return o


def _next_earnings(sym, asof):
    try:
        e = pd.read_parquet(data.EARN / "earnings_calendar" / f"{sym}.parquet")
        e["d"] = pd.to_datetime(e["date"])
        cut = pd.Timestamp(asof) if asof else pd.Timestamp.today()
        fut = e[e["d"] >= cut].sort_values("d")
        return fut["d"].iloc[0].date() if not fut.empty else None
    except Exception:
        return None


def _atm_iv(sym, price, asof):
    try:
        ch = pd.read_parquet(OPT_CHAIN / f"{sym}.parquet", columns=["date", "expiration", "strike", "vol"])
        ch["date"] = pd.to_datetime(ch["date"])
        ch = ch[ch["date"] <= (pd.Timestamp(asof) if asof else ch["date"].max())]
        if ch.empty:
            return None
        d = ch[ch["date"] == ch["date"].max()].copy()
        d["dte"] = (pd.to_datetime(d["expiration"]) - d["date"]).dt.days
        w = d[(d["dte"] >= 15) & (d["dte"] <= 45)]
        w = w if not w.empty else d
        return float(w.iloc[(w["strike"] - price).abs().argmin()]["vol"])
    except Exception:
        return None


def opex_screen(universe, asof=None, top=15) -> pd.DataFrame:
    """Setups to scale into via CALLS before the next monthly opex (the BABA pattern):
    pre-opex catalyst (earnings before expiry) + directional momentum + tradeable IV."""
    opex = next_opex(asof)
    today = (pd.Timestamp(asof) if asof else pd.Timestamp.today()).date()
    spy = data.benchmark()
    if asof is not None:
        spy = spy[spy.index <= asof]
    spy_r60 = float(spy.iloc[-1] / spy.iloc[-61] - 1)
    sect = data.sectors()

    rows = []
    for sym in universe:
        f = _features(sym, asof, spy_r60)
        if not f:
            continue
        earn = _next_earnings(sym, asof)
        f.update({"sector": sect.get(sym, "Unknown"), "earnings": earn,
                  "catalyst": bool(earn and today <= earn <= opex),
                  "atm_iv": _atm_iv(sym, f["price"], asof)})
        rows.append(f)
    df = pd.DataFrame(rows).dropna(subset=["atm_iv"])                  # need options to trade
    if df.empty:
        return df
    blend = 0.2 * df["r5"] + 0.3 * df["r20"] + 0.5 * df["r60"]
    df["mom"] = ((blend - blend.mean()) / blend.std()).round(2)
    df["score"] = (df["mom"] * np.where(df["catalyst"], 1.5, 1.0)).round(2)   # boost pre-opex catalysts
    df["call_strike"] = (np.ceil(df["price"] / 5) * 5).astype(int)            # nearest OTM-ish standard strike
    df["dte_opex"] = (opex - today).days
    df = df[df["mom"] > 0]                                             # directional/bullish only
    df["atm_iv"] = df["atm_iv"].round(2)
    return df.sort_values("score", ascending=False).head(top)[
        ["symbol", "sector", "price", "dte_opex", "earnings", "catalyst", "mom", "atm_iv", "score", "call_strike"]]


def _research_view(asof=None, half_life=45):
    """From recency-weighted research summaries: favored SECTOR weights + mentioned tickers (bonus)."""
    sect = data.sectors()
    tickers = set(sect.index)
    cut = (pd.Timestamp(asof) if asof else pd.Timestamp.today()).date()
    sec_w, mentioned = defaultdict(float), defaultdict(float)
    for p in glob.glob(str(SUMM / "*.json")):
        s = json.loads(open(p, encoding="utf-8").read())
        try:
            age = (cut - date.fromisoformat(s.get("date"))).days
        except Exception:
            continue
        if age < 0:
            continue
        w = 0.5 ** (age / half_life)
        if w < 0.05:
            continue
        cands = {str(c).strip().upper() for c in (s.get("companies") or [])
                 if str(c).strip().isalpha() and 1 <= len(str(c).strip()) <= 5}
        for t in cands & tickers:
            mentioned[t] += w
            sec_w[sect.get(t, "Unknown")] += w
    return sec_w, mentioned


def thesis_screen(universe, asof=None, top=20, n_sectors=4) -> tuple:
    """Names IN the thesis's favored sectors, ranked by momentum; research-mentioned = bonus flag."""
    sec_w, mentioned = _research_view(asof)
    favored = [s for s, _ in sorted(sec_w.items(), key=lambda x: -x[1]) if s != "Unknown"][:n_sectors]
    sect = data.sectors()
    spy = data.benchmark()
    if asof is not None:
        spy = spy[spy.index <= asof]
    spy_r60 = float(spy.iloc[-1] / spy.iloc[-61] - 1)

    rows = []
    for sym in universe:
        if sect.get(sym, "Unknown") not in favored:               # SECTOR is the driver
            continue
        f = _features(sym, asof, spy_r60)
        if not f:
            continue
        f.update({"sector": sect.get(sym), "mentioned": sym in mentioned,
                  "mention_w": round(mentioned.get(sym, 0.0), 2)})
        rows.append(f)
    df = pd.DataFrame(rows)
    if df.empty:
        return favored, df
    blend = 0.2 * df["r5"] + 0.3 * df["r20"] + 0.5 * df["r60"]
    df["mom"] = ((blend - blend.mean()) / blend.std()).round(2)
    df["score"] = (df["mom"] * np.where(df["mentioned"], 1.2, 1.0)).round(2)   # mentioned = bonus
    return favored, df.sort_values("score", ascending=False).head(top)[
        ["symbol", "sector", "mom", "mentioned", "mention_w", "score"]]

