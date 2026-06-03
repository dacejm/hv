"""Multi-source data layer -- redundancy + bad-data detection (the stated goal; also the audit's
silent-failure defense). A second source is only useful if you reconcile it against the first.

The new downloadable history (DoltHub single-name options, via fetch_options_history.py) overlaps
the local index chains on SPY/QQQ/IWM. cross_check_options() reconciles them: if DoltHub's IV
matches the local index IV on the same (date, expiration, strike), the new source is trustworthy
to extend to single names. Divergence = a flag to investigate before relying on it.
"""
from __future__ import annotations

import io
import json
import os
import urllib.request

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from . import data, gex


def cross_check_all(symbol: str = "AAPL", year: int = 2023) -> dict:
    """Run every stream's cross-check at once -- the full multi-source integrity sweep:
    prices (vs Alpha Vantage), fundamentals (vs SEC EDGAR), options IV (DoltHub vs local index),
    rates/macro (vs FRED). Each stream reconciled against an independent source."""
    return {"prices": cross_check_prices(symbol),
            "fundamentals": cross_check_fundamentals(symbol, year),
            "options_iv": cross_check_options("SPY"),
            "rates": cross_check_rates()}


def cross_check_options(symbol: str = "SPY", tol: float = 0.03) -> dict:
    """Reconcile DoltHub single-name option IV (data/options_history) against the local index
    chain (data/QQQ_SPY_IWM) on matched (date, expiration, strike, type). Reports overlap size,
    median |IV_dolthub - IV_local|, and the rank correlation. Pass = sources agree within tol."""
    dolt = data.options_chain(symbol)
    if dolt.empty:
        return {"status": "no DoltHub history downloaded for " + symbol}
    loc = pq.read_table(gex.IDX / f"{symbol}_options.parquet",
                        columns=["date", "type", "strike", "expiration", "implied_volatility"]).to_pandas()
    loc["date"] = pd.to_datetime(loc["date"]).dt.normalize()
    loc["expiration"] = pd.to_datetime(loc["expiration"]).dt.normalize()
    loc["type"] = loc["type"].astype(str).str.lower().str[0]
    d = dolt.copy(); d["type"] = d["type"].str[0]
    key = ["date", "expiration", "strike", "type"]
    m = d.merge(loc, on=key, suffixes=("_dolt", "_loc")).dropna(
        subset=["implied_volatility_dolt", "implied_volatility_loc"])
    m = m[(m["implied_volatility_dolt"] > 0) & (m["implied_volatility_loc"] > 0)]
    if len(m) < 20:
        return {"status": f"insufficient overlap ({len(m)} matched contracts) -- "
                          "check the downloaded date range overlaps the local index data"}
    diff = (m["implied_volatility_dolt"] - m["implied_volatility_loc"]).abs()
    corr = m["implied_volatility_dolt"].corr(m["implied_volatility_loc"], method="spearman")
    med = float(diff.median())
    return {"symbol": symbol, "matched_contracts": int(len(m)),
            "median_abs_IV_diff": round(med, 4), "rank_corr": round(float(corr), 3),
            "agree_within_tol": bool(med <= tol),
            "verdict": "sources agree -> DoltHub history trustworthy" if med <= tol
                       else "DIVERGENCE -> investigate before relying on the new source"}


SEC_FRAMES = data.DATA / "sec_frames"


def _ticker_cik() -> dict:
    """Ticker -> CIK map from SEC (cached). Free, needs a descriptive User-Agent per SEC policy."""
    cache = SEC_FRAMES / "_ticker_cik.json"
    if cache.exists():
        return json.load(open(cache))
    try:
        req = urllib.request.Request("https://www.sec.gov/files/company_tickers.json",
                                     headers={"User-Agent": "hv-research dacejm4@gmail.com"})
        raw = json.load(urllib.request.urlopen(req, timeout=30))
        m = {v["ticker"].upper(): int(v["cik_str"]) for v in raw.values()}
        json.dump(m, open(cache, "w"))
        return m
    except Exception:
        return {}


def cross_check_fundamentals(symbol: str, year: int = 2023, tol: float = 0.05) -> dict:
    """Reconcile DoltHub balance sheet against SEC EDGAR XBRL frames (second source) on the
    ROIC inputs -- total assets and stockholders' equity -- for fiscal year-end CY{year}Q4.
    These feed invested_capital -> ROIC (the flagship DIRECTION signal), so validating them
    matters most. Small relative diff => the fundamental source is trustworthy."""
    cik = _ticker_cik().get(symbol.upper())
    if cik is None:
        return {"status": f"no CIK for {symbol}"}
    bs = data.balance_sheet(symbol)
    if bs.empty:
        return {"status": f"no DoltHub balance sheet for {symbol}"}
    out = {"symbol": symbol, "cik": cik, "year": year}
    checks = {"total_assets": "Assets", "total_equity": "StockholdersEquity"}
    diffs = []
    for dolt_col, sec_tag in checks.items():
        f = SEC_FRAMES / f"{sec_tag}_CY{year}Q4I.json"
        if not f.exists():
            continue
        rec = next((d for d in json.load(open(f))["data"] if d["cik"] == cik), None)
        if rec is None:
            continue
        sec_val, end = float(rec["val"]), pd.Timestamp(rec["end"])
        row = bs.iloc[(bs["period_end"] - end).abs().argmin()]
        if abs((row["period_end"] - end).days) > 45:
            continue
        dolt_val = float(row[dolt_col]) * 1000 if abs(row[dolt_col]) < sec_val / 100 else float(row[dolt_col])
        rel = abs(dolt_val - sec_val) / sec_val if sec_val else np.nan
        diffs.append(rel)
        out[dolt_col] = {"dolthub": dolt_val, "sec": sec_val, "rel_diff": round(rel, 4)}
    if not diffs:
        return {**out, "status": "no overlapping SEC concept/period"}
    med = float(np.median(diffs))
    out["median_rel_diff"] = round(med, 4)
    out["agree_within_tol"] = bool(med <= tol)
    out["verdict"] = "fundamentals agree -> ROIC inputs trustworthy" if med <= tol \
        else "DIVERGENCE -> check units/scaling or filing mismatch"
    return out


def cross_check_rates(tol: float = 0.05) -> dict:
    """Reconcile the local us_treasury 10y against FRED DGS10 (second source) on overlapping
    recent dates. Median |yield diff| in points; small => the rates/macro source agrees."""
    from . import regime
    loc = pd.read_parquet(data.DATA / "rates" / "parquet" / "us_treasury.parquet",
                          columns=["date", "10_year"]).dropna()
    loc["date"] = pd.to_datetime(loc["date"])
    fred = regime._fred_series("DGS10")
    if fred.empty:
        return {"status": "FRED DGS10 unavailable (set FRED_API_KEY / cache)"}
    m = loc.merge(fred.rename("dgs10").reset_index().rename(columns={"index": "date"}), on="date")
    m = m.tail(500)
    if len(m) < 20:
        return {"status": f"insufficient overlap ({len(m)})"}
    diff = (m["10_year"] - m["dgs10"]).abs()
    med = float(diff.median())
    return {"overlap_days": int(len(m)), "median_abs_yield_diff": round(med, 4),
            "agree_within_tol": bool(med <= tol),
            "verdict": "rates agree" if med <= tol else "DIVERGENCE -> check rates source"}


def cross_check_prices(symbol: str, tol: float = 0.005) -> dict:
    """Reconcile local DoltHub ohlcv against a SECOND source (Alpha Vantage free TIME_SERIES_DAILY)
    -- the multi-source redundancy / silent-failure defense. Compares the overlapping ~100 days of
    daily returns; small median |return diff| => sources agree. Needs ALPHAVANTAGE_API_KEY in env."""
    key = os.environ.get("ALPHAVANTAGE_API_KEY", "")
    if not key:
        return {"status": "no ALPHAVANTAGE_API_KEY in env"}
    try:
        u = (f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}"
             f"&outputsize=compact&apikey={key}")
        ts = json.load(urllib.request.urlopen(u, timeout=30)).get("Time Series (Daily)", {})
    except Exception as e:
        return {"status": f"AV fetch failed: {e}"}
    if not ts:
        return {"status": "AV returned no data (rate limit / bad symbol)"}
    av = pd.DataFrame({"date": pd.to_datetime(list(ts)),
                       "av_close": [float(v["4. close"]) for v in ts.values()]}).sort_values("date")
    loc = data.ohlcv(symbol)[["date", "close"]].rename(columns={"close": "loc_close"})
    m = loc.merge(av, on="date").sort_values("date")
    if len(m) < 20:
        return {"status": f"insufficient overlap ({len(m)} days)"}
    rdiff = (m["loc_close"].pct_change() - m["av_close"].pct_change()).abs().dropna()
    med = float(rdiff.median())
    return {"symbol": symbol, "overlap_days": int(len(m)),
            "median_abs_return_diff": round(med, 5),
            "level_corr": round(float(m["loc_close"].corr(m["av_close"])), 4),
            "agree_within_tol": bool(med <= tol),
            "verdict": "sources agree" if med <= tol else "DIVERGENCE -> check local data integrity"}


def vol_surface(symbol: str, asof=None) -> dict:
    """Single-name vol-surface read from the DoltHub history (data.options_asof). DoltHub carries
    only near-dated expirations (<=~50 DTE), so this gives the NEAR-CURVE slope (short vs ~6wk),
    skew (RR25) and ATM IV -- enough for skew/RND work. The 1m-vs-3m (120 DTE) slope needs a paid
    chain provider; reported here as the near-curve only, honestly labelled."""
    d = data.options_asof(symbol, asof)
    if d.empty:
        return {"status": f"no DoltHub options history for {symbol} (run fetch_options_history.py)"}
    px = data.ohlcv(symbol)
    if asof is not None:
        px = px[px["date"] <= pd.Timestamp(asof)]
    spot = float(px["close"].iloc[-1])
    asof_ts = d["date"].max()
    d = d[d["implied_volatility"] > 0].assign(dte=(d["expiration"] - asof_ts).dt.days)
    calls = d[d["type"].str.startswith("c")]

    def atm(lo, hi):
        w = calls[(calls["dte"] >= lo) & (calls["dte"] <= hi)]
        return float(w.iloc[(w["strike"] - spot).abs().argmin()]["implied_volatility"]) if not w.empty else np.nan
    near, mid = atm(7, 21), atm(30, 50)                       # near-curve (both within DoltHub's range)
    nr = d[(d["dte"] >= 20) & (d["dte"] <= 50) & d["delta"].notna()]
    cc, pp = nr[nr["type"].str.startswith("c")], nr[nr["type"].str.startswith("p")]
    rr25 = (float(cc.iloc[(cc["delta"] - 0.25).abs().argmin()]["implied_volatility"])
            - float(pp.iloc[(pp["delta"] + 0.25).abs().argmin()]["implied_volatility"])) \
        if not cc.empty and not pp.empty else np.nan
    return {"symbol": symbol, "asof": str(asof_ts.date()), "spot": round(spot, 2),
            "near_iv": round(near, 3), "mid_iv": round(mid, 3),
            "near_curve_slope": round(mid - near, 3) if pd.notna(near) and pd.notna(mid) else None,
            "rr25": round(rr25, 4) if pd.notna(rr25) else None,
            "max_dte": int(d["dte"].max()), "note": "near-curve only; 120-DTE slope needs paid chains"}

