"""Macro cycle-regime classifier -> drives the screener's MACRO x sector tilt.

Rules-based, transparent (a proper HMM is the upgrade). State variables, all point-in-time:
  curve  = 10y - 2y treasury slope   (steep -> early; flat/inverted -> late)
  growth = SPY 126d return            (>0 expansion, <0 contraction)
  risk   = SPY vs 200dma + realized vol  (ON/OFF)
Phases: EARLY_EXPANSION / MID_EXPANSION / LATE_EXPANSION / CONTRACTION, with confidence.
Each phase carries a sector-tilt table (MID_EXPANSION = read off Mat's screenshot; others =
standard cycle rotation). Whether the tilt actually HELPS is measured separately.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import data

RATES = data.DATA / "rates" / "parquet" / "us_treasury.parquet"

TILTS = {
    "EARLY_EXPANSION": {"Financial": 1.30, "Consumer Cyclical": 1.25, "Industrials": 1.20,
                        "Technology": 1.15, "Real Estate": 1.10, "Basic Materials": 1.05,
                        "Communication Services": 1.0, "Energy": 0.95, "Utilities": 0.85,
                        "Consumer Defensive": 0.85, "Healthcare": 0.90, "Unknown": 1.0},
    "MID_EXPANSION": {"Technology": 1.30, "Industrials": 1.20, "Basic Materials": 1.10,
                      "Consumer Cyclical": 1.10, "Communication Services": 1.05, "Financial": 1.00,
                      "Real Estate": 0.95, "Utilities": 0.95, "Energy": 0.90,
                      "Consumer Defensive": 0.90, "Healthcare": 0.85, "Unknown": 1.0},
    "LATE_EXPANSION": {"Energy": 1.30, "Basic Materials": 1.20, "Consumer Defensive": 1.10,
                       "Healthcare": 1.10, "Utilities": 1.05, "Financial": 1.05, "Industrials": 1.0,
                       "Technology": 0.90, "Consumer Cyclical": 0.85, "Communication Services": 0.90,
                       "Real Estate": 0.95, "Unknown": 1.0},
    "CONTRACTION": {"Consumer Defensive": 1.30, "Utilities": 1.25, "Healthcare": 1.20,
                    "Communication Services": 1.05, "Technology": 1.0, "Financial": 0.90,
                    "Industrials": 0.85, "Consumer Cyclical": 0.85, "Energy": 0.90,
                    "Basic Materials": 0.90, "Real Estate": 0.85, "Unknown": 1.0}}


EC = data.DATA / "economic_calendar_US_EU_2015_today (1).parquet"
FREDDIR = data.DATA / "fred"

# Macro & Micro Frameworks (Book 06) four-quadrant: growth (accel/decel) x inflation
# (rising/falling). Each quadrant favors different RISK PREMIA (Alternative Risk Premia
# Book 16 sec.11), tilted MODESTLY (20-30%) because regime calls are imprecise. This is the
# 2nd axis the growth-only `phase` cycle was missing. Premia, not sectors: carry/value/
# momentum/defensive (incl. their vol/commodity expressions where applicable).
PREMIA_TILT = {
    "RECOVERY":    {"value": 1.25, "momentum": 1.20, "equity_carry": 1.15,
                    "defensive": 0.85, "bond_carry": 0.90},   # Q1 growth^ inflation v/low
    "OVERHEAT":    {"momentum": 1.25, "commodity_carry": 1.25, "value": 1.05,
                    "defensive": 0.85, "bond_carry": 0.80},   # Q2 growth^ inflation^
    "STAGFLATION": {"commodity_carry": 1.30, "fx_carry": 1.10, "defensive": 1.05,
                    "momentum": 0.80, "value": 0.85},         # Q3 growth v inflation^
    "DEFLATION":   {"defensive": 1.30, "bond_carry": 1.25, "fx_carry": 1.10,
                    "equity_carry": 0.80, "value": 0.85},     # Q4 growth v inflation v
}


_FRED_DEAD = set()   # sids that failed this session -> don't retry (avoids stalling)


def _fred(sid, asof=None):
    """FRED series (cached). Uses the authenticated API (FRED_API_KEY env) which works where
    the public CSV gets blocked; falls back to CSV if no key. Latest value <= asof, else None."""
    import io
    import json
    import os
    import urllib.request
    FREDDIR.mkdir(exist_ok=True)
    cache = FREDDIR / f"{sid}.csv"
    if not cache.exists():
        if sid in _FRED_DEAD:
            return None
        try:
            key = os.environ.get("FRED_API_KEY", "")
            if key:
                url = (f"https://api.stlouisfed.org/fred/series/observations?series_id={sid}"
                       f"&api_key={key}&file_type=json")
                d = json.load(urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "hv"}), timeout=15))
                df = pd.DataFrame(d["observations"])[["date", "value"]].rename(columns={"value": "val"})
            else:
                url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}"
                raw = urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"}), timeout=8).read().decode()
                df = pd.read_csv(io.StringIO(raw)); df.columns = ["date", "val"]
            tmp = cache.with_suffix(".csv.tmp")          # atomic write: a killed/timed-out fetch
            df.to_csv(tmp, index=False)                  # leaves a .tmp, never a half-written cache
            os.replace(tmp, cache)
        except Exception:
            _FRED_DEAD.add(sid); return None
    try:
        df = pd.read_csv(cache); df.columns = ["date", "val"]
        if df.empty:
            raise ValueError("empty cache")
    except Exception:                                    # corrupt/0-byte cache -> drop it, refetch next run
        cache.unlink(missing_ok=True); _FRED_DEAD.add(sid); return None
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["val"] = pd.to_numeric(df["val"], errors="coerce")
    df = df.dropna()
    if asof is not None:
        df = df[df["date"] <= pd.Timestamp(asof)]
    return float(df["val"].iloc[-1]) if len(df) else None


def _fred_series(sid, asof=None) -> pd.Series:
    """Full cached FRED series as a date-indexed Series (<= asof). Populates cache via _fred."""
    cache = FREDDIR / f"{sid}.csv"
    if not cache.exists():
        _fred(sid, asof)                                  # triggers the download+cache
    if not cache.exists():
        return pd.Series(dtype=float)
    df = pd.read_csv(cache); df.columns = ["date", "val"]
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["val"] = pd.to_numeric(df["val"], errors="coerce")
    df = df.dropna().sort_values("date")
    if asof is not None:
        df = df[df["date"] <= pd.Timestamp(asof)]
    return df.set_index("date")["val"]


def _inflation_axis(asof=None) -> dict:
    """Inflation DIRECTION (rising/falling), the 4-quadrant's 2nd axis. Market-based:
    10y breakeven inflation (FRED T10YIE), ~3-month change. Falls back to Core CPI YoY
    momentum from the economic calendar when FRED is unreachable."""
    be = _fred_series("T10YIE", asof)
    if len(be) > 70:
        latest = float(be.iloc[-1])
        prior = float(be.iloc[-64]) if len(be) > 64 else float(be.iloc[0])
        chg = latest - prior
        return {"breakeven": round(latest, 2), "breakeven_chg_3m": round(chg, 2),
                "inflation": "rising" if chg > 0.05 else "falling" if chg < -0.05 else "flat",
                "src": "T10YIE"}
    # fallback: Core CPI MoM actuals trend (recent 3 vs prior 3)
    try:
        ec = pd.read_parquet(EC)
        c = ec[ec["event"].astype(str).str.startswith("Core CPI (MoM)", na=False)].copy()
        c["dt"] = pd.to_datetime(c["datetime"]).dt.tz_localize(None)
        if asof is not None:
            c = c[c["dt"] <= pd.Timestamp(asof)]
        v = pd.to_numeric(c.sort_values("dt")["actual"].astype(str).str.rstrip("%"),
                          errors="coerce").dropna()
        if len(v) >= 6:
            chg = v.tail(3).mean() - v.iloc[-6:-3].mean()
            return {"breakeven": None, "breakeven_chg_3m": round(float(chg), 2),
                    "inflation": "rising" if chg > 0.02 else "falling" if chg < -0.02 else "flat",
                    "src": "CoreCPI_MoM"}
    except Exception:
        pass
    return {"breakeven": None, "breakeven_chg_3m": None, "inflation": "unknown", "src": None}


def _latest_pmi(asof=None):
    try:
        ec = pd.read_parquet(EC)
        ec = ec[ec["event"].str.startswith("ISM Manufacturing PMI", na=False)].copy()
        ec["dt"] = pd.to_datetime(ec["datetime"]).dt.tz_localize(None)
        if asof is not None:
            ec = ec[ec["dt"] <= pd.Timestamp(asof)]
        ec = ec.sort_values("dt")
        return float(str(ec["actual"].iloc[-1])) if len(ec) else None
    except Exception:
        return None


def classify(asof=None) -> dict:
    """Macro cycle regime (Mat's framework): yield curve (2s10s, 3m10y) + ISM PMI + growth,
    plus credit (HY) & real rate (TIPS) from FRED when reachable. Recovery/Expansion/
    Slowdown/Contraction. Regime, not forecast (paper §1)."""
    r = pd.read_parquet(RATES, columns=["date", "3_month", "2_year", "10_year"]).sort_values("date")
    r["date"] = pd.to_datetime(r["date"])
    spy = data.benchmark()
    if asof is not None:
        r = r[r["date"] <= pd.Timestamp(asof)]
        spy = spy[spy.index <= pd.Timestamp(asof)]
    c2s10 = float(r["10_year"].iloc[-1] - r["2_year"].iloc[-1])
    c3m10 = float(r["10_year"].iloc[-1] - r["3_month"].iloc[-1])
    growth = float(spy.iloc[-1] / spy.iloc[-127] - 1) if len(spy) > 127 else 0.0
    pmi = _latest_pmi(asof)
    hy = _fred("BAMLH0A0HYM2", asof)        # high-yield OAS (credit risk appetite)
    real_rate = _fred("DFII10", asof)       # 10y TIPS real rate

    pmi_exp = pmi is None or pmi >= 50      # treat missing PMI as neutral-expansion
    credit_stress = hy is not None and hy > 5.5
    if (pmi is not None and pmi < 50 and (c2s10 < 0 or growth < 0)) or (credit_stress and growth < 0):
        phase = "CONTRACTION"
    elif c2s10 < 0.2 and pmi_exp:           # flat/inverted curve, still growing -> late
        phase = "LATE_EXPANSION"
    elif c2s10 > 1.2 and pmi_exp and growth > 0:
        phase = "EARLY_EXPANSION"           # steep curve, recovery
    else:
        phase = "MID_EXPANSION"

    votes = [growth > 0, pmi_exp, c2s10 > 0, not credit_stress]
    agree = sum(votes) if phase != "CONTRACTION" else 4 - sum(votes)

    # ---- second axis: inflation direction -> Book 06 four-quadrant ----------------------
    infl = _inflation_axis(asof)
    growth_up = (growth > 0) and pmi_exp and not credit_stress      # growth accelerating/strong
    infl_up = infl["inflation"] == "rising"
    if growth_up:
        quadrant = "OVERHEAT" if infl_up else "RECOVERY"            # Q2 vs Q1
    else:
        quadrant = "STAGFLATION" if infl_up else "DEFLATION"        # Q3 vs Q4

    return {"phase": phase, "quadrant": quadrant,
            "risk": "ON" if (growth > 0 and not credit_stress) else "OFF",
            "confidence": round(min(0.5 + 0.12 * agree, 0.95), 2),
            "curve_2s10s": round(c2s10, 2), "curve_3m10y": round(c3m10, 2),
            "pmi": pmi, "growth_6m": round(growth, 3),
            "inflation": infl["inflation"], "breakeven": infl["breakeven"],
            "breakeven_chg_3m": infl["breakeven_chg_3m"],
            "hy_oas": hy, "real_rate": real_rate}


def sector_tilt(phase: str) -> dict:
    return TILTS.get(phase, TILTS["MID_EXPANSION"])


def premia_tilt(quadrant: str) -> dict:
    """Risk-premia tilt for the four-quadrant regime (Book 16 sec.11). CONTEXT only --
    modest (20-30%) tilts to a diversified premia base, never a directional signal."""
    return PREMIA_TILT.get(quadrant, {})


# ---- 2-state Gaussian HMM vol-regime (quant-methods paper spec) -------------------------
# hmmlearn won't build here (no compiler), so Baum-Welch EM directly. States = low-vol /
# high-vol on daily returns; the filtered high-vol probability is the regime signal.
def _vol_hmm(r, n_iter=60):
    from scipy.special import logsumexp
    from scipy.stats import norm
    r = np.asarray(r, float) * 100.0                         # to % for conditioning
    T = len(r)
    mu = np.array([r.mean(), r.mean()])
    med = np.median(np.abs(r))
    var = np.array([np.var(r[np.abs(r) < med]) + 1e-6, np.var(r) * 2 + 1e-6])  # low, high
    pi = np.array([0.5, 0.5]); A = np.array([[0.97, 0.03], [0.03, 0.97]])
    gamma = None
    for _ in range(n_iter):
        logB = np.column_stack([norm.logpdf(r, mu[k], np.sqrt(var[k])) for k in (0, 1)])
        logA = np.log(A)
        la = np.zeros((T, 2)); la[0] = np.log(pi) + logB[0]
        for t in range(1, T):
            la[t] = logB[t] + logsumexp(la[t - 1][:, None] + logA, axis=0)
        lb = np.zeros((T, 2))
        for t in range(T - 2, -1, -1):
            lb[t] = logsumexp(logA + logB[t + 1] + lb[t + 1], axis=1)
        ll = logsumexp(la[-1])
        gamma = np.exp(la + lb - ll)
        xi = np.zeros((2, 2))
        for t in range(T - 1):
            xi += np.exp(la[t][:, None] + logA + logB[t + 1][None, :] + lb[t + 1][None, :] - ll)
        pi = gamma[0] / gamma[0].sum()
        A = xi / xi.sum(axis=1, keepdims=True)
        for k in (0, 1):
            w = gamma[:, k]
            mu[k] = (w * r).sum() / w.sum()
            var[k] = (w * (r - mu[k]) ** 2).sum() / w.sum() + 1e-6
    high = int(np.argmax(var))
    return gamma[:, high], high, np.sqrt(var) / 100 * np.sqrt(252)   # high-vol prob series, idx, ann vols


def vol_state(asof=None) -> dict:
    """Fit the 2-state HMM on SPY returns (point-in-time) -> current high-vol probability."""
    spy = data.benchmark()
    if asof is not None:
        spy = spy[spy.index <= pd.Timestamp(asof)]
    r = spy.pct_change().dropna()
    p_high, _, vols = _vol_hmm(r.values)
    return {"high_vol_prob": round(float(p_high[-1]), 3),
            "state": "HIGH-VOL" if p_high[-1] > 0.5 else "LOW-VOL",
            "state_vols": [round(float(v), 2) for v in sorted(vols)]}
