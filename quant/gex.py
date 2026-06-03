"""Index gamma exposure — fragility where Mat says it actually lives.

The options paper: GEX/VEX signal quality is highest in index options (SPX/SPY/QQQ),
where dealer participation is deepest and OI is large enough to matter. Mat uses GEX
"primarily as a vol regime classifier: the sign and percentile rank vs the trailing
30-day distribution." Core claim (paper 8.1): positive-GEX regimes show LOWER realized
vol (dealers sell strength / buy weakness); negative-GEX regimes AMPLIFY.

Naive dealer convention: dealers long call gamma, short put gamma ->
  net GEX = sum( gamma * OI * 100 * spot^2 * 0.01 * (+1 call, -1 put) )
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

IDX = Path(__file__).resolve().parent.parent / "data" / "QQQ_SPY_IWM"


def net_gex(symbol: str = "SPY") -> pd.DataFrame:
    """Daily net dealer gamma exposure and spot for an index ETF (cached)."""
    cache = IDX / f"_gex_cache_{symbol}.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    opt = pd.read_parquet(IDX / f"{symbol}_options.parquet",
                          columns=["date", "type", "gamma", "open_interest", "strike"])
    und = pd.read_parquet(IDX / f"{symbol}_underlying.parquet", columns=["date", "close"])
    und["date"] = pd.to_datetime(und["date"]).dt.normalize()
    spot = und.set_index("date")["close"]

    opt["date"] = pd.to_datetime(opt["date"]).dt.normalize()
    opt["S"] = opt["date"].map(spot)
    sign = np.where(opt["type"].astype(str).str.lower().str.startswith("c"), 1.0, -1.0)
    opt["gex"] = (opt["gamma"].fillna(0) * opt["open_interest"].fillna(0)
                  * 100 * opt["S"] ** 2 * 0.01 * sign)
    daily = opt.groupby("date")["gex"].sum().to_frame("gex")
    daily["spot"] = spot.reindex(daily.index)
    daily["gex_pctile"] = daily["gex"].rolling(30, min_periods=15).apply(
        lambda w: (w < w[-1]).mean(), raw=True)
    out = daily.dropna(subset=["spot"]).reset_index()
    out.to_parquet(IDX / f"_gex_cache_{symbol}.parquet")
    return out


def regime_state(symbol: str = "SPY", asof: pd.Timestamp | None = None) -> dict:
    """Current GEX vol-regime as a point-in-time sizing input (validated component).

    Negative GEX -> amplifying regime (higher realized vol ahead) -> reduce size /
    prefer long-vol structures. Positive GEX -> suppressive -> range-bound, short-vol
    friendlier. Returns sign, 30d percentile, and the historically-typical forward 5d
    realized vol for this regime (from the validated neg/pos split).
    """
    g = net_gex(symbol)
    if asof is not None:
        g = g[g["date"] <= pd.Timestamp(asof)]
    if g.empty:
        return {}
    cur = g.iloc[-1]
    neg = cur["gex"] < 0
    # typical forward 5d RV per regime, using only history up to asof (no peeking)
    rv = realized_vol_forward(symbol, hold=5)
    hist = g.merge(rv, on="date").dropna(subset=["gex", "rv_fwd"])
    if asof is not None:
        hist = hist[hist["date"] <= pd.Timestamp(asof)]
    typ = hist[(hist["gex"] < 0) == neg]["rv_fwd"].median()
    return {"date": cur["date"], "gex": float(cur["gex"]),
            "regime": "amplifying (neg-GEX)" if neg else "suppressive (pos-GEX)",
            "gex_pctile": float(cur["gex_pctile"]) if pd.notna(cur["gex_pctile"]) else None,
            "typical_fwd_5d_rv": round(float(typ), 4) if pd.notna(typ) else None}


def _ols(y: np.ndarray, X: np.ndarray):
    """Returns (beta, t-stats). X must already include an intercept column."""
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = len(y) - X.shape[1]
    sigma2 = (resid @ resid) / dof
    se = np.sqrt(np.diag(sigma2 * np.linalg.inv(X.T @ X)))
    return beta, beta / se


def marginal_value(symbol: str = "SPY", hold: int = 5, split: str = "2018-01-01") -> dict:
    """Does GEX add info about forward vol BEYOND trailing realized vol?

    The endogeneity worry: neg-GEX and high vol are mechanically linked and vol clusters.
    So we (a) bucket by trailing-RV quartile and check the neg-vs-pos gap survives within
    buckets, and (b) regress fwd_RV ~ trailing_RV + neg_GEX_dummy and read the dummy's
    t-stat. If the dummy stays significant after controlling for trailing vol, GEX is not
    just vol-clustering. IS/OOS split confirms it holds out of sample.
    """
    g = net_gex(symbol)
    und = pd.read_parquet(IDX / f"{symbol}_underlying.parquet", columns=["date", "close"])
    und["date"] = pd.to_datetime(und["date"]).dt.normalize()
    und = und.sort_values("date").reset_index(drop=True)
    r = und["close"].pct_change()
    und["trail_rv"] = r.rolling(21).std() * np.sqrt(252)               # known at t
    und["fwd_rv"] = [r.iloc[i + 1:i + 1 + hold].std() * np.sqrt(252)   # t+1..t+hold
                     if i + 1 + hold <= len(r) else np.nan for i in range(len(und))]

    df = g.merge(und[["date", "trail_rv", "fwd_rv"]], on="date").dropna(
        subset=["gex", "trail_rv", "fwd_rv"])
    df["neg"] = (df["gex"] < 0).astype(float)

    # (a) conditional on trailing-vol quartile: median fwd_rv by GEX sign
    df["rv_q"] = pd.qcut(df["trail_rv"], 4, labels=["Q1lo", "Q2", "Q3", "Q4hi"])
    cond = df.pivot_table(index="rv_q", columns="neg", values="fwd_rv",
                          aggfunc="median", observed=True)
    cond.columns = ["pos_GEX", "neg_GEX"]
    cond["neg_minus_pos"] = cond["neg_GEX"] - cond["pos_GEX"]

    # (b) regression with t-stats, full / IS / OOS
    def reg(d):
        z = lambda s: (s - s.mean()) / s.std()
        X = np.column_stack([np.ones(len(d)), z(d["trail_rv"]).values, d["neg"].values])
        _, t = _ols(z(d["fwd_rv"]).values, X)
        return round(float(t[2]), 2), len(d)  # t-stat on neg_GEX dummy

    sp = pd.Timestamp(split)
    t_full, n_full = reg(df)
    t_is, n_is = reg(df[df["date"] < sp])
    t_oos, n_oos = reg(df[df["date"] >= sp])
    return {"symbol": symbol, "conditional": cond.round(4),
            "t_negGEX_full": t_full, "n_full": n_full,
            "t_negGEX_IS": t_is, "t_negGEX_OOS": t_oos, "n_oos": n_oos}


def realized_vol_forward(symbol: str, hold: int = 5) -> pd.DataFrame:
    und = pd.read_parquet(IDX / f"{symbol}_underlying.parquet", columns=["date", "close"])
    und["date"] = pd.to_datetime(und["date"]).dt.normalize()
    und = und.sort_values("date").reset_index(drop=True)
    r = und["close"].pct_change()
    # forward `hold`-day realized vol from t+1..t+hold
    fwd = [r.iloc[i + 1:i + 1 + hold].std() * np.sqrt(252) if i + 1 + hold <= len(r) else np.nan
           for i in range(len(und))]
    und["rv_fwd"] = fwd
    return und[["date", "rv_fwd"]]


def test_gex_regime(symbol: str = "SPY", hold: int = 5) -> pd.DataFrame:
    """Does negative-GEX regime precede higher forward realized vol? (paper's core claim)"""
    g = net_gex(symbol)
    rv = realized_vol_forward(symbol, hold)
    df = g.merge(rv, on="date").dropna(subset=["gex", "rv_fwd"])
    df["regime"] = np.where(df["gex"] < 0, "negative_GEX", "positive_GEX")
    out = df.groupby("regime")["rv_fwd"].agg(["count", "median", "mean"]).round(4)
    return out, df
