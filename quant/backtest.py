"""Replay scorer — evaluated to Mat's quant-methods standard.

One engine, two modes (replay history / run asof=today). Nothing peeks past `asof`.
Upgrades demanded by the quant-methods paper:
  - IC measured across horizons (1/5/21/63d) -> IC-decay curve (signals can flip sign)
  - excess-of-SPY returns, optional sector-neutralization
  - IS/OOS split with an embargo gap (forward labels overlap -> leakage inflates IC)
  - t-stat reported against the t>=3 bar (Harvey-Liu-Zhu) AND a Bonferroni threshold for
    the number of (feature x horizon) tests actually run (multiple-testing deflation)

Features depend only on data at/before each earnings knowledge date, so computing a
symbol's full feature series once and indexing by period_end is point-in-time safe.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from . import data, roc

HORIZONS = (1, 5, 21, 63)
MIN_NAMES = 12
FEATURES = ["growth_accel", "revision"]  # the only signals that cleared the bar


def _entry(px: pd.DataFrame, known_on: pd.Timestamp) -> int | None:
    i = px["date"].searchsorted(pd.Timestamp(known_on), side="left")
    return i if i < len(px) else None


def _fwd(px: pd.DataFrame, spy: pd.Series, e_idx: int, h: int):
    x_idx = e_idx + h
    if x_idx >= len(px):
        return np.nan, np.nan
    entry, exit_ = px["close"].iloc[e_idx], px["close"].iloc[x_idx]
    if entry <= 0 or pd.isna(entry) or pd.isna(exit_):
        return np.nan, np.nan
    fwd = exit_ / entry - 1.0
    ed, xd = px["date"].iloc[e_idx], px["date"].iloc[x_idx]
    # align SPY to the SAME sessions via as-of (<=) lookup. searchsorted("left") grabs the NEXT
    # SPY day when a date is missing (holiday/dirty data) -> stock window and market window drift
    # apart, corrupting the excess. get_indexer(method="ffill") uses the SPY price as-of each date.
    se, sx = spy.index.get_indexer([ed], method="ffill")[0], spy.index.get_indexer([xd], method="ffill")[0]
    mkt = (spy.iloc[sx] / spy.iloc[se] - 1.0) if se >= 0 and sx >= 0 else np.nan
    return fwd, fwd - mkt


def replay(symbols: list[str]) -> pd.DataFrame:
    """One row per (symbol, earnings knowledge date): all features + multi-horizon excess."""
    spy, sect = data.benchmark(), data.sectors()
    rows = []
    for sym in symbols:
        try:
            px = data.ohlcv(sym)
            inc = data.income_statement(sym)
        except FileNotFoundError:
            continue
        if px.empty or inc.empty:
            continue
        rev = roc.revenue_roc(sym).set_index("period_end")
        rm = roc.revision_momentum(sym)

        for _, r in inc.iterrows():
            pe, known_on = r["period_end"], r["known_on"]
            if known_on > px["date"].iloc[-1]:
                continue
            e_idx = _entry(px, known_on)
            if e_idx is None:
                continue
            rev_row = rev.loc[pe] if pe in rev.index else None
            revis = rm[rm["published"] <= known_on]["revision"].tail(4).mean() if not rm.empty else np.nan
            row = {"symbol": sym, "date": known_on, "sector": sect.get(sym, "Unknown"),
                   "growth_accel": rev_row["growth_accel"] if rev_row is not None else np.nan,
                   "revision": revis}
            for h in HORIZONS:
                _, exc = _fwd(px, spy, e_idx, h)
                row[f"excess_{h}"] = exc
            rows.append(row)
    return pd.DataFrame(rows)


def _sector_neutral(panel: pd.DataFrame, col: str) -> pd.Series:
    q = panel["date"].dt.to_period("Q")
    return panel[col] - panel.groupby([q, panel["sector"]])[col].transform("mean")


def _ic_series(panel: pd.DataFrame, feat: str, ret_col: str) -> pd.Series:
    ics = []
    for _, g in panel.groupby(panel["date"].dt.to_period("Q")):
        sub = g[[feat, ret_col]].dropna()
        if len(sub) >= MIN_NAMES and sub[feat].nunique() >= 3:
            ics.append(sub[feat].corr(sub[ret_col], method="spearman"))
    return pd.Series(ics, dtype=float).dropna()


def evaluate(panel: pd.DataFrame, sector_neutral: bool = True,
             split: str = "2023-01-01", features=FEATURES) -> pd.DataFrame:
    """IC-decay across horizons + IS/OOS with embargo + multiple-testing-deflated bar."""
    panel = panel.copy()
    n_tests = len(features) * len(HORIZONS)
    t_bonf = stats.norm.ppf(1 - 0.025 / n_tests)  # two-sided Bonferroni critical t

    out = []
    for feat in features:
        for h in HORIZONS:
            ret = f"excess_{h}"
            col = ret
            if sector_neutral:
                panel["_sn"] = _sector_neutral(panel, ret)
                col = "_sn"
            ics = _ic_series(panel, feat, col)
            if len(ics) < 4:
                continue
            mean_ic, sd = ics.mean(), ics.std()
            t = mean_ic / sd * np.sqrt(len(ics)) if sd else np.nan

            # IS/OOS with one-quarter embargo around the split
            sp = pd.Period(pd.Timestamp(split), "Q")
            q = panel["date"].dt.to_period("Q")
            is_ic = _ic_series(panel[q < sp - 1], feat, col).mean()
            oos_ic = _ic_series(panel[q > sp], feat, col).mean()
            oos_frac = (oos_ic / is_ic) if is_ic and not np.isnan(is_ic) and is_ic != 0 else np.nan

            out.append({"feature": feat, "h": h, "n_q": len(ics),
                        "mean_IC": round(mean_ic, 4), "ICIR": round(mean_ic / sd, 3) if sd else np.nan,
                        "t": round(t, 2), "passes_t3": abs(t) >= 3.0,
                        "passes_bonf": abs(t) >= t_bonf,
                        "IS_IC": round(is_ic, 4) if pd.notna(is_ic) else np.nan,
                        "OOS_IC": round(oos_ic, 4) if pd.notna(oos_ic) else np.nan,
                        "OOS/IS": round(oos_frac, 2) if pd.notna(oos_frac) else np.nan})
    res = pd.DataFrame(out)
    res.attrs["t_bonf"] = round(t_bonf, 2)
    res.attrs["n_tests"] = n_tests
    return res
