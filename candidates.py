"""Direction-candidate gauntlet — test the corpus-surfaced DIRECTION ideas the SAME way
growth_accel was validated: sector-neutral cross-sectional IC across horizons, t>=3
(Harvey-Liu-Zhu), Bonferroni for the 16 (feature x horizon) tests, IS/OOS with embargo.

Candidates (all point-in-time, aligned to the earnings knowledge date):
  acc_quality  -- -accruals (Sloan 1996): low accruals = cash earnings -> outperform.
                  accruals = (net_income - CFO) / avg_total_assets.
  sue          -- standardized earnings surprise (PEAD: beats drift up).
  roic         -- NOPAT / invested capital (quality/value).
  hi52         -- price proximity to the trailing 252d high (George-Hwang 52-week-high).

Nothing is trusted until it clears the bar. Run:  python candidates.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from quant import backtest, data

FEATURES = ["acc_quality", "sue", "roic", "hi52"]


def _roic_row(inc_row) -> float:
    op = inc_row.get("income_after_depreciation_and_amortization")
    pretax, tax = inc_row.get("pretax_income"), inc_row.get("income_taxes")
    if pd.isna(op) or pd.isna(pretax) or pretax == 0:
        return np.nan
    eff_tax = np.clip((tax / pretax) if pd.notna(tax) else 0.21, 0, 0.5)
    return op * (1 - eff_tax)            # NOPAT (invested capital divided in the panel)


def build_panel(symbols: list[str]) -> pd.DataFrame:
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
        inc = inc.set_index("period_end")
        cf = data.cash_flow(sym).set_index("period_end") if _exists(sym, "cash_flow_statement") else None
        bs = data.balance_sheet(sym).set_index("period_end") if _exists(sym, "balance_sheet_assets") else None
        sue_df = data.eps_surprise(sym).set_index("period_end") if _exists(sym, "eps_history") else None
        close = px.set_index("date")["close"]

        for pe, r in inc.iterrows():
            known_on = r["known_on"]
            if known_on > px["date"].iloc[-1]:
                continue
            e_idx = backtest._entry(px, known_on)
            if e_idx is None or e_idx < 1:
                continue
            row = {"symbol": sym, "date": known_on, "sector": sect.get(sym, "Unknown")}

            # 52-week-high proximity at entry (trailing 252 sessions, point-in-time)
            window = close.iloc[max(0, e_idx - 252):e_idx]
            row["hi52"] = float(close.iloc[e_idx] / window.max()) if len(window) > 20 and window.max() > 0 else np.nan

            # SUE (PEAD)
            row["sue"] = float(sue_df.loc[pe, "sue"]) if sue_df is not None and pe in sue_df.index else np.nan

            # ROIC = NOPAT / invested_capital
            row["roic"] = np.nan
            if bs is not None and pe in bs.index:
                ic = bs.loc[pe, "invested_capital"]
                nopat = _roic_row(r)
                if pd.notna(ic) and ic > 0 and pd.notna(nopat):
                    row["roic"] = float(nopat / ic)

            # accruals quality = -(NI - CFO)/avg_assets ; avg of current & year-ago assets
            row["acc_quality"] = np.nan
            if cf is not None and bs is not None and pe in cf.index and pe in bs.index:
                ni = cf.loc[pe, "net_income"]
                cfo = cf.loc[pe, "net_cash_from_operating_activities"]
                ta = bs.loc[pe, "total_assets"]
                yr_ago = pe - pd.Timedelta(days=365)
                near = bs.index[np.argmin(np.abs(bs.index - yr_ago))] if len(bs.index) else None
                ta_prior = bs.loc[near, "total_assets"] if near is not None and abs((near - yr_ago).days) < 60 else ta
                avg_assets = np.nanmean([ta, ta_prior])
                if pd.notna(ni) and pd.notna(cfo) and avg_assets and avg_assets > 0:
                    row["acc_quality"] = -float((ni - cfo) / avg_assets)

            for h in backtest.HORIZONS:
                _, exc = backtest._fwd(px, spy, e_idx, h)
                row[f"excess_{h}"] = exc
            rows.append(row)
    return pd.DataFrame(rows)


def _exists(sym, sub):
    return (data.EARN / sub / f"{sym}.parquet").exists()


def run(symbols: list[str]):
    panel = build_panel(symbols)
    cov = {f: int(panel[f].notna().sum()) for f in FEATURES}
    print(f"panel: {len(panel)} rows, {panel['symbol'].nunique()} symbols  | non-null per feature: {cov}\n")
    res = backtest.evaluate(panel, sector_neutral=True, features=FEATURES)
    if res.empty:
        print("no evaluable features (too few cross-sections)")
        return
    print(res.to_string(index=False))
    print(f"\nbars: t>=3 (Harvey-Liu-Zhu) | Bonferroni t>={res.attrs['t_bonf']} "
          f"for {res.attrs['n_tests']} tests. A candidate is real only if it clears them AND OOS holds.")
    return res


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    inc = {p.stem for p in (data.EARN / "income_statement").glob("*.parquet")}
    uni = sorted(inc & set(data.sectors().index))
    print(f"universe: {len(uni)} symbols with income + sector\n")
    run(uni)
