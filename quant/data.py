"""Point-in-time data accessors over the local parquet sets.

The one rule that keeps backtests honest: a value is only "known" on the date it
was actually published, never on the date it describes.

  - Estimates (eps_estimate/sales_estimate): the `date` column IS the publication
    date -> usable as-is.
  - Actuals (income_statement): the `date` column is the PERIOD-END date, not the
    filing date. We lag it to a conservative report date before it counts as known.
"""
from __future__ import annotations

import functools
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parent.parent / "data"
EARN = DATA / "earnings" / "parquet"
OHLCV = DATA / "stocks" / "parquet" / "ohlcv"
BENCH = DATA / "QQQ_SPY_IWM" / "SPY_underlying.parquet"
SECTORS_CSV = DATA / "sectors.csv"

# Conservative gap between a quarter's period-end and when results are filed.
DEFAULT_REPORT_LAG = pd.Timedelta(days=50)


@functools.lru_cache(maxsize=512)
def income_statement(symbol: str) -> pd.DataFrame:
    """Quarterly income statement with margins and an estimated knowledge date."""
    df = pd.read_parquet(EARN / "income_statement" / f"{symbol}.parquet")
    df = df[df["period"] == "Quarter"].copy()
    df["period_end"] = pd.to_datetime(df["date"])
    df["known_on"] = df["period_end"] + DEFAULT_REPORT_LAG
    df["gross_margin"] = df["gross_profit"] / df["sales"]
    df["net_margin"] = df["net_income"] / df["sales"]
    return df.sort_values("period_end").reset_index(drop=True)


@functools.lru_cache(maxsize=512)
def estimates(symbol: str, kind: str = "sales") -> pd.DataFrame:
    """Analyst estimate snapshots. `date` is the publication date (point-in-time safe)."""
    df = pd.read_parquet(EARN / f"{kind}_estimate" / f"{symbol}.parquet")
    df["published"] = pd.to_datetime(df["date"])
    df["period_end"] = pd.to_datetime(df["period_end_date"])
    return df.sort_values("published").reset_index(drop=True)


@functools.lru_cache(maxsize=512)
def ohlcv(symbol: str) -> pd.DataFrame:
    df = pd.read_parquet(OHLCV / f"{symbol}.parquet")
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


@functools.lru_cache(maxsize=512)
def cash_flow(symbol: str) -> pd.DataFrame:
    """Quarterly cash-flow statement with a knowledge date (for accruals)."""
    df = pd.read_parquet(EARN / "cash_flow_statement" / f"{symbol}.parquet")
    df = df[df["period"] == "Quarter"].copy()
    df["period_end"] = pd.to_datetime(df["date"])
    df["known_on"] = df["period_end"] + DEFAULT_REPORT_LAG
    return df.sort_values("period_end").reset_index(drop=True)


@functools.lru_cache(maxsize=512)
def balance_sheet(symbol: str) -> pd.DataFrame:
    """Quarterly balance sheet (assets+equity+liabilities merged on period-end) with a
    knowledge date, plus derived invested_capital and total_debt (for ROIC/accruals)."""
    a = pd.read_parquet(EARN / "balance_sheet_assets" / f"{symbol}.parquet")
    e = pd.read_parquet(EARN / "balance_sheet_equity" / f"{symbol}.parquet")
    l = pd.read_parquet(EARN / "balance_sheet_liabilities" / f"{symbol}.parquet")
    a, e, l = (d[d["period"] == "Quarter"].copy() for d in (a, e, l))
    m = a[["date", "total_assets"]].merge(
        e[["date", "total_equity"]], on="date").merge(
        l[["date", "long_term_debt", "current_portion_long_term_debt", "notes_payable"]], on="date")
    m["period_end"] = pd.to_datetime(m["date"])
    m["known_on"] = m["period_end"] + DEFAULT_REPORT_LAG
    debt = (m["long_term_debt"].fillna(0) + m["current_portion_long_term_debt"].fillna(0)
            + m["notes_payable"].fillna(0))
    m["total_debt"] = debt
    m["invested_capital"] = m["total_equity"].fillna(0) + debt
    return m.sort_values("period_end").reset_index(drop=True)


@functools.lru_cache(maxsize=512)
def eps_surprise(symbol: str) -> pd.DataFrame:
    """Reported vs estimated EPS -> standardized surprise (SUE-style). period_end_date is
    the quarter end; it becomes known ~report-lag later (PEAD entry)."""
    df = pd.read_parquet(EARN / "eps_history" / f"{symbol}.parquet").copy()
    df["period_end"] = pd.to_datetime(df["period_end_date"])
    df["known_on"] = df["period_end"] + DEFAULT_REPORT_LAG
    df = df.sort_values("period_end")
    rep = pd.to_numeric(df["reported"], errors="coerce")
    est = pd.to_numeric(df["estimate"], errors="coerce")
    surp = rep - est
    df["surprise"] = surp / est.abs().replace(0, np.nan)                  # scaled surprise
    df["sue"] = surp / surp.rolling(8, min_periods=4).std()               # standardized (SUE)
    return df.reset_index(drop=True)


def known_income(symbol: str, asof: pd.Timestamp) -> pd.DataFrame:
    """Income-statement rows whose results were already public on `asof`."""
    df = income_statement(symbol)
    return df[df["known_on"] <= pd.Timestamp(asof)].reset_index(drop=True)


def yoy(df: pd.DataFrame, value_col: str, date_col: str = "period_end",
        tol_days: int = 55) -> pd.Series:
    """Year-over-year growth matched by DATE (period_end - 1yr), not row offset.

    Row-offset YoY (pct_change(4)) silently compares the wrong periods when a
    company skips/merges quarters. Matching on date within a tolerance handles
    irregular reporters (the small-cap universe). Returns a Series aligned to df.
    """
    base = df[[date_col, value_col]].dropna().sort_values(date_col)
    prior = base.rename(columns={value_col: "_prior"}).copy()
    prior["_match"] = prior[date_col] + pd.Timedelta(days=365)
    prior = prior[["_match", "_prior"]].sort_values("_match")
    merged = pd.merge_asof(base, prior, left_on=date_col, right_on="_match",
                           direction="nearest", tolerance=pd.Timedelta(days=tol_days))
    merged["g"] = merged[value_col] / merged["_prior"] - 1.0
    return df[date_col].map(merged.set_index(date_col)["g"])


@functools.lru_cache(maxsize=1)
def benchmark() -> pd.Series:
    """SPY close indexed by date — the market leg for excess returns."""
    df = pd.read_parquet(BENCH)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").set_index("date")["close"]


@functools.lru_cache(maxsize=1)
def sectors() -> pd.Series:
    """Ticker -> sector map."""
    df = pd.read_csv(SECTORS_CSV)
    return df.set_index("ticker")["sector"]
