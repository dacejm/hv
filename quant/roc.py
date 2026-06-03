"""Rate-of-change engine — Mat's spine ("d*ROC").

Terminology fixed to match Mat's usage (verified against the daily brief):
  - growth        : YoY revenue growth = Mat's "first derivative" (now 85% and climbing)
  - growth_accel  : change in growth   = Mat's "second derivative" (what the market discounts)

YoY is matched by DATE, not row offset, so irregular reporters aren't mismeasured.
"""
from __future__ import annotations

import pandas as pd

from . import data


def revenue_roc(symbol: str, asof: pd.Timestamp | None = None) -> pd.DataFrame:
    """Revenue level, YoY growth, and its acceleration (Mat's 1st/2nd derivative)."""
    inc = data.known_income(symbol, asof) if asof is not None else data.income_statement(symbol)
    out = inc[["period_end", "sales", "gross_margin", "net_margin"]].copy()
    out = out.sort_values("period_end").reset_index(drop=True)
    out["growth"] = data.yoy(out, "sales")          # Mat 1st derivative (date-matched YoY)
    # 2nd derivative = change in growth, but ONLY between CONSECUTIVE quarters (~90d apart).
    # Raw .diff() would subtract non-adjacent quarters when a filing is skipped/dropped --
    # the same row-offset trap that yoy() was built to avoid. Null the diff across gaps.
    consec = out["period_end"].diff().dt.days.between(80, 100)
    out["growth_accel"] = out["growth"].diff().where(consec)   # Mat 2nd derivative
    out["margin_trend"] = out["gross_margin"].diff().where(consec)
    return out


def revision_momentum(
    symbol: str, kind: str = "sales", horizon: str = "Current Year",
    asof: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """How the consensus for a FIXED future period has been revised over time.

    Rising consensus for the same target period = forward-looking 'underlying' signal,
    independent of already-reported (backward) actuals. `asof` keeps it point-in-time.
    """
    try:
        est = data.estimates(symbol, kind)
    except FileNotFoundError:
        return pd.DataFrame(columns=["published", "period_end", "consensus", "count", "revision"])
    est = est[est["period"] == horizon].copy()
    if asof is not None:
        est = est[est["published"] <= pd.Timestamp(asof)]
    est = est.sort_values("published").drop_duplicates("published", keep="last")
    est["consensus"] = pd.to_numeric(est["consensus"], errors="coerce")
    est["revision"] = est["consensus"].pct_change(fill_method=None)
    return est[["published", "period_end", "consensus", "count", "revision"]].reset_index(drop=True)
