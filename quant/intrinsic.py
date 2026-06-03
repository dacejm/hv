"""Intrinsic-value scenarios (Mat's IREN-style): WACC + bear/base/bull/re-rate valuation.

Given a view on the exit EV/Revenue multiple, value the equity per share across scenarios:
  EV = multiple x TTM revenue ; equity = EV - net debt ; per-share = equity / shares.
WACC (CAPM cost of equity + after-tax cost of debt, market-value weighted) is reported as
the discount-rate anchor. The multiples are the VIEW input (like Mat's 10x/18x/28x/40x);
defaults are relative to the stock's current EV/Rev.

This is a decision-AID / valuation framework, not a validated alpha signal — "cheap vs DCF
predicts returns" is the value-factor question, which measured weak on this data (fcf_yield).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import data

EARN = data.EARN
RATES = data.DATA / "rates" / "parquet" / "us_treasury.parquet"
ERP = 0.05
SCENARIOS = {"Bear": 0.6, "Base": 1.0, "Bull": 1.6, "Re-rate": 2.2}   # x current EV/Rev (the view)


def _latest_bs(sub, symbol):
    df = pd.read_parquet(EARN / sub / f"{symbol}.parquet")
    df = df[df["period"] == "Quarter"].copy()
    df["pe"] = pd.to_datetime(df["date"])
    return df.sort_values("pe").iloc[-1]


def _rf():
    r = pd.read_parquet(RATES, columns=["date", "10_year"]).dropna()
    return float(r.sort_values("date")["10_year"].iloc[-1]) / 100.0


def _beta(symbol):
    px = data.ohlcv(symbol).set_index("date")["close"].pct_change()
    spy = data.benchmark().pct_change()
    j = pd.concat([px, spy], axis=1, keys=["s", "m"]).dropna().tail(504)
    if len(j) < 60 or j["m"].var() == 0:
        return 1.2
    return float(np.cov(j["s"], j["m"])[0, 1] / np.var(j["m"]))


def value(symbol: str, scenarios: dict = SCENARIOS) -> dict:
    inc = data.income_statement(symbol).sort_values("period_end")
    ttm_sales = inc["sales"].tail(4).sum()
    last = inc.iloc[-1]
    tax = float(np.clip(last["income_taxes"] / last["pretax_income"], 0, 0.5)) if last["pretax_income"] else 0.21
    interest = abs(float(last.get("interest_expense") or 0))

    bse, bsl, bsa = _latest_bs("balance_sheet_equity", symbol), _latest_bs("balance_sheet_liabilities", symbol), _latest_bs("balance_sheet_assets", symbol)
    shares = float(bse["shares_outstanding"])
    debt = float(np.nansum([bsl.get(k) for k in ("long_term_debt", "current_portion_long_term_debt",
                                                 "notes_payable", "convertible_debt")]))
    net_debt = debt - float(bsa.get("cash_and_equivalents") or 0)
    price = float(data.ohlcv(symbol)["close"].iloc[-1])

    rf, beta = _rf(), _beta(symbol)
    ce = rf + beta * ERP
    # bound cost of debt: netting/timing artifacts (e.g. $5M interest on a $10k period-end debt
    # balance) can imply a 50,000% rate and explode WACC. Clip to a sane [rf, 20%] band.
    cd = float(np.clip(interest / debt, rf, 0.20)) if debt > 0 and interest else rf + 0.015
    mktcap = price * shares
    V = mktcap + max(debt, 0)
    wacc = (mktcap / V) * ce + (max(debt, 0) / V) * cd * (1 - tax) if V > 0 else ce
    evrev_now = (mktcap + net_debt) / ttm_sales if ttm_sales else np.nan

    rows = []
    for name, f in scenarios.items():
        mult = evrev_now * f
        eq = mult * ttm_sales - net_debt
        ps = eq / shares
        rows.append({"scenario": name, "ev_rev": round(mult, 1), "target": round(ps, 2),
                     "upside": round(ps / price - 1, 3)})
    return {"symbol": symbol, "price": round(price, 2), "ttm_rev": ttm_sales,
            "ev_rev_now": round(evrev_now, 1), "wacc": round(wacc, 3), "beta": round(beta, 2),
            "scenarios": pd.DataFrame(rows)}
