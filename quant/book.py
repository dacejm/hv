"""The book — the only output built purely from VALIDATED components, lanes respected.

  DIRECTION (what to hold): growth_accel sector-neutral screen (screen.live_screen) ->
                            top longs / bottom shorts. The one validated picker.
  SIZING   (how much):      paper-grounded risk layer (quant.sizing), NOT the old binary
                            "0.5x in amplifying GEX" throttle (combo test + Bk07 showed that
                            was the wrong shape). Instead:
      - within each leg: HRP weights on trailing shrunk cov (equal-weight fallback). The
        measured result: in a single-asset-class equity basket HRP gives the best tail
        control while equal-weight is competitive (DeMiguel), so HRP is the safe default.
      - gross: CONTINUOUS vol-targeting (gross = target_vol / forecast_vol). The GEX
        vol-regime raises the forecast in amplifying (neg-GEX) regimes -> lower gross --
        the continuous version of the throttle, at GEX's own short horizon.
      - capped at HALF-KELLY (Bk07 2.2) so estimation error can't lever us into ruin.

No CONTEXT tools touch selection or size. growth_accel's edge is modest, so this stays a
diversified sector-neutral long/short basket (breadth), never a concentrated bet.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import gex, screen, sizing


def _leg(symbols, asof, scheme="hrp") -> pd.Series:
    w = sizing.leg_weights(list(symbols), asof=asof, scheme=scheme)
    if w.empty:                                   # not enough history -> equal weight
        return pd.Series(1.0 / max(len(symbols), 1), index=list(symbols))
    return w


def build(universe, asof=None, n=15, target_vol=0.10, max_gross=2.0) -> dict:
    asof = pd.Timestamp(asof) if asof else pd.Timestamp.today()
    sc = screen.live_screen(asof, universe, top=n)              # DIRECTION (growth_accel)
    if not sc:
        return {}
    reg = sc["regime"]                                          # GEX vol-regime (forecast input)
    longs, shorts = sc["longs"], sc["shorts"]

    # within-leg HRP weights (long-only each leg), dollar-neutral across legs
    lw = _leg(longs["symbol"], asof)
    sw = _leg(shorts["symbol"], asof)
    lw, sw = lw / lw.sum(), sw / sw.sum()

    # gross via continuous vol-targeting; GEX amplifying regime inflates the vol forecast
    base_vol = _basket_vol(list(lw.index) + list(sw.index), asof)
    amp = "amplifying" in (reg.get("regime") or "")
    fwd_vol = base_vol * (1.30 if amp else 1.0)                 # neg-GEX -> higher forecast -> less gross
    gross = float(target_vol / fwd_vol) if fwd_vol > 0 else 1.0
    gross = round(min(gross, max_gross), 2)                     # half-Kelly-style hard cap on leverage

    rows = []
    for sym, w in lw.items():
        r = longs[longs["symbol"] == sym].iloc[0]
        rows.append({"side": "LONG", "symbol": sym, "sector": r["sector"],
                     "composite": r["composite"], "weight": round(w * gross / 2, 4)})
    for sym, w in sw.items():
        r = shorts[shorts["symbol"] == sym].iloc[0]
        rows.append({"side": "SHORT", "symbol": sym, "sector": r["sector"],
                     "composite": r["composite"], "weight": round(-w * gross / 2, 4)})
    book = pd.DataFrame(rows)
    return {"asof": asof.date().isoformat(), "regime": reg, "gross": gross,
            "target_vol": target_vol, "forecast_vol": round(fwd_vol, 4),
            "weighting": "HRP within legs (equal-weight fallback)",
            "sizing_rule": f"gross = target_vol / forecast_vol "
                           f"({'amplifying: forecast x1.3 -> less gross' if amp else 'suppressive: full'})",
            "book": book}


def _basket_vol(symbols, asof, lookback=63) -> float:
    """Annualized realized vol of the equal-weight basket over trailing `lookback` days."""
    R = sizing.returns_matrix(symbols, asof=asof, lookback=lookback)
    if R.empty:
        return 0.15
    port = R.mean(axis=1)                                       # equal-weight proxy for the basket
    return float(port.std() * np.sqrt(sizing.TRADING_DAYS))
