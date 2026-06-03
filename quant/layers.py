"""Purpose-map registry — which LANE each component belongs to, DERIVED FROM MEASUREMENT.

The discipline (the project's whole point): every tool in its lane. Only DIRECTION signals
may drive what-to-trade; SIZING/VOL scales risk; CONTEXT orients (never a buy signal);
EXPRESSION shapes how to trade a view; FLAGS trigger investigation. "Macro is not a
directional signal" generalizes — almost nothing here is, and the lane says so explicitly.

Lanes verified by: verify.py (index lane test), backtest.evaluate (growth_accel),
gex marginal-value test, regime_macro_test (MACRO x adds nothing), per-component backtests.
"""

LANES = {
    # DIRECTION — the scarce layer; the only thing that may pick longs/shorts
    "roc.growth_accel":   ("DIRECTION",   "validated: sector-neutral IC~0.037, t5.3 @21d, OOS-stable"),
    "roic":               ("DIRECTION",   "validated: NOPAT/invested-capital, sector-neutral t3.81 @63d, OOS-stable (candidates.py). Use STANDALONE @quarterly -- equal-wt blend with growth_accel DILUTES (measured)"),
    "screen":             ("DIRECTION",   "growth_accel sector-neutral live screen"),
    # SIZING / VOL — scale risk, choose vol regime; predicts VOLATILITY not direction
    "gex":                ("SIZING/VOL",  "validated: neg-GEX -> higher fwd realized vol (deflated t 3.8-6.5)"),
    # EXPRESSION — how to express a view (not whether)
    "intrinsic":          ("EXPRESSION",  "valuation scenarios given a multiple view (WACC + bear/base/bull)"),
    "options_vs_stock":   ("EXPRESSION",  "3-factor surface (IVP + term-slope + RR25 skew): long vol when IV cheap; finance puts when skew steep"),
    "sizing":             ("SIZING/VOL",  "vol-target + HRP/min-var + half-Kelly; the validated risk layer (replaces book's binary throttle)"),
    # CONTEXT — orient / idea-generate / label; NEVER a buy signal, NEVER sizes
    "regime":             ("CONTEXT",     "4-quadrant growth x inflation + per-quadrant premia tilt; MACRO x adds nothing to returns -> display only"),
    "screeners":          ("CONTEXT",     "momentum x regime attention board (momentum=noise@21d) -> not a buy list"),
    "nodes":              ("CONTEXT",     "where the OI/volume walls are; no daily directional edge"),
    "rnd":                ("CONTEXT",     "risk-neutral density; right-tail only weak-direction (IC~0.09@5d)"),
    "pcr":                ("CONTEXT",     "put/call deviation; pcr_vol weak-direction (t2.1 non-overlap)"),
    "research_brief":     ("CONTEXT",     "LLM research synthesis; unmeasurable -> context only"),
    "thesis":             ("CONTEXT",     "LLM thesis from research; idea-generation, not validated alpha"),
    # FLAGS — trigger investigation; a flag is not a trade
    "anomaly":            ("FLAGS",       "z-vs-history deviation detector across any stream"),
    # INFRA / MEASUREMENT
    "data":               ("INFRA",       "point-in-time accessors"),
    "backtest":           ("MEASUREMENT", "IC / IS-OOS / t>=3 / deflation gauntlet"),
}


# Validated HORIZON per component — a tool must be USED at the horizon it was TESTED at
# (the GEX lesson: 5-day vol signal must not size a quarterly book). Cross-checked vs the
# options-theory book ("GEX strongest near expiration") and each component's own backtest.
HORIZON = {
    "roc.growth_accel": "quarterly (21-63d fwd, earnings-driven)",
    "roic":             "quarterly (63d fwd) -- standalone ranker, NOT blended with growth_accel",
    "gex":              "short (~5d realized vol, near-expiry) -> size short-dated vol/options ONLY",
    "rnd":              "short (~5d, right-tail)",
    "pcr":              "short (~5d)",
    "options_vs_stock": "weeks (21d IV-rank vs realized) + term-structure slope + RR25 skew (3-factor, done)",
    "sizing":           "matches the basket's horizon; GEX/HMM feed the vol forecast at their OWN short horizon",
    "regime":           "slow (cycle, months) -- but rules-based; HMM/CUSUM is the spec'd upgrade",
    "screeners":        "context (no validated horizon -- attention only)",
}

# Cross-check vs options-theory book (Module 2 Book 03), 2026-validated:
#   RND e^(rT) d2C/dK2 = Dupire (OK) | GEX sign+flip+vol-not-direction (OK) | options-vs-stock low-IV->buy
#   matches paper 1.5 (OK) BUT VRP sign = paper uses IV-RV (mine RV-IV) + missing term-structure contango.
#   regime = rules-based, paper wants HMM. Horizon discipline now explicit above.


def lane(name: str):
    return LANES.get(name, ("UNTAGGED", "not in registry"))


def horizon(name: str):
    return HORIZON.get(name, "untagged")
