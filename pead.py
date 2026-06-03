"""PEAD with a price-confirmation gate -- replicating the TradingView "Earnings Surprise +
Reaction Entry" indicator, measured honestly across the universe.

Logic (from the Pine script):
  surprise   = actual EPS - estimate  ->  direction = sign(surprise)  (beat / miss)
  reaction   = announcement bar (BMO) or the next bar (AMC), per earnings_calendar 'when'
  CONFIRM    = the reaction-day CLOSE must break the reference bar's high (beat) / low (miss),
               reference = the bar immediately before the reaction day. No break -> 'rejected'.
  entry      = OPEN of the session after the reaction bar; hold `window` (60) trading days.

The claim being tested: the breakout confirmation separates real post-earnings drift from
noise -- a beat the market confirms (gap + breakout) drifts; a beat it ignores/fades does not.
So we measure CONFIRMED vs REJECTED vs baseline, long and short, excess-of-SPY. Event-driven and
point-in-time (entry strictly after the reaction; surprise is public at announcement).

  python pead.py            # full universe
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from quant import backtest, data

WINDOWS = (1, 5, 21, 63)                 # aligned to backtest.HORIZONS for the deflated gauntlet
EC = data.EARN / "earnings_calendar"


def _surprises(sym: str) -> pd.DataFrame | None:
    try:
        s = data.eps_surprise(sym)[["period_end", "reported", "estimate", "surprise"]].dropna(
            subset=["reported", "estimate"])
    except FileNotFoundError:
        return None
    return s if not s.empty else None


def events(sym: str, spy: pd.Series) -> pd.DataFrame:
    """One row per earnings announcement with reaction/confirmation/forward excess-return fields.
    Excess = (trade return - SPY over the same span starting at ENTRY), signed by direction."""
    ecf = EC / f"{sym}.parquet"
    if not ecf.exists():
        return pd.DataFrame()
    ec = pd.read_parquet(ecf)
    ec["ann"] = pd.to_datetime(ec["date"])
    ec = ec.dropna(subset=["ann"]).sort_values("ann")
    sur = _surprises(sym)
    if sur is None:
        return pd.DataFrame()
    # attach the most-recent reported quarter's surprise to each announcement date
    ec = pd.merge_asof(ec, sur.sort_values("period_end"), left_on="ann", right_on="period_end",
                       direction="backward", tolerance=pd.Timedelta(days=100)).dropna(subset=["surprise"])
    if ec.empty:
        return pd.DataFrame()
    try:
        px = data.ohlcv(sym)
    except FileNotFoundError:
        return pd.DataFrame()
    sect = data.sectors()
    px = px.reset_index(drop=True)
    dates = px["date"].values
    sidx, sval = spy.index.values, spy.values
    rows = []
    for _, e in ec.iterrows():
        sgn = 1 if e["surprise"] > 0 else -1 if e["surprise"] < 0 else 0
        if sgn == 0:
            continue
        amc = "after" in str(e["when"]).lower()           # AMC -> reaction = next bar; BMO -> same bar
        ann_idx = int(np.searchsorted(dates, np.datetime64(e["ann"]), side="left"))
        if ann_idx >= len(px):
            continue
        rx = ann_idx + 1 if amc else ann_idx               # reaction bar
        ref = rx - 1                                       # reference bar (before reaction)
        ent = rx + 1                                       # entry bar (next open)
        if ref < 0 or ent >= len(px):
            continue
        rx_close = px["close"].iloc[rx]
        ref_hi, ref_lo = px["high"].iloc[ref], px["low"].iloc[ref]
        conf_long = sgn == 1 and rx_close > ref_hi
        conf_short = sgn == -1 and rx_close < ref_lo
        entry_px = px["open"].iloc[ent]
        if entry_px <= 0:
            continue
        si = int(np.searchsorted(sidx, np.datetime64(px["date"].iloc[ent]), side="right")) - 1  # SPY as-of (<=) entry
        # surprise as a continuous feature (winsorized %); the entry date is the panel date
        spct = float(np.clip(e["surprise"] / abs(e["estimate"]), -2, 2)) if e["estimate"] else np.nan
        row = {"symbol": sym, "date": px["date"].iloc[ent], "ann": e["ann"], "dir": sgn,
               "sector": sect.get(sym, "Unknown"), "surprise": spct,
               "confirmed": conf_long or conf_short}
        for w in WINDOWS:
            xi = ent + w
            stock = (px["close"].iloc[xi] / entry_px - 1.0) if xi < len(px) else np.nan
            mkt = (sval[si + w] / sval[si] - 1.0) if 0 <= si and si + w < len(sval) else np.nan
            row[f"exc_{w}"] = (stock - mkt) if pd.notna(stock) and pd.notna(mkt) else np.nan  # UNSIGNED excess
        rows.append(row)
    return pd.DataFrame(rows)


def build(symbols: list[str]) -> pd.DataFrame:
    spy = data.benchmark()
    parts = [e for sym in symbols if not (e := events(sym, spy)).empty]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def run(df: pd.DataFrame):
    if df.empty:
        print("no events"); return
    print(f"=== PEAD reaction-entry: {len(df)} events, {df['symbol'].nunique()} symbols "
          f"({df['ann'].min().date()}->{df['ann'].max().date()}) ===")
    print("excess = (trade return - SPY) signed by direction; +ve = drift in the surprise direction.")
    print("median is the honest center (meme-era 60d returns are fat-tailed); mean winsorized +/-50%.\n")

    def stats(sub, label):
        if sub.empty:
            print(f"{label:32} n=0"); return
        line = f"{label:32} n={len(sub):5}"
        for w in WINDOWS:
            x = (sub[f"exc_{w}"] * sub["dir"]).dropna()      # sign by trade direction = drift
            wins = x.clip(-0.5, 0.5)
            line += f" | {w}d med {x.median():+.4f} mean {wins.mean():+.4f} (hit {(x>0).mean():.0%})"
        print(line)

    conf = df[df["confirmed"]]
    rej = df[~df["confirmed"]]
    print("-- CONFIRMED (reaction broke the reference range) --")
    stats(conf[conf["dir"] == 1], "  beat + breakout (LONG)")
    stats(conf[conf["dir"] == -1], "  miss + breakdown (SHORT)")
    print("-- REJECTED (surprise, no break) -- the filter's control group --")
    stats(rej[rej["dir"] == 1], "  beat, no breakout")
    stats(rej[rej["dir"] == -1], "  miss, no breakdown")
    print("-- baselines --")
    stats(df[df["dir"] == 1], "  all beats")
    stats(df[df["dir"] == -1], "  all misses")
    # robustness: is the beat/miss asymmetry stable across years, or a 2020-21 artifact?
    print("\n-- 60d excess by year (median, hit%) -- is the asymmetry regime-stable? --")
    df["yr"] = df["ann"].dt.year
    for yr, g in df.groupby("yr"):
        b = g[g["dir"] == 1]["exc_63"].dropna()                # beats: drift = +exc (long)
        m = (-g[g["dir"] == -1]["exc_63"]).dropna()            # misses: short drift = -exc
        if len(b) + len(m) < 20:
            continue
        print(f"  {yr}: beats n={len(b):4} med {b.median():+.4f} hit {(b>0).mean():.0%}  | "
              f"short-miss n={len(m):4} med {m.median():+.4f} hit {(m>0).mean():.0%}")

    print("\nread: the confirmation gate earns its keep only if CONFIRMED drift clearly exceeds "
          "REJECTED drift in the same direction. Overlapping windows inflate t -- the gauntlet "
          "below is the deflated, non-overlapping test.")
    return df


def gauntlet(df: pd.DataFrame, split="2023-01-01"):
    """The PROPER test (always, per the project rule):
    (A) cross-sectional IC of signed surprise vs forward excess -- sector-neutral, multi-horizon,
        t>=3 (Harvey-Liu-Zhu), Bonferroni, IS/OOS+embargo (reuses backtest.evaluate).
    (B) NON-OVERLAPPING quarterly portfolio: each quarter, sector-neutralize surprise, form
        terciles, and track three legs -- long-beat, short-miss, long/short. The 63d hold ~= one
        quarter, so the quarterly series is non-overlapping -> the t-stat is honest (no window
        overlap inflation). Reports per-quarter mean, t, annualized Sharpe, hit, and IS/OOS."""
    panel = df.rename(columns={f"exc_{h}": f"excess_{h}" for h in WINDOWS}).copy()
    panel = panel[panel["sector"] != "Unknown"]

    print("\n=== (A) cross-sectional IC gauntlet: signed surprise vs forward excess ===")
    res = backtest.evaluate(panel, sector_neutral=True, split=split, features=["surprise"])
    print(res.to_string(index=False))
    print(f"  bars: t>=3 | Bonferroni t>={res.attrs['t_bonf']} for {res.attrs['n_tests']} tests")

    print("\n=== (B) non-overlapping quarterly portfolio (63d hold ~ 1 quarter) ===")
    panel = panel.dropna(subset=["surprise", "excess_63"]).copy()
    panel["q"] = panel["date"].dt.to_period("Q")
    legs = {"long_beat": [], "short_miss": [], "long_short": []}
    qs = []
    for q, g in panel.groupby("q"):
        if len(g) < 30:
            continue
        g = g.copy()
        g["sn"] = g["surprise"] - g.groupby("sector")["surprise"].transform("mean")  # sector-neutral
        hi = g[g["sn"] >= g["sn"].quantile(0.8)]
        lo = g[g["sn"] <= g["sn"].quantile(0.2)]
        if len(hi) < 3 or len(lo) < 3:
            continue
        lb = hi["excess_63"].mean()                 # long the big beats
        sm = -lo["excess_63"].mean()                # short the big misses
        legs["long_beat"].append(lb)
        legs["short_miss"].append(sm)
        legs["long_short"].append(lb + sm)          # dollar-neutral L/S
        qs.append(q)
    sp = pd.Period(pd.Timestamp(split), "Q")
    print(f"{'leg':12} {'n_q':>4} {'q_mean':>8} {'t':>6} {'annSharpe':>10} {'hit':>5} {'IS':>8} {'OOS':>8}")
    for name, series in legs.items():
        s = pd.Series(series, index=qs).dropna()
        if len(s) < 4:
            continue
        t = s.mean() / s.std() * np.sqrt(len(s)) if s.std() else np.nan
        sharpe = s.mean() / s.std() * 2 if s.std() else np.nan      # quarterly->annual (~x2)
        is_m = s[s.index < sp].mean()
        oos_m = s[s.index > sp].mean()
        print(f"{name:12} {len(s):4d} {s.mean():+8.4f} {t:6.2f} {sharpe:10.2f} "
              f"{(s>0).mean():5.0%} {is_m:+8.4f} {oos_m:+8.4f}")
    print("\nread: long_short is the tradeable spread; short_miss isolates the downside drift "
          "(the part that worked). t and IS/OOS here are non-overlapping -> honest.")
    return res


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    syms = {p.stem for p in EC.glob("*.parquet")}
    uni = sorted(syms & {p.stem for p in (data.EARN / "eps_history").glob("*.parquet")})
    print(f"universe: {len(uni)} symbols with earnings_calendar + eps_history\n")
    panel = build(uni)
    run(panel)
    gauntlet(panel)
