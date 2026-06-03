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

from quant import data

WINDOWS = (5, 21, 60)
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
        si = int(np.searchsorted(sidx, np.datetime64(px["date"].iloc[ent]), side="left"))  # SPY entry
        row = {"symbol": sym, "ann": e["ann"], "dir": sgn, "confirmed": conf_long or conf_short}
        for w in WINDOWS:
            xi = ent + w
            stock = (px["close"].iloc[xi] / entry_px - 1.0) if xi < len(px) else np.nan
            mkt = (sval[si + w] / sval[si] - 1.0) if 0 <= si and si + w < len(sval) else np.nan
            row[f"exc_{w}"] = (stock - mkt) * sgn if pd.notna(stock) and pd.notna(mkt) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def run(symbols: list[str]):
    spy = data.benchmark()
    parts = []
    for sym in symbols:
        e = events(sym, spy)
        if not e.empty:
            parts.append(e)
    if not parts:
        print("no events"); return
    df = pd.concat(parts, ignore_index=True)

    print(f"=== PEAD reaction-entry: {len(df)} events, {df['symbol'].nunique()} symbols "
          f"({df['ann'].min().date()}->{df['ann'].max().date()}) ===")
    print("excess = (trade return - SPY) signed by direction; +ve = drift in the surprise direction.")
    print("median is the honest center (meme-era 60d returns are fat-tailed); mean winsorized +/-50%.\n")

    def stats(sub, label):
        if sub.empty:
            print(f"{label:32} n=0"); return
        line = f"{label:32} n={len(sub):5}"
        for w in WINDOWS:
            x = sub[f"exc_{w}"].dropna()
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
        b, m = g[g["dir"] == 1]["exc_60"].dropna(), g[g["dir"] == -1]["exc_60"].dropna()
        if len(b) + len(m) < 20:
            continue
        print(f"  {yr}: beats n={len(b):4} med {b.median():+.4f} hit {(b>0).mean():.0%}  | "
              f"misses n={len(m):4} med {m.median():+.4f} hit {(m>0).mean():.0%}")

    print("\nread: the confirmation gate earns its keep only if CONFIRMED drift clearly exceeds "
          "REJECTED drift in the same direction. Overlapping windows inflate t -- treat hit-rate "
          "and the confirmed-minus-rejected gap as the honest signal.")
    return df


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    syms = {p.stem for p in EC.glob("*.parquet")}
    uni = sorted(syms & {p.stem for p in (data.EARN / "eps_history").glob("*.parquet")})
    print(f"universe: {len(uni)} symbols with earnings_calendar + eps_history\n")
    run(uni)
