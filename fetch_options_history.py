"""Download the MISSING single-name options HISTORY (IV + greeks + bid/ask) from the free
DoltHub post-no-preference/options DB -- the SAME provider as the local earnings data.

This is downloadable HISTORY (2019-01 .. 2024-06), not the CBOE forward snapshots you had to
wait to accumulate. It unblocks single-name term-structure / skew(RR25) / RND, which needed
deep-dated chains that snapshots can't provide retroactively.

Caveat (honest): this source carries IV + full greeks + bid/ask but NOT open_interest/volume
(DoltHub lacks them). That's fine for the validated stack -- OI is only needed for the INDEX
GEX/nodes/PCR, and those index chains you already have locally. Single-name OI history still
needs a paid provider (AV premium / ORATS).

Access pattern: the table PK leads with `date`, so we query one date at a time for the whole
watchlist (fast, ~2s); symbol-only scans time out. Resumable; writes data/options_history/{SYM}.parquet
with columns normalized to the local index-options schema (type / implied_volatility / greeks).

  python fetch_options_history.py AAPL MSFT NVDA --start 2021-01-01 [--end 2024-06-14]
"""
from __future__ import annotations

import sys
import time
import urllib.parse
import urllib.request
import json
from pathlib import Path

import pandas as pd

from quant import data

API = "https://www.dolthub.com/api/v1alpha1/post-no-preference/options/master"
OUTDIR = data.DATA / "options_history"
COVERAGE_MAX = "2024-06-14"          # DoltHub coverage ends mid-2024
CHUNK = 8                            # symbols per query (keep response well under any row cap)


def _q(sql: str, retries: int = 3):
    u = API + "?" + urllib.parse.urlencode({"q": sql})
    for i in range(retries):
        try:
            r = json.load(urllib.request.urlopen(
                urllib.request.Request(u, headers={"User-Agent": "hv"}), timeout=60))
            if r.get("query_execution_status") == "Success":
                return r.get("rows", [])
        except Exception:
            time.sleep(2 * (i + 1))
    return None


def _trading_dates(start, end) -> list[str]:
    spy = pd.read_parquet(data.DATA / "QQQ_SPY_IWM" / "SPY_underlying.parquet", columns=["date"])
    spy["date"] = pd.to_datetime(spy["date"])
    end = min(pd.Timestamp(end), pd.Timestamp(COVERAGE_MAX))
    d = spy[(spy["date"] >= pd.Timestamp(start)) & (spy["date"] <= end)]["date"]
    return [x.strftime("%Y-%m-%d") for x in sorted(d.unique())]


def fetch(symbols, start="2019-01-01", end=COVERAGE_MAX):
    OUTDIR.mkdir(exist_ok=True)
    symbols = [s.upper() for s in symbols]
    dates = _trading_dates(start, end)
    # resume: skip dates already stored for ALL requested symbols
    existing = {}
    for s in symbols:
        f = OUTDIR / f"{s}.parquet"
        existing[s] = set(pd.read_parquet(f, columns=["date"])["date"].astype(str)) if f.exists() else set()
    todo = [d for d in dates if any(d not in existing[s] for s in symbols)]
    print(f"{len(symbols)} symbols x {len(dates)} dates ({start}..{min(end, COVERAGE_MAX)}); "
          f"{len(todo)} dates to fetch")

    buf = {s: [] for s in symbols}
    for i, d in enumerate(todo):
        for j in range(0, len(symbols), CHUNK):
            chunk = symbols[j:j + CHUNK]
            inlist = ",".join(f"'{s}'" for s in chunk)
            rows = _q(f"SELECT date,act_symbol,expiration,strike,call_put,bid,ask,vol,"
                      f"delta,gamma,theta,vega,rho FROM option_chain "
                      f"WHERE date='{d}' AND act_symbol IN ({inlist})")
            if rows is None:
                print(f"  ! query failed {d} {chunk}"); continue
            for r in rows:
                buf[r["act_symbol"]].append(r)
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(todo)} dates...")
        time.sleep(0.15)
    # write per symbol (merge with existing, normalize columns to the local options schema)
    for s in symbols:
        if not buf[s]:
            continue
        df = pd.DataFrame(buf[s]).rename(columns={"call_put": "type", "vol": "implied_volatility"})
        for c in ["strike", "bid", "ask", "implied_volatility", "delta", "gamma", "theta", "vega", "rho"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df["type"] = df["type"].str.lower()                       # 'Call'->'call'
        f = OUTDIR / f"{s}.parquet"
        if f.exists():
            df = pd.concat([pd.read_parquet(f), df], ignore_index=True)
        df = df.drop_duplicates(["date", "expiration", "strike", "type"]).sort_values(["date", "expiration", "strike"])
        df.to_parquet(f)
        print(f"  wrote {s}: {len(df)} rows -> {f.name}")


if __name__ == "__main__":
    argv = sys.argv[1:]
    start = next((argv[i + 1] for i, a in enumerate(argv) if a == "--start"), "2019-01-01")
    end = next((argv[i + 1] for i, a in enumerate(argv) if a == "--end"), COVERAGE_MAX)
    flagvals = {start, end}
    syms = [a for a in argv if not a.startswith("--") and a not in flagvals]   # drop flags AND their values
    if not syms:
        print(__doc__); sys.exit(0)
    fetch(syms, start=start, end=end)
