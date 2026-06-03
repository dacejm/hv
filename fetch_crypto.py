"""Fetch BTC/ETH daily history from Binance klines -- free, no API key. Feeds the book's trend
sleeve (crypto trend is uncorrelated to equities and high-Sharpe; vol-scaled so it can't dominate).
Writes data/crypto/{SYM}.parquet.

  python fetch_crypto.py [BTCUSDT ETHUSDT ...]
"""
import json
import sys
import time
import urllib.request
from pathlib import Path

import pandas as pd

OUT = Path("data/crypto")


def klines(sym, start_ms=1451606400000):       # 2016-01-01
    rows, start = [], start_ms
    while True:
        u = f"https://api.binance.com/api/v3/klines?symbol={sym}&interval=1d&startTime={start}&limit=1000"
        try:
            d = json.load(urllib.request.urlopen(urllib.request.Request(u, headers={"User-Agent": "hv"}), timeout=30))
        except Exception as e:
            print(f"{sym} fetch error: {e}"); return None
        if not d:
            break
        rows += d
        start = d[-1][0] + 86_400_000
        if len(d) < 1000:
            break
        time.sleep(0.2)
    df = pd.DataFrame(rows, columns=["t", "o", "h", "l", "close", "v", "ct", "qv", "n", "tb", "tq", "ig"])
    df["date"] = pd.to_datetime(df["t"], unit="ms")
    df["close"] = df["close"].astype(float)
    return df[["date", "close"]]


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    syms = sys.argv[1:] or ["BTCUSDT", "ETHUSDT"]
    for s in syms:
        df = klines(s)
        if df is not None:
            df.to_parquet(OUT / f"{s}.parquet")
            print(f"{s}: {len(df)} rows {df['date'].min().date()}->{df['date'].max().date()}")
