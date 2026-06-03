"""Put/call ratio + deviations (index ETFs), built AND measured.

pcr_oi  = sum put OI  / sum call OI   (positioning)
pcr_vol = sum put vol / sum call vol  (daily flow)
_z      = deviation vs trailing `window` days (point-in-time: window excludes today)

measure(): does a PCR deviation predict forward returns? (contrarian view: extreme put
demand = fear = bullish forward). Reports rank-IC + spike-conditional forward returns.

  python pcr.py
"""
import sys
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, ".")
from quant import nodes

IDX = nodes.IDX


def pcr_series(symbol="SPY", window=60) -> pd.DataFrame:
    opt = pq.read_table(IDX / f"{symbol}_options.parquet",
                        columns=["date", "type", "open_interest", "volume", "delta"]).to_pandas()
    opt["date"] = pd.to_datetime(opt["date"]).dt.normalize()
    opt["call"] = opt["type"].astype(str).str.lower().str.startswith("c")
    # delta-weighted volume: weight each contract by |delta| so the ratio reflects directional
    # conviction, not far-OTM lottery tickets (markets Bk: Pan-Poteshman use conviction-weighted
    # option flow). Measured against raw pcr_vol below.
    opt["dvol"] = opt["volume"].fillna(0) * opt["delta"].abs().fillna(0)
    g = opt.groupby(["date", "call"])[["open_interest", "volume", "dvol"]].sum().unstack("call")
    df = pd.DataFrame({"call_oi": g[("open_interest", True)], "put_oi": g[("open_interest", False)],
                       "call_vol": g[("volume", True)], "put_vol": g[("volume", False)],
                       "call_dvol": g[("dvol", True)], "put_dvol": g[("dvol", False)]}).dropna()
    df["pcr_oi"] = df["put_oi"] / df["call_oi"]
    df["pcr_vol"] = df["put_vol"] / df["call_vol"]
    df["pcr_dw"] = df["put_dvol"] / df["call_dvol"]          # delta-weighted (conviction) PCR
    df = df.replace([np.inf, -np.inf], np.nan)               # zero call OI/vol -> NaN, not inf (don't poison rolling stats)
    for c in ("pcr_oi", "pcr_vol", "pcr_dw"):
        m = df[c].rolling(window).mean().shift(1)        # trailing, excludes today (point-in-time)
        s = df[c].rolling(window).std().shift(1)
        df[c + "_z"] = (df[c] - m) / s
    return df.reset_index()


def measure(symbol="SPY", hold=5):
    df = pcr_series(symbol)
    und = pd.read_parquet(IDX / f"{symbol}_underlying.parquet", columns=["date", "close"])
    und["date"] = pd.to_datetime(und["date"]).dt.normalize()
    und = und.sort_values("date")
    und["fwd"] = und["close"].shift(-hold) / und["close"] - 1
    m = df.merge(und[["date", "fwd"]], on="date")
    print(f"=== {symbol} PCR vs fwd {hold}d ({m['date'].min().date()}->{m['date'].max().date()}) ===")
    base = m["fwd"].mean()
    for col in ("pcr_oi_z", "pcr_vol_z", "pcr_dw_z"):
        s = m.dropna(subset=[col, "fwd"])
        ic = s[col].corr(s["fwd"], method="spearman")
        hi, lo = s[s[col] > 2], s[s[col] < -2]
        print(f"\n{col}: n={len(s)}  rank-IC={ic:+.3f}  (t~{ic*np.sqrt(len(s)):+.1f})")
        print(f"  spike  z>+2 (fear):  n={len(hi):4} mean fwd {hi['fwd'].mean():+.4f}")
        print(f"  spike  z<-2 (greed): n={len(lo):4} mean fwd {lo['fwd'].mean():+.4f}")
        print(f"  baseline:                          {base:+.4f}")


if __name__ == "__main__":
    measure(sys.argv[1] if len(sys.argv) > 1 else "SPY")
