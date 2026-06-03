"""Risk-neutral density anomaly (the red/blue/star heatmap).

Breeden-Litzenberger: the risk-neutral PDF is the discounted second derivative of the
call price wrt strike, RND(K) = e^{rT} d2C/dK2. We build C(K) from the IV smile, take
its RND, and compare to the RND a FLAT (ATM) vol would give (a lognormal). The excess is
the market's extra/less probability mass at each strike; z-scored across strikes:

  red  (z>0)  = excess density   (market prices MORE probability here than lognormal)
  blue (z<0)  = deficit density
  star |z|>=1.8 = anomaly
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.parquet as pq
from scipy.stats import norm

IDX = Path(__file__).resolve().parent.parent / "data" / "QQQ_SPY_IWM"
R = 0.04
STAR = 1.8


def _bs_call(S, K, T, iv):
    iv = np.clip(iv, 1e-3, 5); T = max(T, 1e-4)
    d1 = (np.log(S / K) + (R + 0.5 * iv**2) * T) / (iv * np.sqrt(T))
    d2 = d1 - iv * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-R * T) * norm.cdf(d2)


def rnd(symbol: str = "SPY", target_dte: int = 30, asof: pd.Timestamp | None = None) -> pd.DataFrame:
    path = IDX / f"{symbol}_options.parquet"
    dts = pq.read_table(path, columns=["date"])["date"]
    maxd = pc.max(dts).as_py() if asof is None else \
        pc.max(pc.filter(dts, pc.less_equal(dts, pd.Timestamp(asof)))).as_py()
    df = pq.read_table(path, columns=["date", "type", "strike", "expiration", "implied_volatility"],
                       filters=[("date", "=", maxd)]).to_pandas()
    und = pd.read_parquet(IDX / f"{symbol}_underlying.parquet", columns=["date", "close"])
    und["date"] = pd.to_datetime(und["date"]).dt.normalize()
    spot = float(und[und["date"] <= pd.Timestamp(maxd)]["close"].iloc[-1])

    df["dte"] = (pd.to_datetime(df["expiration"]) - pd.Timestamp(maxd)).dt.days
    exp = df.iloc[(df["dte"] - target_dte).abs().argmin()]["expiration"]      # nearest target expiry
    c = df[(df["expiration"] == exp) & (df["type"].astype(str).str.lower().str.startswith("c"))
           & (df["implied_volatility"] > 0)].copy()
    c = c[(c["strike"] >= 0.6 * spot) & (c["strike"] <= 1.4 * spot)].sort_values("strike")
    if len(c) < 8:
        return pd.DataFrame()
    T = float(c["dte"].iloc[0]) / 365.0

    K = np.linspace(c["strike"].min(), c["strike"].max(), 200)
    iv_smile = np.interp(K, c["strike"], c["implied_volatility"])
    atm_iv = float(np.interp(spot, c["strike"], c["implied_volatility"]))

    def density(ivs):
        C = _bs_call(spot, K, T, ivs)
        d2 = np.gradient(np.gradient(C, K), K)
        d2 = np.clip(np.exp(R * T) * d2, 0, None)
        area = np.sum((d2[:-1] + d2[1:]) / 2 * np.diff(K))                   # trapezoid integral
        return d2 / area if area > 0 else d2                                 # normalize to a PDF

    rnd_smile = density(iv_smile)
    rnd_flat = density(np.full_like(K, atm_iv))                              # lognormal baseline
    excess = rnd_smile - rnd_flat
    z = (excess - excess.mean()) / excess.std()
    out = pd.DataFrame({"strike": K, "moneyness": K / spot, "rnd": rnd_smile,
                        "lognormal": rnd_flat, "excess": excess, "z": z})
    out["color"] = np.where(out["z"] >= 0, "red", "blue")
    out["star"] = out["z"].abs() >= STAR
    out.attrs["spot"], out.attrs["date"], out.attrs["dte"] = spot, str(pd.Timestamp(maxd).date()), int(c["dte"].iloc[0])
    return out
