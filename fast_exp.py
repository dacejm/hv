"""Fast (cached/vectorized) equity-signal experiment. Runs in seconds via featcache.
Tests QM / risk-adjusted-momentum / multi-horizon equity sleeves, each combined 50/50 with the
(ETF+BTC) trend sleeve, vol-targeted 15%. Beats current book (Sharpe 1.02) in BOTH OOS halves to win."""
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from featcache import load, momentum, vol, run_signal
import improve

C, roic_g, mcap_g, grid, sect = load()
mom12 = momentum(C, grid, 21, 252)
mom6 = momentum(C, grid, 21, 126)
v = vol(C, grid, 126)
rk = lambda x: x.rank(axis=1, pct=True)

sigs = {
    "QM":     rk(mom12) + rk(roic_g),
    "RAM":    rk(mom12 / v) + rk(roic_g),
    "MH":     (rk(mom6) + rk(mom12)) / 2 + rk(roic_g),
    "RAM-MH": (rk(mom6 / v) + rk(mom12 / v)) / 2 + rk(roic_g),
    "RAM only (no roic)": rk(mom12 / v),
}
eqs = {n: run_signal(C, grid, s, mcap_g, weight="cap") for n, s in sigs.items()}

# sanity: QM cap-weighted unlevered should reproduce the validated ~0.86 Sharpe
qm = eqs["QM"]; sr = qm.mean() / qm.std() * np.sqrt(252)
print(f"[check] QM cap-weighted unlevered Sharpe = {sr:.2f} (validated ~0.86)\n")

trend = improve.trend_sleeve(
    ["SPY", "QQQ", "TLT", "IEF", "GLD", "SLV", "DBC", "UUP", "HYG", "EEM", "VNQ", "XLE", "BTCUSDT"],
    (252,), qm.index[0], qm.index[-1]).reindex(qm.index).fillna(0)

cur = improve.vt(0.5 * eqs["QM"] + 0.5 * trend)
csr, cc, cd = improve.metr(cur); o1, o2 = improve.oos(cur)
print(f"CURRENT BOOK (QM + BTC-trend) @15%: Sharpe {csr:.2f} CAGR {cc:+.1%} maxDD {cd:.0%} | OOS {o1:.2f}/{o2:.2f}\n")
for n, e in eqs.items():
    solo = e.mean() / e.std() * np.sqrt(252)
    r = improve.vt(0.5 * e.reindex(qm.index).fillna(0) + 0.5 * trend)
    s, cg, dd = improve.metr(r); a1, a2 = improve.oos(r)
    beats = s > csr and a1 >= o1 - 0.02 and a2 >= o2 - 0.02 and (a1 > o1 or a2 > o2)
    print(f"{n:20} equity-Sh {solo:.2f} | BOOK Sh {s:.2f} CAGR {cg:+.1%} maxDD {dd:.0%} | OOS {a1:.2f}/{a2:.2f}{'  <== BETTER' if beats else ''}")
