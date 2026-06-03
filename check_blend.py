"""Is ROIC complementary to growth_accel, or redundant? Proper test (per the project rule):
  1. cross-sectional correlation (per-quarter spearman, averaged) -- low corr => blending raises
     IR (Grinold's Fundamental Law); high corr => redundant.
  2. the deflated gauntlet on growth_accel / roic / a sector-neutral z-blend -- only wire the
     blend if its ICIR beats the better single signal AND it still clears t>=3 / OOS.
"""
import sys
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
warnings.filterwarnings("ignore")

import candidates
from quant import backtest, data

inc = {p.stem for p in (data.EARN / "income_statement").glob("*.parquet")}
uni = sorted(inc & set(data.sectors().index))
print(f"universe {len(uni)} symbols")
p = candidates.build_panel(uni)
p = p[p["sector"] != "Unknown"].copy()
both = p.dropna(subset=["growth_accel", "roic"])
print(f"panel {len(p)} rows | both growth_accel & roic present: {len(both)} rows\n")

# 1. cross-sectional correlation per quarter
q = p["date"].dt.to_period("Q")
corrs = []
for _, g in both.groupby(both["date"].dt.to_period("Q")):
    if len(g) >= 12 and g["growth_accel"].nunique() > 3 and g["roic"].nunique() > 3:
        corrs.append(g["growth_accel"].corr(g["roic"], method="spearman"))
corr = float(np.nanmean(corrs))
print(f"mean within-quarter rank corr(growth_accel, roic) = {corr:+.3f}  "
      f"({'complementary -> blend can help' if abs(corr) < 0.3 else 'overlapping -> blend may not help'})\n")

# 2. sector-neutral z-blend (requires BOTH signals present)
for f in ["growth_accel", "roic"]:
    gz = p.groupby([q, p["sector"]])[f]
    p[f"z_{f}"] = (p[f] - gz.transform("mean")) / gz.transform("std")
p["combo"] = (p["z_growth_accel"] + p["z_roic"]) / 2          # NaN unless both present

res = backtest.evaluate(p, sector_neutral=True, features=["growth_accel", "roic", "combo"])
print(res.to_string(index=False))
print(f"\nbars: t>=3 | Bonferroni t>={res.attrs['t_bonf']} for {res.attrs['n_tests']} tests")
print("decide: wire the blend only if combo ICIR/t beats BOTH singles at a shared horizon and OOS holds.")
