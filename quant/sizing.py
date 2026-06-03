"""Position sizing & basket weighting — the paper-grounded risk layer.

Replaces book.py's crude "0.5x in amplifying GEX" throttle, which the risk-and-execution
book (Bk07) and the combo test showed is the wrong SHAPE. The validated recipe, assembled
from the corpus:

  - vol-targeting (Bk07 3): size each name ~ target_vol / sigma_i, so a 1-sigma move costs
    the same dollars regardless of the name's vol. Portfolio levered to a constant target vol.
  - basket weights via MIN-VARIANCE / RISK-PARITY / HRP (portfolio-construction Bk12 Part VI-VIII)
    on a LEDOIT-WOLF shrunk covariance (94% of sample-cov eigenvalues are MP-bulk noise; raw
    inverse is "error maximization", Michaud). HRP (Lopez de Prado 2016) never inverts the matrix.
  - half-Kelly cap (Bk07 2.2): full Kelly = mu/sigma^2; estimation CV ~40% -> half-Kelly is
    optimal and roughly halves drawdown. Never lever past it.
  - GEX/HMM feed the vol FORECAST that scales gross -- at their OWN (short) horizon (the lane rule).
  - judged by Calmar / max-drawdown / CVaR (Bk07 4, Bk12 29), not raw Sharpe.

Everything trailing-only (no lookahead); compare() walks forward to MEASURE that min-var/HRP
actually cut realized vol and lift Calmar over equal-weight on the real universe.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import data

TRADING_DAYS = 252


# ---- covariance ------------------------------------------------------------------------
def returns_matrix(symbols, asof=None, lookback=TRADING_DAYS) -> pd.DataFrame:
    """Daily-return panel (cols=symbols), trailing `lookback` rows up to asof. Inner-joined
    on common dates so every name has the full window (point-in-time: nothing after asof)."""
    cols = {}
    for s in symbols:
        try:
            px = data.ohlcv(s)
        except FileNotFoundError:
            continue
        if asof is not None:
            px = px[px["date"] <= pd.Timestamp(asof)]
        r = px.set_index("date")["close"].pct_change()
        if r.notna().sum() >= lookback:
            cols[s] = r
    if not cols:
        return pd.DataFrame()
    R = pd.DataFrame(cols).dropna()
    return R.tail(lookback)


def ledoit_wolf_cov(R: pd.DataFrame) -> np.ndarray:
    """Ledoit-Wolf shrunk covariance (well-conditioned, invertible). Falls back to sample
    cov if sklearn is unavailable. Annualized."""
    try:
        from sklearn.covariance import LedoitWolf
        cov = LedoitWolf().fit(R.values).covariance_
    except Exception:
        cov = np.cov(R.values, rowvar=False)
    return cov * TRADING_DAYS


def _stress_mask(R: pd.DataFrame, recent=21, q=0.66):
    """Per-row stress flag from the equal-weight basket's rolling 21d realized vol (top
    tercile of the window = stress). Also the current stress probability = recent stress
    fraction. Used to split the window for regime-conditional covariance."""
    port = R.mean(axis=1)
    rv = port.rolling(21).std()
    mask = rv > rv.quantile(q)
    p_now = float(mask.tail(recent).mean()) if mask.notna().any() else 0.0
    return mask.fillna(False), p_now


def conditional_cov(R: pd.DataFrame) -> np.ndarray:
    """Stress-correlation covariance (Bk07 6.1): use full-window VOLATILITIES but STRESS-period
    CORRELATIONS, always. The 2022 failure was that bonds look uncorrelated-to-equities in calm
    data, so min-var/risk-parity pile into duration -- then correlations spike positive in the
    shock. Assuming stress-level correlations structurally removes that diversification illusion
    (a reactive recent-vol blend lags the shock and makes it worse -- measured)."""
    mask, _ = _stress_mask(R)
    stress = R[mask]
    full = ledoit_wolf_cov(R)
    if len(stress) < 30:
        return full
    Ss = ledoit_wolf_cov(stress)
    ds = np.sqrt(np.diag(Ss))
    corr_s = Ss / np.outer(ds, ds)                      # stress correlation matrix
    vol_full = np.sqrt(np.diag(full))                   # full-window vols (realistic scaling)
    return corr_s * np.outer(vol_full, vol_full)


def cov_estimate(R: pd.DataFrame, conditional=False) -> np.ndarray:
    return conditional_cov(R) if conditional else ledoit_wolf_cov(R)


# ---- weighting schemes (all return a weight vector summing to 1, long-only) -------------
def equal_weight(cov) -> np.ndarray:
    n = cov.shape[0]
    return np.full(n, 1.0 / n)


def inverse_vol(cov) -> np.ndarray:
    d = np.sqrt(np.diag(cov))
    iv = np.where(d > 0, 1.0 / np.where(d > 0, d, 1.0), 0.0)   # zero-variance (halted/flatlined) -> 0 weight
    s = iv.sum()
    return iv / s if s > 0 else equal_weight(cov)


def min_variance(cov, long_only=True) -> np.ndarray:
    """w = Sigma^-1 1 / (1' Sigma^-1 1); long-only via simple active-set clipping iteration."""
    n = cov.shape[0]
    ones = np.ones(n)
    try:
        inv = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        return equal_weight(cov)
    w = inv @ ones
    w = w / w.sum()
    if long_only and (w < 0).any():
        # drop the most-negative name and re-solve until all non-negative (cheap projected solve)
        active = np.ones(n, bool)
        for _ in range(n):
            sub = cov[np.ix_(active, active)]
            try:
                wi = np.linalg.inv(sub) @ np.ones(active.sum())
            except np.linalg.LinAlgError:
                break
            wi = wi / wi.sum()
            full = np.zeros(n)
            full[active] = wi
            if (wi >= -1e-9).all():
                return np.clip(full, 0, None) / np.clip(full, 0, None).sum()
            active[np.where(active)[0][wi.argmin()]] = False
            if active.sum() <= 1:
                break
        w = np.clip(w, 0, None)
        w = w / w.sum() if w.sum() > 0 else equal_weight(cov)
    return w


def risk_parity(cov, iters=500, tol=1e-8) -> np.ndarray:
    """Equal risk contribution: iterate w_i <- (1/MCR_i), normalize, until stable."""
    n = cov.shape[0]
    w = inverse_vol(cov)
    for _ in range(iters):
        mrc = cov @ w
        rc = np.clip(w * mrc, 1e-12, None)        # risk contributions, guarded positive
        target = rc.mean()
        w_new = w * (target / rc) ** 0.5
        w_new = np.clip(w_new, 1e-12, None)
        w_new /= w_new.sum()
        if np.abs(w_new - w).max() < tol:
            break
        w = w_new
    return w


def hrp(R: pd.DataFrame, cov=None) -> np.ndarray:
    """Hierarchical Risk Parity (Lopez de Prado 2016): cluster by correlation distance,
    quasi-diagonalize, recursive inverse-variance bisection. Never inverts the covariance.
    If `cov` is supplied (e.g. regime-conditional), it drives both the clustering correlations
    and the bisection variances; otherwise the sample correlation/cov of R is used."""
    from scipy.cluster.hierarchy import leaves_list, linkage
    from scipy.spatial.distance import squareform
    if cov is None:
        corr = R.corr().values
        cov = (R.cov() * TRADING_DAYS).values
    else:
        d = np.sqrt(np.diag(cov))
        corr = cov / np.outer(d, d)
    n = cov.shape[0]
    if n < 3:
        return inverse_vol(cov)
    dist = np.sqrt(np.clip(0.5 * (1 - corr), 0, None))
    link = linkage(squareform(dist, checks=False), method="single")
    # quasi-diagonalization via scipy's iterative leaf-order recovery (a recursive tree walk
    # blows the Python recursion limit past ~500 assets; leaves_list is iterative + C-backed).
    order = list(leaves_list(link))

    # recursive bisection
    w = np.ones(n)
    clusters = [order]
    while clusters:
        c = clusters.pop()
        if len(c) <= 1:
            continue
        half = len(c) // 2
        c0, c1 = c[:half], c[half:]

        def _ivp_var(idx):
            sub = cov[np.ix_(idx, idx)]
            ivp = 1.0 / np.diag(sub)
            ivp /= ivp.sum()
            return ivp @ sub @ ivp
        v0, v1 = _ivp_var(c0), _ivp_var(c1)
        alpha = 1 - v0 / (v0 + v1)
        for i in c0:
            w[i] *= alpha
        for i in c1:
            w[i] *= (1 - alpha)
        clusters += [c0, c1]
    return w / w.sum()


# ---- gross leverage: vol-target + half-Kelly cap ---------------------------------------
def portfolio_vol(w, cov) -> float:
    return float(np.sqrt(w @ cov @ w))


def vol_target_leverage(w, cov, target_vol=0.10) -> float:
    pv = portfolio_vol(w, cov)
    return target_vol / pv if pv > 0 else 1.0


def half_kelly_leverage(w, mu_annual, cov) -> float:
    """Full Kelly leverage for a fixed-mix portfolio = mu_p / sigma_p^2; half-Kelly caps it."""
    mu_p = float(w @ mu_annual)
    var_p = float(w @ cov @ w)
    if var_p <= 0 or mu_p <= 0:
        return 0.0
    return 0.5 * mu_p / var_p


# ---- performance evaluation (Bk07 / Bk12 metrics) --------------------------------------
def evaluate(rets: pd.Series) -> dict:
    rets = pd.Series(rets).dropna()
    if len(rets) < 2:
        return {}
    eq = (1 + rets).cumprod()
    yrs = len(rets) / TRADING_DAYS
    cagr = eq.iloc[-1] ** (1 / yrs) - 1 if yrs > 0 else np.nan
    vol = rets.std() * np.sqrt(TRADING_DAYS)
    sharpe = rets.mean() / rets.std() * np.sqrt(TRADING_DAYS) if rets.std() else np.nan
    dd = eq / eq.cummax() - 1
    max_dd = dd.min()
    calmar = cagr / abs(max_dd) if max_dd < 0 else np.nan
    cvar = rets[rets <= rets.quantile(0.05)].mean()   # expected shortfall, 5%
    return {"CAGR": round(cagr, 4), "vol": round(vol, 4), "Sharpe": round(sharpe, 2),
            "maxDD": round(float(max_dd), 4), "Calmar": round(float(calmar), 2) if pd.notna(calmar) else np.nan,
            "CVaR5": round(float(cvar), 4)}


# ---- the measurement: walk-forward compare weighting schemes ---------------------------
def compare(symbols, lookback=TRADING_DAYS, hold=21, target_vol=0.10, start="2016-01-01",
            conditional=False):
    """Walk forward: each `hold` days estimate trailing cov, form weights under each scheme,
    hold, record. Then evaluate. Trailing-only -> the vol/Calmar gaps are honest, not fit.
    conditional=True uses regime-conditional covariance (calm/stress blend) for the cov-based
    schemes -- A/B this to see the 2022 bond-concentration drawdown shrink."""
    # build a clean common-history panel once
    full = {}
    for s in symbols:
        try:
            px = data.ohlcv(s)
        except FileNotFoundError:
            continue
        r = px.set_index("date")["close"].pct_change()
        if r.notna().sum() > lookback + 252:
            full[s] = r
    R = pd.DataFrame(full).dropna()
    R = R[R.index >= pd.Timestamp(start)]
    if R.shape[1] < 5 or len(R) < lookback + hold:
        print("insufficient panel:", R.shape)
        return
    schemes = {"equal": [], "inv_vol": [], "min_var": [], "risk_parity": [], "HRP": []}
    dates = R.index
    i = lookback
    while i + hold <= len(R):
        train = R.iloc[i - lookback:i]
        fwd = R.iloc[i:i + hold]
        cov = cov_estimate(train, conditional=conditional)
        wmap = {"equal": equal_weight(cov), "inv_vol": inverse_vol(cov),
                "min_var": min_variance(cov), "risk_parity": risk_parity(cov),
                "HRP": hrp(train, cov=cov if conditional else None)}
        for name, w in wmap.items():
            lev = vol_target_leverage(w, cov, target_vol)            # constant-vol target
            port = (fwd.values @ w) * lev
            schemes[name].append(pd.Series(port, index=fwd.index))
        i += hold
    print(f"=== weighting-scheme walk-forward ({R.index[lookback].date()}->{R.index[-1].date()}, "
          f"{R.shape[1]} names, hold={hold}d, target_vol={target_vol:.0%}) ===\n")
    out = {}
    for name, parts in schemes.items():
        s = pd.concat(parts)
        out[name] = evaluate(s)
    tbl = pd.DataFrame(out).T[["CAGR", "vol", "Sharpe", "maxDD", "Calmar", "CVaR5"]]
    print(tbl.to_string())
    print("\nread: vol-targeting pins realized vol near target; min_var/HRP should show the "
          "LOWEST maxDD/CVaR and HIGHEST Calmar -- the paper's claim, measured.")
    return tbl


# ---- live application: leg weights for the book ----------------------------------------
def leg_weights(symbols, asof=None, scheme="hrp", lookback=TRADING_DAYS, conditional=True) -> pd.Series:
    """Long-only basket weights for one leg of the book (min-var/HRP/inv-vol on trailing,
    shrunk cov). conditional=True uses regime-conditional cov (calm/stress blend) for the
    cov-based schemes -- the 2022 fix. Returns a Series indexed by the symbols with history."""
    R = returns_matrix(symbols, asof=asof, lookback=lookback)
    if R.empty or R.shape[1] < 2:
        return pd.Series(dtype=float)
    cov = cov_estimate(R, conditional=conditional)
    if scheme == "hrp":
        w = hrp(R, cov=cov if conditional else None)
    else:
        w = {"min_var": min_variance, "inv_vol": inverse_vol,
             "risk_parity": risk_parity}.get(scheme, min_variance)(cov)
    return pd.Series(w, index=R.columns)
