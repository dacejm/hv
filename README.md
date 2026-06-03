# hv — a measurement-first quant research framework

A Python research stack for equity/options signal construction, built on one rule:
**nothing is trusted until it has been measured.** Every component is assigned a *lane*
(DIRECTION / SIZING-VOL / EXPRESSION / CONTEXT / FLAGS) and a validated *horizon*, and only
signals that clear a strict statistical bar are allowed to drive decisions. The organizing
principle — "macro is not a directional signal" — generalizes: almost nothing is, and the
registry says so explicitly.

> Data, third-party research, and API keys are intentionally excluded (see `.gitignore`).
> This repo is the **code only**. Accessors expect local parquet datasets under `data/`.

## The measurement bar (`quant/backtest.py`)
Cross-sectional IC across horizons (1/5/21/63d), sector-neutral, IS/OOS with an embargo,
t-stat against the **t ≥ 3** threshold (Harvey-Liu-Zhu) *and* a Bonferroni threshold for the
number of (feature × horizon) tests run. A signal is real only if it clears both **and** holds OOS.

## Lanes (`quant/layers.py` — derived from measurement)
| Lane | Components | Verdict |
|------|-----------|---------|
| **DIRECTION** | `roc.growth_accel`, `roic` | validated cross-sectional pickers (sector-neutral, t≥3, OOS-stable) |
| **SIZING/VOL** | `gex`, `sizing` | GEX vol-regime (neg-GEX → higher fwd realized vol); vol-target + HRP + half-Kelly |
| **EXPRESSION** | `options_vs_stock`, `intrinsic` | 3-factor surface (IVP + term-slope + RR25); WACC scenarios |
| **CONTEXT** | `regime`, `screeners`, `rnd`, `pcr`, `nodes` | orient/label only — never a buy signal |
| **FLAGS** | `anomaly` | z-vs-history deviation detector |

## Key modules
- `quant/data.py` — point-in-time accessors (filing-lag, no lookahead).
- `quant/roc.py` — revenue growth / growth-acceleration ("2nd derivative").
- `quant/gex.py` — index gamma exposure → vol-regime conditioner.
- `quant/regime.py` — 4-quadrant macro (growth × inflation) + 2-state HMM vol-state + premia tilt.
- `quant/sizing.py` — vol-targeting, min-var / risk-parity / **HRP**, Ledoit-Wolf shrinkage,
  regime-conditional (stress-correlation) covariance, half-Kelly, Calmar/CVaR eval.
- `quant/rnd.py` — Breeden-Litzenberger risk-neutral density anomaly heatmap.
- `quant/book.py` — the live book: validated DIRECTION × validated SIZING, lanes respected.
- `candidates.py` — runs new direction ideas through the full gauntlet (accruals, PEAD/SUE,
  ROIC, 52-week-high). ROIC validated (t=3.81 @63d, OOS-stable); the others did not clear.

## Running
Components are import-and-call (`from quant import book; book.build(universe, asof=...)`).
Standalone measurement scripts: `python candidates.py`, `python pcr.py`, `python verify.py`,
`python -c "from quant import sizing; sizing.compare([...])"`.

Status: research framework, not investment advice. NFA.
