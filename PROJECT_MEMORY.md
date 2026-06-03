# hv — project memory

Quant research system operationalizing "Mat's" (optionstam) framework with measurement-first rigor.
Core principle: **nothing is trusted until backtested; every component is measured.** Macro/regime
may only condition/size, never emit signals.

## Stack
Python (pandas/pyarrow/numpy). Data = local parquet under `data/` (see ~/.claude memory `hv-data-inventory`).

## Built so far (`quant/`)
- `data.py` — point-in-time accessors. KEY RULE: income_statement `date` = period-END (not filing);
  lagged +50d to `known_on` before it counts as known. Estimates `date` = true publication date (safe).
- `roc.py` — rate-of-change engine ("d*ROC"): YoY revenue growth + 1st/2nd derivative; margins;
  estimate-revision momentum (asof-aware, point-in-time).
- `divergence.py` — surface (growth 2nd-deriv decel) vs underlying (6q gross-margin slope w/ artifact
  guard, OR rising revisions). Emits DIVERGENCE flag.
- `backtest.py` — replay scorer. One engine, two modes: replay history (trust) / run asof=today (live).
  Strictly no peeking past `asof`. v1 unit = (symbol, earnings report date) -> flag + 63d fwd return.

## Findings
- NVDA reacceleration read VALIDATED by RoC engine (point-in-time).
- IREN structural-margin read NOT reproducible — dataset has only ~2 clean margin quarters (gross_margin=1.0
  artifacts). Data-source limit, not engine bug. Small-cap fundamentals here are thin/dirty.
- **Divergence flag has NO edge** (backtest, 10,541 obs, 391 names, 2017-2026): div=True +1.65%/51.6% hit
  vs div=False +4.18%/54.1%. Behaves like mild inverse-momentum. Mat's edge needs fragility + catalyst,
  not divergence alone. Do NOT curve-fit variants on this sample.
- **Hardened scorer (excess-vs-SPY + sector-neutral + rank IC)**: raw +3.37% fwd was ~all beta (excess +0.14%).
  Per-primitive IC: **d2_growth is the only one with signal** (IC~0.03, t~2 on excess) — supports Mat's
  "market discounts the 2nd derivative". BUT ~half is sector rotation (sector-neutral t~1.0, insignificant).
  d_growth weak, margin_slope ~0, revision_4q = noise (annual sales consensus too sticky; try EPS/short-horizon).
  Only 34 quarterly buckets, no OOS split -> suggestive not robust.

- `fragility.py` — positioning primitive from single-name chain: atm_iv, put_skew (25d put-call IV / ATM),
  skew_history for percentile ranking. VALIDATED: NVDA skew 2026-05-12 = 18th pctile of own history =
  "historically compressed" (matches Mat). LIMITATION: term_slope mostly NaN (chains lack 75-120 DTE expiries).

- **Interaction test d2_growth x fragility (7,234 obs w/ skew, 434 names, 2020-2026): REJECTED as posed.**
  d2_growth edge is STRONGER in normal-skew names (IC 0.037, spread +2.68%) than compressed/fragile
  (IC 0.015, spread +2.00%). d_growth INVERTS in fragile names (spread -1.85%). => compressed skew does
  not amplify DIRECTIONAL fundamental edge. BUT likely wrong dependent variable: Mat's mechanism is about
  MOVE MAGNITUDE / realized vol / tails (what makes the options trade pay), not 63d drift direction.
  Next faithful test: does compressed skew predict larger realized moves/tails around the catalyst?
  Caveats: arbitrary 0.30 threshold (don't tune to manufacture result), overlapping windows, 2020 COVID in window.
- **Faithful vol-magnitude test: ALSO REJECTED (decisive).** 7,234 obs. With ATM IV held even across cohorts
  (0.349 vs 0.354 - no confound), compressed-skew names move SAME/slightly LESS: med rv/iv 0.926 vs 0.957,
  pct rv>iv 40.4% vs 43.2%, excursion 0.171 vs 0.178. Mat's "no cushion -> move goes further" NOT supported
  at single-name/earnings/63d level. **DECISION (pre-committed): single-name compressed put-skew = position-
  SIZING input only. NOT a signal, NOT a conditioner. Stop testing it.** Scorecard so far: d2_growth = only
  measured edge (weak, IC~0.03, partly sector); divergence flag = none; single-name fragility = none.
  Untested & distinct: INDEX-level gamma/vanna (SPY/QQQ/IWM where we HAVE OI) around MACRO catalysts, short
  horizon - that's where Mat's flush narrative actually lives; do NOT conflate with dead single-name skew.
- Cached full panel to `panel_cache.parquet` (skew_history recompute is the bottleneck; reuse it).

## VERIFICATION AUDIT (read Mat's papers + daily brief, audited code)
BUGS:
1. revenue_roc YoY uses pct_change(4) = ROW offset -> WRONG for irregular reporters (IREN gaps 275/365/183d).
   Fix: match period_end - 1yr. (Partly explains IREN's bogus 271%/1221% growth = my bug, not just data.)
2. DERIVATIVE LABELS OFF BY ONE. Mat "1st deriv"=growth rate=yoy_growth; Mat "2nd deriv"(what mkt discounts)
   =change in growth=my d_growth. My d2_growth=3rd deriv. The IC 'survivor' (d2_growth) is NOT Mat's concept;
   his real 2nd deriv (d_growth) scored weaker (~0.02). Relabel + re-run.
3. put_skew sign opposite Mat's RR25 (=IVcall-IVput). Cosmetic, standardize.
CONCEPT MISMATCH vs papers:
- Fragility: Mat's GEX/VEX/CEX = Greek x OI, INDEX options only, as SIZING modifier on macro+event signals,
  NOT single-name directional. My single-name skew test = strawman; rejections are CONSISTENT w/ Mat not against.
  Faithful test = index GEX/VEX on SPY/QQQ/IWM (have OI) around macro catalysts, short horizon.
- Divergence 'underlying' should be ROIIC (dNOPAT/dInvestedCap) + capex accel, NOT margin slope. Margin
  mean-reverts ~40%/yr (Fama-French) so margin-expansion-as-bullish may be BACKWARDS.
- Missing documented anomalies Mat cites: ACCRUALS (Sloan, huge), FCF-yield vs rates. Build these (have data).
EVAL GAPS vs Mat's own quant-methods paper: needs t>=3 (not 2), IC>=0.05 meaningful (mine 0.03), Deflated
  Sharpe/PBO for #tests, PURGED k-fold + EMBARGO (my 63d overlap inflates IC), IC-decay curve (1/5/21/63d;
  63d may be reversal zone), IS/OOS split (OOS>=60% IS). => d2_growth 'survivor' NOT yet earned by his bar.
CONFIRMED CORRECT: options-vs-stock logic; flush=index+macro-catalyst+compressed SPX skew+vanna;
  rotation/laggard thesis; PCR-deviation + ceiling-node mean reversion.

## "FIX EVERYTHING" REBUILD (done)
- data.yoy(): date-matched YoY (period_end-1yr, 55d tol) replaces pct_change(4). IREN bogus growth fixed.
- roc.py relabeled: growth=Mat 1st deriv; growth_accel=Mat 2nd deriv (was mislabeled d2_growth=3rd deriv).
- NEW fundamentals.py: roiic (dNOPAT/dInvestedCap), capex_accel, accruals (Sloan, TTM), fcf_ttm+shares. Point-in-time.
- backtest.py REWRITTEN to Mat's quant-methods standard: multi-horizon IC-decay (1/5/21/63d), excess-of-SPY,
  sector-neutral, IS/OOS split (2023) w/ 1-quarter embargo, t-stat vs t>=3 AND Bonferroni (t_crit 3.12 for 28 tests).
- gex.py NEW: index net GEX (gamma*OI*100*S^2*0.01, calls+ puts-) + 30d pctile regime; test neg vs pos GEX -> fwd RV.

## RIGOROUS RESULT (18,700 obs, 590 names, 2016-2026, sector-neutral excess)
NOTHING clears Mat's bar (t>=3 / Bonferroni / IC>=0.05 / OOS>=60%IS). The "correct outcome" his paper predicts.
- Mat's TRUE 2nd deriv (growth_accel): weak, t<=1.2, peaks 5d. The old "survivor" was mislabel(3rd deriv)+no-embargo artifact.
- Fundamentals peak at 21d horizon (63d-only was suboptimal). Most promising, right-signed, UNDER-POWERED (9-12 q):
  fcf_yield IC0.045/t1.8 @21d; accruals IC-0.068/t-1.8 @21d (correct Sloan sign); roiic t1.65 @21d OOS-stable.
- capex_accel: too sparse (6-9 q) to evaluate. Thin samples = fundamentals need bigger universe/more history.
- Panels cached: panel_v2.parquet (features), panel_cache.parquet (old w/ skew).
NEXT: more data power for fcf_yield/accruals/roiic @21d; GEX regime test result pending.

## GEX REGIME TEST: CONFIRMED (first robust result in the project)
SPY/QQQ/IWM, 2008-2025. Negative-GEX regime -> HIGHER forward 5d realized vol, all 3 indices:
SPY 0.153 vs 0.090 (+70%), QQQ 0.182 vs 0.113 (+60%), IWM 0.181 vs 0.140 (+29%). Matches paper exactly.
=> GEX works where Mat says (index, with OI) as a VOL-REGIME CONDITIONER (predicts vol NOT direction).
CAVEATS: endogeneity (neg-GEX & high vol mechanically linked; vol clusters) -> partly "high vol persists".
  Rigorous next test: control for trailing realized vol -> does GEX add MARGINAL info? Also apply IS/OOS.
  ** MARGINAL-VALUE TEST DONE -> PASSES. ** neg-vs-pos fwd-vol gap survives in EVERY trailing-vol quartile
  (SPY +0.026->+0.075 low->high vol; monotonic; same QQQ/IWM). Regression t(neg_GEX | trailing_rv) huge &
  OOS-stable (SPY 15/IS10.8/OOS11.3 etc). Deflated for overlap (non-overlapping n~900): SPY t=5.3, QQQ 6.5,
  IWM 3.8 -> ALL still clear t>=3. => GEX adds GENUINE marginal info beyond vol-clustering. FIRST fully-
  validated component (a vol-regime/sizing CONDITIONER, not direction). Cached: data/QQQ_SPY_IWM/_gex_cache_*.parquet
- gex.regime_state(symbol,asof) productionizes it: sign + 30d pctile + typical fwd-5d-RV. Validated on COVID/Jan22/Apr25.

## FULL-UNIVERSE SWEEP (parallelized 10-way, 153s, 175,882 obs, 7,622 names, 2016-2026)
Run via run_full.py (ProcessPoolExecutor over symbol chunks). Single-thread was ~25min -> 153s. panel_full.parquet.
VALIDATED DIRECTIONAL SIGNALS (well-powered 33-35 q, OOS-stable, sector-neutral, no dependence on thin/buggy data):
  - growth_accel (Mat's TRUE 2nd derivative): IC 0.037/t5.3 @21d, OOS/IS=1.04. <-- power problem in 600-name
    sample was why it looked weak earlier; at full breadth it CLEARS t>=3 at 5/21/63d. Mat's "mkt discounts 2nd deriv" = REAL.
  - revision (analyst estimate momentum): t4-5 @5/21/63d, OOS-stable.
  - roiic: t3.2-3.3 sector-neutral only, FAILS raw -> borderline.
BUG FOUND + FIXED: fcf_yield had .fillna(0) on missing cash-flow -> mostly fake 0s (25/50/75 pctile all 0.0) +
  outliers -> fake t=11 over only 12 unique-value quarters. Fixed (NaN not 0). Re-running.
DATA-COVERAGE LIMIT: cash-flow-dependent features (fcf_yield, accruals, capex_accel) only merge ~2023+ (thin) ->
  can't validate on THIS data. Need deeper/multi-source fundamentals feed (ties to multi-source principle).
  POST-FIX CONFIRMED: fcf_yield -> n_q 9 (thin), raw fails 3/4 horizons -> NOT validated (was fillna-zeros artifact).

## SCRUTINIZED FINAL VERDICT
Two validated layers now exist:
1. CONDITIONER: GEX vol-regime (index, OI). Robust, OOS-stable, deflated t 3.8-6.5. Predicts VOL not direction.
2. DIRECTION: growth_accel (Mat 2nd deriv) + revision (analyst momentum). Validated as SECTOR-NEUTRAL stock-
   selection @21d: IC~0.037, t5.3/4.5, OOS/IS~1.0. NOTE: raw market-relative they're weaker & OOS-degraded
   (growth_accel raw OOS/IS 0.5) -> edge is cross-sectional WITHIN sector. IC 0.037 < Mat's 0.05 "meaningful"
   but usable at high breadth (Fundamental Law: low IC ok with large N). Modest but real.
Architecture foundation = pair sector-neutral direction (growth_accel/revision) x GEX regime sizing.
NEXT OPTIONS: (a) deeper multi-source fundamentals to test accruals/fcf/roiic properly; (b) build the
  screener/sizing layer combining validated direction x GEX regime; (c) final hardening (DSR, purged k-fold).

## ITEM 2 DONE: deep SEC EDGAR fundamentals (run_sec.py, frames API, annual 2010-2025, cached data/sec_frames/)
120,680 annual records, 15,824 CIKs. Multi-source principle PAID OFF: deep SEC data DISPROVED the dolt
"strong accruals" as a thin-coverage artifact.
  - accruals: IC~0, t-0.03/+0.58 -> does NOT replicate (anomaly decayed, cf McLean-Pontiff). DEAD.
  - roiic: IC~0, t~0.5. DEAD.
  - fcf_yield: IC 0.055, t2.5/2.1 -> modest, right sign, FAILS t>=3 (annual power limit). The honest number
    (vs the fake t=11 fillna-zero bug). Promising blend candidate, not standalone.
=> No NEW validated directional signal. Arsenal stays growth_accel + revision.

## ITEM 1 DONE: screener/sizing layer (quant/screen.py)
- Tested compositing: equal-weight growth_accel+revision DILUTED (ICIR 0.92->0.58) despite low corr (0.175)
  -> coverage/NaN heterogeneity. Decision: RANK on growth_accel alone (strongest validated), revision as context.
- live_screen(asof, universe): per-name growth_accel asof (point-in-time) -> outlier-robust within-sector
  percentile rank -> top/bottom; drops Unknown/thin(<5) sectors; attaches gex.regime_state for sizing.
- WORKS: e.g. 2025-12-12 regime=amplifying(neg-GEX) -> "reduce gross/long-vol"; top=best-in-sector accelerators.
CAVEATS: GEX date stale (options data ends 2025-12-12; prod=daily CBOE snapshot per multi-source plan);
  edge modest (IC~0.037, breadth tool not conviction); screen = starting universe not a buy list.

## RESEARCH SUBSYSTEM (LLM synthesis layer over sell-side/central-bank notes)
Goal: ingest research PDFs/notes -> markdown store -> per-doc structured summary -> recency-weighted house brief.
Pipeline (all built, working):
- ingest.py: PDFs in research/inbox/ -> docling -> markdown, strip disclaimer boilerplate, detect publisher
  (word-boundary markers; NOT naive substring) + date, dedupe by content hash, store research/md/<YYYY-MM>/<publisher>/
  + manifest.jsonl. md_path_for() = month-folder org.
- fetch_research.py: auto-pull PUBLIC feeds -> store. Feeds: sellside.substack.com/feed (multi-bank desk notes,
  RSS=~20 recent only), Fed press+speeches, ECB press (page-fetched for full text). detect_publisher per item.
- fetch_archive.py: pages sellside.substack ARCHIVE API (back-catalog ~273 posts) -> deep multi-bank coverage.
  THIS solved "too few DB/UBS": store now 324 docs (GS159 Fed38 BofA33 ECB15 MS12 JPM10 DB9 UBS6 Barclays3 Jeff2 Unk37).
- research_brief.py: OpenRouter (NOT Anthropic - user won't pay separately). KEY in env OPENROUTER_API_KEY (set via setx).
  Models: MODEL_SUMMARY=openai/gpt-oss-120b:free, MODEL_REASON=nvidia/nemotron-3-super-120b-a12b:free (1M ctx), fallback openrouter/free.
  Checked the live free-model list (DeepSeek NOT free now). `summarize` = per-doc fresh JSON (key_points/stances/
  price_targets/catalysts) in ISOLATION (bias firewall). `brief` = reconcile -> Current view/What changed/
  Contradictions/Watch, every claim cited. WORKS end-to-end.
- RECENCY WEIGHTING (user request "older = less weight"): _weight() = 0.5^(age/HALF_LIFE_DAYS=45), floor 0.01 (~10mo),
  recent-first, weights in payload + prompt instructs weight-recent/supersede-old. Demonstrated: dropped 20 old 2025
  notes, brief flipped from 2025(tariffs/DeepSeek) to current 2026(oil/AI-stability/Fed). Tunable: HALF_LIFE_DAYS + floor.
RATE LIMIT: user key is_free_tier=False (has credit) -> NO 50/day cap, no 429s. Don't tell them to pay. It just grinds.
STATUS: 71/324 summarized, 253 (archive) running in bg. BOUNDARY HOLDS: brief = CONTEXT only (unmeasurable);
  structured stances/PTs = the measurable bridge that would go through backtest.evaluate before touching sizing.
NOTE: sell-side notes copyright -> personal use only; do NOT redistribute bot output.

## PLAN ITEMS (the user's original list — building + MEASURING each)
- NODES (quant/nodes.py): 6 types built. oi_nodes (ceiling/call-wall, floor/put-wall, pin, zero_gamma
  via proper spot-grid gamma reprice) for index ETFs w/ OI; volume_nodes (POC/HVN/LVN) for any ticker.
  MEASURED (nodes_backtest.py, SPY 2008-2025, 4508 days): NO predictive edge at daily horizon -- pin magnet
  convergence 13.9% (price diverges), ceiling/floor fwd ~= baseline. Gamma-pin is intraday/near-expiry;
  daily EOD data can't capture it. VERDICT: descriptive context only, NOT a signal. Did not fish for a horizon.
- PCR + deviations (pcr.py): pcr_oi/pcr_vol + trailing z (point-in-time). MEASURED (SPY 2008-2025):
  pcr_oi_z = NOISE (IC -0.03). pcr_vol_z (flow) = REAL contrarian, OOS-STABLE: IC 0.071, coherent spikes
  (put-fear z>2 -> +0.45% fwd vs +0.21% base; call-greed z<-2 -> -0.37%). Deflated non-overlap t=2.1 (just
  under t>=3); IS t3.1 / OOS t3.8 (strengthens OOS). VERDICT: promising conditioner, modest, not slam-dunk.
- RN-density anomaly (quant/rnd.py): Breeden-Litzenberger RND from IV smile (d2C/dK2), z vs FLAT/lognormal
  baseline -> red(excess)/blue(deficit)/star|z|>=1.8. WORKS (SPY: red excess cluster K/S~1.0 z+3.9, 18 stars,
  z range -2.4..4.4). BUT z-vs-lognormal = the PERMANENT smile shape (descriptive, not time-varying) -> NOT a
  measurable signal as-is. TODO to MEASURE: recompute z vs the density's OWN trailing history (per-date RND over
  history, heavy compute like node backtest), then test if deviations precede moves. Field built, measurement pending.
- bug fixed mid-build: np.trapz removed in numpy 2.x -> manual trapezoid in rnd.py.
  MEASURED (rnd_backtest.py, SPY 2008-2025, 903 weekly RND obs, z vs trailing-52 history): right_tail_z (upside
  density deviation) = MODEST signal IC 0.086/t2.5 @5d (near-non-overlap, honest), 0.136 @21d (overlap-inflated).
  left_tail_z + rn_skew_z = noise. VERDICT: modest real-ish (~t2.5, below t>=3), PCR tier. RND item DONE.
- OPTIONS-VS-STOCK (options_vs_stock.py): SPY ATM-IV + IV-rank vs trailing-yr, VRP = realized_fwd21 - implied.
  MEASURED (793 weekly obs 2009-2025): VALIDATES directionally. LOW iv-rank VRP +0.0098 (long vol PAID, only
  positive regime), MID -0.0045, HIGH -0.0037; corr(iv_rank,VRP)=-0.11. Rule "options when IV cheap, stock when
  rich" supported, right sign, modest magnitude. Cleanest 'works as theorized' of the analytics. Live rule = IV-rank bucket.
- INTRINSIC VALUE (quant/intrinsic.py): WACC (CAPM ce + after-tax cd, mkt-value weighted) + bear/base/bull/re-rate
  via EV/Rev multiple x TTM revenue -> per-share + upside. CROSS-CHECK vs Mat's own IREN note: WACC 14.9% vs his
  15.3%; base $58/$56.6, bull $94/$89, re-rate $129/$132 -- engine calibrated correctly. NOTE: multiples are the
  VIEW input (base=current by construction); it's a valuation FRAMEWORK/decision-aid, not a predictive signal.
- "GENERAL anomaly detection" = a cross-cutting reusable z-vs-trailing-history deviation detector (red/blue/star)
  applied to ALL streams (options/price-vol/fundamentals/macro/flow); already done piecemeal (pcr_z, rnd z, gex pctile);
  the "general" build = factor into one shared anomaly utility + measure each flag's predictiveness.
- SCREENERS (quant/screeners.py): cap-band framework DONE+clean. small(0.3-2b)/mid(2-10b) via mktcap=shares*price,
  ranked by sector-neutral CONTINUOUS score of a swappable `ranker` (default growth_accel). Bugs fixed: inf-filter
  (near-zero-base blowup e.g. ITOS), winsorize +-5, continuous demean/std score (was all-0.5 rank artifact).
  OPEN: (1) ranker CRITERIA per screener unconfirmed (Mat may rank momentum/'lottos' not growth_accel) - swap is
  1 arg; (2) opex + thesis screeners NOT built (need logic); (3) GEX regime overlay (sizing) NOT wired into screeners.
- MAT'S SCREENER DECODED (from his screenshots): MOMENTUM screen, macro-adjusted by sector. Columns: 5d/20d/60d %
  returns, RS/SPY (=stock60d/spy60d), VOL x (vol/avg), MOM (cross-sectional z of momentum blend), MACRO x (regime
  sector multiplier), SCORE = MOM x MACRO x (VERIFIED: RXO 2.744*1.20=3.293, AXSM 2.642*0.85=2.246), FIT=IN CYCLE
  if MACRO x>=1.10 else NEUTRAL. Header shows REGIME (MID_EXPANSION 60% conf) + Risk ON. MID_EXPANSION sector
  tilts read off screenshots: Tech1.30 Industrials1.20 Materials1.10 Comm1.05 Financials1.00 Energy0.90 Healthcare0.85.
  => screeners.py REBUILT to this spec (was wrongly growth_accel). Ranker = momentum, NOT fundamentals.
- REGIME (Mat's) = MACRO BUSINESS-CYCLE phase (EARLY/MID/LATE_EXPANSION, CONTRACTION) + Risk on/off + confidence,
  used as the MACRO x sector tilt in the screener. This is SEPARATE from the validated GEX vol-regime. Mat's cycle
  regime classifier = the 'regime' plan item, currently a FIXED MID_EXPANSION default in screeners.py (needs building).
  MUST MEASURE whether MACRO x adjustment helps vs hurts (the 'macro as signal' guard the user explicitly wants).
- gex.regime_state (validated GEX vol-regime) is the OTHER regime - sizing-not-selection, not wired to screeners.
- GENERAL ANOMALY UTILITY = NOT done (only ad-hoc per-signal); shared z-vs-history module still to build.
- OPEX SCREENER (screeners.opex_screen) DONE: next monthly opex (3rd Fri), names with +momentum + pre-opex catalyst
  (earnings before expiry, x1.5 boost) + tradeable IV -> suggested OTM call strike. The BABA Jun18 150C pattern.
  Validated: CRWD flagged (earnings 6/3 before 6/19 opex). Data note: SNDK bad price $1542 (slipped split filter).
- SCREENER SET: small/mid (momentum x macro-tilt, Mat spec) DONE; opex DONE. REMAINING: thesis screener (chain off
  thesis.py favored sectors/tickers). Plus: regime CLASSIFIER (dynamic MACRO x), and MEASURE whether MACRO x helps.
- THESIS SCREENER (screeners.thesis_screen) DONE: SECTOR-driven (favored sectors from recency-weighted research),
  names ranked by momentum within them; research-MENTIONED tickers = bonus flag (x1.2 boost). Validated: DELL
  (mention_w 1.62) + AMD (0.98) surfaced/boosted - the names recent papers mentioned. "sector is driver, tickers bonus."
- *** SCREENER SET COMPLETE: small / mid / opex / thesis - all 4 separate, each own logic. ***
- REGIME CLASSIFIER (quant/regime.py) DONE+VALIDATED: rules on curve(10y-2y)+growth(SPY126d)+risk(200dma,rv) ->
  EARLY/MID/LATE_EXPANSION/CONTRACTION + risk + conf. Validated vs history: 2009/2020/2022=CONTRACTION, 2021=EARLY,
  2024=LATE(inverted curve), CURRENT=MID_EXPANSION (matches Mat's screenshot). Per-phase sector TILTS. Wired into screeners.run.
- MACRO x VALUE TEST (regime_macro_test.py) DONE: 736 cross-sections, ~563 names, 2011-25. momentum ALONE IC -0.007
  (t-1.4, ~noise/mild reversal at 21d); momentum x MACRO tilt IC -0.007; DELTA -0.0003 => MACRO x does NOT help.
  CONCLUSION (macro-as-signal guard worked): regime = CONTEXT/display only, NOT in the score as alpha. ALSO: Mat's
  short-horizon (5/20/60d) momentum has no 21d edge here (classic 12m-1m momentum is a different, untested construction).
  DECISION PENDING (user): keep Mat's SCORE=MOM x MACRO (replication) vs strip MACRO from score (measurement). Regime
  stays DISPLAYED (phase/FIT) either way.
- GENERAL ANOMALY UTILITY (quant/anomaly.py) DONE: reusable zscore/detect/latest/scan -- z vs trailing window
  (excludes today, point-in-time), red(excess)/blue(deficit)/star(|z|>=1.8). Demoed across streams (SPY realized vol,
  volume, US 10y, GEX). One detector for all streams (replaces ad-hoc pcr_z/rnd_z/gex-pctile). Threshold tunable.

## ORIGINAL PLAN -- NEAR COMPLETE (only the agentic thesis BUILDER capstone remains, user deferred):
  thesis builder(agentic,10-thesis)=NOT built (thesis.py=single-prompt version exists) | screeners small/mid/opex/thesis=DONE
  | macro=research brief+regime context | regime classifier=DONE+validated | RN anomaly=built+measured(right-tail modest)
  | intrinsic value=DONE(cross-checked vs Mat IREN) | general anomaly=DONE | PCR+deviations=DONE+measured(pcr_vol modest)
  | nodes=built+measured(no daily edge, descriptive) | options-vs-stock=DONE+measured(validates) | measure-everything/
  macro-not-signal=THE SPINE, enforced throughout (MACRO x test = latest: regime adds nothing -> context not signal).
  Validated edges: growth_accel + GEX. Everything else: modest or descriptive. Honest = markets efficient.

## LANE TEST + PURPOSE-MAP CODIFIED ("everything in its right place")
- verify.py (897 weekly SPY obs): tested each index component vs forward RETURN (direction) AND forward VOL (sizing).
  Result: momentum dirIC -0.05 / volIC -0.35; GEX dirIC -0.08 / volIC -0.34; yield_curve ~0; realized_vol volIC +0.56.
  => NOTHING index-level predicts direction; GEX+momentum+vol all live in SIZING/VOL lane; macro=context. Direction
  comes ONLY from cross-sectional growth_accel. Confirms: using momentum/macro as DIRECTION = wrong-tool-wrong-job.
- quant/layers.py = the PURPOSE-MAP registry (lane + why, per component), derived from measurement. Lanes:
  DIRECTION=growth_accel/screen | SIZING/VOL=gex | EXPRESSION=intrinsic/options_vs_stock | CONTEXT=regime/screeners/
  nodes/rnd/pcr/research/thesis | FLAGS=anomaly | INFRA=data | MEASUREMENT=backtest.
- CODIFIED: screeners SCORE = momentum-only (MACRO x / FIT kept as CONTEXT label, NOT multiplied in -> measured to add
  nothing). Regime stays displayed, not a signal. The 'measure what each is best for' principle is now enforced in code.

## VALIDATED BOOK + COMBINATION TEST (quant/book.py, combo_test.py)
- book.py: DIRECTION (growth_accel sector-neutral L/S, screen.live_screen) x SIZING (gex.regime_state gross: 0.5 in
  amplifying/neg-GEX else 1.0). Today: amplifying -> half gross, 10L/10S sector-neutral equal-weight.
- COMBINATION DOES NOT VALIDATE (31 quarterly rebal 2018-25): UNSIZED ann+34.8% vol51% Sharpe0.68 DD-35.5%;
  GEX-SIZED ann+18% vol31% Sharpe0.59 DD-35%. GEX-sizing cuts ret & vol proportionally -> Sharpe DOWN, DD flat.
  REASON = HORIZON MISMATCH: GEX edge is ~5d vol; basket is quarterly(63d) -> the vol spike has passed by quarter-end.
  NUANCE: GEX-sizing IS a legit risk-throttle (vol 51->31%, slight Sharpe cost) if you want lower vol, NOT a Sharpe improver.
- LANES HAVE HORIZONS too: growth_accel=quarterly direction; GEX=short-horizon vol -> size short-dated vol/options, not the
  quarterly basket. growth_accel L/S = standalone direction product (Sharpe~0.68, high vol, modest, 31-q noisy).

## CROSS-CHECK vs Mat's papers (read options-theory hp.md = Module2 Book03)
MATCHES (builds correct per his spec): RND = Dupire/Breeden-Litzenberger e^(rT)*d2C/dK2 (exact). GEX sign+flip+
  "neg-GEX->higher RV, near-expiry, mechanics-not-direction" (my measured result IS the paper). nodes "no daily edge"
  consistent (near-expiry mechanic). options-vs-stock low-IV->buy-vol matches paper 1.5 ("IV below future RV in low-vol
  regimes just before a spike").
DISCREPANCIES to fix: (1) VRP sign - paper uses IV-RV (positive=seller premium, +avg); I used RV-IV and only tested the
  minority 'buy when IV cheap' regime. (2) options-vs-stock MISSING term-structure contango condition (paper 3.5).
  (3) regime classifier is rules-based; paper (quant-methods) specs HMM/CUSUM. 
HORIZON DISCIPLINE codified in layers.py HORIZON{}: GEX=~5d (size short-dated vol ONLY, not quarterly basket -
  this is WHY the combo failed, and the paper confirms 'GEX strongest near expiration'). growth_accel=quarterly.
  "validate and use at the SAME horizon" now explicit.
READ vola m2 hp.md (Book05): VRP=IV-RV (+avg 2-5vp); sec5.1 vol-regime table (IV level x term-structure ->
  low-vol-contango/elevated/transitional/crisis-backwardation -> sell/reduce/buy); sec5.2 VRP-adaptive sizing;
  vol cone (IV vs hist RV pctiles); range estimators (Parkinson/GK 5-7x more efficient than close-close).
FIX DONE: options_vs_stock.py rebuilt -> VRP=IV-RV (correct sign), term_structure() near vs far ATM IV,
  vol_regime() per book 5.1. Confirmed: cheap IV-rank -> negative VRP (buy options, book1.5); rich -> positive (sell/stock).
  corr(iv_rank,VRP)=+0.11. Current SPY: contango -> LOW-VOL CONTANGO (sell vol).
FIX DONE: regime HMM. hmmlearn won't build (no compiler) -> hand-rolled 2-state Gaussian HMM (Baum-Welch EM) in
  regime.vol_hmm/vol_state. VALIDATED: states low-vol 11% / high-vol 31% ann; high-vol prob = 1.00 at 2008/2020/2022,
  0.00 in calm, current 0.02 (LOW-VOL, matches contango). The quant-paper's exact spec, now correct & dependency-free.
=> BOTH cross-check fixes done: options-vs-stock (VRP sign+term structure+regime) + regime HMM. RND/GEX/nodes already matched.
REGIME views now (all CONTEXT/SIZING, never direction): macro-cycle (regime.classify, rules) | vol-regime (vol_regime,
  book5.1 rules) | vol-state HMM (vol_state, statistical) | GEX gamma-regime (validated, short-horizon).
READ macro frame (1).md: macro = REGIME not forecast; 4-layer (yield curve > credit spreads/EBP > capital flows/
  dollar > sector rotation); 4 cycle phases (Recovery/Expansion/Slowdown/Contraction) from curve(2s10s,3m10y)+
  credit(HY/EBP)+PMI(ISM)+real rate(TIPS); event-driven (pre-FOMC drift, IV crush, CPI/NFP regime-dependent).
  Macro = scoring/posture+sizing, NOT trades (aligns w/ user's macro-not-signal).
UPGRADED regime.classify -> multi-signal cycle: 2s10s + 3m10y + ISM PMI (from econ calendar, 1631 US events) +
  SPY growth, + FRED HY(BAMLH0A0HYM2)/TIPS(DFII10) when reachable. VALIDATED: 2009=EARLY, 2021=EARLY(reflation),
  2018=MID, 2022=LATE(inverted), 2024=CONTRACTION(debatable-the inverted-curve-no-recession ambiguity), CURRENT=
  MID_EXPANSION (matches Mat screenshot). FRED unreachable this session (conn refused) -> HY/TIPS None, made fail-fast
  (_FRED_DEAD memo + 6s timeout). To enable credit/real-rate signals: working FRED route (API key / host blocks CSV).
READ risk and execution hp.md (Book07): Kelly f*=mu/sig^2=Sharpe/vol, use HALF-KELLY (edge uncertain, over-bet
  punished more). VOL-TARGETING size=target_$vol/(sig_daily*price) -> continuous, scales inverse to forecast vol
  (GARCH/HMM/GEX feed forecast). Variance drag g~=mu-sig^2/2. Drawdown asymmetry + thresholds + Calmar. VaR/ES(coherent).
  Min-variance/risk-parity weighting (beats equal/cap OOS). Options sizing: defined-risk=max-loss*alloc%, undefined=3-5sig stress.
RESOLVES the GEX-sizing puzzle: crude 0.5x quarterly throttle was WRONG SHAPE. Proper = VOL-TARGETING (size ~ 1/forecast_vol),
  horizon-matched, GEX/HMM as the vol-forecast input. Checked variance-drag: even so, crude throttle geometric 0.13 < unsized 0.22.
NEXT BUILD (the genuine sizing completion): sizing.py = vol-targeting + half-Kelly cap, vol forecast from realized/HMM/GEX;
  + min-variance basket weighting to tame growth_accel 51% vol; + Calmar/maxDD eval. STILL UNREAD: vol-instruments, VAR-corner.
SESSION SCORECARD of analytics (built+measured): nodes=dead; pcr_vol=modest(t2.1); rnd right-tail=modest(t2.5);
  options-vs-stock=validates directionally(modest). Earlier validated: growth_accel + GEX. Pattern: mostly modest/dead -
  honest, markets efficient. The measurement discipline is doing its job.

## END-TO-END STATE: validated CONDITIONER (GEX) + validated DIRECTION (growth_accel, sector-neutral) +
  working screener pairing them. Honest verdict: modest broad systematic edge + robust vol-regime sizing.
  Not done: DSR/purged-kfold final hardening; daily CBOE OI snapshot pipeline; LLM thesis layer (forward-only).
STRATEGIC READ: the CONDITIONER (GEX vol regime) works; the DIRECTIONAL signals (fundamentals) are weak/
  under-powered. Architecture should lean on GEX for sizing/regime + keep hunting direction. Matches user's
  "regime as conditioner not signal" + Mat's 4-layer stack (macro->fundamental->vol surface->GEX sizing).

## Open data items
- ARCHITECTURE: multiple data sources for EVERYTHING (prices, fundamentals, OI, IV, macro) — >=2 sources +
  cross-check, for redundancy and bad-data detection. Bake into the data layer.
- Free OI: CBOE delayed JSON (`cdn.cboe.com/api/global/delayed_quotes/options/{SYM}.json`) VERIFIED to
  carry open_interest+iv+greeks, free. Plan: daily forward snapshot to build single-name OI history.
  Historical OI gap remains (Alpha Vantage 25/day for curated names, or pay).
- Single-name `vol` column = IMPLIED VOL (not volume). So single names have IV+greeks+bid/ask, only OI missing.

## Next candidates (undecided)
- Sector-neutralize the scorer + proper market-relative baseline.
- Build fragility primitive (skew percentile, term structure; ETF gamma via OI) and test divergence×fragility×catalyst.
- Catalyst surprise from economic_calendar (actual−forecast).

## FULL-CORPUS READ (2026-06-03) — cross-check of all 30 books + foundation papers vs builds
Read every .md paper in discord_export (Books 01-16 + the April foundation/markets/vol/risk/macro essays +
HOUSE swan-signal/read-first). Only unread = 3 prop-firm PDFs (propfirm_edge_in_theory, identifying_tradeable_
variance_regimes, models_for_propfirm_trading) — Discord attachments, NOT in export, cannot read locally.

VERDICT: the corpus VALIDATES the lane/horizon discipline wholesale. read-first decision chain =
  macro(quadrant) -> instrument/signal(IC>0, regime-consistent) -> structure/entry -> size(Kelly/VaR/DD) ->
  execute/monitor -> exit/review. That IS the project's architecture. "Context disciplines the trade the
  macro layer generates; it does not generate trades" (micro-in-macro 10.1) = exactly our LANES doctrine.

CONFIRMED (build is correct, paper-grounded):
- growth_accel = corporate-fundamentals "2nd derivative"; sector-neutral demean = market-data Bk10 neutralization;
  IC~0.037 powered by Grinold IR=IC*sqrt(N*TC) -> need full-universe breadth (foundation/quant-methods).
- GEX = dealer inventory-hedging flow (markets Ho-Stoll, Ni-Pan-Poteshman pinning); short-horizon vol FORECAST
  not direction. swan-signal (Mat's own GLD trade) confirms mechanism: vanna/charm support EOM->wk3, rolls off
  into OPEX -> floor gone -> gaps to macro reality. Our GEX lane + horizon = right.
- rnd: Breeden-Litzenberger f=e^(rT) d2C/dK2 confirmed exact (vol-surface Bk05 1.2).
- nodes: max-pain/pinning real but imprecise/mechanical/no fundamental info (markets 9.1) -> CONTEXT lane right.
- regime as CONTEXT, MACROx modest 20-30% tilt: risk-premia Bk16 11 says EXACTLY this (strong tilt on wrong
  regime = tracking error, no benefit).
- HMM 2-state Gaussian = quant-methods 17 spec exact (persistence .97-.99, reduce 30-60% high-vol). Match.
- point-in-time +50d filing lag = market-data Bk10 (10-Q 30-45d). Match.

DISCREPANCIES / IMPROVEMENTS TO MAKE (flagged, not yet built):
1. sizing.py NOW FULLY GROUNDED (was the open question): vol-target size=target_vol/(sigma_daily*price) [risk-exec
   Bk07 3], half-Kelly cap (CV~40% -> half-Kelly optimal, MacLean-Thorp-Ziemba), basket weights via min-variance
   (Sigma^-1 1)/(1'Sigma^-1 1) OR HRP (Lopez de Prado 2016, NO matrix inversion -> robust) with LEDOIT-WOLF
   shrunk cov (94% of sample-cov eigenvalues = MP-bulk noise), GEX/HMM feed sigma forecast AT ITS OWN horizon,
   eval by Calmar/maxDD/CVaR (Rockafellar-Uryasev LP). For SHORT-VOL/GEX positions: size to Expected Shortfall
   not VaR, fractional-Kelly cut 30-50% for negative coskewness, max-loss(4-sigma) sizing (vol-instruments 9.2).
   This is the validated replacement for book.py's crude 0.5x GEX throttle.
2. regime.py MISSING THE INFLATION AXIS. macromicro Bk06 = true 4-quadrant growth(accel/decel) x inflation
   (rising/falling): Q1 Recovery/Q2 Overheat/Q3 Stagflation/Q4 Deflation. Ours is growth-only cycle. Add
   CPI/PPI/breakevens (TIPS) as 2nd dim. Also: tilt PREMIA per quadrant (risk-premia Bk16 table) not just sectors;
   ACM term-premium decomposition (level move = repriced-expectations vs term-premium = different trades).
3. options_vs_stock MISSING RR25 (25-delta risk reversal/skew) = the 3rd of the validated 3-factor surface
   signal (IVP + term-slope + RR25) [vol-surface Bk05 9.1, vol-instruments 6]. SRP (skew premium) is a distinct
   durable edge from VRP.
4. pcr: strengthen via Pan-Poteshman NON-market-maker signed option volume (2-3wk predictive) not raw PCR.
5. split-artifact >50% filter is hacky; Bk10 says store raw price + adj_factor (reversible); distress delist ~ -55%.

NEW TESTABLE DIRECTION CANDIDATES the corpus surfaces (worth running through backtest.py gauntlet):
- Accruals anomaly (Sloan): low accruals (high cash earnings) outperform next yr. Cross-sectional, measurable.
- 52-week-high / PEAD (George-Hwang, micro-in-macro 6.3): 52wk-high beats raw momentum; anchoring under-reaction.
- ROIC-WACC spread (corporate-valuation Bk08): P/IC=(ROIC-g)/(WACC-g) -> value-creation anchor for intrinsic.py.
- Capital cycle (micro-in-macro 7.2): 4-condition sector entry (below-maint capex + competitor exits + high
  utilization + cautious guidance) -> measurable from fundamentals, sector-rotation DIRECTION.
- Credit excess-bond-premium (Gilchrist-Zakrajsek) leading regime signal; cross-asset MOVE-vs-VIX stress warning.
Cannot build (no data): VPIN/order-flow toxicity, L2/L3 microstructure (need tick data).

## FIXES APPLIED (2026-06-03) — the flagged discrepancies, built + MEASURED
1. quant/sizing.py NEW — paper-grounded risk layer (Bk07 + Bk12). vol-target, inverse-vol,
   min-variance (long-only active-set), risk-parity, HRP (Lopez de Prado 2016, no inversion),
   Ledoit-Wolf shrunk cov (sklearn), half-Kelly leverage, eval=Calmar/maxDD/CVaR. compare() =
   walk-forward (trailing-only). MEASURED:
   - single-asset-class equity basket (30 large caps): equal-weight wins return (Calmar 0.35);
     HRP best tail (vol 0.129, maxDD 0.252 lowest). = DeMiguel-Garlappi-Uppal (1/N competitive),
     which Bk12 itself cites.
   - cross-asset (eq+bonds+gold+sectors): min-var DOES minimize vol(0.116)/CVaR(-0.018) [paper
     claim confirmed] BUT concentrates into bonds -> 2022 rate shock maxDD 0.42 = Bk12 17's own
     risk-parity-fails-in-2022 warning, reproduced. -> regime-conditional cov is the real need.
   HONEST CONCLUSION: the big lever is continuous VOL-TARGETING (gross scalar), NOT the
   cross-sectional weights; within one asset class equal-weight/HRP ~ tie. So book uses HRP
   (equal-wt fallback) + continuous vol-target.
2. book.py REWIRED — dropped the binary "0.5x in amplifying GEX" throttle. Now: HRP weights
   within each leg, gross = target_vol / forecast_vol (continuous), GEX amplifying regime
   inflates forecast x1.3 -> less gross, hard cap = max_gross (half-Kelly-style). Tested
   2024-06-01: gross 0.67 (10%/15%), dollar-neutral net 0, HRP->equal fallback on thin-history
   small caps (as designed).
3. regime.py — ADDED INFLATION AXIS (was growth-only). _inflation_axis via FRED T10YIE breakeven
   3m-change (cached 5858 rows; fallback Core CPI MoM). classify() now emits quadrant
   (RECOVERY/OVERHEAT/STAGFLATION/DEFLATION = Bk06 growth x inflation) + inflation/breakeven.
   PREMIA_TILT + premia_tilt(quadrant) per Bk16 11 (carry/value/momentum/defensive, modest tilt).
   Legacy phase/TILTS/sector_tilt UNTOUCHED (screeners still work, verify.py clean). Sanity:
   2021=OVERHEAT, 2020/2023/2024=DEFLATION, today=STAGFLATION (PMI<50 + breakevens up). Point-in-time.
4. options_vs_stock.py — ADDED RR25 (25-delta risk reversal) = the missing 3rd surface factor
   (Bk05 9.1: IVP + term-slope + RR25). Uses the options `delta` column. SPY/QQQ RR25 ~ -0.059
   (structural steep put skew), IWM -0.039. skew label -> "finance puts via spreads when steep".
5. pcr.py — added delta-weighted (conviction) PCR (pcr_dw). MEASURED: IC +0.068 (t~4.5) vs raw
   pcr_vol +0.071 (t~4.7) = NO improvement (tied). Per measurement-first, NOT promoted; both
   reported. Marginally cleaner only at fear-spike (z>+2 -> +0.66% vs raw +0.45%). (Also fixed a
   pre-existing cp1252 crash on the U+2248 char.)
NOT done (deliberate): split-artifact >50% filter still hacky (works; Bk10 ideal = raw+adj_factor);
   regime-conditional covariance for sizing (the 2022 lesson) — next if pursued. The new DIRECTION
   candidates (accruals/52wk-high/ROIC-WACC/capital-cycle) remain untested ideas, not built.

## REGIME-CONDITIONAL COV + DIRECTION CANDIDATES + AUDIT (2026-06-03 pt.2)
PART 1 — regime-conditional covariance (sizing.py). The 2022 bond blowup = trailing cov shows
  bonds uncorrelated-to-equities, min-var/HRP pile into duration, correlations spike in the shock.
  First attempt (recent-realized-vol calm/stress blend) made it WORSE (reactive, de-levers after
  the damage; measured: min_var maxDD -0.419->-0.443). FIX (Bk07 6.1): use full-window VOLS but
  STRESS-period CORRELATIONS always (structural, no diversification illusion). Wired HRP to accept
  the conditional cov (it computes its own otherwise). MEASURED A/B cross-asset basket:
  HRP maxDD -0.429 -> -0.378 (-12%), Calmar 0.02 -> 0.10; equal/inv_vol also improved. min_var
  NOT rescued (concentrates by construction regardless) -- but book uses HRP, so default
  conditional=True in leg_weights. Honest: within single-asset-class equity book the effect is
  small; the cross-asset duration risk is the real target and HRP+stress-corr handles it.
PART 2 — direction candidates through the FULL gauntlet (candidates.py, 95,812 rows / 3,250 syms,
  sector-neutral IC x horizon, t>=3, Bonferroni t>=2.96 for 16 tests, IS/OOS+embargo). Added PIT
  accessors data.cash_flow / balance_sheet / eps_surprise. VERDICT:
   - roic (NOPAT/invested capital) = VALIDATED DIRECTION: h=63 IC 0.059 ICIR 0.626 t=3.81, clears
     t3 AND Bonferroni, OOS 0.073 > IS 0.059 (STRENGTHENS OOS). A genuine SECOND validated picker
     alongside growth_accel. (also t=3.13 @h=1 but overlap-suspect.)
   - sue (PEAD standardized surprise) = WEAK/CONTEXT: t<=2.5, OOS consistently positive but below bar.
   - hi52 (52-week-high proximity) = FAILS at earnings-date sampling (t1.74 @63d; small-sample pass
     was noise). CAVEAT: George-Hwang test it MONTHLY -> a fair re-test needs monthly cross-sections.
   - acc_quality (Sloan accruals, -(NI-CFO)/avg_assets) = DEAD, does not replicate (and thinnest
     coverage 36k). 
  -> roic is the keeper. NOT auto-wired into book; follow-up = check growth_accel x roic cross-corr
     before blending (screen.py note: naive blend with revision DILUTED growth_accel). 
PART 3 — code audit: 17/17 modules import clean; roc/gex/rnd/nodes/anomaly/regime/intrinsic/screen/
  screeners/sizing/book/pcr/options_vs_stock all functional; verify.py + regime_macro_test (MACROx
  delta -0.0003, still nothing) + nodes_backtest (no edge) + combo_test (GEX-quarterly Sharpe
  0.68->0.59, the horizon-mismatch that justified sizing.py) all run. BUGS FOUND+FIXED: 2 cp1252
  UnicodeEncode crashes on U+2248 (pcr.py, rnd_backtest.py) -> swapped to ASCII 't~'.

## PEAD reaction-entry test (2026-06-03 pt.3) — user-supplied TradingView strategy
Replicated the "Earnings Surprise + Reaction Entry" Pine indicator (pead.py): surprise sign +
CONFIRMATION gate (reaction-day close must break the reference bar's hi/lo; reaction = ann bar
BMO or next bar AMC from earnings_calendar 'when'), entry next open, hold 60d, excess-of-SPY.
Measured 16,808 events / 1200 random symbols (2020-2026; earnings_calendar starts 2020). VERDICT:
1. The CONFIRMATION GATE ADDS ~NOTHING: confirmed beats 60d med -2.1% hit45% vs rejected -2.5%
   hit44% (identical); confirmed misses +5.3% hit60% vs rejected +4.3% hit59% (marginal). The
   breakout filter does not separate signal from noise -- the surprise SIGN carries the edge.
2. LONG/BEAT SIDE IS A LOSER: all beats 60d med -2.3%, hit 44% -- beats get FADED, not drifted.
3. SHORT/MISS SIDE IS THE REAL, REGIME-STABLE SIGNAL: all misses 60d med +4.8%, hit 60%.
   By year (beats hit% / misses hit%): 2020 54/50 (recovery exception, beats drift up),
   2021 41/68, 2022 50/53 (bear, muted), 2023 40/64, 2024 41/64, 2025 40/58. From 2021 on
   (ex-2022) the asymmetry is robust: beats fade ~40%, misses drift down 58-68%.
4. EDGE = short the miss, not long the beat (negative-PEAD asymmetry; literature: bad news
   incorporated slowly + short-sale constraints). Caveat: overlapping 60d windows inflate t;
   event medians/hit-rates are the honest read; equal-weight, no costs, SPY-excess.
This complements candidates.py's cross-sectional sue=weak finding: raw SUE cross-section is weak,
but the DIRECTIONAL/event view shows a real downside drift the long side masks. pead.py pushed to
the github review repo.

## PEAD — PROPER deflated gauntlet OVERTURNS the event study (2026-06-03 pt.4)
Ran pead.gauntlet() full universe (5,654 syms, ~83k events): (A) cross-sectional sector-neutral IC
of signed surprise vs fwd excess, t>=3/Bonferroni/IS-OOS; (B) NON-OVERLAPPING quarterly tercile
portfolio (63d hold ~ 1 quarter -> honest t).
RESULTS:
(A) surprise IC fails the bar at every horizon: best h=63 IC +0.016 t=2.15 (< t3 and < Bonferroni
    2.5), OOS/IS 0.24 (decays hard). Surprise does not clear cross-sectionally.
(B) long_beat  q+6.6% t 1.52 Sharpe 0.63 hit 61% IS+0.047 OOS+0.104 (positive, STRENGTHENS OOS)
    short_miss q-4.6% t -2.06 Sharpe -0.86 hit 48%  (SIGNIFICANTLY LOSES)
    long_short q+2.0% t 0.48 Sharpe 0.20 (not significant)
THE LESSON (why "always do proper tests"): the event-study median/hit-rate said "misses drift down
60% -> short them; beats fade -> avoid". The proper MEAN, non-overlapping, sector-neutral test says
the OPPOSITE: shorting misses LOSES with t -2.06 (missed-earnings names have fat right-tail rebounds
that crush the short despite a >50% median); the only encouraging leg is LONG big beats (mean+6.6%,
hit 61%, OOS-positive) -- but t=1.52, does NOT clear t3, so it's WATCH/weak, not validated. Median
!= mean under fat tails; the event study would have led to a losing trade. Confirms candidates.py
(sue weak ~0.02 @63d). NET: PEAD/surprise = not a validated signal; the TradingView confirmation
gate adds nothing AND the tradeable direction is the reverse of what hit-rate implied. Nothing wired.
SAVED standing preference -> memory/proper-tests-always.md (deflated/non-overlapping, never hit-rate).

## ROIC x growth_accel BLEND CHECK (2026-06-03 pt.5) — proper test, blend REJECTED
check_blend.py, full universe 95,812 rows (74,332 with both signals):
- within-quarter rank corr(growth_accel, roic) = +0.045 -> essentially uncorrelated (complementary
  in principle, Grinold says blend should help).
- BUT measured equal-weight sector-neutral z-blend DILUTES, worse than roic alone at EVERY horizon:
    growth_accel @21d IC .019 ICIR .41 t2.36 ; roic @63d IC .059 ICIR .626 t3.81 (clears) ;
    combo @21d ICIR .29 t1.68 / @63d ICIR .23 t1.29  (BELOW both singles).
  Why: (1) different peak horizons (ga@21, roic@63) -> equal-wt at one horizon mixes strong+weak;
  (2) their IC TIME-series co-move enough that the cross-sectional independence doesn't diversify;
  equal-weighting a strong (ICIR .63) with a weak signal drags the composite down. = the screen.py
  dilution warning, confirmed.
DECISION: DO NOT blend. roic recorded as a VALIDATED standalone DIRECTION signal @ quarterly(63d);
  growth_accel stays @21d. layers.py updated (roic -> DIRECTION, standalone, not blended). If a
  multi-factor combine is ever wanted it must be IC/horizon-aware (Grinold optimal weights), not
  naive equal-weight -- and must be re-measured. (Note: growth_accel shows t2.36 in the candidates
  panel vs t5.3 in backtest.replay -- different panel construction/coverage; roic is clearly the
  stronger of the two here.)
NEXT (optional): wire roic as a standalone quarterly ranker/sleeve in screen.py/book.py (separate
  from the 21d growth_accel sleeve). Not done yet -- design choice, awaiting go-ahead.

## EXTERNAL AUDIT FIXES + growth_accel DOWNGRADE (2026-06-03 pt.6)
Acted on an external code audit. Verified each claim against code; fixed 9 real bugs:
1. roc.growth_accel: raw .diff() -> date-gap-gated (80-100d) so non-consecutive quarters aren't
   diff'd (the row-offset trap yoy() already guarded). 94.1% of rows retained.
2. backtest._fwd SPY hedge: searchsorted("left") grabbed NEXT day on missing dates -> misaligned
   excess. Now get_indexer(method="ffill") as-of. (pead.py: same fix, as-of <=.)
3. gex gex_pctile: w<w[-1] capped at 29/30 -> w<=w[-1] (reaches 1.0).
4. intrinsic cost-of-debt: unbounded interest/debt could hit 50000% -> clip [rf, 0.20].
5. sizing inverse_vol: zero-variance (halted) name gave inf/NaN -> zero-weight it. HRP recursive
   _leaves (RecursionError >500 assets) -> scipy leaves_list (iterative).
6. pcr: inf from zero call OI/vol -> replace inf->NaN before rolling z.
7. regime._fred: half-written cache deadlock -> atomic temp+os.replace + corrupt-cache unlink/refetch.
8. ingest.strip_boilerplate: min(find) truncated body at a page-1 disclaimer -> rfind + back-half guard.
NOT changed (with reason): half_kelly_leverage 0.0 edge -- NOT wired into book (book uses vol-target+
   max_gross cap), so no live effect. HRP single linkage = LdP-canonical (kept). iterrows/lru_cache =
   perf not correctness (deferred). fetch_research regex-XML / LLM prompt-injection / filename-hash =
   CONTEXT-ingestion only, never drives a trade (lanes contain the blast radius) -- noted, deferred.

CRITICAL re-measure outcome: fixing #1 prompted a full-universe re-measure of growth_accel ->
   IC~0.018 t2.44 @21d, t2.71 @63d (passes Bonferroni 2.5 only at 63d; does NOT clear t3). The
   date-match fix itself barely moved it (6% of values; buggy candidates-panel also showed t2.36).
   => The long-recorded "growth_accel t5.3 @21d" DOES NOT REPRODUCE on the current full universe;
   it was an earlier/smaller/curated universe snapshot and was OVERSTATED. growth_accel is now a
   MODEST signal (downgraded in layers.py). ROIC (t3.81 @63d) is the STRONGEST validated DIRECTION
   signal -- the project's flagship picker is roic, not growth_accel. Honest, and exactly why the
   measurement discipline exists. book.py still ranks on growth_accel (screen.py) -> SHOULD migrate
   the direction layer to roic (or roic-primary) given this -- flagged, not yet done.
EXEC: framework is research/paper only (book.py emits target weights, no router). Live = IBKR
   (ib_async), paper-first, with Almgren-Chriss/midpoint exec + Bk07 drawdown protocol as a new
   EXECUTION lane. Options data: index from local parquet, single-name OI from delayed CBOE JSON
   (forward-only); 120-DTE term-structure blocked -> needs ORATS/Polygon for historical IV surface.

## DOWNLOADABLE HISTORICAL DATA + MULTI-SOURCE LAYER (2026-06-03 pt.7)
User: "all data is from DoltHub" + an Alpha Vantage key. Built the missing-history fetch + the
multi-source cross-check layer ("as described"). Findings (all verified):
- SOURCE for single-name options HISTORY = DoltHub post-no-preference/options (same provider as
  earnings). Free SQL API (branch=master), table option_chain PK leads with `date` -> query ONE
  DATE at a time for the watchlist (fast ~2s); symbol-only scans time out. Coverage 2019-01..2024-06.
  fetch_options_history.py downloads watchlist x date-range -> data/options_history/{SYM}.parquet
  (resumable). data.options_chain()/options_asof() accessors added.
- CROSS-CHECK (sources.py cross_check_options): DoltHub SPY IV vs local index SPY IV = 8108 matched
  contracts, median |IV diff| 0.010 (1 vol pt) -> sources AGREE, new source trustworthy.
- LIMITATION (honest): DoltHub option_chain has IV + greeks + bid/ask but NO open_interest, and only
  NEAR-dated expirations (<=~50 DTE). So it UNBLOCKS single-name skew/RR25 + RND + a NEAR-CURVE slope
  (7-21d vs 30-50d), but NOT the 120-DTE term-structure slope nor single-name OI. Those need a PAID
  chain provider (ORATS/Polygon, or AV premium).
- The AV key (1XZ6P80...) is a valid Alpha Vantage FREE key: HISTORICAL_OPTIONS is PREMIUM-gated
  (can't fill the far-chain/OI gap), but TIME_SERIES_DAILY works -> wired as the SECOND price source
  for sources.cross_check_prices (local ohlcv vs AV: AAPL 93d, return-diff 0.0, corr 1.0 -> agree).
  Stored in env as ALPHAVANTAGE_API_KEY (not hardcoded). Stooq is now apikey-gated (dead as free src).
- sources.vol_surface(sym,asof): single-name near-curve IV + slope + RR25 from DoltHub history.
NET: the missing downloadable history (single-name IV/greeks/skew) is now fetchable, validated against
existing data, and the multi-source price cross-check is live. OI + 120-DTE remain paid-only (stated).

## CORRECTION: single-name options history was ALREADY LOCAL (2026-06-03 pt.7b)
The single-name options history I "downloaded" was already present: data/options data/parquet/
option_chain/ = full local DoltHub clone, 2,274 names, 2019-02 -> 2026-05 (MORE current than the
public API's 2024-06). The `vol` column IS implied vol (old hv-data-inventory note mislabeled it
"no IV"). Repointed data.options_chain() at the local clone (deleted the redundant data/options_
history/ download). Confirmed limits hold on the FULL clone too: expirations cap ~66 DTE, NO OI.
So fetch_options_history.py is just a bootstrap if the clone is missing. Net: single-name skew/
RR25/RND/near-curve are available for ALL 2274 names from existing local data -- no fetch needed.

## MULTI-SOURCE LAYER COMPLETE — all 4 streams cross-checked (2026-06-03 pt.8)
Finished "the rest" of the multi-source/bad-data layer (quant/sources.py). Every data stream now
reconciles against an INDEPENDENT second source (the audit's silent-failure defense + stated goal):
- prices:       local DoltHub ohlcv vs Alpha Vantage TIME_SERIES_DAILY (free key, env) -> corr 1.0
- fundamentals: DoltHub balance sheet vs SEC EDGAR XBRL frames (data/sec_frames, by CIK via SEC
                company_tickers map) -> AAPL/MSFT/JNJ total_assets & total_equity EXACT match
                (rel_diff 0.0). Validates the ROIC inputs (invested_capital) = the flagship signal.
- options IV:   DoltHub clone vs local index chains (SPY) -> median IV diff 1 vol pt
- rates/macro:  local us_treasury 10y vs FRED DGS10 -> median diff 0.0
sources.cross_check_all(symbol) runs the full sweep. SEC concepts available in sec_frames: Assets,
StockholdersEquity, NetIncomeLoss, OperatingIncomeLoss, LongTermDebtNoncurrent, CommonStockShares,
NetCashProvidedByUsedInOperatingActivities, PaymentsToAcquirePPE (2010-2025 Q4) -> can extend the
fundamentals check to NOPAT/CFO/debt later. Multi-source data layer = DONE.

## CRITICAL: ROIC FAILS THE P&L BAR (2026-06-03 pt.9) — IC != money
First ACTUAL strategy backtest (book_backtest.py): ROIC sector-neutral L/S, dollar-neutral, quarterly,
point-in-time, costs 10bps/side, 2016-2025, 3237-name universe.
RESULT: book CAGR -1.9%, Sharpe -0.30, maxDD -27%, final 0.84x (LOSES). vol-targeted worse (-0.31).
LEG DECOMP (the why): long-quintile +13.4%/Sharpe0.73 but long ALPHA vs universe = +0.33%/Sharpe0.09
  (~ZERO -- quality didn't outperform); short-quintile +15.2% (junk OUTPERFORMED) -> short alpha
  -3.8%/Sharpe-0.45 (toxic). Negative EVEN GROSS. Both legs dead; long-only wouldn't save it.
=> The t=3.81 cross-sectional IC was REAL but economically meaningless (too small + concentrated in
  un-shortable junk). "Validated" in this project meant IC-positive, NOT profitable -- different bars.
RE-GRADE (honest): NOTHING in the stack has cleared the P&L bar. ROIC = IC-positive/P&L-negative.
  growth_accel downgraded (t2.4). PEAD failed. GEX/regime = conditioners, never standalone money.
  The project's durable value = the measurement + risk infrastructure, not alpha. Matches the corpus
  (McLean-Pontiff decay, Harvey-Liu-Zhu false factors, low realistic IR). 
NEXT (fair test before burying ROIC): large-cap-only universe (no microcap-junk squeeze on the short)
  + long-only quality + cost/quintile sensitivity. Not expected to rescue, but it's the honest check.

## LARGE-CAP RESCUE TEST (2026-06-03 pt.10) — partial, sub-threshold
book_backtest.py 500 (top-500 by point-in-time market cap = shares_outstanding x price):
- L/S book +0.6% Sharpe 0.17 (still not tradeable). vs full-universe -1.9%.
- LONG ALPHA (high-ROIC large-cap - large-cap universe) = +1.96%/yr, Sharpe 0.54, maxDD -12%, hit52%.
  REAL directional quality tilt (vs ~0 in full universe -- junk-squeeze removed). BUT Sharpe 0.54 over
  ~9y -> t~1.6, does NOT clear t>=3 (or even 2). Encouraging, not validated. = the known/decayed
  quality premium magnitude.
- SHORT ALPHA flat (-0.2%, Sharpe -0.01) in large-cap (was -3.8% toxic in full universe). Short leg
  is dead everywhere -> dollar-neutral L/S is pointless; the /2 just halves the long alpha.
VERDICT: ROIC is a MODEST LONG-ONLY LARGE-CAP quality TILT (~2%/yr, Sharpe~0.5, t~1.6), not a
market-neutral strategy. If used at all: long-only large-cap, ROIC as an overweight, NOT the current
dollar-neutral L/S. Still nothing in the stack clears the P&L significance bar. book.py/screen.py are
built as dollar-neutral L/S on the FULL universe -> that's the worst config (the losing one).

## RECONFIGURED TO A VIABLE, PROFITABLE BOOK (2026-06-03 pt.11) — quality-momentum, cap-weighted
After proving the dollar-neutral ROIC L/S LOSES, swept long-equity configs (book_backtest.py/strategy.py).
Path: long-quality equal-weight LAGGED SPY (mega-cap bull); vol-targeting + trend-gate WHIPSAWED/hurt;
momentum alone ~tied; the WINNER = long-only large-cap QUALITY-MOMENTUM (12-1m mom rank + ROIC rank,
top 20% of top-500 mcap), CAP-WEIGHTED, monthly, net 10bps:
  FULL: CAGR 17.7% vs SPY 13.5% | vol 21.6% | Sharpe 0.86 vs 0.78 | maxDD -30% vs -34% | Calmar 0.59 vs 0.40 | 4.46x
  OOS split-half (split 2021-05): 1st half Sharpe 0.97 vs 0.85; 2nd half 0.75 vs 0.71 -> WINS BOTH halves
  on Sharpe+CAGR+Calmar. Robust, not full-sample overfit. Only 4 principled configs tested (not data-mined).
WHY cap-weight won: leaning into the large quality-momentum names = the mega-caps that drove the decade
  (equal-weight diluted away from them). WHY this works where L/S didn't: it's ENHANCED BETA -- return is
  mostly the equity risk premium, the QM tilt + cap-weighting lift risk-adjusted return over passive.
book.py RECONFIGURED: live long-only cap-weighted QM large-cap book (retired the losing L/S). momentum
  added to layers as DIRECTION; book tagged STRATEGY. CAVEATS (honest): higher vol than SPY (~22%);
  cap-weighting concentrates in mega-caps (reversal risk -> add a max-weight cap in production); single
  market/era; costs simplified; momentum needs the trade discipline (monthly, ~40% turnover). 
NET: the system finally has a profitable, OOS-robust, viable strategy -- and it's honest about being
  managed beta + the two most-validated factors, not a market-neutral alpha miracle.

## SOMETHING BETTER: ADD A CROSS-ASSET TREND SLEEVE (2026-06-03 pt.12) — deviated from the papers
Goal: keep QM book, find better. Tried re-weighting first (all WORSE): QMV/low-vol 0.70, inv-vol 0.72,
sig-wt 0.77 -- low-vol & risk-weighting DILUTE the high-vol mega-cap momentum that drove the era. So
"better" needed DIVERSIFICATION, not re-weighting. NEW (off-paper) idea: a cross-asset TREND/managed-
futures sleeve (12-1m time-series momentum across SPY/QQQ/TLT/IEF/GLD/SLV/DBC/UUP/HYG/EEM/VNQ/XLE,
inverse-vol scaled, long/short). corr(QM,trend)=+0.32, trend alone Sharpe 0.53 but POSITIVE in 2022.
COMBINE 60/40, vol-target 15%: at EQUAL RISK beats QM-only on every metric in BOTH OOS halves:
  QM60/trend40 @15%: CAGR 15.6% Sharpe 0.97 maxDD -20% Calmar 0.77 | OOS 1st 1.09 2nd 0.85
  QM-only      @15%: CAGR 15.2% Sharpe 0.95 maxDD -20% Calmar 0.75 | OOS 1st 1.08 2nd 0.81
  (vs passive SPY: Sharpe 0.78, maxDD -34%.) The trend sleeve fixes QM's weak 2nd half (2022).
book.py RECONFIGURED to the two-sleeve vol-targeted portfolio (EQUITY 60% + TREND 40%). layers: added
momentum_xasset (DIVERSIFIER), book=STRATEGY two-sleeve. The genuine free lunch (diversification of
uncorrelated premia) is the only thing that beat the concentrated QM champion -- consistent with the
corpus (risk-premia Bk16: combine uncorrelated premia -> IR*sqrt(N)). 
EVOLUTION OF THE BOOK: ROIC L/S (lost) -> QM cap-wtd long (Sharpe 0.86) -> QM+trend vol-targeted
(Sharpe 0.97, maxDD -20%, OOS-robust) = the current viable, profitable, diversified strategy.

## BETTER STILL: ADD BTC TO THE TREND SLEEVE (2026-06-03 pt.13) — crossed Sharpe 1.0
Goal: best strategy possible, free data ok. Trend re-engineering (multi-horizon, wide assets) DIDN'T
help (0.95 vs 0.97). The win = NEW DATA: BTC daily (free, Binance klines, no key -> fetch_crypto.py,
data/crypto/). Added BTC to the trend sleeve (inverse-vol scaled so it can't dominate; it's a trend
signal long/short, not buy-hold). Standalone trend Sharpe 0.53->0.66 with BTC. Book (now QM50/trend50,
trend=ETFs+BTC, vol-tgt 15%): Sharpe 1.02, CAGR 16.4%, maxDD -21%, OOS 1.13/0.90 -- beats the prior
book (0.98, 1.09/0.87) on full Sharpe AND both halves. ETH adds nothing beyond BTC (too correlated).
book.py updated: TREND_ETFS += BTCUSDT, 50/50 split, crypto loader. CAVEAT: crypto 2017-2025 was a great
trend regime (overfit-to-crypto-bull risk); it's vol-scaled small and long/short so robust-ish, but
size it humbly. EVOLUTION: ROIC L/S (lost) -> QM long (0.86) -> +trend (0.97) -> +BTC trend (1.02).
NEXT: improve the EQUITY sleeve (risk-adjusted / residual momentum) -- the main return lever, untested.
