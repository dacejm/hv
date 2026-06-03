"""Thesis builder — the core idea: read the recency-weighted research store and output a
concrete TRADEABLE thesis (which SECTORS and TICKERS to trade), grounded in the notes.

Each note is matched to the tickers it's about (resolve names/symbols against the ticker
universe), tallied by sector with recency weight, then an LLM synthesizes the thesis.

  python thesis.py
"""
import glob, json, sys
from collections import defaultdict
from datetime import date
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, ".")
import research_brief as rb
from quant import data

NAME2TKR = {  # common names that appear in prose rather than ticker form
    "nvidia": "NVDA", "microsoft": "MSFT", "apple": "AAPL", "meta": "META", "amazon": "AMZN",
    "alphabet": "GOOGL", "google": "GOOGL", "tesla": "TSLA", "intel": "INTC", "broadcom": "AVGO",
    "micron": "MU", "dell": "DELL", "oracle": "ORCL", "palantir": "PLTR", "netflix": "NFLX",
    "advanced micro devices": "AMD", "amd": "AMD", "taiwan semiconductor": "TSM", "tsmc": "TSM",
}

THESIS_SYS = (
    "You are a portfolio strategist. From the recency-weighted research below (higher weight = "
    "more recent = more important), produce ONE concrete tradeable thesis. Ground every claim in "
    "the notes and cite source titles in (parentheses). Output exactly these sections:\n"
    "## The trade  (1-2 sentences: the single clearest call)\n"
    "## Sectors  (ranked long/short, one-line rationale each)\n"
    "## Tickers  (specific names the research supports, long/short, grouped by sector)\n"
    "## Reasoning  (the connected narrative, cited)\n"
    "## Risks / what invalidates it\n"
    "Weight recent notes far more than old ones; treat an old view as superseded unless reaffirmed. "
    "Only name a ticker if the notes support it."
)


def _match_tickers(summ, tickers):
    cands = set()
    for c in (summ.get("companies") or []):
        c = str(c).strip()
        if c.lower() in NAME2TKR:
            cands.add(NAME2TKR[c.lower()]); continue
        t = c.upper()
        if t.isalpha() and 1 <= len(t) <= 5:
            cands.add(t)
    for pt in (summ.get("price_targets") or []):
        n = str(pt.get("name", "")).upper()
        if n.isalpha() and 1 <= len(n) <= 5:
            cands.add(n)
    return sorted(cands & tickers)


def main():
    sect = data.sectors()
    tickers = set(sect.index)
    summ = [json.loads(Path(p).read_text(encoding="utf-8")) for p in glob.glob("research/summaries/*.json")]
    for s in summ:
        s["_w"] = rb._weight(s.get("date"))
    summ = [s for s in summ if s["_w"] >= 0.05]            # recent only -> a CURRENT thesis
    summ.sort(key=lambda s: s.get("date") or "", reverse=True)
    if not summ:
        print("no recent summaries"); return

    sec_w, tkr_w = defaultdict(float), defaultdict(float)
    for s in summ:
        s["_tickers"] = _match_tickers(s, tickers)
        for t in s["_tickers"]:
            tkr_w[t] += s["_w"]; sec_w[sect.get(t, "?")] += s["_w"]
    top_sec = sorted(sec_w.items(), key=lambda x: -x[1])[:8]
    top_tkr = sorted(tkr_w.items(), key=lambda x: -x[1])[:25]

    payload = "RECENCY-WEIGHTED RESEARCH (recent first):\n"
    for s in summ[:60]:
        payload += (f"\n### {s['title']} ({s['publisher']}, {s.get('date')}) [w {s['_w']}] "
                    f"tickers={s['_tickers']}\n" + json.dumps({k: s.get(k) for k in ("key_points", "stances")}))
    payload += f"\n\nWEIGHTED SECTOR TALLY: {top_sec}\nWEIGHTED TICKER TALLY: {top_tkr}"

    out = rb.chat(rb.MODEL_REASON, THESIS_SYS, payload[:120000])
    d = Path("research/theses"); d.mkdir(parents=True, exist_ok=True)
    f = d / f"thesis_{date.today().isoformat()}.md"
    f.write_text(out, encoding="utf-8")
    print(f"ok    thesis -> {f} ({len(summ)} recent notes, {len(tkr_w)} tickers tagged)")
    print("top sectors:", [f'{k}({round(v,1)})' for k, v in top_sec])


if __name__ == "__main__":
    main()
