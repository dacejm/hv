"""LLM layer over the research store — fresh summary + reconcile into a house brief.

Uses OpenRouter free models (no Anthropic). Set OPENROUTER_API_KEY in the env.
  python research_brief.py summarize       # summarize any not-yet-summarized docs
  python research_brief.py brief [days]    # reconcile recent summaries -> house brief

Design (bias firewall): each doc is summarized IN ISOLATION (store not in context).
The store/trajectory only re-enters at the reconcile step. The brief is CONTEXT only
-- it never feeds sizing; the structured stance/PT fields are the measurable bridge.
"""
import json, os, sys, time, urllib.request
from datetime import datetime, date, timedelta
from pathlib import Path

HALF_LIFE_DAYS = 45   # a note's weight halves every ~45 days; older = less influence

ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "research" / "manifest.jsonl"
SUMM = ROOT / "research" / "summaries"; SUMM.mkdir(parents=True, exist_ok=True)
BRIEFS = ROOT / "research" / "briefs"; BRIEFS.mkdir(parents=True, exist_ok=True)

MODEL_SUMMARY = "openai/gpt-oss-120b:free"
MODEL_REASON = "nvidia/nemotron-3-super-120b-a12b:free"   # 1M ctx for reconcile
FALLBACK = "openrouter/free"
KEY = os.environ.get("OPENROUTER_API_KEY", "")


def chat(model: str, system: str, user: str, temperature=0.2, retries=3) -> str:
    if not KEY:
        sys.exit("set OPENROUTER_API_KEY (free key at openrouter.ai)")
    body = json.dumps({"model": model, "temperature": temperature,
                       "messages": [{"role": "system", "content": system},
                                    {"role": "user", "content": user}]}).encode()
    req = urllib.request.Request("https://openrouter.ai/api/v1/chat/completions", data=body,
                                 headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"})
    for i in range(retries):
        try:
            r = json.load(urllib.request.urlopen(req, timeout=120))
            return r["choices"][0]["message"]["content"]
        except Exception as e:
            if i == retries - 1:
                raise
            time.sleep(2 ** i)


SUMMARY_SYS = (
    "You summarize a single financial research note. Use ONLY the text provided -- never "
    "add outside knowledge or speculation. Output strict JSON with keys: "
    "key_points (list of <=6 short strings), stances (list of {dimension, score -1..1, note}; "
    "dimensions like monetary/risk/oil/growth/<sector>), companies (list of names mentioned), "
    "price_targets (list of {name, rating, target} ONLY if explicitly stated, else []), "
    "catalysts (list of upcoming events mentioned). If something isn't in the text, use empty."
)
RECONCILE_SYS = (
    "You are a research desk synthesizing notes into a house view. Each note is tagged with a "
    "recency weight (1.0 = today, halving every ~45 days). WEIGHT RECENT NOTES MUCH MORE HEAVILY: "
    "the current view should reflect high-weight notes; treat low-weight (older) notes as fading "
    "background, and consider an older stance SUPERSEDED by a newer one unless the newer reaffirms it. "
    "Ground EVERY claim in the provided summaries and cite the source title in (parentheses). Output "
    "sections: ## Current view  ## What changed  ## Contradictions / disagreements  ## Watch. Be concise."
)


def _manifest():
    return [json.loads(l) for l in MANIFEST.read_text(encoding="utf-8").splitlines() if l.strip()] \
        if MANIFEST.exists() else []


def summarize():
    docs = _manifest()
    for d in docs:
        out = SUMM / f"{d['hash']}.json"
        if out.exists():
            print(f"skip  {d['title']}"); continue
        md = (ROOT / d["md_path"]).read_text(encoding="utf-8")
        raw = chat(MODEL_SUMMARY, SUMMARY_SYS, md[:120000])
        try:
            parsed = json.loads(raw[raw.find("{"):raw.rfind("}") + 1])
        except Exception:
            parsed = {"raw": raw}
        out.write_text(json.dumps({"hash": d["hash"], "title": d["title"],
                                   "publisher": d["publisher"], "date": d["date"], **parsed}, indent=2))
        print(f"ok    summarized {d['title']}")


def _weight(date_str: str | None) -> float:
    """Exponential recency weight: 1.0 today, halving every HALF_LIFE_DAYS."""
    try:
        age = (date.today() - date.fromisoformat(date_str)).days
    except Exception:
        return 0.05
    return round(0.5 ** (max(age, 0) / HALF_LIFE_DAYS), 3)


def brief(days=None):
    summ = [json.loads(p.read_text()) for p in SUMM.glob("*.json")]
    for s in summ:
        s["_w"] = _weight(s.get("date"))
    summ = [s for s in summ if s["_w"] >= 0.01]                 # drop negligibly-old notes
    summ.sort(key=lambda s: s.get("date") or "", reverse=True)  # recent first
    payload = "\n\n".join(
        f"### {s['title']} ({s['publisher']}, {s.get('date')}) [recency weight {s['_w']}]\n"
        + json.dumps({k: s.get(k) for k in ("key_points", "stances", "catalysts")})
        for s in summ)
    out = chat(MODEL_REASON, RECONCILE_SYS, payload)
    f = BRIEFS / f"brief_{datetime.now().date().isoformat()}.md"
    f.write_text(out, encoding="utf-8")
    wmax = max((s["_w"] for s in summ), default=0)
    print(f"ok    brief -> {f.relative_to(ROOT)} ({len(summ)} notes, top weight {wmax})")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "brief"
    if cmd == "summarize":
        summarize()
    else:
        brief(sys.argv[2] if len(sys.argv) > 2 else 7)
