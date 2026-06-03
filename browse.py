"""Browse the stored research summaries by publisher (and/or ticker) -- age-independent.

The recency-weighted brief is for "current macro view". This is for reading a specific
shop's theses regardless of date (e.g. Deutsche Bank's stock picks).

  python browse.py "Deutsche"            # all DB notes, newest first
  python browse.py "Morgan" 8            # latest 8 MS notes
  python browse.py "" NVDA               # any publisher mentioning NVDA
"""
import json, sys
from pathlib import Path

SUMM = Path(__file__).resolve().parent / "research" / "summaries"


def main(pub="", ticker="", limit=15):
    rows = [json.loads(p.read_text(encoding="utf-8")) for p in SUMM.glob("*.json")]
    pub, ticker = pub.lower(), ticker.upper()
    sel = [r for r in rows
           if (not pub or pub in (r.get("publisher", "") + " " + r.get("title", "")).lower())
           and (not ticker or ticker in [c.upper() for c in r.get("companies", [])]
                or any(ticker in str(pt.get("name", "")).upper() for pt in r.get("price_targets", []) or []))]
    sel.sort(key=lambda r: r.get("date") or "", reverse=True)
    print(f"{len(sel)} matching notes\n" + "=" * 70)
    for r in sel[:int(limit)]:
        print(f"\n[{r.get('date')}] {r.get('publisher')} - {r.get('title')}")
        for kp in (r.get("key_points") or [])[:5]:
            print(f"  - {kp}")
        pts = r.get("price_targets") or []
        if pts:
            print("  PRICE TARGETS: " + "; ".join(
                f"{pt.get('name')} {pt.get('rating','')} {pt.get('target','')}" for pt in pts))
        st = r.get("stances") or []
        if st:
            print("  stance: " + ", ".join(f"{s.get('dimension')}={s.get('score')}" for s in st))


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0] if len(a) > 0 else "",
         a[1] if len(a) > 1 and a[1].isupper() else "",
         next((x for x in a if x.isdigit()), 15))
