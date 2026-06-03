"""Pull the sell-side Substack ARCHIVE (full back-catalog, not just the RSS window).

The RSS feed only exposes ~20 recent posts -> thin DB/UBS. The archive API pages the
whole history (BofA, MS, JPM, DB, UBS, Barclays...). Each post -> month-foldered store,
deduped by URL, same schema as ingest/fetch_research.

  python fetch_archive.py [max_posts]
"""
import hashlib, json, sys, time, urllib.request
from datetime import datetime

import ingest
import fetch_research as fr

ARCHIVE = "https://sellside.substack.com/api/v1/archive?sort=new&limit=50&offset={}"
UA = {"User-Agent": "Mozilla/5.0 hv-research"}


def _json(url):
    return json.load(urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=30))


def main(max_posts=300):
    seen = {r.get("source_url") for r in ingest.load_manifest().values()}
    n = 0
    with ingest.MANIFEST.open("a", encoding="utf-8") as mf:
        for off in range(0, int(max_posts), 50):
            try:
                arr = _json(ARCHIVE.format(off))
            except Exception as e:
                print(f"archive ERR off={off}: {str(e)[:80]}"); break
            if not arr:
                break
            for p in arr:
                url = p.get("canonical_url")
                if not url or url in seen:
                    continue
                text = fr.html_to_text(p.get("body_html") or "")
                if len(text) < 200:                      # list body truncated -> fetch the page
                    try:
                        text = fr.html_to_text(fr._get(url))[:60000]
                    except Exception:
                        continue
                if len(text) < 200:
                    continue
                dt = (p.get("post_date") or "")[:10] or None
                title = p.get("title") or url.rsplit("/", 1)[-1]
                pub = ingest.detect_publisher(title, text)
                h = hashlib.sha256(url.encode()).hexdigest()
                safe = ("".join(c for c in title if c.isalnum() or c in " -_")[:110].strip() or h[:12])
                out = ingest.md_path_for(pub, dt, safe)
                out.write_text(f"# {title}\n\n{text}", encoding="utf-8")
                rec = {"hash": h, "title": title, "publisher": pub, "date": dt,
                       "tickers": ingest.extract_tickers(text), "md_path": str(out.relative_to(ingest.ROOT)),
                       "chars": len(text), "ingested_at": datetime.now().isoformat(timespec="seconds"),
                       "source_url": url}
                mf.write(json.dumps(rec) + "\n"); seen.add(url); n += 1
                if n % 20 == 0:
                    print(f"  ... {n} pulled")
            time.sleep(0.5)
    print(f"--- archive: {n} new docs ---")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else 300)
