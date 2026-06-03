"""Auto-fetch research from PUBLIC sources into the store ingest.py uses.

Multiple sources (do not rely on one):
  - sellside.substack.com : republished sell-side desk notes -- MS, BofA, DB, UBS, GS,
    (delayed) JPM & Barclays. Full body in the feed; publisher read from each title.
  - Federal Reserve press + speeches, ECB press : feed gives a summary + link, so we
    fetch the linked page for full text.

Each new item -> research/md/<YYYY-MM>/<publisher>/<title>.md + manifest.jsonl, deduped
by URL. Flows straight into summarize/brief. Add more feeds to FEEDS below.

  python fetch_research.py
"""
import hashlib, html, json, re, sys, urllib.request
from datetime import datetime
from email.utils import parsedate_to_datetime

sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # never let a non-ASCII title crash the run

import ingest  # reuse store paths + helpers (MANIFEST, ROOT, md_path_for, detect_publisher, extract_tickers, load_manifest)

UA = {"User-Agent": "Mozilla/5.0 hv-research"}

FEEDS = [
    {"name": "Sell-side desk notes", "url": "https://sellside.substack.com/feed",
     "full_body": True, "default_pub": None},                 # multi-bank; detect per title
    {"name": "Federal Reserve press", "url": "https://www.federalreserve.gov/feeds/press_all.xml",
     "full_body": False, "default_pub": "Federal Reserve"},
    {"name": "Federal Reserve speeches", "url": "https://www.federalreserve.gov/feeds/speeches.xml",
     "full_body": False, "default_pub": "Federal Reserve"},
    {"name": "ECB press", "url": "https://www.ecb.europa.eu/rss/press.html",
     "full_body": False, "default_pub": "ECB"},
    {"name": "NY Fed Liberty Street", "url": "https://libertystreeteconomics.newyorkfed.org/feed/",
     "full_body": True, "default_pub": "Federal Reserve"},
    {"name": "Bank of England", "url": "https://www.bankofengland.co.uk/rss/news",
     "full_body": False, "default_pub": "Bank of England"},
    {"name": "Bank of Japan", "url": "https://www.boj.or.jp/en/rss/whatsnew.xml",
     "full_body": False, "default_pub": "Bank of Japan"},
    {"name": "Bank of Canada", "url": "https://www.bankofcanada.ca/feed/",
     "full_body": False, "default_pub": "Bank of Canada"},
]


def html_to_text(s: str) -> str:
    s = re.sub(r"(?is)<(script|style).*?>.*?</\1>", "", s)
    s = re.sub(r"(?i)<br\s*/?>", "\n", s)
    s = re.sub(r"(?i)</(p|div|li|h\d)>", "\n\n", s)
    s = re.sub(r"<[^>]+>", "", s)
    return re.sub(r"\n{3,}", "\n\n", html.unescape(s)).strip()


def _get(url: str) -> str:
    return urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=30).read().decode("utf-8", "ignore")


def parse_feed(raw: str) -> list[dict]:
    out = []
    for it in re.findall(r"<item>(.*?)</item>", raw, re.S):
        def g(tag):
            m = re.search(rf"<{tag}>(?:<!\[CDATA\[)?(.*?)(?:\]\]>)?</{tag}>", it, re.S)
            return m.group(1).strip() if m else ""
        body = re.search(r"<content:encoded>(?:<!\[CDATA\[)?(.*?)(?:\]\]>)?</content:encoded>", it, re.S)
        out.append({"title": g("title"), "link": g("link"), "pubdate": g("pubDate"),
                    "desc": g("description"), "body": body.group(1) if body else ""})
    return out


def content_for(feed: dict, it: dict) -> str:
    if feed["full_body"] and it["body"]:
        return html_to_text(it["body"])
    if it["link"]:                                  # follow the link for full text
        try:
            return html_to_text(_get(it["link"]))[:60000]
        except Exception:
            pass
    return html_to_text(it["desc"])


def main():
    seen_urls = {r.get("source_url") for r in ingest.load_manifest().values()}
    n = 0
    with ingest.MANIFEST.open("a", encoding="utf-8") as mf:
        for feed in FEEDS:
            try:
                items = parse_feed(_get(feed["url"]))
            except Exception as e:
                print(f"FEED ERR {feed['name']}: {str(e)[:80]}"); continue
            for it in items:
                if not it["link"] or it["link"] in seen_urls:
                    continue
                text = content_for(feed, it)
                if len(text) < 200:
                    continue
                try: dt = parsedate_to_datetime(it["pubdate"]).date().isoformat()
                except Exception: dt = None
                pub = ingest.detect_publisher(it["title"], text)
                if pub == "Unknown" and feed["default_pub"]:
                    pub = feed["default_pub"]
                h = hashlib.sha256(it["link"].encode()).hexdigest()
                safe = (re.sub(r"[^\w\- ]", "", it["title"])[:110].strip() or h[:12])
                out = ingest.md_path_for(pub, dt, safe)
                out.write_text(f"# {it['title']}\n\n{text}", encoding="utf-8")
                rec = {"hash": h, "title": it["title"], "publisher": pub, "date": dt,
                       "tickers": ingest.extract_tickers(text), "md_path": str(out.relative_to(ingest.ROOT)),
                       "chars": len(text), "ingested_at": datetime.now().isoformat(timespec="seconds"),
                       "source_url": it["link"]}
                mf.write(json.dumps(rec) + "\n"); seen_urls.add(it["link"]); n += 1
                print(f"ok    {pub:16} | {dt} | {it['title'][:55]}")
    print(f"--- fetched {n} new ---")


if __name__ == "__main__":
    main()
