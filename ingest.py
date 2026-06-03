"""Research ingestion pipeline.

  drop research PDFs (GS, BofA, DB, MS, JPM, ...) into research/inbox/
  python ingest.py

For each NEW pdf (deduped by content hash): detect publisher + date, convert to
markdown via docling, write research/md/<publisher>/<file>.md, and append a record
to research/manifest.jsonl. The framework (store / summarize / reconcile) reads the
manifest -- it never re-parses PDFs.
"""
import hashlib, json, re, sys, time
from datetime import datetime, date
from pathlib import Path

ROOT = Path(__file__).resolve().parent
INBOX = ROOT / "research" / "inbox"
MD = ROOT / "research" / "md"
MANIFEST = ROOT / "research" / "manifest.jsonl"
for d in (INBOX, MD):
    d.mkdir(parents=True, exist_ok=True)

PUBLISHERS = [  # (canonical, [distinctive markers — matched with word boundaries])
    ("Goldman Sachs", ["goldman", "gs basics", "[gs]"]),
    ("BofA", ["bofa", "baml", "bank of america", "merrill lynch"]),
    ("Deutsche Bank", ["deutsche", "early morning reid", "[db]"]),
    ("Morgan Stanley", ["morgan stanley", "[ms]"]),
    ("JPMorgan", ["jpmorgan", "jp morgan", "j.p. morgan", "[jpm]"]),
    ("UBS", ["ubs"]), ("Citigroup", ["citigroup", "citi research"]),
    ("Barclays", ["barclays"]), ("Jefferies", ["jefferies"]),
    ("Wells Fargo", ["wells fargo"]),
    ("Federal Reserve", ["federal reserve", "fomc", "liberty street"]),
    ("ECB", ["european central bank"]),
    ("Bank of England", ["bank of england"]),
    ("Bank of Japan", ["bank of japan"]),
    ("Bank of Canada", ["bank of canada"]),
    ("BIS", ["bank for international settlements"]),
]


def file_hash(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def detect_publisher(name: str, text: str) -> str:
    # title is the reliable signal; word boundaries stop "holdings " matching a "gs" marker
    hay = (name + "  " + text[:600]).lower()
    for canon, markers in PUBLISHERS:
        for m in markers:
            if re.search(r"(?<![a-z0-9])" + re.escape(m) + r"(?![a-z0-9])", hay):
                return canon
    return "Unknown"


def detect_date(name: str, text: str) -> str | None:
    for pat in (r"(20\d{2})[-_]?(\d{2})[-_]?(\d{2})",):       # 20260527 / 2026-05-27
        m = re.search(pat, name)
        if m:
            try: return date(*map(int, m.groups())).isoformat()
            except ValueError: pass
    head = text[:1500]
    for pat in (r"\b(\d{1,2})\s+(January|February|March|April|May|June|July|August|"
                r"September|October|November|December)\s+(20\d{2})\b",
                r"\b(January|February|March|April|May|June|July|August|September|"
                r"October|November|December)\s+(\d{1,2}),?\s+(20\d{2})\b"):
        m = re.search(pat, head, re.I)
        if m:
            for fmt in ("%d %B %Y", "%B %d %Y"):
                try: return datetime.strptime(re.sub(r",", "", m.group(0)), fmt).date().isoformat()
                except ValueError: continue
    return None


BOILERPLATE = ["Additional Disclaimers", "This message has been prepared by personnel",
               "This material has been prepared", "Non-Reliance and Risk Disclosure",
               "Legal Entities Disseminating"]


def strip_boilerplate(md: str) -> str:
    """Cut the legal/disclaimer TAIL -- it's most of the file and pure noise. Use the LAST
    occurrence of each marker (rfind) and only cut in the back half of the document: a page-1
    'This material has been prepared' must not truncate the other 49 pages (min(find) did)."""
    half = len(md) // 2
    cuts = [c for m in BOILERPLATE if (c := md.rfind(m)) > half]
    cut = min(cuts) if cuts else -1
    return md[:cut].rstrip() if cut > 0 else md


def extract_tickers(text: str) -> list[str]:
    # regex over prose is weak (notes name companies, e.g. "Samsung", not tickers);
    # real resolution is an NER/dictionary job in the LLM layer. This catches the explicit ones.
    cands = set(re.findall(r"\$([A-Z]{1,5})\b", text))                 # $NVDA
    cands |= set(re.findall(r"\(([A-Z]{2,5})\)", text))                # (NVDA)
    stop = {"AI", "US", "EU", "UK", "CEO", "GDP", "CPI", "ETF", "OW", "PT", "FICC",
            "BBG", "FT", "WSJ", "CNBC", "BBC", "FAIS", "JCR", "ESG", "FX", "MOC"}
    return sorted(c for c in cands if c not in stop)


def md_path_for(publisher: str, dt: str | None, stem: str) -> Path:
    """research/md/<YYYY-MM>/<publisher>/<stem>.md  (month-first, per the bot's UX)."""
    month = dt[:7] if dt else "undated"
    folder = MD / month / publisher
    folder.mkdir(parents=True, exist_ok=True)
    return folder / (stem + ".md")


def load_manifest() -> dict:
    seen = {}
    if MANIFEST.exists():
        for line in MANIFEST.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line); seen[r["hash"]] = r
    return seen


def main(args):
    pdfs = [Path(a) for a in args] if args else sorted(INBOX.glob("*.pdf"))
    if not pdfs:
        print(f"no PDFs (drop them in {INBOX} or pass paths)"); return
    seen = load_manifest()
    from docling.document_converter import DocumentConverter
    conv = DocumentConverter()

    with MANIFEST.open("a", encoding="utf-8") as mf:
        for pdf in pdfs:
            h = file_hash(pdf)
            if h in seen:
                print(f"dup   {pdf.name} (already ingested as '{seen[h]['title']}')"); continue
            t = time.time()
            try:
                md = conv.convert(str(pdf)).document.export_to_markdown()
            except Exception as e:
                print(f"FAIL  {pdf.name}: {e}"); continue
            md = strip_boilerplate(md)
            pub = detect_publisher(pdf.name, md)
            dt = detect_date(pdf.name, md)
            out = md_path_for(pub, dt, pdf.stem)
            out.write_text(md, encoding="utf-8")
            rec = {"hash": h, "title": pdf.stem, "publisher": pub, "date": dt,
                   "tickers": extract_tickers(md), "md_path": str(out.relative_to(ROOT)),
                   "chars": len(md), "ingested_at": datetime.now().isoformat(timespec="seconds")}
            mf.write(json.dumps(rec) + "\n"); seen[h] = rec
            print(f"ok    {pdf.name} -> {pub} | {dt} | {len(rec['tickers'])} tickers | {len(md)} chars | {time.time()-t:.1f}s")


if __name__ == "__main__":
    main(sys.argv[1:])
