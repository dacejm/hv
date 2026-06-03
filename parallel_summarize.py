"""Parallel summarizer — many concurrent workers (the calls are I/O-bound API waits).

Additive: reuses research_brief's chat/prompt/store, doesn't modify it. Skips docs
already summarized, so it's safe to run alongside or after the sequential one.

  python parallel_summarize.py [workers]   # default 8 (stays under the ~20 req/min cap)
"""
import glob, json, sys
from concurrent.futures import ThreadPoolExecutor

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, ".")
import research_brief as rb


def work(d):
    out = rb.SUMM / f"{d['hash']}.json"
    if out.exists():
        return None
    try:
        md = (rb.ROOT / d["md_path"]).read_text(encoding="utf-8")
        raw = rb.chat(rb.MODEL_SUMMARY, rb.SUMMARY_SYS, md[:120000])
        try:
            parsed = json.loads(raw[raw.find("{"):raw.rfind("}") + 1])
        except Exception:
            parsed = {"raw": raw}
        out.write_text(json.dumps({"hash": d["hash"], "title": d["title"],
                                   "publisher": d["publisher"], "date": d["date"], **parsed}, indent=2))
        return True
    except Exception as e:
        print(f"FAIL {d['title'][:40]}: {str(e)[:60]}", flush=True)
        return False


def main(workers=8):
    docs = rb._manifest()
    todo = [d for d in docs if not (rb.SUMM / f"{d['hash']}.json").exists()]
    print(f"{len(todo)} to summarize, {workers} workers", flush=True)
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for r in ex.map(work, todo):
            if r:
                done += 1
                if done % 25 == 0:
                    print(f"  ...{done}/{len(todo)}", flush=True)
    print(f"done. summaries: {len(glob.glob('research/summaries/*.json'))}/{len(docs)}", flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 8)
