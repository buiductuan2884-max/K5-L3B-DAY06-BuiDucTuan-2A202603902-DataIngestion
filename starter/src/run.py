"""Pipeline CLI, run from starter/: python -m src.run {ingest,evaluate,verify-benchmark,audit-sheets}"""
import argparse
import csv
import json
import statistics
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from .chunking import by_paragraph, fixed_size
from .evaluation import evaluate_query, norm, summarize
from .ingestion import load_pdf
from .models import Chunk, DocumentPage

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
PDF = ROOT / "data" / "Economics2e.pdf"
BENCHMARK = ROOT.parent / "benchmark" / "questions.json"
DOC_ID = "openstax-econ2e"
CHAPTERS = [
    {"chapter": 3, "title": "Demand and Supply", "pages": [55, 84]},
    {"chapter": 4, "title": "Labor and Financial Markets", "pages": [93, 112]},
    {"chapter": 5, "title": "Elasticity", "pages": [117, 136]},
]
COST_PER_M_TOKENS = 0.02  # text-embedding-3-small, USD


def write_json(name: str, data) -> None:
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / name).write_text(json.dumps(data, indent=1, ensure_ascii=False), encoding="utf-8")


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def load_pages() -> list[DocumentPage]:
    return [DocumentPage(**p) for p in read_json(RESULTS / "pages_normalized.json")]


def cmd_ingest(args) -> None:
    import tiktoken
    from .embeddings import MyEmbedder
    from .store import ChromaStore

    log: list[dict] = []
    pages = load_pdf(PDF, DOC_ID, CHAPTERS, log)
    write_json("pages_normalized.json", [asdict(p) for p in pages])
    write_json("ingestion_log.json", log)
    strategies = {"A": fixed_size(pages, 500, 50), "B": by_paragraph(pages, 1000)}
    enc = tiktoken.get_encoding("cl100k_base")
    stats = {}
    for name, chunks in strategies.items():
        (RESULTS / f"chunks_{name}.jsonl").write_text(
            "".join(json.dumps(asdict(c), ensure_ascii=False) + "\n" for c in chunks), encoding="utf-8")
        lens = [len(c.text) for c in chunks]
        stats[name] = {"count": len(chunks), "min_chars": min(lens), "median_chars": statistics.median(lens),
                       "mean_chars": round(statistics.mean(lens), 1), "max_chars": max(lens),
                       "total_chars": sum(lens), "total_tokens_cl100k": sum(len(enc.encode(c.text)) for c in chunks)}
    write_json("chunk_stats.json", stats)

    embedder, store = MyEmbedder(), ChromaStore()
    run = {"timestamp": datetime.now(timezone.utc).isoformat(), "model": embedder.model_name, "strategies": {}}
    for name, chunks in strategies.items():
        t0, tok0 = time.time(), embedder.total_tokens
        vectors = embedder.embed_documents([c.text for c in chunks])
        counts = store.upsert(name, chunks, vectors)
        tokens = embedder.total_tokens - tok0
        run["strategies"][name] = {"seconds": round(time.time() - t0, 2), "embedding_tokens": tokens,
                                   "est_cost_usd": tokens * COST_PER_M_TOKENS / 1e6, "upsert": counts}
    report_path = RESULTS / "ingest_report.json"
    history = read_json(report_path) if report_path.exists() else []
    write_json("ingest_report.json", history + [run])
    print(json.dumps(run, indent=1))


def cmd_evaluate(args) -> None:
    from .embeddings import MyEmbedder
    from .store import ChromaStore

    questions = read_json(BENCHMARK)
    embedder, store = MyEmbedder(), ChromaStore()
    traces = {"A": [], "B": [], "C": []}
    for q in questions:
        vec = embedder.embed_query(q["question"])  # embedded once, reused for A/B/C
        flt = q.get("filter")
        traces["A"].append(evaluate_query(q, store.search("A", vec), "A"))
        traces["B"].append(evaluate_query(q, store.search("B", vec), "B"))
        traces["C"].append(evaluate_query(q, store.search("B", vec, where=flt or None), "C", flt))
    rows, comp = [], []
    for s, ts in traces.items():
        (RESULTS / f"traces_{s}.jsonl").write_text(
            "".join(json.dumps(t, ensure_ascii=False) + "\n" for t in ts), encoding="utf-8")
        for r in summarize(ts):
            rows.append({"strategy": s, "split": r["split"], "hit1": f"{r['n_hit1']}/{r['N']}",
                         "hit3": f"{r['n_hit3']}/{r['N']}", "evidence_split": r["evidence_split"]})
    for q in questions:
        if q.get("filter"):
            for s, ts in traces.items():
                t = next(t for t in ts if t["id"] == q["id"])
                ok = all(h[k] == v for h in t["top3"] for k, v in q["filter"].items())
                comp.append({"strategy": s, "question_id": q["id"], "split": q["split"],
                             "filter": json.dumps(q["filter"]), "top3_all_match": ok})
    write_csv(RESULTS / "metrics.csv", rows)
    if comp:
        write_csv(RESULTS / "filter_compliance.csv", comp)
    for r in rows:
        print(r)


def cmd_verify(args) -> None:
    if not BENCHMARK.exists():
        sys.exit(f"{BENCHMARK} not found")
    texts = {p["page"]: norm(p["text"]) for p in read_json(RESULTS / "pages_normalized.json")}
    failed = 0
    for q in read_json(BENCHMARK):
        gp = sorted(q.get("gold_pages") or [q["gold_page"]])
        quote = norm(q["evidence_quote"])
        spans = [texts.get(p, "") for p in gp]
        spans += [texts.get(p, "") + " " + texts.get(p + 1, "") for p in gp if p + 1 in gp]  # spans consecutive gold pages
        if not any(quote in s for s in spans):
            failed += 1
            print(f"FAIL {q['id']} pages={gp}: {q['evidence_quote'][:80]!r}")
    print(f"{failed} failed")
    sys.exit(1 if failed else 0)


def cmd_audit(args) -> None:
    out = RESULTS / "audit"
    pages = read_json(RESULTS / "pages_normalized.json")
    chunks = {s: [Chunk(**c) for c in read_jsonl(RESULTS / f"chunks_{s}.jsonl")] for s in "AB"}
    positions = []
    for p in pages:  # first paragraph >= 200 chars on each of the first 10 counted pages
        para = next((x for x in p["text"].split("\n\n") if len(x) >= 200), None)
        if para:
            positions.append((p["page"], para, " ".join(p["text"].split("\n\n")).find(para)))
        if len(positions) == 10:
            break
    for s, src in (("A", "A"), ("B", "B"), ("C", "B")):
        rows = []
        for i, (page, para, off) in enumerate(positions, 1):
            if src == "A":
                c = next(c for c in chunks["A"] if c.page == page and c.metadata["char_start"] <= off < c.metadata["char_start"] + 500)
            else:
                c = next(c for c in chunks["B"] if c.page <= page <= c.metadata["page_end"] and para[:80] in c.text)
            rows.append({"position_id": i, "page": page, "paragraph_start": para[:80], "chunk_id": c.chunk_id,
                         "chunk_text": c.text, "score_0_2": "", "reason": ""})
        write_csv(out / f"coherence_{s}.csv", rows)
        tfile = RESULTS / f"traces_{s}.jsonl"
        if not tfile.exists():
            print(f"no {tfile.name} yet: run evaluate, then audit-sheets again for grounding/metadata sheets")
            continue
        test = [t for t in read_jsonl(tfile) if t["split"] == "test"]
        write_csv(out / f"grounding_{s}.csv", [
            {"question_id": t["id"], "question": t["question"],
             "top3_text": "\n---\n".join(f"[{h['rank']}] {h['text']}" for h in t["top3"]),
             "score_0_2": "", "reason": ""} for t in test])
        write_csv(out / f"metadata_{s}.csv", [
            {"question_id": t["id"], "rank": h["rank"], "source": h["source"], "page": h["page"],
             "page_end": h["page_end"], "chapter": h["chapter"], "section": h["section"],
             "metadata_correct_yn": "", "note": ""} for t in test for h in t["top3"]])


def main() -> None:
    ap = argparse.ArgumentParser(prog="python -m src.run")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in (("ingest", cmd_ingest), ("evaluate", cmd_evaluate),
                     ("verify-benchmark", cmd_verify), ("audit-sheets", cmd_audit)):
        sub.add_parser(name).set_defaults(fn=fn)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
