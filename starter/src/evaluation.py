"""Hit@1 / Evidence Hit@3 with per-query traces."""
import re
import unicodedata
from typing import Any

# curly quotes and dashes -> ascii
_PUNCT = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"',
                        "–": "-", "—": "-", "−": "-"})


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKC", s).translate(_PUNCT).lower()
    return re.sub(r"\s+", " ", s).strip()


def gold_pages(q: dict[str, Any]) -> list[int]:
    return list(q.get("gold_pages") or [q["gold_page"]])


def is_hit(result: dict[str, Any], q: dict[str, Any]) -> bool:
    m = result["metadata"]
    return (m["document_id"] == q["gold_source_id"]
            and any(m["page"] <= p <= m["page_end"] for p in gold_pages(q))
            and norm(q["evidence_quote"]) in norm(result["text"]))


def evaluate_query(question: dict[str, Any], hits: list[dict[str, Any]], strategy: str = "",
                   filter_used: dict | None = None) -> dict[str, Any]:
    top3 = []
    for rank, h in enumerate(hits[:3], 1):
        m = h["metadata"]
        top3.append({"rank": rank, "chunk_id": h["chunk_id"], "score": h["score"], "source": m["source"],
                     "page": m["page"], "page_end": m["page_end"], "chapter": m["chapter"],
                     "section": m["section"], "text": h["text"], "is_hit": is_hit(h, question)})
    hit3 = any(t["is_hit"] for t in top3)
    page_seen = any(t["page"] <= p <= t["page_end"] for t in top3 for p in gold_pages(question))
    return {"strategy": strategy, "id": question["id"], "split": question["split"],
            "question": question["question"], "filter_applied": filter_used, "top3": top3,
            "hit1": bool(top3 and top3[0]["is_hit"]), "hit3": hit3,
            "evidence_split_flag": (not hit3) and page_seen}


def summarize(traces: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: dict[tuple, dict] = {}
    for t in traces:
        r = rows.setdefault((t["strategy"], t["split"]), {"strategy": t["strategy"], "split": t["split"],
                            "n_hit1": 0, "n_hit3": 0, "N": 0, "evidence_split": 0})
        r["N"] += 1
        r["n_hit1"] += t["hit1"]
        r["n_hit3"] += t["hit3"]
        r["evidence_split"] += t["evidence_split_flag"]
    return sorted(rows.values(), key=lambda r: (r["strategy"], r["split"]))
