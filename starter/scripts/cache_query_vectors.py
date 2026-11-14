"""Save query embeddings of the benchmark questions so the demo still works if the API is down.

Run from starter/: python scripts/cache_query_vectors.py
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from src.embeddings import MyEmbedder  # noqa: E402

questions = json.loads((HERE.parent / "benchmark" / "questions.json").read_text(encoding="utf-8"))
texts = [q["question"] for q in questions]
vectors = MyEmbedder().embed_documents(texts)  # same model; no query/document prefix difference for this model
out = HERE / "results" / "query_vectors.json"
out.write_text(json.dumps(dict(zip(texts, vectors))), encoding="utf-8")
assert len(json.loads(out.read_text(encoding="utf-8"))) == len(set(texts))
print(f"saved {len(texts)} vectors -> {out}")
