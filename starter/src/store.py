"""Chroma persistent store, one collection per strategy, cosine space."""
from pathlib import Path
from typing import Any

import chromadb

from .models import Chunk

DB_PATH = Path(__file__).resolve().parents[1] / "chroma_db"


class ChromaStore:
    def __init__(self, path: Path = DB_PATH, dimensions: int = 1536) -> None:
        self.client = chromadb.PersistentClient(path=str(path))
        self.dimensions = dimensions

    def collection(self, strategy: str):
        return self.client.get_or_create_collection(f"strategy_{strategy}", metadata={"hnsw:space": "cosine"})

    def upsert(self, strategy: str, chunks: list[Chunk], vectors: list[list[float]]) -> dict[str, int]:
        """Upsert by chunk_id, then delete stale ids of the same document_id. Returns before/after/deleted."""
        if len(chunks) != len(vectors):
            raise ValueError(f"{len(chunks)} chunks but {len(vectors)} vectors")
        if any(len(v) != self.dimensions for v in vectors):
            raise ValueError(f"vector dimension != {self.dimensions}")
        col = self.collection(strategy)
        before = col.count()
        for i in range(0, len(chunks), 500):
            part = chunks[i:i + 500]
            col.upsert(ids=[c.chunk_id for c in part], embeddings=vectors[i:i + 500],
                       documents=[c.text for c in part], metadatas=[c.metadata for c in part])
        new_ids = {c.chunk_id for c in chunks}
        deleted = 0
        for doc in {c.document_id for c in chunks}:
            stale = [i for i in col.get(where={"document_id": doc}, include=[])["ids"] if i not in new_ids]
            if stale:
                col.delete(ids=stale)
                deleted += len(stale)
        return {"before": before, "after": col.count(), "deleted": deleted}

    def search(self, strategy: str, query_vector: list[float], k: int = 3,
               where: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """Top-k by cosine; score = 1 - cosine distance. Chroma uses approximate (HNSW) search, not exact."""
        r = self.collection(strategy).query(query_embeddings=[query_vector], n_results=k, where=where)
        return [{"chunk_id": i, "text": t, "score": 1 - d, "metadata": m}
                for i, t, d, m in zip(r["ids"][0], r["documents"][0], r["distances"][0], r["metadatas"][0])]
