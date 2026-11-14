"""OpenAI text-embedding-3-small (1536 dims). Raises on missing key or API error; never fakes vectors."""
import os
from pathlib import Path
from typing import Protocol

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(Path(__file__).resolve().parents[1] / ".env")


class Embedder(Protocol):
    model_name: str
    dimensions: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, question: str) -> list[float]: ...


class MyEmbedder:
    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or os.getenv("EMBEDDING_MODEL") or "text-embedding-3-small"
        self.dimensions = 1536
        self.total_tokens = 0
        key = os.getenv("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("OPENAI_API_KEY is missing (set it in starter/.env)")
        self.client = OpenAI(api_key=key)

    def _embed(self, texts: list[str]) -> list[list[float]]:
        resp = self.client.embeddings.create(model=self.model_name, input=texts)
        self.total_tokens += resp.usage.total_tokens
        vectors = [d.embedding for d in sorted(resp.data, key=lambda d: d.index)]
        if len(vectors) != len(texts) or any(len(v) != self.dimensions for v in vectors):
            raise RuntimeError(f"unexpected embedding count/dimensions from {self.model_name}")
        return vectors

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        out = []
        for i in range(0, len(texts), 100):
            out += self._embed(texts[i:i + 100])
        return out

    def embed_query(self, question: str) -> list[float]:
        return self._embed([question])[0]
