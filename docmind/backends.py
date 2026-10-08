"""Vector-store backends.

`ChromaBackend` is the real one (LangChain + ChromaDB + Ollama embeddings).
`InMemoryBackend` is a hashed bag-of-words test double so tests, CI and the
evaluation script run with no model server. It is NOT semantic search.
"""
from __future__ import annotations

import hashlib
import math
from collections.abc import Iterator, Sequence
from typing import Protocol

from .bm25 import tokenize
from .config import Settings
from .models import Chunk

_BATCH = 128


def _batches(items: Sequence, size: int = _BATCH) -> Iterator[Sequence]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


class VectorBackend(Protocol):
    def add(self, chunks: Sequence[Chunk]) -> int:
        """Store chunks that are not already present; return how many were new."""

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        """Return (chunk_id, similarity) pairs, most similar first."""

    def all_chunks(self) -> list[Chunk]: ...

    def delete(self, ids: Sequence[str]) -> None: ...


class InMemoryBackend:
    def __init__(self, dim: int = 512) -> None:
        self._dim = dim
        self._chunks: dict[str, Chunk] = {}
        self._vecs: dict[str, list[float]] = {}

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * self._dim
        for tok in tokenize(text):
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
            vec[h % self._dim] += 1.0 if (h >> 64) & 1 else -1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def add(self, chunks: Sequence[Chunk]) -> int:
        new = 0
        for c in chunks:
            if c.id not in self._chunks:
                self._chunks[c.id] = c
                self._vecs[c.id] = self._embed(c.text)
                new += 1
        return new

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        q = self._embed(query)
        scored = [(cid, sum(a * b for a, b in zip(q, v))) for cid, v in self._vecs.items()]
        scored = [s for s in scored if s[1] > 0]
        scored.sort(key=lambda x: (-x[1], x[0]))
        return scored[:k]

    def all_chunks(self) -> list[Chunk]:
        return list(self._chunks.values())

    def delete(self, ids: Sequence[str]) -> None:
        for i in ids:
            self._chunks.pop(i, None)
            self._vecs.pop(i, None)


class ChromaBackend:
    """Persistent ChromaDB collection, embedded with a local Ollama model via LangChain."""

    def __init__(self, settings: Settings, embeddings=None) -> None:
        from langchain_chroma import Chroma

        if embeddings is None:
            from langchain_ollama import OllamaEmbeddings

            embeddings = OllamaEmbeddings(model=settings.embed_model, base_url=settings.ollama_base_url)
        self._db = Chroma(
            collection_name=settings.collection,
            embedding_function=embeddings,
            persist_directory=str(settings.persist_dir),
        )

    def add(self, chunks: Sequence[Chunk]) -> int:
        new_total = 0
        for batch in _batches(list(chunks)):
            existing = set(self._db.get(ids=[c.id for c in batch])["ids"])
            new = [c for c in batch if c.id not in existing]
            if not new:
                continue
            self._db.add_texts(
                texts=[c.text for c in new],
                metadatas=[{"chunk_id": c.id, "source": c.source, "page": c.page} for c in new],
                ids=[c.id for c in new],
            )
            new_total += len(new)
        return new_total

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        results = self._db.similarity_search_with_score(query, k=k)
        # Chroma returns a distance (lower = closer); convert to a similarity in (0, 1].
        return [(doc.metadata["chunk_id"], 1.0 / (1.0 + float(dist))) for doc, dist in results]

    def all_chunks(self) -> list[Chunk]:
        data = self._db.get(include=["documents", "metadatas"])
        return [
            Chunk(id=i, text=t, source=m.get("source", "unknown"), page=int(m.get("page", 0)))
            for i, t, m in zip(data["ids"], data["documents"], data["metadatas"])
        ]

    def delete(self, ids: Sequence[str]) -> None:
        for batch in _batches(list(ids)):
            if batch:
                self._db.delete(ids=list(batch))
