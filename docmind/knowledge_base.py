"""Hybrid retrieval: vector search + BM25, merged with Reciprocal Rank Fusion."""
from __future__ import annotations

from collections import Counter
from collections.abc import Iterable

from .backends import VectorBackend
from .bm25 import BM25Index
from .chunking import Splitter, make_chunks
from .fusion import reciprocal_rank_fusion
from .models import Chunk, Page, Retrieved

MODES = ("hybrid", "vector", "bm25")


class KnowledgeBase:
    def __init__(self, backend: VectorBackend, rrf_k: int = 60) -> None:
        self._backend = backend
        self._rrf_k = rrf_k
        self._chunks: dict[str, Chunk] | None = None
        self._bm25: BM25Index | None = None
        self._bm25_ids: list[str] = []

    # ---- cache -----------------------------------------------------------------
    def _invalidate(self) -> None:
        self._chunks = None
        self._bm25 = None
        self._bm25_ids = []

    def _chunk_map(self) -> dict[str, Chunk]:
        if self._chunks is None:
            self._chunks = {c.id: c for c in self._backend.all_chunks()}
        return self._chunks

    def _bm25_index(self) -> BM25Index:
        if self._bm25 is None:
            chunks = self._chunk_map()
            self._bm25_ids = list(chunks)
            self._bm25 = BM25Index([chunks[i].text for i in self._bm25_ids])
        return self._bm25

    # ---- writes ----------------------------------------------------------------
    def add_pages(self, pages: Iterable[Page], split: Splitter) -> int:
        """Chunk and index pages. Returns the number of NEW chunks (re-adding is a no-op)."""
        added = self._backend.add(make_chunks(pages, split))
        self._invalidate()
        return added

    def delete_source(self, source: str) -> int:
        ids = [c.id for c in self._chunk_map().values() if c.source == source]
        self._backend.delete(ids)
        self._invalidate()
        return len(ids)

    def clear(self) -> None:
        self._backend.delete(list(self._chunk_map()))
        self._invalidate()

    # ---- reads -----------------------------------------------------------------
    def count(self) -> int:
        return len(self._chunk_map())

    def sources(self) -> dict[str, int]:
        return dict(sorted(Counter(c.source for c in self._chunk_map().values()).items()))

    def chunks(self) -> list[Chunk]:
        return list(self._chunk_map().values())

    def search(self, query: str, k: int = 4, mode: str = "hybrid") -> list[Retrieved]:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        chunks = self._chunk_map()
        if not chunks or not query.strip() or k <= 0:
            return []
        fetch_k = max(k * 3, 10)  # over-fetch so fusion has candidates to work with

        vector: list[tuple[str, float]] = []
        lexical: list[tuple[str, float]] = []
        if mode in ("vector", "hybrid"):
            vector = self._backend.search(query, fetch_k)
        if mode in ("bm25", "hybrid"):
            index = self._bm25_index()
            lexical = [(self._bm25_ids[i], s) for i, s in index.search(query, fetch_k)]

        if mode == "vector":
            ranked = vector
        elif mode == "bm25":
            ranked = lexical
        else:
            ranked = reciprocal_rank_fusion([[i for i, _ in vector], [i for i, _ in lexical]], self._rrf_k)

        return [Retrieved(chunks[cid], score, mode) for cid, score in ranked[:k] if cid in chunks]
