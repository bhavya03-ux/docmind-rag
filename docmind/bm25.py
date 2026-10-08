"""A small, dependency-free BM25 (Okapi) implementation for keyword retrieval."""
from __future__ import annotations

import math
import re
from collections import Counter

_TOKEN = re.compile(r"[a-z0-9]+")
STOPWORDS = frozenset(
    "a an and are as at be by for from has have in is it its of on or that the this to was were will with "
    "what which who how why when where do does did can could should would about into than then there their".split()
)


def tokenize(text: str) -> list[str]:
    """Lower-case alphanumeric tokens with common English stop-words removed."""
    return [t for t in _TOKEN.findall(text.lower()) if t not in STOPWORDS]


class BM25Index:
    """BM25 ranking over a fixed list of documents.

    score(q, d) = sum_t IDF(t) * tf * (k1 + 1) / (tf + k1 * (1 - b + b * |d| / avgdl))
    """

    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._tf = [Counter(tokenize(d)) for d in docs]
        self._len = [sum(c.values()) for c in self._tf]
        n = len(docs)
        self._avgdl = (sum(self._len) / n) if n else 0.0
        df: Counter[str] = Counter()
        for c in self._tf:
            df.update(c.keys())
        # "+1" inside the log keeps IDF positive even for very common terms.
        self._idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def __len__(self) -> int:
        return len(self._tf)

    def search(self, query: str, k: int = 5) -> list[tuple[int, float]]:
        """Return up to k (document_index, score) pairs, best first, positive scores only."""
        terms = [t for t in set(tokenize(query)) if t in self._idf]
        if not terms or not self._avgdl:
            return []
        scored: list[tuple[int, float]] = []
        for i, tf in enumerate(self._tf):
            norm = self.k1 * (1 - self.b + self.b * self._len[i] / self._avgdl)
            score = 0.0
            for t in terms:
                f = tf.get(t, 0)
                if f:
                    score += self._idf[t] * f * (self.k1 + 1) / (f + norm)
            if score > 0:
                scored.append((i, score))
        scored.sort(key=lambda x: (-x[1], x[0]))
        return scored[:k]
