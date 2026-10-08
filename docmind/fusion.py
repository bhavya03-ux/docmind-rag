"""Reciprocal Rank Fusion (RRF) for merging ranked lists from different retrievers."""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence


def reciprocal_rank_fusion(rankings: Sequence[Sequence[str]], k: int = 60) -> list[tuple[str, float]]:
    """Fuse several best-first rankings into one.

    Each item scores sum(1 / (k + rank)) over the lists it appears in (rank starts at 1).
    RRF needs no score normalisation, which makes it a robust way to combine
    cosine similarity with BM25 scores that live on different scales.
    """
    if k < 0:
        raise ValueError("k must be non-negative")
    scores: dict[str, float] = defaultdict(float)
    for ranking in rankings:
        seen: set[str] = set()
        for rank, item in enumerate(ranking, start=1):
            if item in seen:  # ignore duplicates within one list
                continue
            seen.add(item)
            scores[item] += 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
