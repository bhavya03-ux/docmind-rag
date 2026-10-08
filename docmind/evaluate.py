"""Measure retrieval quality (hit rate, MRR, latency) and optionally answer quality."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

from .knowledge_base import MODES, KnowledgeBase
from .models import Chunk
from .rag import RAGPipeline


@dataclass(frozen=True)
class QAItem:
    question: str
    source: str
    must_contain: str  # text that a relevant chunk from `source` must contain
    answer_contains: tuple[str, ...] = ()  # keywords a correct answer should mention


@dataclass(frozen=True)
class RetrievalScore:
    n: int
    hit_rate: float  # share of questions with a relevant chunk in the top-k
    mrr: float  # mean of 1/rank of the first relevant chunk (0 if none)
    avg_ms: float  # mean retrieval latency per question


def load_qa(path: str | Path) -> list[QAItem]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return [
        QAItem(r["question"], r["source"], r["must_contain"], tuple(r.get("answer_contains", ()))) for r in raw
    ]


def is_relevant(chunk: Chunk, item: QAItem) -> bool:
    return chunk.source == item.source and item.must_contain.lower() in chunk.text.lower()


def evaluate_retrieval(
    kb: KnowledgeBase, items: list[QAItem], modes: tuple[str, ...] = MODES, k: int = 4
) -> dict[str, RetrievalScore]:
    if not items:
        raise ValueError("No evaluation questions given.")
    results: dict[str, RetrievalScore] = {}
    for mode in modes:
        hits = 0
        rr_sum = 0.0
        elapsed = 0.0
        for item in items:
            start = time.perf_counter()
            retrieved = kb.search(item.question, k=k, mode=mode)
            elapsed += time.perf_counter() - start
            rank = next((i for i, r in enumerate(retrieved, 1) if is_relevant(r.chunk, item)), None)
            if rank:
                hits += 1
                rr_sum += 1.0 / rank
        n = len(items)
        results[mode] = RetrievalScore(n, hits / n, rr_sum / n, 1000 * elapsed / n)
    return results


def evaluate_answers(pipeline: RAGPipeline, items: list[QAItem], mode: str = "hybrid", k: int = 4) -> dict:
    """End-to-end check: answer mentions the expected keywords AND cites a relevant chunk."""
    keyword_ok = cite_ok = 0
    for item in items:
        answer = pipeline.ask(item.question, mode=mode, k=k)
        text = answer.text.lower()
        if all(kw.lower() in text for kw in item.answer_contains):
            keyword_ok += 1
        if any(is_relevant(r.chunk, item) for r in answer.cited_sources):
            cite_ok += 1
    n = len(items)
    return {"n": n, "keyword_accuracy": keyword_ok / n, "citation_accuracy": cite_ok / n}


def to_markdown(results: dict[str, RetrievalScore], k: int) -> str:
    rows = [f"| Mode | Hit@{k} | MRR@{k} | Avg latency (ms) |", "| --- | --- | --- | --- |"]
    for mode, s in results.items():
        rows.append(f"| {mode} | {s.hit_rate:.2f} | {s.mrr:.2f} | {s.avg_ms:.1f} |")
    return "\n".join(rows)
