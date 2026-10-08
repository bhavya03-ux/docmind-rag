"""The RAG pipeline: (optionally) condense -> retrieve -> prompt -> generate -> cite."""
from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass

from .knowledge_base import KnowledgeBase
from .llm import LLM
from .models import Answer, Retrieved
from .prompts import (
    NO_ANSWER,
    Message,
    build_condense_prompt,
    build_messages,
    extract_citations,
)

log = logging.getLogger(__name__)


@dataclass
class Prepared:
    """Everything needed to call the LLM. `messages` is None when nothing relevant was found."""

    query: str
    retrieved: list[Retrieved]
    messages: list[Message] | None


class RAGPipeline:
    def __init__(self, kb: KnowledgeBase, llm: LLM, top_k: int = 4, max_history_turns: int = 3) -> None:
        self.kb = kb
        self.llm = llm
        self.top_k = top_k
        self.max_history_turns = max_history_turns

    def condense(self, question: str, history: Sequence[Message]) -> str:
        """Make a follow-up ("and its limits?") standalone so retrieval can find it."""
        if not history:
            return question
        try:
            rewritten = self.llm.generate(
                [("human", build_condense_prompt(history[-2 * self.max_history_turns :], question))]
            ).strip().strip('"')
        except Exception:  # noqa: BLE001 - retrieval with the raw question is a fine fallback
            log.warning("Question condensing failed; using the original question.", exc_info=True)
            return question
        return rewritten if 0 < len(rewritten) <= 500 else question

    def prepare(
        self, question: str, mode: str = "hybrid", k: int | None = None, history: Sequence[Message] = ()
    ) -> Prepared:
        query = self.condense(question, history)
        retrieved = self.kb.search(query, k=k or self.top_k, mode=mode)
        if not retrieved:
            return Prepared(query, [], None)
        return Prepared(query, retrieved, build_messages(question, retrieved, history, self.max_history_turns))

    def ask(
        self, question: str, mode: str = "hybrid", k: int | None = None, history: Sequence[Message] = ()
    ) -> Answer:
        prepared = self.prepare(question, mode, k, history)
        if prepared.messages is None:
            return Answer(NO_ANSWER)
        text = self.llm.generate(prepared.messages).strip()
        return Answer(text, prepared.retrieved, extract_citations(text, len(prepared.retrieved)))
