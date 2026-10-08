"""LLM access. The rest of the code only needs `generate` and `stream`."""
from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Protocol

from .config import Settings
from .prompts import Message


class LLM(Protocol):
    def generate(self, messages: Sequence[Message]) -> str: ...

    def stream(self, messages: Sequence[Message]) -> Iterator[str]: ...


class OllamaLLM:
    """Local chat model served by Ollama, called through LangChain."""

    def __init__(self, settings: Settings) -> None:
        from langchain_ollama import ChatOllama

        self._chat = ChatOllama(
            model=settings.llm_model, base_url=settings.ollama_base_url, temperature=settings.temperature
        )

    def generate(self, messages: Sequence[Message]) -> str:
        return str(self._chat.invoke(list(messages)).content)

    def stream(self, messages: Sequence[Message]) -> Iterator[str]:
        for chunk in self._chat.stream(list(messages)):
            if chunk.content:
                yield str(chunk.content)
