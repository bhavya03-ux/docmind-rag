"""Wire the pieces together."""
from __future__ import annotations

from .backends import ChromaBackend, InMemoryBackend, VectorBackend
from .config import Settings
from .knowledge_base import KnowledgeBase
from .llm import OllamaLLM
from .rag import RAGPipeline

BACKENDS = ("chroma", "memory")


def build_backend(settings: Settings, name: str = "chroma") -> VectorBackend:
    if name == "chroma":
        return ChromaBackend(settings)
    if name == "memory":
        return InMemoryBackend()
    raise ValueError(f"backend must be one of {BACKENDS}")


def build_knowledge_base(settings: Settings, backend: str = "chroma") -> KnowledgeBase:
    return KnowledgeBase(build_backend(settings, backend), rrf_k=settings.rrf_k)


def build_pipeline(settings: Settings, backend: str = "chroma") -> RAGPipeline:
    return RAGPipeline(build_knowledge_base(settings, backend), OllamaLLM(settings), top_k=settings.top_k)
