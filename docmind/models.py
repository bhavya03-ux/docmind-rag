"""Plain data classes shared across the project (no heavy dependencies)."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Page:
    """Text extracted from one page of a document (page 0 = not paginated)."""

    text: str
    source: str
    page: int = 0


@dataclass(frozen=True)
class Chunk:
    """A retrievable piece of a document."""

    id: str
    text: str
    source: str
    page: int = 0

    @property
    def label(self) -> str:
        return f"{self.source}, p.{self.page}" if self.page > 0 else self.source


@dataclass(frozen=True)
class Retrieved:
    """A chunk returned by retrieval, with its score and the method that found it."""

    chunk: Chunk
    score: float
    via: str  # "vector" | "bm25" | "hybrid"


@dataclass
class Answer:
    text: str
    sources: list[Retrieved] = field(default_factory=list)
    cited: list[int] = field(default_factory=list)  # 1-based indices into `sources`

    @property
    def cited_sources(self) -> list[Retrieved]:
        return [self.sources[i - 1] for i in self.cited if 1 <= i <= len(self.sources)]
