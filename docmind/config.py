"""Runtime settings, overridable with environment variables."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _env(name: str, default, cast=str):
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return cast(raw)
    except ValueError as exc:
        raise ValueError(f"Invalid value for {name}: {raw!r}") from exc


@dataclass(frozen=True)
class Settings:
    ollama_base_url: str = "http://localhost:11434"
    llm_model: str = "llama3.2"
    embed_model: str = "nomic-embed-text"
    temperature: float = 0.1
    chunk_size: int = 800
    chunk_overlap: int = 120
    top_k: int = 4
    rrf_k: int = 60
    persist_dir: Path = Path(".chroma")
    collection: str = "docmind"
    max_upload_mb: int = 20

    def __post_init__(self) -> None:
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if not 0 <= self.chunk_overlap < self.chunk_size:
            raise ValueError("chunk_overlap must be >= 0 and smaller than chunk_size")
        if self.top_k <= 0:
            raise ValueError("top_k must be positive")

    @classmethod
    def from_env(cls) -> "Settings":
        d = cls()
        return cls(
            ollama_base_url=_env("OLLAMA_BASE_URL", d.ollama_base_url).rstrip("/"),
            llm_model=_env("DOCMIND_LLM_MODEL", d.llm_model),
            embed_model=_env("DOCMIND_EMBED_MODEL", d.embed_model),
            temperature=_env("DOCMIND_TEMPERATURE", d.temperature, float),
            chunk_size=_env("DOCMIND_CHUNK_SIZE", d.chunk_size, int),
            chunk_overlap=_env("DOCMIND_CHUNK_OVERLAP", d.chunk_overlap, int),
            top_k=_env("DOCMIND_TOP_K", d.top_k, int),
            rrf_k=_env("DOCMIND_RRF_K", d.rrf_k, int),
            persist_dir=Path(_env("DOCMIND_PERSIST_DIR", str(d.persist_dir))),
            collection=_env("DOCMIND_COLLECTION", d.collection),
            max_upload_mb=_env("DOCMIND_MAX_UPLOAD_MB", d.max_upload_mb, int),
        )
