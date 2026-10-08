"""Turn pages into stable, de-duplicable chunks."""
from __future__ import annotations

import hashlib
import logging
from collections.abc import Callable, Iterable

from .models import Chunk, Page

log = logging.getLogger(__name__)
Splitter = Callable[[str], list[str]]


def greedy_splitter(size: int, overlap: int = 0) -> Splitter:
    """Dependency-free fallback: pack whole lines into chunks of at most `size` characters."""
    if size <= 0 or not 0 <= overlap < size:
        raise ValueError("need size > 0 and 0 <= overlap < size")

    def split(text: str) -> list[str]:
        pieces: list[str] = []
        buf = ""
        for line in (ln.strip() for ln in text.splitlines()):
            if not line:
                continue
            if len(line) > size:  # hard-slice very long lines, keeping some overlap
                if buf:
                    pieces.append(buf)
                    buf = ""
                step = size - overlap
                for i in range(0, len(line), step):
                    pieces.append(line[i : i + size])
                    if i + size >= len(line):
                        break
            elif buf and len(buf) + 1 + len(line) > size:
                pieces.append(buf)
                buf = line
            else:
                buf = f"{buf}\n{line}" if buf else line
        if buf:
            pieces.append(buf)
        return pieces

    return split


def langchain_splitter(size: int, overlap: int) -> Splitter:
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=size, chunk_overlap=overlap, separators=["\n\n", "\n", ". ", " ", ""]
    )
    return splitter.split_text


def get_splitter(size: int, overlap: int) -> Splitter:
    """LangChain's RecursiveCharacterTextSplitter when installed, else the simple fallback."""
    try:
        return langchain_splitter(size, overlap)
    except ImportError:
        log.warning("langchain-text-splitters not installed; using the simple fallback splitter.")
        return greedy_splitter(size, overlap)


def chunk_id(source: str, page: int, index: int, text: str) -> str:
    """Deterministic id, so re-indexing the same file never creates duplicates."""
    raw = f"{source}\x00{page}\x00{index}\x00{text}".encode("utf-8")
    return hashlib.sha1(raw).hexdigest()[:20]


def make_chunks(pages: Iterable[Page], split: Splitter) -> list[Chunk]:
    chunks: list[Chunk] = []
    for page in pages:
        pieces = [p.strip() for p in split(page.text) if p and p.strip()]
        for i, piece in enumerate(pieces):
            chunks.append(Chunk(chunk_id(page.source, page.page, i, piece), piece, page.source, page.page))
    return chunks
