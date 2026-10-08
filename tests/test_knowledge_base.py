import pytest

from docmind.backends import InMemoryBackend
from docmind.chunking import greedy_splitter
from docmind.knowledge_base import MODES, KnowledgeBase
from docmind.models import Page

DOCS = [
    Page("Docker containers package applications.\nVolumes keep data after removal.", "docker.md", 0),
    Page("Git branches isolate features.\nRebase rewrites history.", "git.md", 0),
]
SPLIT = greedy_splitter(60, 0)  # -> 2 docker chunks + 1 git chunk


def make_kb():
    kb = KnowledgeBase(InMemoryBackend())
    kb.add_pages(DOCS, SPLIT)
    return kb


def test_counts_and_sources():
    kb = make_kb()
    assert kb.count() == 3
    assert kb.sources() == {"docker.md": 2, "git.md": 1}


def test_every_mode_finds_the_relevant_chunk():
    kb = make_kb()
    for mode in MODES:
        top = kb.search("volumes data", k=1, mode=mode)
        assert top and top[0].chunk.source == "docker.md", mode
        assert top[0].via == mode


def test_reindexing_is_idempotent():
    kb = make_kb()
    assert kb.add_pages(DOCS, SPLIT) == 0
    assert kb.count() == 3


def test_delete_source_and_clear():
    kb = make_kb()
    assert kb.delete_source("docker.md") == 2
    assert kb.sources() == {"git.md": 1}
    kb.clear()
    assert kb.count() == 0 and kb.search("git") == []


def test_invalid_inputs():
    kb = make_kb()
    with pytest.raises(ValueError):
        kb.search("x", mode="nope")
    assert kb.search("   ") == []
    assert kb.search("git", k=0) == []


def test_hybrid_combines_both_retrievers():
    kb = make_kb()
    git_chunk = next(c for c in kb.chunks() if c.source == "git.md")
    kb._backend.search = lambda query, k: [(git_chunk.id, 0.9)]  # "vector" side only knows the git chunk
    ids = {r.chunk.id for r in kb.search("volumes data", k=2, mode="hybrid")}  # BM25 finds the docker chunk
    assert git_chunk.id in ids and len(ids) == 2
