import pytest

from docmind.chunking import chunk_id, greedy_splitter, make_chunks
from docmind.models import Page


def test_greedy_respects_size():
    text = "\n".join(f"line number {i} " + "x" * 30 for i in range(40))
    pieces = greedy_splitter(200, 20)(text)
    assert len(pieces) > 1
    assert all(len(p) <= 200 for p in pieces)


def test_greedy_keeps_all_content():
    text = "alpha beta\n\n\ngamma delta\nepsilon"
    joined = "\n".join(greedy_splitter(15, 0)(text))
    for word in ("alpha", "beta", "gamma", "delta", "epsilon"):
        assert word in joined


def test_long_lines_sliced_with_overlap():
    pieces = greedy_splitter(50, 10)("".join(chr(97 + i % 26) for i in range(120)))
    assert all(len(p) <= 50 for p in pieces)
    assert pieces[0][-10:] == pieces[1][:10]


def test_invalid_parameters():
    with pytest.raises(ValueError):
        greedy_splitter(0)
    with pytest.raises(ValueError):
        greedy_splitter(10, 10)


def test_chunk_ids_deterministic_and_unique():
    pages = [Page("alpha\nbeta", "a.txt", 1), Page("alpha\nbeta", "a.txt", 2)]
    split = greedy_splitter(5, 0)
    first, second = make_chunks(pages, split), make_chunks(pages, split)
    assert [c.id for c in first] == [c.id for c in second]
    assert len({c.id for c in first}) == len(first) == 4


def test_blank_pieces_dropped():
    chunks = make_chunks([Page("x", "a", 0)], lambda t: ["", "   ", "real"])
    assert [c.text for c in chunks] == ["real"]


def test_chunk_id_changes_with_each_input():
    base = chunk_id("a", 1, 0, "t")
    assert len({base, chunk_id("b", 1, 0, "t"), chunk_id("a", 2, 0, "t"),
                chunk_id("a", 1, 1, "t"), chunk_id("a", 1, 0, "u")}) == 5


def test_langchain_splitter_when_installed():
    pytest.importorskip("langchain_text_splitters")
    from docmind.chunking import langchain_splitter

    pieces = langchain_splitter(100, 10)("word " * 100)
    assert len(pieces) > 1
    assert all(len(p) <= 100 for p in pieces)
