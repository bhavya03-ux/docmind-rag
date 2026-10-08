from docmind.models import Chunk, Retrieved
from docmind.prompts import (
    NO_ANSWER,
    SYSTEM_PROMPT,
    build_condense_prompt,
    build_context,
    build_messages,
    extract_citations,
    trim_history,
)


def R(text, source="a.md", page=0):
    return Retrieved(Chunk("id-" + text, text, source, page), 1.0, "hybrid")


def test_extract_citations_filters_invalid_and_dedups():
    assert extract_citations("Yes [2] and [1][2], also [9] and [0].", 3) == [2, 1]
    assert extract_citations("no citations", 3) == []


def test_context_numbers_sources_and_pages():
    ctx = build_context([R("alpha"), R("beta", "b.pdf", 7)])
    assert "[1] (source: a.md)\nalpha" in ctx
    assert "[2] (source: b.pdf, p.7)\nbeta" in ctx


def test_message_structure():
    msgs = build_messages("Why?", [R("alpha")], [("human", "q1"), ("ai", "a1")])
    assert msgs[0][0] == "system" and msgs[-1][0] == "human"
    assert "Question: Why?" in msgs[-1][1] and "alpha" in msgs[-1][1]
    assert msgs[1:3] == [("human", "q1"), ("ai", "a1")]


def test_history_is_trimmed():
    hist = [("human" if i % 2 == 0 else "ai", f"m{i}") for i in range(10)]
    assert trim_history(hist, 2) == hist[-4:]
    assert trim_history(hist, 0) == []


def test_system_prompt_rules():
    assert NO_ANSWER in SYSTEM_PROMPT
    assert "untrusted" in SYSTEM_PROMPT


def test_condense_prompt_labels_roles():
    p = build_condense_prompt([("human", "What is RAG?"), ("ai", "A method.")], "And limits?")
    assert "User: What is RAG?" in p and "Assistant: A method." in p and "And limits?" in p
