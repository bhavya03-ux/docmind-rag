from pathlib import Path

from docmind.backends import InMemoryBackend
from docmind.chunking import greedy_splitter
from docmind.evaluate import (
    QAItem,
    evaluate_answers,
    evaluate_retrieval,
    is_relevant,
    load_qa,
    to_markdown,
)
from docmind.knowledge_base import KnowledgeBase
from docmind.loader import discover, load_path
from docmind.rag import RAGPipeline

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "data" / "sample_docs"
QA = ROOT / "eval" / "qa_set.json"


def sample_kb():
    kb = KnowledgeBase(InMemoryBackend())
    split = greedy_splitter(800, 120)
    for f in discover(DOCS):
        kb.add_pages(load_path(f), split)
    return kb


def test_every_expected_phrase_exists_in_its_source_file():
    for item in load_qa(QA):
        text = (DOCS / item.source).read_text(encoding="utf-8").lower()
        assert item.must_contain.lower() in text, item.question


def test_every_expected_phrase_lands_inside_a_single_chunk():
    kb = sample_kb()
    for item in load_qa(QA):
        assert any(is_relevant(c, item) for c in kb.chunks()), item.question


def test_metrics_for_hit_and_miss():
    kb = sample_kb()
    hit = QAItem("How do I keep data after a container is removed?", "docker_basics.md", "A volume stores data")
    miss = QAItem("How do I keep data after a container is removed?", "git_workflow.md", "never appears")
    best = evaluate_retrieval(kb, [hit], modes=("bm25",), k=4)["bm25"]
    assert best.hit_rate == 1.0 and 0 < best.mrr <= 1.0
    none = evaluate_retrieval(kb, [miss], modes=("bm25",), k=4)["bm25"]
    assert none.hit_rate == 0.0 and none.mrr == 0.0


def test_keyword_retrieval_is_strong_on_the_sample_set():
    scores = evaluate_retrieval(sample_kb(), load_qa(QA), modes=("bm25", "hybrid"), k=4)
    assert scores["bm25"].hit_rate >= 0.8
    assert scores["hybrid"].hit_rate >= 0.8


def test_markdown_table():
    table = to_markdown(evaluate_retrieval(sample_kb(), load_qa(QA)[:3]), 4)
    assert table.startswith("| Mode |") and "hybrid" in table and "bm25" in table


class EchoLLM:
    def generate(self, messages):
        return "Volumes persist data [1]."

    def stream(self, messages):
        yield "x"


def test_answer_evaluation_returns_rates_between_0_and_1():
    items = load_qa(QA)[:4]
    result = evaluate_answers(RAGPipeline(sample_kb(), EchoLLM()), items, mode="bm25")
    assert result["n"] == 4
    assert 0 <= result["keyword_accuracy"] <= 1 and 0 <= result["citation_accuracy"] <= 1
