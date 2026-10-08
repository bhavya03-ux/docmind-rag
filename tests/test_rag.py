from docmind.backends import InMemoryBackend
from docmind.chunking import greedy_splitter
from docmind.knowledge_base import KnowledgeBase
from docmind.models import Page
from docmind.prompts import NO_ANSWER
from docmind.rag import RAGPipeline


class FakeLLM:
    def __init__(self, reply="Volumes keep data [1]."):
        self.reply = reply
        self.calls = []

    def generate(self, messages):
        self.calls.append(list(messages))
        if isinstance(self.reply, Exception):
            raise self.reply
        return self.reply

    def stream(self, messages):
        yield from self.reply.split(" ")


def make_pipeline(reply="Volumes keep data [1]."):
    kb = KnowledgeBase(InMemoryBackend())
    kb.add_pages(
        [Page("Docker containers package applications.\nVolumes keep data after removal.", "docker.md", 0)],
        greedy_splitter(60, 0),
    )
    llm = FakeLLM(reply)
    return RAGPipeline(kb, llm), llm


def test_ask_returns_text_and_resolved_citations():
    pipeline, llm = make_pipeline()
    answer = pipeline.ask("What keeps data after removal?", mode="bm25")
    assert answer.text == "Volumes keep data [1]."
    assert answer.cited == [1]
    assert answer.cited_sources[0].chunk.source == "docker.md"
    assert len(llm.calls) == 1


def test_no_relevant_context_skips_the_llm():
    pipeline, llm = make_pipeline()
    answer = pipeline.ask("zzzz qqqq")
    assert answer.text == NO_ANSWER and answer.sources == []
    assert llm.calls == []
    assert pipeline.prepare("zzzz qqqq").messages is None


def test_no_condense_call_without_history():
    pipeline, llm = make_pipeline()
    pipeline.prepare("volumes data")
    assert llm.calls == []


def test_followup_is_condensed_with_history():
    pipeline, llm = make_pipeline("volumes data")
    prepared = pipeline.prepare("and what else?", history=[("human", "q"), ("ai", "a")])
    assert prepared.query == "volumes data"
    assert len(llm.calls) == 1


def test_condense_failure_falls_back_to_original_question():
    pipeline, _ = make_pipeline(RuntimeError("model down"))
    assert pipeline.condense("original?", [("human", "q"), ("ai", "a")]) == "original?"


def test_overlong_rewrite_is_rejected():
    pipeline, _ = make_pipeline("x" * 600)
    assert pipeline.condense("original?", [("human", "q"), ("ai", "a")]) == "original?"
