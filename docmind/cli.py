"""Command-line interface:  python -m docmind.cli <command> --help"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .chunking import get_splitter
from .config import Settings
from .evaluate import evaluate_answers, evaluate_retrieval, load_qa, to_markdown
from .factory import BACKENDS, build_knowledge_base, build_pipeline
from .health import check_ollama
from .knowledge_base import MODES, KnowledgeBase
from .loader import discover, load_path

ROOT = Path(__file__).resolve().parent.parent
SAMPLE_DOCS = ROOT / "data" / "sample_docs"
SAMPLE_QA = ROOT / "eval" / "qa_set.json"


def ingest_paths(kb: KnowledgeBase, paths: list[Path], settings: Settings) -> int:
    split = get_splitter(settings.chunk_size, settings.chunk_overlap)
    total = 0
    for root in paths:
        files = discover(root)
        if not files:
            print(f"! nothing to index in {root}")
        for f in files:
            try:
                added = kb.add_pages(load_path(f), split)
            except (ValueError, OSError) as exc:
                print(f"! skipped {f}: {exc}")
                continue
            total += added
            print(f"+ {f.name}: {added} new chunks")
    return total


def _cmd_ingest(args, settings) -> int:
    kb = build_knowledge_base(settings, args.backend)
    ingest_paths(kb, [Path(p) for p in args.paths], settings)
    print(f"Indexed documents: {kb.sources()}")
    return 0


def _cmd_ask(args, settings) -> int:
    status = check_ollama(settings.ollama_base_url, [settings.llm_model, settings.embed_model])
    if not status.ok:
        print(status.message)
        return 1
    pipeline = build_pipeline(settings, args.backend)
    answer = pipeline.ask(args.question, mode=args.mode, k=args.k)
    print(answer.text)
    if answer.sources:
        print("\nSources:")
        for i, r in enumerate(answer.sources, 1):
            mark = "*" if i in answer.cited else " "
            print(f" {mark}[{i}] {r.chunk.label}  (score {r.score:.3f})")
    return 0


def _cmd_eval(args, settings) -> int:
    kb = build_knowledge_base(settings, args.backend)
    ingest_paths(kb, [Path(p) for p in args.docs], settings)
    items = load_qa(args.qa)
    print(f"\nRetrieval on {len(items)} questions (backend={args.backend}):\n")
    print(to_markdown(evaluate_retrieval(kb, items, k=args.k), args.k))
    if args.backend == "memory":
        print("\nNote: the memory backend is a hashed bag-of-words stand-in, not real embeddings.")
    if args.with_answers:
        status = check_ollama(settings.ollama_base_url, [settings.llm_model])
        if not status.ok:
            print("\n" + status.message)
            return 1
        from .llm import OllamaLLM
        from .rag import RAGPipeline

        scores = evaluate_answers(RAGPipeline(kb, OllamaLLM(settings), top_k=args.k), items, "hybrid", args.k)
        print(f"\nAnswers (hybrid): keyword accuracy {scores['keyword_accuracy']:.2f}, "
              f"citation accuracy {scores['citation_accuracy']:.2f}")
    return 0


def _cmd_sources(args, settings) -> int:
    kb = build_knowledge_base(settings, args.backend)
    for name, n in kb.sources().items():
        print(f"{name}: {n} chunks")
    if not kb.count():
        print("(empty)")
    return 0


def _cmd_clear(args, settings) -> int:
    kb = build_knowledge_base(settings, args.backend)
    kb.clear()
    print("Index cleared.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="docmind", description="Local hybrid-RAG chatbot")
    p.add_argument("--backend", choices=BACKENDS, default="chroma", help="vector store (default: chroma)")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("ingest", help="index files or folders (pdf, txt, md)")
    s.add_argument("paths", nargs="+")
    s.set_defaults(func=_cmd_ingest)

    s = sub.add_parser("ask", help="ask a question about the indexed documents")
    s.add_argument("question")
    s.add_argument("--mode", choices=MODES, default="hybrid")
    s.add_argument("-k", type=int, default=None, help="chunks to retrieve")
    s.set_defaults(func=_cmd_ask)

    s = sub.add_parser("eval", help="benchmark vector vs BM25 vs hybrid retrieval")
    s.add_argument("--docs", nargs="+", default=[str(SAMPLE_DOCS)])
    s.add_argument("--qa", default=str(SAMPLE_QA))
    s.add_argument("-k", type=int, default=4)
    s.add_argument("--with-answers", action="store_true", help="also grade LLM answers (needs Ollama)")
    s.set_defaults(func=_cmd_eval)

    s = sub.add_parser("sources", help="list indexed documents")
    s.set_defaults(func=_cmd_sources)
    s = sub.add_parser("clear", help="delete everything from the index")
    s.set_defaults(func=_cmd_clear)
    return p


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    args = build_parser().parse_args(argv)
    try:
        return args.func(args, Settings.from_env())
    except KeyboardInterrupt:
        return 130
    except Exception as exc:  # noqa: BLE001 - show a friendly message instead of a traceback
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
