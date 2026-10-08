"""Streamlit UI.  Run with:  streamlit run app.py"""
from __future__ import annotations

import os
from pathlib import Path

import streamlit as st

from docmind.chunking import get_splitter
from docmind.config import Settings
from docmind.factory import build_pipeline
from docmind.health import check_ollama
from docmind.knowledge_base import MODES
from docmind.loader import discover, load_bytes
from docmind.prompts import NO_ANSWER, extract_citations

SAMPLE_DIR = Path(__file__).parent / "data" / "sample_docs"
settings = Settings.from_env()
BACKEND = os.environ.get("DOCMIND_BACKEND", "chroma")

st.set_page_config(page_title="DocMind", page_icon="📄", layout="wide")


@st.cache_resource(show_spinner="Loading index...")
def get_pipeline():
    return build_pipeline(settings, BACKEND)


@st.cache_data(ttl=15, show_spinner=False)
def get_status(base_url: str, models: tuple):
    return check_ollama(base_url, list(models))


pipeline = get_pipeline()
kb = pipeline.kb
split = get_splitter(settings.chunk_size, settings.chunk_overlap)


def index_bytes(name: str, data: bytes) -> None:
    if len(data) > settings.max_upload_mb * 1024 * 1024:
        st.toast(f"{name}: larger than {settings.max_upload_mb} MB, skipped", icon="⚠️")
        return
    try:
        added = kb.add_pages(load_bytes(name, data), split)
    except ValueError as exc:
        st.toast(str(exc), icon="⚠️")
        return
    except Exception as exc:  # noqa: BLE001 - e.g. Ollama not running while embedding
        st.toast(f"Could not index {name}: {exc}", icon="❌")
        return
    st.toast(f"{name}: {added} new chunks" if added else f"{name}: already indexed", icon="✅")


def render_sources(sources: list[dict]) -> None:
    if not sources:
        return
    with st.expander(f"Sources ({len(sources)})"):
        for s in sources:
            tag = "cited" if s["cited"] else "retrieved"
            st.markdown(f"**[{s['n']}] {s['label']}** · {tag} · score {s['score']:.3f}")
            snippet = s["text"][:600] + ("..." if len(s["text"]) > 600 else "")
            st.text(snippet)


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("📄 DocMind")
    status = get_status(settings.ollama_base_url, (settings.llm_model, settings.embed_model))
    if status.ok:
        st.success(f"Ollama ready · {settings.llm_model}")
    else:
        st.error(status.message)

    st.subheader("Add documents")
    uploads = st.file_uploader("PDF, Markdown or text", type=["pdf", "md", "txt"], accept_multiple_files=True)
    if st.button("Index uploaded files", disabled=not uploads, use_container_width=True):
        with st.spinner("Chunking and embedding..."):
            for f in uploads:
                index_bytes(f.name, f.getvalue())
    if st.button("Load sample documents", use_container_width=True):
        with st.spinner("Chunking and embedding..."):
            for path in discover(SAMPLE_DIR):
                index_bytes(path.name, path.read_bytes())

    st.subheader("Indexed documents")
    sources = kb.sources()
    if not sources:
        st.caption("Nothing indexed yet.")
    for name, n in sources.items():
        left, right = st.columns([4, 1])
        left.write(f"**{name}** · {n} chunks")
        if right.button("🗑", key=f"del_{name}", help=f"Remove {name}"):
            kb.delete_source(name)
            st.rerun()

    st.subheader("Retrieval")
    mode = st.radio(
        "Mode",
        MODES,
        help="hybrid = vector + BM25 merged with Reciprocal Rank Fusion; vector = meaning only; bm25 = keywords only",
    )
    k = st.slider("Chunks to retrieve (top-k)", 1, 10, min(settings.top_k, 10))
    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ------------------------------------------------------------------ chat
st.title("Chat with your documents")
st.caption("Local hybrid RAG. Answers cite their sources, and nothing leaves your machine.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        render_sources(m.get("sources", []))

prompt = st.chat_input("Ask a question about your documents")
if prompt:
    history = [("human" if m["role"] == "user" else "ai", m["content"]) for m in st.session_state.messages]
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        sources: list[dict] = []
        try:
            if kb.count() == 0:
                reply = "No documents are indexed yet. Upload a file or load the samples from the sidebar."
                st.markdown(reply)
            else:
                prepared = pipeline.prepare(prompt, mode=mode, k=k, history=history)
                if prepared.messages is None:
                    reply = NO_ANSWER
                    st.markdown(reply)
                else:
                    reply = str(st.write_stream(pipeline.llm.stream(prepared.messages)))
                    cited = set(extract_citations(reply, len(prepared.retrieved)))
                    sources = [
                        {
                            "n": i,
                            "label": r.chunk.label,
                            "text": r.chunk.text,
                            "score": r.score,
                            "cited": i in cited,
                        }
                        for i, r in enumerate(prepared.retrieved, start=1)
                    ]
                    render_sources(sources)
        except Exception as exc:  # noqa: BLE001 - show the problem instead of crashing the app
            reply = f"Something went wrong: {exc}"
            st.error(reply + " Check that Ollama is running and the models are pulled.")
    st.session_state.messages.append({"role": "assistant", "content": reply, "sources": sources})
