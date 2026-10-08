import contextlib
import io
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from unittest import mock

import pytest

from docmind.cli import main
from docmind.config import Settings
from docmind.health import check_ollama, has_model


def test_has_model_matching():
    installed = ["llama3.2:latest", "nomic-embed-text:latest", "qwen2.5:7b"]
    assert has_model(installed, "llama3.2")
    assert has_model(installed, "qwen2.5:7b")
    assert not has_model(installed, "qwen2.5:3b")
    assert not has_model(installed, "mistral")


def test_unreachable_server():
    status = check_ollama("http://127.0.0.1:9", ["llama3.2"], timeout=0.5)
    assert not status.ok and not status.reachable and "Cannot reach" in status.message


def test_reachable_server_with_missing_model():
    body = json.dumps({"models": [{"name": "llama3.2:latest"}]}).encode()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        url = f"http://127.0.0.1:{server.server_port}"
        assert check_ollama(url, ["llama3.2"]).ok
        status = check_ollama(url, ["llama3.2", "nomic-embed-text"])
        assert status.reachable and status.missing == ["nomic-embed-text"] and not status.ok
    finally:
        server.shutdown()


def test_settings_validation():
    with pytest.raises(ValueError):
        Settings(chunk_size=100, chunk_overlap=100)
    with pytest.raises(ValueError):
        Settings(top_k=0)


def test_settings_from_env():
    env = {"DOCMIND_CHUNK_SIZE": "500", "DOCMIND_CHUNK_OVERLAP": "50", "OLLAMA_BASE_URL": "http://box:11434/"}
    with mock.patch.dict(os.environ, env):
        s = Settings.from_env()
    assert (s.chunk_size, s.chunk_overlap, s.ollama_base_url) == (500, 50, "http://box:11434")
    with mock.patch.dict(os.environ, {"DOCMIND_TOP_K": "abc"}):
        with pytest.raises(ValueError):
            Settings.from_env()


def run_cli(*argv):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        code = main(list(argv))
    return code, out.getvalue()


def test_cli_eval_with_memory_backend():
    code, out = run_cli("--backend", "memory", "eval")
    assert code == 0
    assert "| hybrid |" in out and "| bm25 |" in out and "| vector |" in out


def test_cli_sources_empty_memory_backend():
    code, out = run_cli("--backend", "memory", "sources")
    assert code == 0 and "(empty)" in out
