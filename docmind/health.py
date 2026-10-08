"""Check that Ollama is reachable and the needed models are pulled."""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass, field


@dataclass
class OllamaStatus:
    reachable: bool
    missing: list[str] = field(default_factory=list)
    message: str = ""

    @property
    def ok(self) -> bool:
        return self.reachable and not self.missing


def has_model(installed: list[str], wanted: str) -> bool:
    """`llama3.2` matches `llama3.2:latest`; `llama3.2:1b` must match exactly."""
    if ":" in wanted:
        return wanted in installed
    return any(name == wanted or name.split(":")[0] == wanted for name in installed)


def check_ollama(base_url: str, models: list[str], timeout: float = 2.0) -> OllamaStatus:
    url = base_url.rstrip("/") + "/api/tags"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            data = json.load(resp)
    except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
        return OllamaStatus(False, list(models), f"Cannot reach Ollama at {base_url} ({exc}). Is it running?")
    installed = [m.get("name", "") for m in data.get("models", [])]
    missing = [m for m in models if not has_model(installed, m)]
    msg = "Ollama is ready." if not missing else "Missing models: " + ", ".join(f"`ollama pull {m}`" for m in missing)
    return OllamaStatus(True, missing, msg)
