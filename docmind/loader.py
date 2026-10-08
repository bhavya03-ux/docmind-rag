"""Read PDF / TXT / Markdown files into `Page` objects."""
from __future__ import annotations

import io
import re
from pathlib import Path

from .models import Page

SUPPORTED = (".pdf", ".txt", ".md")


def _clean(text: str) -> str:
    text = text.replace("\x00", "")
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _read_pdf(source: str, data: bytes) -> list[Page]:
    from pypdf import PdfReader

    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted:
            try:
                ok = reader.decrypt("")
            except Exception:  # noqa: BLE001 - any decrypt failure means we cannot read it
                ok = 0
            if not ok:
                raise ValueError(f"{source} is password protected.")
        pages = []
        for number, page in enumerate(reader.pages, start=1):
            text = _clean(page.extract_text() or "")
            if text:
                pages.append(Page(text, source, number))
        return pages
    except ValueError:
        raise
    except Exception as exc:  # noqa: BLE001 - pypdf raises several unrelated error types
        raise ValueError(f"{source} is not a readable PDF ({exc.__class__.__name__}).") from exc


def load_bytes(name: str, data: bytes) -> list[Page]:
    source = Path(name).name
    ext = Path(name).suffix.lower()
    if ext not in SUPPORTED:
        raise ValueError(f"Unsupported file type {ext!r}. Use one of: {', '.join(SUPPORTED)}")
    if ext == ".pdf":
        pages = _read_pdf(source, data)
    else:
        text = _clean(data.decode("utf-8", errors="replace"))
        pages = [Page(text, source, 0)] if text else []
    if not pages:
        raise ValueError(f"No extractable text in {source} (scanned PDFs need OCR first).")
    return pages


def load_path(path: str | Path) -> list[Page]:
    p = Path(path)
    return load_bytes(p.name, p.read_bytes())


def discover(path: str | Path) -> list[Path]:
    """A single file, or every supported file under a directory (sorted)."""
    p = Path(path)
    if p.is_dir():
        return sorted(f for f in p.rglob("*") if f.is_file() and f.suffix.lower() in SUPPORTED)
    return [p]
