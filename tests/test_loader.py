import pytest

from docmind.loader import SUPPORTED, discover, load_bytes


def make_pdf(text: str) -> bytes:
    stream = f"BT /F1 18 Tf 20 80 Td ({text}) Tj ET"
    objs = [
        "<< /Type /Catalog /Pages 2 0 R >>",
        "<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 300 144] /Contents 4 0 R "
        "/Resources << /Font << /F1 5 0 R >> >> >>",
        f"<< /Length {len(stream)} >>\nstream\n{stream}\nendstream",
        "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out = b"%PDF-1.4\n"
    offsets = []
    for i, obj in enumerate(objs, 1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n{obj}\nendobj\n".encode()
    xref = len(out)
    out += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode()
    for off in offsets:
        out += f"{off:010d} 00000 n \n".encode()
    out += f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF".encode()
    return out


def test_text_and_markdown():
    pages = load_bytes("notes.md", b"# Title\n\n\n\nBody   text")
    assert len(pages) == 1 and pages[0].page == 0 and pages[0].source == "notes.md"
    assert pages[0].text == "# Title\n\nBody text"
    assert load_bytes("a.TXT", b"hello")[0].text == "hello"


def test_source_is_basename_only():
    assert load_bytes("some/dir/notes.txt", b"hi")[0].source == "notes.txt"


def test_unsupported_extension():
    with pytest.raises(ValueError):
        load_bytes("report.docx", b"x")


def test_empty_text_rejected():
    with pytest.raises(ValueError):
        load_bytes("empty.txt", b"   \n  ")


def test_pdf_pages_and_text():
    pages = load_bytes("doc.pdf", make_pdf("Hello DocMind"))
    assert pages[0].page == 1
    assert "Hello DocMind" in pages[0].text


def test_garbage_pdf_gives_value_error():
    with pytest.raises(ValueError):
        load_bytes("broken.pdf", b"this is not a pdf")


def test_discover_filters_and_sorts(tmp_path):
    (tmp_path / "b.txt").write_text("b")
    (tmp_path / "a.md").write_text("a")
    (tmp_path / "image.png").write_bytes(b"x")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "c.pdf").write_bytes(b"x")
    found = [p.name for p in discover(tmp_path)]
    assert found == sorted(found) and set(found) == {"a.md", "b.txt", "c.pdf"}
    assert discover(tmp_path / "a.md") == [tmp_path / "a.md"]
    assert ".pdf" in SUPPORTED
