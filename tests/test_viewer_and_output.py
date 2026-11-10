import json
from pathlib import Path

from deltaparse.output import unique_stem, write_outputs
from deltaparse.pdf import render_pdf
from deltaparse.viewer import build_html, rebuild, render_block


def test_render_block_kinds():
    assert render_block({"class": "Title", "content": "# Report"}) == "<h1>Report</h1>"
    assert render_block({"class": "Section-header", "content": "## Notes"}) == "<h2>Notes</h2>"
    table = "<table><tr><td>1</td></tr></table>"
    assert render_block({"class": "Table", "content": table}) == table  # HTML tables pass through
    assert render_block({"class": "Text", "content": "**bold** and *it*"}) == "<p><b>bold</b> and <i>it</i></p>"
    assert "[picture]" in render_block({"class": "Picture", "content": ""})


def test_render_block_escapes_html():
    out = render_block({"class": "Text", "content": "<script>alert(1)</script>"})
    assert "<script>" not in out and "&lt;script&gt;" in out


def test_build_html_has_page_boxes_and_blocks():
    pages = [{"page": 0, "jpg": b"\xff\xd8fake", "blank": False,
              "blocks": [{"class": "Title", "bbox": [0.1, 0.1, 0.9, 0.2], "content": "Hello"}]}]
    out = build_html("doc.pdf", pages)
    assert out.startswith("<!doctype html>")
    assert "<title>doc.pdf</title>" in out
    assert "1 pages | 1 blocks" in out
    assert "Hello" in out


def test_unique_stem_dedupes_same_names():
    used = set()
    assert unique_stem(Path("a/report.pdf"), used) == "report"
    assert unique_stem(Path("b/report.pdf"), used) == "report_"
    assert unique_stem(Path("c/report.pdf"), used) == "report__"


def test_write_outputs_then_rebuild_viewer(tmp_path, sample_pdf):
    pages = render_pdf(sample_pdf.read_bytes())
    for p in pages:
        p.pop("png")
    pages[0].update(markdown="# Quarterly Portfolio Summary",
                    blocks=[{"class": "Title", "bbox": [0.1, 0.08, 0.6, 0.11],
                             "content": "Quarterly Portfolio Summary"}])
    write_outputs(tmp_path, "sample", sample_pdf, pages, title="sample.pdf", extra={"model": "m1"})

    md = (tmp_path / "sample.md").read_text(encoding="utf-8")
    assert md == "<!-- page 1 -->\n\n# Quarterly Portfolio Summary"
    data = json.loads((tmp_path / "sample.json").read_text(encoding="utf-8"))
    assert list(data) == ["source", "model", "pages"]
    assert data["pages"][0]["page"] == 0 and "jpg" not in data["pages"][0]
    assert "Quarterly Portfolio Summary" in (tmp_path / "sample.html").read_text(encoding="utf-8")

    (tmp_path / "sample.html").unlink()
    out = rebuild(tmp_path / "sample.json", sample_pdf)
    assert out == tmp_path / "sample.html" and "data:image/jpeg;base64," in out.read_text(encoding="utf-8")
