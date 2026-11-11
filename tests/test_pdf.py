import pytest
from deltaparse.pdf import batch_by_pages, collect_pdfs, render_pdf


def test_render_pdf_gives_png_and_viewer_jpg(sample_pdf):
    pages = render_pdf(sample_pdf.read_bytes(), dpi=150)
    assert len(pages) == 1
    page = pages[0]
    assert page["blank"] is False
    assert page["png"].startswith(b"\x89PNG")
    assert page["jpg"].startswith(b"\xff\xd8")  # JPEG magic
    assert page["markdown"] == "" and page["blocks"] == [] and page["complete"] is True


def test_render_pdf_skips_blank_pages(blank_then_text_pdf):
    first, second = render_pdf(blank_then_text_pdf.read_bytes())
    assert first["blank"] is True and first["png"] is None
    assert first["jpg"]  # the viewer still shows the blank page
    assert second["blank"] is False and second["png"]


def test_collect_pdfs_from_file_and_folder(tmp_path, sample_pdf):
    assert collect_pdfs(sample_pdf) == [sample_pdf]
    sub = tmp_path / "nested"
    sub.mkdir()
    (sub / "b.PDF").write_bytes(sample_pdf.read_bytes())  # upper-case suffix still counts
    (tmp_path / "notes.txt").write_text("not a pdf")
    found = collect_pdfs(tmp_path)
    assert [p.name for p in found] == ["b.PDF", "sample.pdf"]
    assert len(collect_pdfs(tmp_path, limit=1)) == 1


def test_batch_by_pages_groups_and_skips_unreadable(tmp_path, sample_pdf, blank_then_text_pdf, capsys):
    broken = tmp_path / "broken.pdf"
    broken.write_bytes(b"this is not a pdf")
    batches = batch_by_pages([sample_pdf, broken, blank_then_text_pdf, sample_pdf], target=2)
    # 1 page + 2 pages reaches the target of 2, then the last 1-page PDF starts a new batch
    assert batches == [[sample_pdf, blank_then_text_pdf], [sample_pdf]]
    assert "skip unreadable broken.pdf" in capsys.readouterr().out


def test_render_without_view_skips_the_jpg(sample_pdf):
    from deltaparse.pdf import render_pdf

    page = render_pdf(sample_pdf.read_bytes(), view=False)[0]
    assert page["jpg"] is None and page["png"]


def test_fast_png_has_exactly_the_same_pixels(sample_pdf):
    pytest.importorskip("PIL")
    import pymupdf

    from deltaparse.pdf import render_pdf

    slow = render_pdf(sample_pdf.read_bytes())[0]["png"]
    fast = render_pdf(sample_pdf.read_bytes(), fast_png=True)[0]["png"]
    a, b = pymupdf.Pixmap(slow), pymupdf.Pixmap(fast)
    assert (a.width, a.height, a.n) == (b.width, b.height, b.n) and a.samples == b.samples
