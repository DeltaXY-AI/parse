import pytest

from deltaparse.chunkr import (DPI, SEGMENT_TYPE, build_output, for_api, plain_text, to_bbox,
                               to_chunks, to_segments)


def test_plain_text_strips_markdown_and_html():
    assert plain_text("## **Notes** and *more*") == "Notes and more"
    table = "<table><tr><th>A</th><th>B</th></tr><tr><td>1 &amp; 2</td><td>3</td></tr></table>"
    assert plain_text(table) == "A\tB\n1 & 2\t3"
    assert plain_text("") == ""


def test_to_bbox_converts_normalised_corners_to_pixels():
    assert to_bbox([0.1, 0.2, 0.6, 0.3], 1000, 2000) == {"left": 100.0, "top": 400.0,
                                                         "width": 500.0, "height": 200.0}


def test_every_nemotron_class_maps_to_a_chunkr_type():
    chunkr_types = {"Caption", "Footnote", "Formula", "ListItem", "Page", "PageFooter", "PageHeader",
                    "Picture", "SectionHeader", "Table", "Text", "Title"}
    nemotron = {"Title", "Text", "Picture", "Section-header", "List-item", "Caption", "Page-footer",
                "Table", "Page-header", "Formula", "Footnote", "Chart"}
    assert nemotron <= set(SEGMENT_TYPE)
    assert set(SEGMENT_TYPE.values()) <= chunkr_types


PAGE = {"width": 1224, "height": 1584, "markdown": "# Report\n\nHello world", "blocks": [
    {"class": "Title", "bbox": [0.1, 0.05, 0.9, 0.1], "content": "# Report"},
    {"class": "Text", "bbox": [0.1, 0.2, 0.9, 0.3], "content": "Hello **world**"},
    {"class": "Table", "bbox": [0.1, 0.4, 0.9, 0.6],
     "content": "<table><tr><td>a</td><td>b</td></tr></table>"},
    {"class": "Text", "bbox": [0.1, 0.7, 0.9, 0.8], "content": "   "},  # empty: dropped
]}


def test_to_segments_layout_analysis():
    title, text, table = to_segments([PAGE])
    assert title["segment_type"] == "Title" and title["page_number"] == 1
    assert title["bbox"] == {"left": 122.4, "top": 79.2, "width": 979.2, "height": 79.2}
    assert text["content"] == text["text"] == "Hello world"
    assert text["markdown"] == "Hello **world**"
    assert text["html"] == "<p><b>world</b></p>".replace("<p>", "<p>Hello ")
    assert table["segment_type"] == "Table" and table["html"].startswith("<table>")
    assert table["content"] == "a\tb" and table["segment_length"] == 2
    assert len({s["segment_id"] for s in (title, text, table)}) == 3


def test_to_segments_page_strategy_gives_one_segment_per_page():
    (seg,) = to_segments([PAGE, {**PAGE, "markdown": "", "blocks": []}], strategy="Page")
    assert seg["segment_type"] == "Page"
    assert seg["bbox"] == {"left": 0, "top": 0, "width": 1224, "height": 1584}
    assert seg["markdown"] == PAGE["markdown"] and "<h1>Report</h1>" in seg["html"]


def _seg(kind, words):
    return {"segment_type": kind, "segment_length": words, "content": " ".join(["w"] * words)}


def test_to_chunks_respects_target_and_keeps_headings_with_text():
    segs = [_seg("Title", 2), _seg("Text", 50), _seg("SectionHeader", 2), _seg("Text", 50), _seg("Text", 60)]
    chunks = to_chunks(segs, target_length=100)
    assert [len(c["segments"]) for c in chunks] == [2, 2, 1]
    assert [c["chunk_length"] for c in chunks] == [52, 52, 60]
    assert chunks[0]["content"] == chunks[0]["embed"]
    assert [len(c["segments"]) for c in to_chunks(segs, target_length=0)] == [1, 1, 1, 1, 1]
    assert to_chunks([], 100) == []


def test_build_output_scales_points_to_144_dpi_pixels():
    page = {**PAGE, "width": 612, "height": 792}  # US Letter in points
    out = build_output([page], "report.pdf")
    seg = out["chunks"][0]["segments"][0]
    assert (seg["page_width"], seg["page_height"]) == (612 * DPI / 72, 792 * DPI / 72) == (1224, 1584)
    assert out["file_name"] == "report.pdf" and out["page_count"] == 1
    assert out["pages"] is None and out["pdf_url"] is None


def test_build_output_honours_chunk_target_zero():
    out = build_output([{**PAGE, "width": 612, "height": 792}], "r.pdf",
                       {"chunk_processing": {"target_length": 0}})
    assert len(out["chunks"]) == 3


@pytest.mark.parametrize("fmt, expected", [(None, "Hello **world**"), ("Html", "<p>Hello <b>world</b></p>")])
def test_for_api_v2_content_follows_requested_format(fmt, expected):
    out = build_output([{**PAGE, "width": 612, "height": 792}], "r.pdf")
    config = {"segment_processing": {"Text": {"format": fmt}}} if fmt else {}
    v2 = for_api(out, "v2", config)
    text = v2["chunks"][0]["segments"][1]
    assert text["content"] == expected
    assert text["text"] == "Hello world"
    v1 = for_api(out, "v1", config)
    assert v1["chunks"][0]["segments"][1]["content"] == "Hello world"  # v1: plain text
    assert out["chunks"][0]["segments"][1]["content"] == "Hello world"  # stored output unchanged
    assert for_api(out, "v2", include_chunks=False)["chunks"] == []
    assert for_api(None, "v2") is None
