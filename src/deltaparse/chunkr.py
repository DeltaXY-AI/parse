"""Map Parse output (Nemotron blocks) to Chunkr's task and output format.

Chunkr's hosted API shuts down on 2026-11-15. This module lets existing Chunkr code keep
working against a self-hosted Parse server (see deltaparse.server and docs/chunkr_api_reference.md).

Parse block:    {"class": "Section-header", "bbox": [x0, y0, x1, y1] (0-1, page-relative),
                 "content": markdown (tables are HTML)}
Chunkr segment: {"segment_type": "SectionHeader", "bbox": {left, top, width, height} in pixels
                 at DPI, "page_width", "page_height", "content", "text", "html", "markdown", ...}

Segments are stored in the v1 shape (content = plain text). `for_api(output, "v2", config)` gives
the v2 shape, where content is markdown or HTML (see docs/chunkr_api_reference.md).
"""
import html as htmllib
import re
import uuid

from deltaparse.viewer import render_block

# Nemotron (and OpenDocRouter) block classes -> Chunkr segment types
SEGMENT_TYPE = {
    "Title": "Title", "Section-header": "SectionHeader", "Text": "Text", "List-item": "ListItem",
    "Table": "Table", "Picture": "Picture", "Chart": "Picture", "Caption": "Caption",
    "Formula": "Formula", "Footnote": "Footnote", "Page-header": "PageHeader",
    "Page-footer": "PageFooter", "Code": "Text", "Form": "Text", "Key-value": "Text",
}
DEFAULT_TARGET_LENGTH = 4096  # words per chunk (Chunkr default; 0 = one segment per chunk)
DPI = 144  # Chunkr reports page pixels at 144 dpi (1224 x 1584 for US Letter)


def plain_text(content: str) -> str:
    """Markdown or HTML -> plain text: tables become one line per row, markup is removed."""
    s = content or ""
    s = re.sub(r"</t[dh]>\s*", "\t", s, flags=re.I)
    s = re.sub(r"</tr>|<br\s*/?>|</p>|</li>", "\n", s, flags=re.I)
    s = re.sub(r"<[^>]+>", "", s)
    s = htmllib.unescape(s)
    s = re.sub(r"^\s{0,3}#{1,6}\s*", "", s, flags=re.M)          # headings
    s = re.sub(r"(\*\*|__)(.+?)\1", r"\2", s, flags=re.S)        # bold
    s = re.sub(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])", r"\1", s, flags=re.S)  # italic
    s = re.sub(r"`([^`]+)`", r"\1", s)
    s = re.sub(r"[ \t]*\t[ \t]*", "\t", s)
    s = "\n".join(line.strip(" \t") for line in s.splitlines())
    return re.sub(r"\n{3,}", "\n\n", s).strip()


def word_count(text: str) -> int:
    return len(text.split())


def to_bbox(norm: list[float], page_width: float, page_height: float) -> dict:
    """[x0, y0, x1, y1] in 0-1 -> Chunkr {left, top, width, height} in page units."""
    x0, y0, x1, y1 = (float(v) for v in norm)
    return {"left": round(x0 * page_width, 2), "top": round(y0 * page_height, 2),
            "width": round((x1 - x0) * page_width, 2), "height": round((y1 - y0) * page_height, 2)}


def _segment(segment_type: str, bbox: dict, markdown: str, html: str, page_number: int,
             page_width: float, page_height: float, confidence: float | None) -> dict:
    text = plain_text(markdown)
    return {
        "segment_id": str(uuid.uuid4()), "segment_type": segment_type, "bbox": bbox,
        "page_number": page_number, "page_width": page_width, "page_height": page_height,
        "content": text, "text": text, "html": html, "markdown": markdown,
        "confidence": confidence, "segment_length": word_count(text),
        "image": None, "llm": None, "description": None, "embed": None, "ocr": [],
    }


def to_segments(pages: list[dict], strategy: str = "LayoutAnalysis") -> list[dict]:
    """pages: [{"blocks", "markdown", "width", "height"}] in order (width/height in pixels).

    strategy "LayoutAnalysis" gives one segment per block; "Page" gives one segment per page.
    """
    segments = []
    for i, page in enumerate(pages):
        w, h, num = page["width"], page["height"], i + 1
        if strategy == "Page":
            md = page.get("markdown") or ""
            if md.strip():
                segments.append(_segment("Page", {"left": 0, "top": 0, "width": w, "height": h},
                                         md, "", num, w, h, None))
                segments[-1]["html"] = "".join(render_block(b) for b in page.get("blocks", []))
            continue
        for block in page.get("blocks", []):
            md = (block.get("content") or block.get("text") or "").strip()
            cls = block.get("class", "Text")
            if not md and cls != "Picture":
                continue
            segments.append(_segment(SEGMENT_TYPE.get(cls, "Text"), to_bbox(block["bbox"], w, h),
                                     md, render_block(block), num, w, h, block.get("confidence")))
    return segments


def to_chunks(segments: list[dict], target_length: int = DEFAULT_TARGET_LENGTH) -> list[dict]:
    """Group segments into chunks of about `target_length` words, in reading order.

    A new chunk starts before a title or section header once the current chunk is a quarter
    full, so a heading stays with the text under it. target_length 0: one segment per chunk.
    """
    chunks, cur = [], []

    def flush():
        if cur:
            content = "\n\n".join(s["content"] for s in cur if s["content"])
            chunks.append({"chunk_id": str(uuid.uuid4()), "segments": list(cur),
                           "chunk_length": sum(s["segment_length"] or 0 for s in cur),
                           "content": content, "embed": content})
            cur.clear()

    length = 0
    for seg in segments:
        heading = seg["segment_type"] in ("Title", "SectionHeader")
        n = seg["segment_length"] or 0
        if cur and (target_length <= 0 or length + n > target_length or (heading and length > 0 and length >= target_length // 4)):
            flush()
            length = 0
        cur.append(seg)
        length += n
    flush()
    return chunks


def build_output(pages: list[dict], file_name: str | None, config: dict | None = None,
                 mime_type: str = "application/pdf", dpi: int = DPI) -> dict:
    """Chunkr OutputResponse (v1 shape) for a parsed document.

    pages: render_pdf pages after parsing, with width/height in PDF points.
    """
    config = config or {}
    strategy = config.get("segmentation_strategy") or "LayoutAnalysis"
    target = (config.get("chunk_processing") or {}).get("target_length")
    target = DEFAULT_TARGET_LENGTH if target is None else target
    scale = dpi / 72
    sized = [{**p, "width": round(p["width"] * scale, 2), "height": round(p["height"] * scale, 2)}
             for p in pages]
    return {
        "chunks": to_chunks(to_segments(sized, strategy), target),
        "file_name": file_name, "mime_type": mime_type,
        "page_count": len(pages), "pages": None, "pdf_url": None,
    }


def _format_for(segment_type: str, config: dict) -> str:
    """v2 `content` format for a segment type: segment_processing[type].format, else Markdown."""
    per_type = (config.get("segment_processing") or {}).get(segment_type) or {}
    return per_type.get("format") or "Markdown"


def for_api(output: dict | None, api: str, config: dict | None = None,
            include_chunks: bool = True) -> dict | None:
    """Output as the given API version expects it ("v1" or "v2")."""
    if output is None:
        return None
    config = config or {}
    chunks = []
    if include_chunks:
        for chunk in output["chunks"]:
            segs = []
            for seg in chunk["segments"]:
                seg = dict(seg)
                if api == "v2":
                    fmt = _format_for(seg["segment_type"], config)
                    seg["content"] = seg["html"] if fmt == "Html" else seg["markdown"]
                segs.append(seg)
            content = "\n\n".join(s["content"] for s in segs if s["content"])
            chunks.append({**chunk, "segments": segs, "content": content})
    return {**output, "chunks": chunks}
