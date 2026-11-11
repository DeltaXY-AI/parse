"""PDF helpers shared by both backends: find PDFs, render pages, group them into batches."""
from pathlib import Path

BLANK_INK_SHARE = 0.00002  # below this share of dark pixels a page counts as blank


def render_pdf(data: bytes, dpi: int = 150, view: bool = True, fast_png: bool = False) -> list[dict]:
    """Per page: png at `dpi` for the model (None if blank) + 100 dpi jpg for the viewer.

    view=False skips the viewer jpg (the API does not use it). fast_png=True encodes the png
    with Pillow at zlib level 1: the same pixels, about 5x faster than PyMuPDF's encoder.
    """
    import numpy as np
    import pymupdf

    pages = []
    with pymupdf.open(stream=data, filetype="pdf") as pdf:
        for page in pdf:
            pix = page.get_pixmap(dpi=dpi, colorspace=pymupdf.csRGB, alpha=False)
            gray = np.frombuffer(pymupdf.Pixmap(pymupdf.csGRAY, pix).samples, dtype=np.uint8)
            blank = bool((gray < 200).mean() < BLANK_INK_SHARE)  # model loops on blank pages
            jpg = None
            if view:
                jpg = page.get_pixmap(dpi=100, colorspace=pymupdf.csRGB, alpha=False).tobytes(
                    "jpeg", jpg_quality=75)
            pages.append({
                "png": None if blank else _png(pix, fast_png),
                "jpg": jpg,
                "blank": blank, "markdown": "", "blocks": [], "complete": True,
                "width": round(page.rect.width, 2), "height": round(page.rect.height, 2),  # points
            })
    return pages


def _png(pix, fast: bool) -> bytes:
    if not fast:
        return pix.tobytes("png")
    import io

    from PIL import Image

    out = io.BytesIO()
    Image.frombytes("RGB", (pix.width, pix.height), pix.samples).save(out, format="PNG", compress_level=1)
    return out.getvalue()


def collect_pdfs(path: Path, limit: int = 0) -> list[Path]:
    """A single file, or every PDF under a folder (sorted). `limit` 0 means no limit."""
    if path.is_dir():
        files = sorted(p for p in path.rglob("*") if p.suffix.lower() == ".pdf")
        return files[:limit] if limit else files
    return [path]


def batch_by_pages(pdfs: list[Path], target: int) -> list[list[Path]]:
    """Group PDFs into batches of about `target` pages. Unreadable PDFs are skipped."""
    import pymupdf

    batches, cur, n = [], [], 0
    for pdf in pdfs:
        try:
            with pymupdf.open(str(pdf)) as d:
                count = d.page_count
        except Exception as e:  # noqa: BLE001
            print(f"skip unreadable {pdf.name}: {e}")
            continue
        cur.append(pdf)
        n += count
        if n >= target:
            batches.append(cur)
            cur, n = [], 0
    if cur:
        batches.append(cur)
    return batches
