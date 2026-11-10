"""PDF helpers shared by both backends: find PDFs, render pages, group them into batches."""
from pathlib import Path

BLANK_INK_SHARE = 0.00002  # below this share of dark pixels a page counts as blank


def render_pdf(data: bytes, dpi: int = 150) -> list[dict]:
    """Per page: png at `dpi` for the model (None if blank) + 100 dpi jpg for the viewer."""
    import numpy as np
    import pymupdf

    pages = []
    with pymupdf.open(stream=data, filetype="pdf") as pdf:
        for page in pdf:
            pix = page.get_pixmap(dpi=dpi, colorspace=pymupdf.csRGB, alpha=False)
            gray = np.frombuffer(pymupdf.Pixmap(pymupdf.csGRAY, pix).samples, dtype=np.uint8)
            blank = bool((gray < 200).mean() < BLANK_INK_SHARE)  # model loops on blank pages
            view = page.get_pixmap(dpi=100, colorspace=pymupdf.csRGB, alpha=False)
            pages.append({
                "png": None if blank else pix.tobytes("png"),
                "jpg": view.tobytes("jpeg", jpg_quality=75),
                "blank": blank, "markdown": "", "blocks": [], "complete": True,
                "width": round(page.rect.width, 2), "height": round(page.rect.height, 2),  # points
            })
    return pages


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
