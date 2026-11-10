"""Scan a folder of PDFs and rank them by how hard they are to parse.

Signals per PDF (PyMuPDF, no OCR):
  scanned_pages : pages with < 50 chars of text but an image covering most of the page
  garbled       : share of text chars that are replacement / private-use glyphs (bad font encoding)
  multi_col     : pages whose text blocks sit in 2+ distinct x-columns
  tables        : vector-drawn table candidates (many lines/rects on a page)
Usage: python find_tough.py <folder> [report.csv]
"""
import csv
import sys
from pathlib import Path

import pymupdf

pymupdf.TOOLS.mupdf_display_errors(False)


def analyse(path: Path) -> dict:
    r = {"path": str(path), "pages": 0, "scanned_pages": 0, "garbled": 0.0,
         "multi_col": 0, "table_pages": 0, "error": ""}
    try:
        with pymupdf.open(str(path)) as doc:
            r["pages"] = doc.page_count
            bad = total = 0
            for page in doc:
                area = abs(page.rect)
                text = page.get_text()
                total += len(text)
                bad += sum(1 for ch in text if ch == "�" or 0xE000 <= ord(ch) <= 0xF8FF)
                img_area = 0.0
                for info in page.get_image_info():
                    img_area += abs(pymupdf.Rect(info["bbox"]) & page.rect)
                if len(text.strip()) < 50 and img_area > 0.5 * area:
                    r["scanned_pages"] += 1
                blocks = [b for b in page.get_text("blocks") if b[6] == 0 and len(b[4].strip()) > 20]
                lefts = sorted({round(b[0] / (page.rect.width / 6)) for b in blocks})
                if len(blocks) >= 6 and len(lefts) >= 2 and lefts[-1] >= 3:
                    r["multi_col"] += 1
                drawings = page.get_drawings()
                lines = sum(len(d["items"]) for d in drawings)
                if lines >= 40:
                    r["table_pages"] += 1
            r["garbled"] = round(bad / total, 3) if total else 0.0
    except Exception as e:  # noqa: BLE001
        r["error"] = str(e)[:100]
    p = max(r["pages"], 1)
    r["score"] = round(
        3 * r["scanned_pages"] / p + 3 * min(r["garbled"] * 10, 1)
        + 1.5 * r["multi_col"] / p + 1.5 * r["table_pages"] / p, 2)
    return r


def main():
    root = Path(sys.argv[1])
    out = Path(sys.argv[2] if len(sys.argv) > 2 else "tough_report.csv")
    files = sorted(p for p in root.rglob("*") if p.suffix.lower() == ".pdf")
    rows = []
    for i, f in enumerate(files, 1):
        rows.append(analyse(f))
        if i % 200 == 0:
            print(f"{i}/{len(files)}", flush=True)
    rows.sort(key=lambda r: -r["score"])
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    n = len(rows)
    print(f"\n{n} PDFs | fully scanned: {sum(r['scanned_pages'] == r['pages'] > 0 for r in rows)}"
          f" | some scanned pages: {sum(r['scanned_pages'] > 0 for r in rows)}"
          f" | garbled text: {sum(r['garbled'] > 0.02 for r in rows)}"
          f" | errors: {sum(bool(r['error']) for r in rows)}")
    print("\nTop 25 toughest:")
    for r in rows[:25]:
        print(f"{r['score']:5.2f} pg={r['pages']:3} scan={r['scanned_pages']:3} garb={r['garbled']:.2f} "
              f"mcol={r['multi_col']:3} tbl={r['table_pages']:3}  {Path(r['path']).relative_to(root)}")


if __name__ == "__main__":
    main()
