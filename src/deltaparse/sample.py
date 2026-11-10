"""Create a small sample PDF (heading, paragraph, table) for testing the parser."""
import argparse
from pathlib import Path

import pymupdf


def make_sample(path: Path) -> Path:
    """Write the one-page sample PDF to `path` and return it."""
    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)  # A4 in points

    page.insert_text((60, 80), "Quarterly Portfolio Summary", fontsize=22, fontname="hebo")
    page.insert_textbox(
        pymupdf.Rect(60, 100, 535, 170),
        "This report summarises holdings across mutual funds and equities as of "
        "October 2026. Values are in Indian Rupees. The table below lists the top "
        "holdings by current market value.",
        fontsize=11,
    )

    rows = [
        ["Holding", "Units", "NAV / Price", "Value (Rs)"],
        ["Nifty 50 Index Fund", "12,450.32", "245.10", "30,51,523"],
        ["PFC", "500", "412.65", "2,06,325"],
        ["REC", "400", "538.20", "2,15,280"],
        ["HDFC Bank", "150", "1,742.00", "2,61,300"],
        ["Total", "", "", "37,34,428"],
    ]
    x0, y0, col_w, row_h = 60, 200, [190, 80, 100, 105], 24
    for r, row in enumerate(rows):
        x = x0
        for c, cell in enumerate(row):
            rect = pymupdf.Rect(x, y0 + r * row_h, x + col_w[c], y0 + (r + 1) * row_h)
            page.draw_rect(rect, color=(0, 0, 0), width=0.7)
            page.insert_textbox(rect + (4, 6, -4, 0), cell, fontsize=10,
                                fontname="hebo" if r in (0, len(rows) - 1) else "helv")
            x += col_w[c]

    page.insert_text((60, 380), "Notes", fontsize=14, fontname="hebo")
    page.insert_textbox(
        pymupdf.Rect(60, 390, 535, 460),
        "1. SIP of Rs 40,000 per month continues.\n2. Rebalance once a year.",
        fontsize=11,
    )
    doc.save(str(path))
    return path


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Write a one-page sample PDF.")
    ap.add_argument("out", nargs="?", default="sample.pdf")
    a = ap.parse_args(argv)
    print(f"wrote {make_sample(Path(a.out))}")


if __name__ == "__main__":
    main()
