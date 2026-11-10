"""Write the three output files per PDF: <stem>.md, <stem>.json and <stem>.html (viewer)."""
import json
from pathlib import Path

from deltaparse.viewer import build_html


def unique_stem(src: Path, used: set[str]) -> str:
    """File stem that is not in `used` yet (same file name in different folders), then mark it used."""
    stem = src.stem
    while stem in used:
        stem += "_"
    used.add(stem)
    return stem


def write_outputs(out: Path, stem: str, src: Path, pages: list[dict], title: str,
                  extra: dict | None = None) -> None:
    """pages: one dict per page, in order, with markdown, blocks and jpg (viewer image)."""
    combined = [{"page": i, **pg} for i, pg in enumerate(pages)]
    md = "\n\n".join(f"<!-- page {p['page'] + 1} -->\n\n{p['markdown']}" for p in combined)
    (out / f"{stem}.md").write_text(md, encoding="utf-8")
    meta = [{k: v for k, v in p.items() if k != "jpg"} for p in combined]
    doc = {"source": str(src), **(extra or {}), "pages": meta}
    (out / f"{stem}.json").write_text(
        json.dumps(doc, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    (out / f"{stem}.html").write_text(build_html(title, combined), encoding="utf-8")
