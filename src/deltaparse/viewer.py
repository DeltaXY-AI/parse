"""Build a self-contained HTML viewer: PDF page (with block boxes) side by side with parsed output.

build_html(title, pages) -> str
  pages: list of {"page": int, "jpg": bytes, "blank": bool, "seconds": float, "blocks": [...]}
  block: {"class", "bbox" (normalised x0,y0,x1,y1), "content"}
"""
import base64
import html
import json
import re
from pathlib import Path

COLORS = {
    "Title": "#d62728", "Section-header": "#ff7f0e", "Text": "#1f77b4",
    "List-item": "#17becf", "Table": "#2ca02c", "Picture": "#9467bd",
    "Chart": "#8c564b", "Caption": "#e377c2", "Formula": "#bcbd22",
    "Page-header": "#7f7f7f", "Page-footer": "#7f7f7f", "Footnote": "#a0a0a0",
}
DEFAULT_COLOR = "#555555"


def _inline_md(text: str) -> str:
    """Escape, then apply bold/italic/code markdown."""
    s = html.escape(text)
    s = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s, flags=re.S)
    s = re.sub(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])", r"<i>\1</i>", s, flags=re.S)
    s = re.sub(r"`([^`]+)`", r"<code>\1</code>", s)
    return s.replace("\n", "<br>")


def render_block(block: dict) -> str:
    cls = block.get("class", "")
    content = (block.get("content") or block.get("text") or "").strip()
    if cls in ("Table",) and content.lstrip().startswith("<table"):
        return content  # model/postprocessor HTML table
    body = re.sub(r"^#{1,6}\s*", "", content)
    if cls == "Title":
        return f"<h1>{_inline_md(body)}</h1>"
    if cls == "Section-header":
        return f"<h2>{_inline_md(body)}</h2>"
    if cls == "Formula":
        return f"<pre class='formula'>{html.escape(content)}</pre>"
    if cls == "Picture" and not content:
        return "<div class='muted'>[picture]</div>"
    return f"<p>{_inline_md(body)}</p>" if body else "<div class='muted'>[empty]</div>"


CSS = """
*{box-sizing:border-box}body{margin:0;font:14px/1.5 system-ui,Segoe UI,Arial,sans-serif;background:#eef0f3;color:#222}
header{position:sticky;top:0;z-index:10;background:#1e293b;color:#fff;padding:10px 18px;display:flex;gap:18px;align-items:center;flex-wrap:wrap}
header h1{font-size:16px;margin:0;font-weight:600}header .stats{opacity:.8;font-size:12px}
header label{font-size:12px;cursor:pointer;user-select:none}
.legend{display:flex;gap:8px;flex-wrap:wrap;font-size:11px}.legend span{padding:1px 6px;border-radius:3px;color:#fff}
nav{font-size:12px}nav a{color:#93c5fd;margin-right:6px}
.page{display:grid;grid-template-columns:1fr 1fr;gap:16px;padding:16px 18px;border-bottom:2px solid #cbd5e1}
.page h3{grid-column:1/-1;margin:0;font-size:13px;color:#475569}
.imgwrap{position:relative;align-self:start;position:sticky;top:60px;background:#fff;box-shadow:0 1px 4px #0003}
.imgwrap img{display:block;width:100%}
.box{position:absolute;border:2px solid;cursor:pointer;transition:background .1s}
.box .tag{position:absolute;top:-15px;left:-2px;font-size:9px;color:#fff;padding:0 3px;white-space:nowrap;display:none}
.box:hover .tag,.box.hl .tag,body.tags .box .tag{display:block}
.box.hl{background:#facc1540;border-width:3px}
.out{background:#fff;padding:12px 18px;box-shadow:0 1px 4px #0003;min-width:0;overflow-x:auto}
.blk{border-left:4px solid;padding:2px 10px;margin:6px 0;cursor:pointer;border-radius:2px}
.blk.hl{background:#fef9c3}
.blk .cls{font-size:10px;font-weight:600;text-transform:uppercase;letter-spacing:.04em}
.blk h1{font-size:20px;margin:2px 0}.blk h2{font-size:16px;margin:2px 0}.blk p{margin:2px 0}
.blk table{border-collapse:collapse;margin:4px 0;font-size:13px}.blk td,.blk th{border:1px solid #94a3b8;padding:3px 7px;vertical-align:top}
.blk th{background:#f1f5f9}.muted{color:#94a3b8;font-style:italic}
pre.src{white-space:pre-wrap;font-size:12px;background:#f8fafc;margin:2px 0;display:none}
body.src pre.src{display:block}body.src .rendered{display:none}
.blankpage{padding:40px;text-align:center;color:#94a3b8}
"""

JS = """
function link(id){document.querySelectorAll('[data-id="'+id+'"]').forEach(e=>e.classList.add('hl'))}
function unlink(){document.querySelectorAll('.hl').forEach(e=>e.classList.remove('hl'))}
document.addEventListener('mouseover',e=>{const t=e.target.closest('[data-id]');unlink();if(t)link(t.dataset.id)});
document.addEventListener('click',e=>{const t=e.target.closest('[data-id]');if(!t)return;
  const other=t.classList.contains('box')?document.querySelector('.blk[data-id="'+t.dataset.id+'"]')
                                         :document.querySelector('.box[data-id="'+t.dataset.id+'"]');
  if(other)other.scrollIntoView({behavior:'smooth',block:'center'})});
document.getElementById('tags').onchange=e=>document.body.classList.toggle('tags',e.target.checked);
document.getElementById('src').onchange=e=>document.body.classList.toggle('src',e.target.checked);
"""


def build_html(title: str, pages: list[dict]) -> str:
    used = sorted({b.get("class", "") for p in pages for b in p.get("blocks", [])})
    n_blocks = sum(len(p.get("blocks", [])) for p in pages)
    secs = sum(p.get("seconds") or 0 for p in pages)
    gpu = f" | GPU {secs:.1f}s" if secs else ""
    legend = "".join(f"<span style='background:{COLORS.get(c, DEFAULT_COLOR)}'>{html.escape(c)}</span>" for c in used)
    nav = "".join(f"<a href='#p{p['page']}'>{p['page'] + 1}</a>" for p in pages)
    parts = [f"<!doctype html><html><head><meta charset='utf-8'><title>{html.escape(title)}</title>"
             f"<style>{CSS}</style></head><body><header><h1>{html.escape(title)}</h1>"
             f"<span class='stats'>{len(pages)} pages | {n_blocks} blocks{gpu}</span>"
             f"<label><input type='checkbox' id='tags'> show labels</label>"
             f"<label><input type='checkbox' id='src'> markdown source</label>"
             f"<div class='legend'>{legend}</div><nav>Page: {nav}</nav></header>"]
    for p in pages:
        n = p["page"]
        img = base64.b64encode(p["jpg"]).decode()
        meta = "blank - skipped" if p.get("blank") else f"{len(p['blocks'])} blocks"
        if p.get("seconds"):
            meta += f", {p['seconds']}s"
        if p.get("error"):
            meta += f" - FAILED ({p['error']})"
        elif p.get("complete") is False:
            meta += " - INCOMPLETE (hit token limit)"
        boxes, blks = [], []
        for i, b in enumerate(p.get("blocks", [])):
            bid = f"{n}-{i}"
            cls = b.get("class", "")
            col = COLORS.get(cls, DEFAULT_COLOR)
            x0, y0, x1, y1 = b["bbox"]
            boxes.append(
                f"<div class='box' data-id='{bid}' style='left:{x0 * 100:.3f}%;top:{y0 * 100:.3f}%;"
                f"width:{(x1 - x0) * 100:.3f}%;height:{(y1 - y0) * 100:.3f}%;border-color:{col}'>"
                f"<span class='tag' style='background:{col}'>{i + 1} {html.escape(cls)}</span></div>")
            src = b.get("content") or b.get("text") or ""
            blks.append(
                f"<div class='blk' data-id='{bid}' style='border-color:{col}'>"
                f"<div class='cls' style='color:{col}'>{i + 1}. {html.escape(cls)}</div>"
                f"<div class='rendered'>{render_block(b)}</div><pre class='src'>{html.escape(src)}</pre></div>")
        right = "".join(blks) or "<div class='blankpage'>No content</div>"
        parts.append(
            f"<section class='page' id='p{n}'><h3>Page {n + 1} - {meta}</h3>"
            f"<div class='imgwrap'><img src='data:image/jpeg;base64,{img}'>{''.join(boxes)}</div>"
            f"<div class='out'>{right}</div></section>")
    parts.append(f"<script>{JS}</script></body></html>")
    return "".join(parts)


def rebuild(json_path: Path, pdf_path: Path) -> Path:
    """Rebuild <json>.html from a saved combined JSON file and its source PDF."""
    import pymupdf

    data = json.loads(json_path.read_text(encoding="utf-8"))
    with pymupdf.open(str(pdf_path)) as pdf:
        for p in data["pages"]:
            p["jpg"] = pdf[p["page"]].get_pixmap(dpi=100).tobytes("jpeg", jpg_quality=75)
    out = json_path.with_suffix(".html")
    out.write_text(build_html(data["source"], data["pages"]), encoding="utf-8")
    return out


def main(argv: list[str] | None = None) -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Rebuild the HTML viewer from a saved JSON file.")
    ap.add_argument("json", type=Path, help="<name>.json written by a backend")
    ap.add_argument("pdf", type=Path, help="the source PDF")
    a = ap.parse_args(argv)
    print(f"wrote {rebuild(a.json, a.pdf)}")


if __name__ == "__main__":
    main()
