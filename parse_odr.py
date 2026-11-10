"""Parse PDFs through the OpenDocRouter hosted API (default model: Nemotron 3.5 Parse).

Same outputs as parse_vllm.py: <stem>.md, <stem>.json, <stem>.html (side-by-side viewer).
No GPU of our own: OpenDocRouter runs the model, bills per page, failed pages are free.

    python parse_odr.py <pdf-or-folder> [--out-dir out_odr] [--model nvidia/nemotron-3.5-parse]
                        [--limit N] [--workers 8]

Needs OPEN_DOC_ROUTER_API_KEY in the environment or in .env next to this file.
"""
import argparse
import base64
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx

from parse_vllm import collect_pdfs, render_pdf
from viewer import build_html

API = "https://www.opendocrouter.ai/v1"
INLINE_MAX = 2_800_000  # inline base64 limit is about 3 MB
SYNC_MAX_PAGES = 50

# OpenDocRouter layout types -> class names used by the viewer
CLASS = {
    "title": "Title", "section_header": "Section-header", "text": "Text", "list_item": "List-item",
    "table": "Table", "picture": "Picture", "chart": "Chart", "formula": "Formula",
    "caption": "Caption", "footnote": "Footnote", "page_header": "Page-header",
    "page_footer": "Page-footer", "code": "Code", "form": "Form", "key_value": "Key-value",
}


def api_key() -> str:
    key = os.environ.get("OPEN_DOC_ROUTER_API_KEY")
    env = Path(__file__).with_name(".env")
    if not key and env.exists():
        for line in env.read_text().splitlines():
            if line.startswith("OPEN_DOC_ROUTER_API_KEY="):
                key = line.split("=", 1)[1].strip()
    if not key:
        raise SystemExit("OPEN_DOC_ROUTER_API_KEY not set")
    return key


def page_spec(nums: list[int]) -> str:
    """[1,2,3,7] -> '1-3,7' (1-based)."""
    out, start, prev = [], None, None
    for n in nums:
        if start is None:
            start = prev = n
        elif n == prev + 1:
            prev = n
        else:
            out.append(f"{start}-{prev}" if prev > start else str(start))
            start = prev = n
    if start is not None:
        out.append(f"{start}-{prev}" if prev > start else str(start))
    return ",".join(out)


def request(client: httpx.Client, method: str, url: str, **kw) -> dict:
    """HTTP call with retry on 429/503 (Retry-After)."""
    for attempt in range(8):
        r = client.request(method, url, **kw)
        if r.status_code in (429, 503):
            wait = float(r.headers.get("Retry-After", 10))
            print(f"  {r.status_code} {r.json().get('error', {}).get('code')}, retry in {wait:.0f}s")
            time.sleep(min(wait, 120))
            continue
        if r.status_code >= 400:
            raise RuntimeError(f"{method} {url} -> {r.status_code}: {r.text[:300]}")
        return r.json()
    raise RuntimeError(f"{method} {url}: gave up after retries")


def document_ref(client: httpx.Client, data: bytes) -> dict:
    if len(data) <= INLINE_MAX:
        return {"data": base64.b64encode(data).decode(), "mime_type": "application/pdf"}
    up = request(client, "POST", f"{API}/uploads")
    httpx.put(up["upload_url"], content=data, timeout=300).raise_for_status()
    return {"upload_id": up["upload_id"]}


def to_blocks(page: dict) -> list[dict]:
    """OpenDocRouter layout elements -> viewer blocks (class, bbox x0,y0,x1,y1, content)."""
    lines = (page.get("markdown") or "").split("\n")
    layout = page.get("layout") or {}
    blocks = []
    for el in layout.get("elements") or []:
        boxes = el.get("boxes") or []
        if not boxes:
            continue
        x0 = min(b["x"] for b in boxes)
        y0 = min(b["y"] for b in boxes)
        x1 = max(b["x"] + b["w"] for b in boxes)
        y1 = max(b["y"] + b["h"] for b in boxes)
        span = el.get("lines")
        content = "\n".join(lines[span[0]:span[1] + 1]) if span else ""
        blocks.append({"class": CLASS.get(el.get("type"), el.get("type", "")),
                       "bbox": [x0, y0, x1, y1], "content": content,
                       "confidence": el.get("confidence")})
    return blocks


def parse_one(client: httpx.Client, src: Path, model: str) -> tuple[list[dict], float, dict]:
    data = src.read_bytes()
    pages = render_pdf(data, 150)  # blank detection + 100 dpi jpg for the viewer
    for p in pages:
        p.pop("png", None)
    wanted = [i + 1 for i, p in enumerate(pages) if not p["blank"]]
    stats = {"pages": len(pages), "blank": len(pages) - len(wanted), "failed": 0, "errors": {}}
    if not wanted:
        return pages, 0.0, stats

    body = {"model": model, "document": document_ref(client, data), "pages": page_spec(wanted),
            "layout": True}
    if len(wanted) > SYNC_MAX_PAGES:
        body.update(mode="async", cache=True)
        res = request(client, "POST", f"{API}/parse", json=body)
        while res["status"] == "processing":
            time.sleep(5)
            res = request(client, "GET", f"{API}/parse/{res['id']}")
        res = request(client, "GET", f"{API}/parse/{res['id']}", params={"expand": "markdown,layout"})
        got = list(res["pages"])
        while res.get("next_cursor"):
            res = request(client, "GET", f"{API}/parse/{res['id']}",
                          params={"expand": "markdown,layout", "cursor": res["next_cursor"]})
            got += res["pages"]
        request(client, "DELETE", f"{API}/parse/{res['id']}")  # drop the 24h cached copy
    else:
        res = request(client, "POST", f"{API}/parse", json=body)
        got = res["pages"]

    cost = 0.0
    for pg in got:
        p = pages[pg["page"] - 1]
        cost += pg.get("charge_usd") or 0
        if pg["status"] == "ok":
            p["markdown"] = pg.get("markdown") or ""
            p["blocks"] = to_blocks(pg)
        else:
            code = (pg.get("error") or {}).get("code", "error")
            p["complete"] = False
            p["error"] = code
            stats["failed"] += 1
            stats["errors"][code] = stats["errors"].get(code, 0) + 1
    return pages, cost, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--out-dir", default="out_odr")
    ap.add_argument("--model", default="nvidia/nemotron-3.5-parse")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()

    out = Path(a.out_dir)
    out.mkdir(exist_ok=True)
    pdfs = collect_pdfs(Path(a.path), a.limit)
    client = httpx.Client(headers={"Authorization": f"Bearer {api_key()}"}, timeout=600)
    credits = request(client, "GET", f"{API}/credits")
    print(f"{len(pdfs)} PDF(s), model {a.model}, credits {credits}")

    t0 = time.time()
    used, total = set(), {"pages": 0, "blank": 0, "failed": 0, "cost": 0.0, "errors": {}}
    with ThreadPoolExecutor(a.workers) as pool:
        futs = {pool.submit(parse_one, client, p, a.model): p for p in pdfs}
        for fut in as_completed(futs):
            src = futs[fut]
            try:
                pages, cost, st = fut.result()
            except Exception as e:  # noqa: BLE001
                print(f"FAIL {src.name}: {e}")
                continue
            for k in ("pages", "blank", "failed"):
                total[k] += st[k]
            total["cost"] += cost
            for k, v in st["errors"].items():
                total["errors"][k] = total["errors"].get(k, 0) + v
            stem = src.stem
            while stem in used:
                stem += "_"
            used.add(stem)
            combined = [{"page": i, **pg} for i, pg in enumerate(pages)]
            md = "\n\n".join(f"<!-- page {p['page'] + 1} -->\n\n{p['markdown']}" for p in combined)
            (out / f"{stem}.md").write_text(md, encoding="utf-8")
            meta = [{k: v for k, v in p.items() if k != "jpg"} for p in combined]
            (out / f"{stem}.json").write_text(
                json.dumps({"source": str(src), "model": a.model, "pages": meta},
                           indent=2, ensure_ascii=False), encoding="utf-8")
            (out / f"{stem}.html").write_text(build_html(f"{src.name} ({a.model})", combined),
                                              encoding="utf-8")
            print(f"{src.name}: {st['pages']} pages, {st['blank']} blank, {st['failed']} failed "
                  f"{st['errors'] or ''} ${cost:.4f}  [{time.time() - t0:.0f}s]")

    wall = time.time() - t0
    parsed = total["pages"] - total["blank"] - total["failed"]
    per_k = total["cost"] / max(parsed, 1) * 1000
    print(f"done: {total['pages']} pages ({total['blank']} blank, {total['failed']} failed "
          f"{total['errors'] or ''}) in {wall:.0f}s = {total['pages'] / max(wall, 1e-9):.1f} pages/s, "
          f"cost ${total['cost']:.4f} (${per_k:.2f} per 1k parsed pages) -> {out}/")


if __name__ == "__main__":
    main()
