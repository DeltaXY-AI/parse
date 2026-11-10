"""Run nvidia/Nemotron-3.5-Parse on Modal GPU.

Usage:
    modal run parse_modal.py --path sample.pdf
    modal run parse_modal.py --path page.png --out-dir out

PDF pages are rendered locally (PyMuPDF, 150 dpi) and sent as PNG bytes.
Outputs (all pages combined): <name>.md, <name>.json (pages + blocks + raw), <name>.html (viewer)
"""
import json
import time
from pathlib import Path

import modal

MODEL_ID = "nvidia/Nemotron-3.5-Parse"
MODEL_DIR = "/model"
TASK_PROMPT = "</s><s><predict_bbox><predict_classes><output_markdown><predict_text_in_pic>"


def download_model():
    from huggingface_hub import snapshot_download

    snapshot_download(MODEL_ID, local_dir=MODEL_DIR)


image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch",
        "torchvision",
        "transformers",
        "timm",
        "accelerate",
        "huggingface_hub[hf_transfer]",
        "pillow",
        "numpy",
        "beautifulsoup4",
        "httpx",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .run_function(download_model)
)

app = modal.App("nemotron-parse", image=image)


@app.cls(gpu="H100", cpu=4, memory=16384, timeout=1800, scaledown_window=120, max_containers=4)
class Parser:
    @modal.enter()
    def load(self):
        import sys

        import torch
        from transformers import AutoModel, AutoProcessor, GenerationConfig

        sys.path.insert(0, MODEL_DIR)
        t = time.time()
        self.model = AutoModel.from_pretrained(
            MODEL_DIR, trust_remote_code=True, dtype=torch.bfloat16
        ).to("cuda").eval()
        self.processor = AutoProcessor.from_pretrained(MODEL_DIR, trust_remote_code=True)
        self.generation = GenerationConfig.from_pretrained(MODEL_DIR)
        self.generation.max_new_tokens = 8744
        print(f"model loaded in {time.time() - t:.1f}s on {torch.cuda.get_device_name(0)}")

    @modal.method()
    def parse(self, png_bytes: bytes) -> dict:
        import io

        import torch
        from nemotron_parse import postprocess_document
        from PIL import Image

        image = Image.open(io.BytesIO(png_bytes)).convert("RGB")
        inputs = self.processor(
            images=[image], text=TASK_PROMPT,
            return_tensors="pt", add_special_tokens=False,
        ).to("cuda")
        t = time.time()
        with torch.inference_mode():
            output_ids = self.model.generate(**inputs, generation_config=self.generation)
        raw = self.processor.batch_decode(output_ids, skip_special_tokens=False)[0]
        elapsed = time.time() - t
        n_tok = output_ids.shape[-1]
        print(f"{n_tok} tokens in {elapsed:.1f}s ({n_tok / elapsed:.0f} tok/s)")
        doc = postprocess_document(raw, image_size=image.size)
        return {
            "raw": raw,
            "markdown": doc["markdown"],
            "blocks": doc["blocks"],
            "seconds": round(elapsed, 2),
            "tokens": int(n_tok),
        }


def is_blank(pix) -> bool:
    """True if almost every pixel is near-white (model loops on blank pages)."""
    import pymupdf

    samples = pymupdf.Pixmap(pymupdf.csGRAY, pix).samples
    dark = sum(1 for b in samples if b < 200)
    return dark / max(len(samples), 1) < 0.00002


def render_pages(path: Path, dpi: int = 150) -> list[dict]:
    """Per page: png (model input, None if blank) and jpg (smaller copy for the HTML viewer)."""
    import pymupdf

    if path.suffix.lower() != ".pdf":
        pix = pymupdf.Pixmap(path.read_bytes())
        return [{"png": path.read_bytes(), "jpg": pix.tobytes("jpeg", jpg_quality=75)}]
    pages = []
    with pymupdf.open(str(path)) as pdf:
        for page in pdf:
            pix = page.get_pixmap(dpi=dpi, colorspace=pymupdf.csRGB, alpha=False)
            view = page.get_pixmap(dpi=100, colorspace=pymupdf.csRGB, alpha=False)
            pages.append({
                "png": None if is_blank(pix) else pix.tobytes("png"),
                "jpg": view.tobytes("jpeg", jpg_quality=75),
            })
    return pages


@app.local_entrypoint()
def main(path: str, out_dir: str = "out", dpi: int = 150):
    """Writes <name>.md (all pages), <name>.json (all pages + blocks), <name>.html (viewer)."""
    from viewer import build_html

    src = Path(path)
    out = Path(out_dir)
    out.mkdir(exist_ok=True)
    t0 = time.time()
    pages = render_pages(src, dpi)
    todo = [i for i, p in enumerate(pages) if p["png"] is not None]
    skipped = [i for i, p in enumerate(pages) if p["png"] is None]
    print(f"{src.name}: {len(pages)} page(s), blank skipped: {skipped or 'none'}")

    combined = [{"page": i, "blank": True, "seconds": 0, "blocks": [], "markdown": "", "raw": ""}
                for i in range(len(pages))]
    for i, result in zip(todo, Parser().parse.map([pages[i]["png"] for i in todo])):
        combined[i].update(blank=False, **result)
        print(f"page {i}: {len(result['blocks'])} blocks, {result['seconds']}s")

    md = "\n\n".join(f"<!-- page {p['page'] + 1} -->\n\n{p['markdown']}" for p in combined)
    (out / f"{src.stem}.md").write_text(md, encoding="utf-8")
    (out / f"{src.stem}.json").write_text(
        json.dumps({"source": src.name, "pages": combined}, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    for p in combined:
        p["jpg"] = pages[p["page"]]["jpg"]
    (out / f"{src.stem}.html").write_text(build_html(src.name, combined), encoding="utf-8")
    print(f"done in {time.time() - t0:.1f}s -> {out / src.stem}.md / .json / .html")
