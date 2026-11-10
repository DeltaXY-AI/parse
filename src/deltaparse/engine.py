"""Nemotron-3.5-Parse on one GPU: start `vllm serve` and parse PDFs through it.

No Modal here, so the same code runs on Modal (deltaparse.gpu) and in the Docker image
(deltaparse.selfhost). Needs the GPU environment: vllm, the model's nemotron_parse package
(installed from the model folder) and pymupdf.
"""
import os
import subprocess
import sys
import time

MODEL_ID = "nvidia/Nemotron-3.5-Parse"
MODEL_DIR = os.environ.get("DELTAPARSE_MODEL_DIR", "/model")
PORT = 8000  # vLLM's own port, on 127.0.0.1 only


def download_model(model_dir: str = MODEL_DIR) -> None:
    from huggingface_hub import snapshot_download

    snapshot_download(MODEL_ID, local_dir=model_dir)


def vllm_command(model_dir: str = MODEL_DIR, port: int = PORT) -> list[str]:
    """NVIDIA's reference MTP serving profile for one H100."""
    return [
        "vllm", "serve", model_dir,
        "--served-model-name", "nemotron-parse",
        "--trust-remote-code",
        "--dtype", "bfloat16",
        "--distributed-executor-backend", "uni",
        "--tensor-parallel-size", "1",
        "--data-parallel-size", "1",
        "--max-model-len", "16384",
        "--max-num-seqs", "192",
        "--max-num-batched-tokens", "16384",
        "--no-enable-prefix-caching",
        "--limit-mm-per-prompt", '{"image":1}',
        "--attention-config", '{"flash_attn_version":3}',
        "--chat-template", f"{model_dir}/chat_template.jinja",
        "--host", "127.0.0.1", "--port", str(port),
    ]


def start_vllm(model_dir: str = MODEL_DIR, port: int = PORT, timeout: int = 900) -> subprocess.Popen:
    """Start `vllm serve` and wait until /health answers; raises if it exits or times out."""
    import httpx

    t = time.time()
    proc = subprocess.Popen(vllm_command(model_dir, port))
    while True:
        if time.time() - t > timeout:
            proc.kill()
            raise RuntimeError(f"vllm serve not healthy after {timeout // 60} min")
        if proc.poll() is not None:
            raise RuntimeError(f"vllm serve exited with {proc.returncode}")
        try:
            if httpx.get(f"http://127.0.0.1:{port}/health", timeout=2).status_code == 200:
                break
        except httpx.HTTPError:
            pass
        time.sleep(2)
    print(f"vLLM ready in {time.time() - t:.0f}s", flush=True)
    return proc


def parse_pdfs(pdfs: list[bytes], mode: str = "single", dpi: int = 150, model_dir: str = MODEL_DIR,
               port: int = PORT, workers: int = 8) -> dict:
    """Render PDFs on the CPUs, parse all non-blank pages in one batch on the running vLLM.

    Returns {"pdfs": [[page, ...] per pdf], "metrics": {...}};
    page = {jpg, blank, markdown, blocks, complete}.
    """
    import io
    from concurrent.futures import ProcessPoolExecutor

    from PIL import Image

    from deltaparse.pdf import render_pdf

    if model_dir not in sys.path:
        sys.path.insert(0, model_dir)
    from nemotron_parse.inference import DocumentParser

    t = time.time()
    with ProcessPoolExecutor(workers) as pool:
        rendered = list(pool.map(render_pdf, pdfs, [dpi] * len(pdfs)))
    render_s = time.time() - t
    jobs = [(d, i) for d, pages in enumerate(rendered) for i, pg in enumerate(pages) if pg["png"]]
    images = [Image.open(io.BytesIO(rendered[d][i]["png"])).convert("RGB") for d, i in jobs]

    metrics = {}
    if images:
        with DocumentParser(f"http://127.0.0.1:{port}/v1", "nemotron-parse") as parser:
            result = parser.parse_prepared(
                parser.prepare(images), mode=mode, concurrency=192,
                postprocess_workers=workers, allow_partial=True,
            )
        metrics = result.metrics
        for (d, i), doc in zip(jobs, result.documents):
            rendered[d][i].update(
                markdown=doc.get("markdown", ""), blocks=doc.get("blocks", []),
                complete=doc.get("complete", doc.get("incomplete") is not True),
            )
    for pages in rendered:
        for pg in pages:
            pg.pop("png", None)
    metrics["render_seconds"] = round(render_s, 1)
    return {"pdfs": rendered, "metrics": metrics}
