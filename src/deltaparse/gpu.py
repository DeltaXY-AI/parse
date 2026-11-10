"""Nemotron-3.5-Parse on Modal H100 with vLLM (NVIDIA reference MTP serving profile).

Usage (needs `pip install -e ".[gpu]"` and `modal setup`):
    deltaparse gpu some.pdf
    deltaparse gpu "C:/folder/with/pdfs" --out-dir out_vllm --limit 20
or directly:
    modal run -m deltaparse.gpu --path some.pdf

One H100 container runs `vllm serve` plus the repo's DocumentParser batch client
(up to 192 requests in flight). PDFs are uploaded as-is in batches of ~400 pages and
rendered on the container CPUs (blank pages skipped), so the GPU stays saturated. Per PDF it writes <name>.md, <name>.json and <name>.html (viewer).
"""
import subprocess
import time
from pathlib import Path

import modal

from deltaparse.output import unique_stem, write_outputs
from deltaparse.pdf import batch_by_pages, collect_pdfs, render_pdf

MODEL_ID = "nvidia/Nemotron-3.5-Parse"
MODEL_DIR = "/model"
PORT = 8000


def download_model():
    from huggingface_hub import snapshot_download

    snapshot_download(MODEL_ID, local_dir=MODEL_DIR)


image = (
    # CUDA devel image: FlashInfer JIT-compiles its sampling kernels with nvcc at warm-up
    modal.Image.from_registry("nvidia/cuda:13.0.2-devel-ubuntu24.04", add_python="3.12")
    .entrypoint([])
    .apt_install("build-essential")
    .pip_install("huggingface-hub==1.24.0")
    .run_function(download_model)
    .run_commands(
        f"cd {MODEL_DIR} && pip install '.[vllm,client]' 'vllm==0.26.0' 'transformers==5.14.1'",
        "pip check || true",
    )
    .pip_install("pymupdf==1.27.2.3")
    .add_local_python_source("deltaparse")  # our package (render_pdf) inside the container
)

vllm_cache = modal.Volume.from_name("nemotron-parse-vllm-cache", create_if_missing=True)
app = modal.App("nemotron-parse-vllm", image=image)


@app.cls(
    gpu="H100", retries=0, cpu=8, memory=32768, timeout=3600, scaledown_window=300,
    max_containers=1, volumes={"/root/.cache/vllm": vllm_cache},
)
@modal.concurrent(max_inputs=4)  # several chunks share one vLLM server
class VllmParser:
    @modal.enter()
    def start(self):
        # Never let a failed start crash the container: Modal would restart it in a loop
        # (each restart burns H100 time). Record the error and fail the call instead.
        self.error = None
        try:
            self._start()
        except Exception as e:  # noqa: BLE001
            self.error = repr(e)
            print("vLLM failed to start:", self.error)

    def _start(self):
        import httpx

        t = time.time()
        cmd = [
            "vllm", "serve", MODEL_DIR,
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
            "--chat-template", f"{MODEL_DIR}/chat_template.jinja",
            "--host", "127.0.0.1", "--port", str(PORT),
        ]
        self.proc = subprocess.Popen(cmd)
        while True:
            if time.time() - t > 900:
                self.proc.kill()
                raise RuntimeError("vllm serve not healthy after 15 min")
            if self.proc.poll() is not None:
                raise RuntimeError(f"vllm serve exited with {self.proc.returncode}")
            try:
                if httpx.get(f"http://127.0.0.1:{PORT}/health", timeout=2).status_code == 200:
                    break
            except httpx.HTTPError:
                pass
            time.sleep(2)
        vllm_cache.commit()
        print(f"vLLM ready in {time.time() - t:.0f}s")

    @modal.exit()
    def stop(self):
        if getattr(self, "proc", None):
            self.proc.terminate()

    @modal.method()
    def parse_pdfs(self, pdfs: list[bytes], mode: str = "single", dpi: int = 150) -> dict:
        """Render PDFs on the container CPUs, parse all non-blank pages in one batch.

        Returns {"pdfs": [[page, ...] per pdf], "metrics": {...}};
        page = {jpg, blank, markdown, blocks, complete}.
        """
        import io
        import sys
        from concurrent.futures import ProcessPoolExecutor

        from PIL import Image

        if self.error:
            raise RuntimeError("vLLM server failed to start: " + self.error)
        sys.path.insert(0, MODEL_DIR)
        from nemotron_parse.inference import DocumentParser

        t = time.time()
        with ProcessPoolExecutor(8) as pool:
            rendered = list(pool.map(render_pdf, pdfs, [dpi] * len(pdfs)))
        render_s = time.time() - t
        jobs = [(d, i) for d, pages in enumerate(rendered) for i, pg in enumerate(pages) if pg["png"]]
        images = [Image.open(io.BytesIO(rendered[d][i]["png"])).convert("RGB") for d, i in jobs]

        metrics = {}
        if images:
            with DocumentParser(f"http://127.0.0.1:{PORT}/v1", "nemotron-parse") as parser:
                result = parser.parse_prepared(
                    parser.prepare(images), mode=mode, concurrency=192,
                    postprocess_workers=8, allow_partial=True,
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


# ---------------------------------------------------------------- local side


@app.local_entrypoint()
def main(path: str, out_dir: str = "out_vllm", mode: str = "single", limit: int = 0,
         dpi: int = 150, batch_pages: int = 400):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pdfs = collect_pdfs(Path(path), limit)
    batches = batch_by_pages(pdfs, batch_pages)
    mb = sum(p.stat().st_size for b in batches for p in b) / 1e6
    print(f"{sum(map(len, batches))} PDF(s) in {len(batches)} batch(es), {mb:.1f} MB to upload")

    t0 = time.time()
    used, total_pages, parsed = set(), 0, 0
    payloads = [[p.read_bytes() for p in b] for b in batches]
    results = VllmParser().parse_pdfs.map(payloads, kwargs={"mode": mode, "dpi": dpi})
    for batch, res in zip(batches, results):
        m = res["metrics"]
        n = sum(len(pages) for pages in res["pdfs"])
        k = sum(1 for pages in res["pdfs"] for pg in pages if not pg["blank"])
        total_pages += n
        parsed += k
        print(f"batch: {len(batch)} PDFs, {n} pages ({n - k} blank), render {m.get('render_seconds')}s, "
              f"GPU {m.get('processed_pages_per_second', 0):.1f} pages/s, "
              f"incomplete {m.get('incomplete_pages', 0)}, elapsed {time.time() - t0:.0f}s")
        for src, pages in zip(batch, res["pdfs"]):
            write_outputs(out, unique_stem(src, used), src, pages, title=src.name)
    wall = time.time() - t0
    print(f"done: {total_pages} pages ({parsed} parsed) in {wall:.0f}s = "
          f"{total_pages / max(wall, 1e-9):.1f} pages/s end-to-end (incl. cold start) -> {out}/")
