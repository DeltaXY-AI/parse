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


def vllm_command(model_dir: str = MODEL_DIR, port: int = PORT, sleep_mode: bool = False) -> list[str]:
    """NVIDIA's reference MTP serving profile for one H100.

    sleep_mode adds --enable-sleep-mode, so vLLM can free the GPU before a Modal memory snapshot
    and take it back on restore (needs VLLM_SERVER_DEV_MODE=1 for /sleep and /wake_up).
    """
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
    ] + (["--enable-sleep-mode"] if sleep_mode else [])


def start_vllm(model_dir: str = MODEL_DIR, port: int = PORT, timeout: int = 900,
               sleep_mode: bool = False) -> subprocess.Popen:
    """Start `vllm serve` and wait until /health answers; raises if it exits or times out."""
    t = time.time()
    proc = subprocess.Popen(vllm_command(model_dir, port, sleep_mode))
    wait_healthy(proc, port, timeout)
    print(f"vLLM ready in {time.time() - t:.0f}s", flush=True)
    return proc


def wait_healthy(proc: subprocess.Popen, port: int = PORT, timeout: int = 900) -> None:
    """Wait until vLLM's /health answers; raises if the process exits or the wait times out."""
    import httpx

    t = time.time()
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
        time.sleep(0.5)


def sleep_vllm(port: int = PORT, level: int = 1) -> None:
    """Free the GPU (weights to CPU memory, empty KV cache dropped) before a memory snapshot."""
    import httpx

    httpx.post(f"http://127.0.0.1:{port}/sleep", params={"level": level}, timeout=300).raise_for_status()


def wake_vllm(proc: subprocess.Popen, port: int = PORT) -> None:
    """Take the GPU back after a snapshot restore and wait until vLLM serves again."""
    import httpx

    httpx.post(f"http://127.0.0.1:{port}/wake_up", timeout=300).raise_for_status()
    wait_healthy(proc, port, timeout=300)


def warm_up(model_dir: str = MODEL_DIR, port: int = PORT, runs: int = 2) -> None:
    """Parse a realistic page `runs` times in a child process, so vLLM compiles every kernel it
    needs. A child, because the parser starts process pools whose helper processes would
    otherwise stay in the container (and in its memory snapshot)."""
    code = ("import sys, tempfile; from pathlib import Path; from deltaparse import engine; "
            "from deltaparse.sample import make_sample; d = tempfile.mkdtemp(); "
            "pdf = make_sample(Path(d) / 'warm.pdf').read_bytes(); "
            f"[engine.parse_pdfs([pdf], model_dir={model_dir!r}, port={port}) for _ in range({runs})]")
    from pathlib import Path

    pkg_root = str(Path(__file__).resolve().parents[1])  # so the child can import deltaparse
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, [pkg_root, os.environ.get("PYTHONPATH")]))}
    subprocess.run([sys.executable, "-c", code], check=True, timeout=600, env=env)



def parse_pdfs(pdfs: list[bytes], mode: str = "single", dpi: int = 150, model_dir: str = MODEL_DIR,
               port: int = PORT, workers: int = 8, progress=None, view: bool = True) -> dict:
    """Render PDFs on the CPUs, parse all non-blank pages in one batch on the running vLLM.

    Returns {"pdfs": [[page, ...] per pdf], "metrics": {...}};
    page = {jpg, blank, markdown, blocks, complete}.
    progress(done, total), if given, is called as page requests finish (counted from vLLM
    responses, so with mode="double"/"triple" it can reach total before the last stage ends).
    view=False skips the viewer jpg per page (the API does not use it).
    """
    import tempfile
    from concurrent.futures import ProcessPoolExecutor
    from pathlib import Path

    from deltaparse.pdf import render_pdf

    if model_dir not in sys.path:
        sys.path.insert(0, model_dir)
    from nemotron_parse.inference import DocumentParser

    t = time.time()
    n = len(pdfs)
    if n == 1:  # work is split per PDF, so a pool only costs its start-up here
        rendered = [render_pdf(pdfs[0], dpi, view, True)]
    else:
        with ProcessPoolExecutor(workers) as pool:
            rendered = list(pool.map(render_pdf, pdfs, [dpi] * n, [view] * n, [True] * n))
    render_s = time.time() - t
    jobs = [(d, i) for d, pages in enumerate(rendered) for i, pg in enumerate(pages) if pg["png"]]
    # The parser keeps a PNG it is given as a file; a PIL image it would encode again (slowly).
    tmp = tempfile.TemporaryDirectory()
    images = []
    for k, (d, i) in enumerate(jobs):
        f = Path(tmp.name) / f"{k}.png"
        f.write_bytes(rendered[d][i]["png"])
        images.append(f)

    metrics = {}
    t = time.time()
    if images:
        import threading

        import httpx

        total, done, lock = len(images), [0], threading.Lock()

        def on_response(response):  # runs on the parser's request threads
            if progress and response.request.url.path.endswith("/chat/completions"):
                with lock:
                    done[0] += 1
                    n = min(done[0], total)
                progress(n, total)

        if progress:
            progress(0, total)
        client = httpx.Client(timeout=1800, event_hooks={"response": [on_response]},
                              limits=httpx.Limits(max_connections=512, max_keepalive_connections=256))
        with client, DocumentParser(f"http://127.0.0.1:{port}/v1", "nemotron-parse", client=client) as parser:
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
    tmp.cleanup()
    for pages in rendered:
        for pg in pages:
            pg.pop("png", None)
    metrics["render_seconds"] = round(render_s, 2)
    metrics["parse_seconds"] = round(time.time() - t, 2)
    return {"pdfs": rendered, "metrics": metrics}
