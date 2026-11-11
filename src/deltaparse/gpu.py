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
import time
from pathlib import Path

import modal

from deltaparse import engine
from deltaparse.output import unique_stem, write_outputs
from deltaparse.pdf import batch_by_pages, collect_pdfs

MODEL_ID = engine.MODEL_ID
MODEL_DIR = "/model"
PORT = engine.PORT
SCALEDOWN = 300  # seconds an idle H100 stays up

# Modal Dicts of the hosted API (deltaparse.api); the GPU writes task results straight into them.
TASKS = "deltaparse-tasks"      # task status and output
KEYS = "deltaparse-keys"        # sha256(key) -> customer (see deltaparse.keys)
USAGE = "deltaparse-usage"      # task_id -> pages parsed, per customer
TIMINGS = "deltaparse-timings"  # task_id -> seconds per step (see `deltaparse timings`)
STATE = "deltaparse-state"      # "gpu": is a GPU up; "cold_starts": recent cold waits; "warm:<customer>"
DEFAULT_COLD_START_S = 150      # used for the status hint until real cold starts are recorded


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
    .env({
        "VLLM_SERVER_DEV_MODE": "1",            # /sleep and /wake_up (vLLM listens on 127.0.0.1 only)
        "TORCHINDUCTOR_COMPILE_THREADS": "1",   # serial compilation restores reliably
        # JIT and compile caches on the container disk: a cache Volume that changes after the
        # snapshot breaks its restore.
        "VLLM_CACHE_ROOT": "/tmp/deltaparse-cache/vllm",
        "TORCHINDUCTOR_CACHE_DIR": "/tmp/deltaparse-cache/torchinductor",
        "TRITON_CACHE_DIR": "/tmp/deltaparse-cache/triton",
        # One GPU, no peers: a restored NCCL heartbeat would poll a pre-snapshot socket.
        "TORCH_NCCL_DUMP_ON_TIMEOUT": "0", "TORCH_NCCL_ENABLE_MONITORING": "0",
        "TORCH_NCCL_TRACE_BUFFER_SIZE": "0",
    })
    .add_local_python_source("deltaparse")  # our package (render_pdf) inside the container
)

app = modal.App("nemotron-parse-vllm", image=image)


def _dict(name: str):
    return modal.Dict.from_name(name, create_if_missing=True)


# Cold start (#76): a deployed app takes a CPU and GPU memory snapshot once vLLM has started,
# parsed a warm-up page twice and gone to sleep; a new container restores it and wakes vLLM.
# What makes the snapshot restorable (learned in DeltaXY-AI/suvadi): no mutable cache Volume
# (compile caches live in /tmp, set in the image env), nothing left running from the warm-up,
# and a failed boot raises instead of being captured. `modal run` takes no snapshot.
@app.cls(
    gpu="H100", retries=0, cpu=8, memory=32768, timeout=3600, scaledown_window=SCALEDOWN,
    max_containers=1, startup_timeout=15 * 60,
    enable_memory_snapshot=True, experimental_options={"enable_gpu_snapshot": True},
)
@modal.concurrent(max_inputs=4)  # several chunks share one vLLM server
class VllmParser:
    @modal.enter(snap=True)
    def boot(self):
        """Once per snapshot: start vLLM, warm it up, put it to sleep (raises if any step fails)."""
        t = time.time()
        self.proc = engine.start_vllm(MODEL_DIR, PORT, sleep_mode=True)
        engine.warm_up(MODEL_DIR, PORT)
        engine.sleep_vllm(PORT)
        self.booted_at = time.time()
        self.boot_s = round(self.booted_at - t, 1)
        print(f"vLLM started, warmed and asleep in {self.boot_s}s; ready for the snapshot", flush=True)

    @modal.enter(snap=False)
    def wake(self):
        """In every container: wake vLLM (restored from the snapshot, or just booted)."""
        # Never let a failed wake crash the container: Modal would restart it in a loop (each
        # restart burns H100 time). Fall back to a plain start; record the error if that fails.
        t = time.time()
        self.restored = t - self.booted_at > 60  # boot() ran in an earlier container
        self.error, self.first = None, True
        try:
            engine.wake_vllm(self.proc, PORT)
        except Exception as e:  # noqa: BLE001
            print("wake failed, starting vLLM again:", repr(e), flush=True)
            if self.proc.poll() is None:
                self.proc.kill()
            try:
                self.proc = engine.start_vllm(MODEL_DIR, PORT)
            except Exception as e2:  # noqa: BLE001
                self.error = repr(e2)
                print("vLLM failed to start:", self.error, flush=True)
        self.wake_s = round(time.time() - t, 1)
        print(f"vLLM {'restored from the snapshot' if self.restored else 'booted'}; "
              f"awake in {self.wake_s}s", flush=True)
        self.state, self.tasks = _dict(STATE), _dict(TASKS)
        self.usage, self.timings = _dict(USAGE), _dict(TIMINGS)
        self.state["gpu"] = {"up": self.error is None, "last_used": time.time()}

    @modal.exit()
    def stop(self):
        try:
            self.state["gpu"] = {"up": False, "last_used": time.time()}
        except Exception:  # noqa: BLE001
            pass
        if getattr(self, "proc", None):
            self.proc.terminate()

    def _check(self):
        if self.error:
            raise RuntimeError("vLLM server failed to start: " + self.error)

    @modal.method()
    def parse_pdfs(self, pdfs: list[bytes], mode: str = "single", dpi: int = 150) -> dict:
        """See deltaparse.engine.parse_pdfs."""
        self._check()
        return engine.parse_pdfs(pdfs, mode=mode, dpi=dpi, model_dir=MODEL_DIR, port=PORT)

    @modal.method()
    def warm(self) -> bool:
        """Start the GPU without parsing (POST /warm). True if vLLM is ready."""
        self.first = False  # the next task finds a warm GPU: not a cold start for the hint
        self.state["gpu"] = {**(self.state.get("gpu") or {}), "last_used": time.time()}
        return self.error is None

    @modal.method()
    def run_task(self, task_id: str, pdf: bytes, file_name: str, config: dict) -> None:
        """Parse one API task and store the Chunkr output in the task record (no CPU hop)."""
        from deltaparse.server import run_task

        first, self.first = self.first, False

        def parse(data: bytes, progress) -> list[dict]:
            self._check()
            progress.timing(cold=first, restored=self.restored if first else None,
                            wake_s=self.wake_s if first else None,
                            boot_s=self.boot_s if first and not self.restored else None)
            res = engine.parse_pdfs([data], mode="single", dpi=150, model_dir=MODEL_DIR, port=PORT,
                                    progress=progress.pages, view=False)
            m = res["metrics"]
            progress.timing(render_s=m.get("render_seconds"), parse_s=m.get("parse_seconds"))
            return res["pdfs"][0]

        timings = self.timings
        try:
            run_task(self.tasks, task_id, pdf, file_name, config, parse, usage=self.usage,
                     timings=timings)
        finally:
            self.state.pop(f"call:{task_id}", None)
            self.state["gpu"] = {"up": self.error is None, "last_used": time.time()}
        if first:  # a cold start: what the status hint says next time
            waited = (timings.get(task_id) or {}).get("queued_s")
            if waited is not None:
                prior = self.state.get("cold_starts") or []
                if self.restored:  # snapshot restores from now on: drop the full boots (over 60 s)
                    prior = [w for w in prior if w < 60]
                self.state["cold_starts"] = (prior + [waited])[-9:]


class CachedState:
    """Reads the GPU state at most every `ttl` seconds (the API creates tasks faster than it changes)."""

    def __init__(self, state, ttl: float = 5.0):
        self.state, self.ttl, self.at, self.values = state, ttl, 0.0, {}

    def get(self, key, default=None):
        if time.monotonic() - self.at > self.ttl:
            self.values = {k: self.state.get(k) for k in ("gpu", "cold_starts")}
            self.at = time.monotonic()
        v = self.values.get(key)
        return default if v is None else v


def queued_message(state) -> str:
    """Status hint for a new task: no hint when a GPU is up, else the expected cold start."""
    import statistics

    gpu = state.get("gpu") or {}
    if gpu.get("up") and time.time() - gpu.get("last_used", 0) < SCALEDOWN - 15:
        return "Task queued"
    cold = state.get("cold_starts") or []
    n = statistics.median(cold) if cold else DEFAULT_COLD_START_S
    return f"Starting GPU (about {max(5, round(n))} s)"


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
