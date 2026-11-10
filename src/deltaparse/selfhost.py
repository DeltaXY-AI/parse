"""The Chunkr-compatible API on your own GPU, without Modal (what the Docker image runs).

    DELTAPARSE_API_KEY=<a long random string> deltaparse serve      # port 8000

One process: `vllm serve` on 127.0.0.1, the API on 0.0.0.0, tasks parsed in a thread pool.
The API answers at once; tasks wait in "Starting" until vLLM is ready (GET /ready says when).
Tasks are JSON files in DELTAPARSE_DATA_DIR (default /data), so they survive a restart.

Environment:
    DELTAPARSE_API_KEY             required; clients send it as the Authorization header
    DELTAPARSE_DATA_DIR            task store (default /data; mount a volume to keep it)
    DELTAPARSE_DEFAULT_EXPIRES_IN  seconds to keep a task when the request does not say
    DELTAPARSE_MODEL_DIR           the model folder (default /model, baked into the image)
"""
import json
import os
import re
import threading
from collections.abc import Iterator, MutableMapping
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

SAFE_ID = re.compile(r"^[A-Za-z0-9_-]{1,128}$")
UNFINISHED = ("Starting", "Processing")


class DirStore(MutableMapping):
    """Task store: one JSON file per task, written atomically."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()

    def _file(self, key: str) -> Path | None:
        return self.path / f"{key}.json" if isinstance(key, str) and SAFE_ID.match(key) else None

    def __getitem__(self, key: str) -> dict:
        f = self._file(key)
        if f is None:
            raise KeyError(key)
        try:
            with self.lock:  # Windows cannot replace a file while another thread reads it
                text = f.read_text(encoding="utf-8")
            return json.loads(text)
        except (FileNotFoundError, json.JSONDecodeError):
            raise KeyError(key) from None

    def __setitem__(self, key: str, value: dict) -> None:
        f = self._file(key)
        if f is None:
            raise KeyError(f"invalid task id: {key!r}")
        with self.lock:
            tmp = f.with_name(f"{f.name}.{threading.get_ident()}.tmp")
            tmp.write_text(json.dumps(value), encoding="utf-8")
            os.replace(tmp, f)

    def __delitem__(self, key: str) -> None:
        f = self._file(key)
        if f is None or not f.exists():
            raise KeyError(key)
        with self.lock:
            f.unlink(missing_ok=True)

    def __iter__(self) -> Iterator[str]:
        return (f.stem for f in self.path.glob("*.json"))

    def __len__(self) -> int:
        return sum(1 for _ in self)


def fail_interrupted(store: MutableMapping) -> int:
    """Tasks that were running when the server stopped will never finish: mark them Failed."""
    from deltaparse.server import now

    n = 0
    for tid in list(store):
        rec = store.get(tid)
        if rec and rec.get("status") in UNFINISHED:
            rec.update(status="Failed", message="the server restarted before the task finished",
                       finished_at=now())
            store[tid] = rec
            n += 1
    return n


class ThreadRunner:
    """Runs tasks in a thread pool; each waits for `ready` (vLLM up) before parsing."""

    def __init__(self, store: MutableMapping, parse, workers: int = 4, ready: threading.Event | None = None):
        self.store, self.parse = store, parse
        self.ready = ready or threading.Event()
        if ready is None:
            self.ready.set()
        self.error: str | None = None  # set if the GPU backend could not start
        self.pool = ThreadPoolExecutor(workers, thread_name_prefix="task")
        self.futures: dict[str, Future] = {}

    def _parse(self, pdf: bytes) -> list[dict]:
        self.ready.wait()
        if self.error:
            raise RuntimeError("the GPU backend failed to start: " + self.error)
        return self.parse(pdf)

    def _run(self, task_id: str, pdf: bytes, file_name: str, config: dict) -> None:
        from deltaparse.server import run_task

        try:
            run_task(self.store, task_id, pdf, file_name, config, self._parse)
        finally:
            self.futures.pop(task_id, None)

    def start(self, task_id: str, pdf: bytes, file_name: str, config: dict) -> str:
        self.futures[task_id] = self.pool.submit(self._run, task_id, pdf, file_name, config)
        return task_id

    def cancel(self, call_id: str) -> None:
        # A queued task never starts; a running one finishes on the GPU, but run_task keeps
        # the Cancelled status and drops the result.
        fut = self.futures.get(call_id)
        if fut:
            fut.cancel()


def build(data_dir: str, api_key: str, parse, ready: threading.Event | None = None,
          default_expires_in: int | None = None, workers: int = 4):
    """The FastAPI app plus its runner (split out so tests can use a fake parse)."""
    from deltaparse.server import create_app

    store = DirStore(data_dir)
    fail_interrupted(store)
    runner = ThreadRunner(store, parse, workers=workers, ready=ready)
    app = create_app(store, runner, api_key=api_key, default_expires_in=default_expires_in)

    def is_ready():
        from fastapi.responses import JSONResponse

        if runner.error:
            return JSONResponse({"ready": False, "error": runner.error}, status_code=500)
        return JSONResponse({"ready": runner.ready.is_set()}, status_code=200 if runner.ready.is_set() else 503)

    app.get("/ready")(is_ready)
    return app, runner, store


def _sweep_forever(store: MutableMapping, every: int = 3600) -> None:
    import time

    from deltaparse.server import sweep_expired

    while True:
        time.sleep(every)
        try:
            n = sweep_expired(store)
            if n:
                print(f"removed {n} expired task(s)", flush=True)
        except Exception as e:  # noqa: BLE001
            print("sweep failed:", repr(e), flush=True)


def serve(host: str = "0.0.0.0", port: int = 8000) -> int:
    import uvicorn

    from deltaparse import engine

    api_key = os.environ.get("DELTAPARSE_API_KEY", "").strip()
    if not api_key:
        print("set DELTAPARSE_API_KEY (clients send it as the Authorization header)", flush=True)
        return 2
    vllm_port = 8001 if port != 8001 else 8002  # vLLM listens on 127.0.0.1 only, next to the API
    expires = int(os.environ.get("DELTAPARSE_DEFAULT_EXPIRES_IN") or 0) or None
    ready = threading.Event()

    def parse(pdf: bytes) -> list[dict]:
        return engine.parse_pdfs([pdf], mode="single", dpi=150, port=vllm_port)["pdfs"][0]

    app, runner, store = build(os.environ.get("DELTAPARSE_DATA_DIR", "/data"), api_key, parse,
                               ready=ready, default_expires_in=expires)

    def start_gpu():
        try:
            app.state.vllm = engine.start_vllm(engine.MODEL_DIR, vllm_port)
        except Exception as e:  # noqa: BLE001
            runner.error = repr(e)
            print("vLLM failed to start:", runner.error, flush=True)
        ready.set()

    threading.Thread(target=start_gpu, daemon=True).start()
    threading.Thread(target=_sweep_forever, args=(store,), daemon=True).start()
    try:
        uvicorn.run(app, host=host, port=port, proxy_headers=True, forwarded_allow_ips="*")
    finally:
        proc = getattr(app.state, "vllm", None)
        if proc:
            proc.terminate()
    return 0
