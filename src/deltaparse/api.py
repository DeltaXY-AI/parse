"""Deploy the Chunkr-compatible API to your own Modal account (H100 + vLLM; the GPU scales to zero).

One-time setup, then deploy:
    pip install -e ".[gpu]"
    modal setup
    modal secret create deltaparse-api-key DELTAPARSE_API_KEY=<a long random string>
    modal deploy -m deltaparse.api        # prints your API URL

Then point Chunkr code at it: CHUNKR_URL=<url> (v1 client) or CHUNKR_BASE_URL=<url> (v2 client),
and CHUNKR_API_KEY=<the same random string>.

Hosting for several customers (each sees only its own tasks):
    deltaparse keys add <customer>      # prints a new key, shown once
    deltaparse keys list | revoke <customer>
    deltaparse usage                    # pages parsed per customer
    deltaparse timings [task_id]        # seconds per step of recent tasks
Optional default retention: add DELTAPARSE_DEFAULT_EXPIRES_IN=<seconds> to the secret;
an hourly sweep removes expired tasks.

A task goes straight from the API container to the H100 (VllmParser.run_task), which writes the
result into the task store; POST /warm starts the GPU ahead of the first file.
"""
import os

import modal

from deltaparse import gpu

TASKS, KEYS, USAGE, TIMINGS, STATE = gpu.TASKS, gpu.KEYS, gpu.USAGE, gpu.TIMINGS, gpu.STATE

api_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("fastapi[standard]", "httpx", "numpy", "pymupdf==1.27.2.3")  # pymupdf as on the GPU
    .add_local_python_source("deltaparse")
)

app = modal.App("deltaparse")
app.include(gpu.app)  # the H100 VllmParser class


@app.function(image=api_image, schedule=modal.Period(hours=1))
def sweep() -> int:
    """Remove expired tasks even if nobody reads them again."""
    from deltaparse.server import prune_timings, sweep_expired

    prune_timings(modal.Dict.from_name(TIMINGS, create_if_missing=True))
    return sweep_expired(modal.Dict.from_name(TASKS, create_if_missing=True))


class ModalRunner:
    """Spawns each task on the H100; call ids live in the state Dict (the GPU removes them).

    The spawn (it uploads the file to Modal) runs on a background thread, so the create
    request answers at once; if it fails, the task is marked Failed.
    """

    def __init__(self, store, state):
        from concurrent.futures import ThreadPoolExecutor

        self.store, self.state = store, state
        self.pool = ThreadPoolExecutor(16, thread_name_prefix="spawn")

    def start(self, task_id: str, pdf: bytes, file_name: str, config: dict) -> None:
        self.pool.submit(self._spawn, task_id, pdf, file_name, config)

    def _spawn(self, task_id: str, pdf: bytes, file_name: str, config: dict) -> None:
        from deltaparse.server import now

        try:
            call = gpu.VllmParser().run_task.spawn(task_id, pdf, file_name, config)
            self.state[f"call:{task_id}"] = call.object_id
        except Exception as e:  # noqa: BLE001
            rec = self.store.get(task_id)
            if rec and rec["status"] == "Starting":
                rec.update(status="Failed", message=f"could not start the task: {e}"[:500], finished_at=now())
                self.store[task_id] = rec

    def cancel(self, task_id: str) -> None:
        call_id = self.state.pop(f"call:{task_id}", None)
        if call_id:
            modal.FunctionCall.from_id(call_id).cancel()


# One small CPU container stays up (the H100 still scales to zero): a cold start takes about
# 7.5 s, longer than the 5 s timeout of Chunkr's 0.3.x client on its upload call.
# Pinned next to the us-east ingress and the task Dicts: unpinned, it ran in ap-south and each
# Dict call took about 220 ms instead of 25 ms (several per request).
@app.function(image=api_image, secrets=[modal.Secret.from_name("deltaparse-api-key")],
              min_containers=1, scaledown_window=300, region="us-east")
@modal.concurrent(max_inputs=100)
@modal.asgi_app(label="deltaparse")
def web():
    from deltaparse.server import create_app

    store = modal.Dict.from_name(TASKS, create_if_missing=True)
    keys = modal.Dict.from_name(KEYS, create_if_missing=True)
    state = modal.Dict.from_name(STATE, create_if_missing=True)
    hint_state = gpu.CachedState(state)
    expires = int(os.environ.get("DELTAPARSE_DEFAULT_EXPIRES_IN") or 0) or None
    return create_app(store, ModalRunner(store, state), api_key=os.environ["DELTAPARSE_API_KEY"], keys=keys,
                      default_expires_in=expires, queued_message=lambda: gpu.queued_message(hint_state),
                      warmer=lambda: gpu.VllmParser().warm.spawn(), warm_log=state, poll_hold=2.0)
