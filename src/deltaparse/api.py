"""Deploy the Chunkr-compatible API to your own Modal account (H100 + vLLM, scales to zero).

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
Optional default retention: add DELTAPARSE_DEFAULT_EXPIRES_IN=<seconds> to the secret;
an hourly sweep removes expired tasks.
"""
import os

import modal

from deltaparse import gpu

TASKS = "deltaparse-tasks"  # Modal Dict holding task status and output
KEYS = "deltaparse-keys"    # Modal Dict: sha256(key) -> customer (see deltaparse.keys)
USAGE = "deltaparse-usage"  # Modal Dict: task_id -> pages parsed, per customer

api_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("fastapi[standard]", "httpx", "numpy", "pymupdf==1.27.2.3")  # pymupdf as on the GPU
    .add_local_python_source("deltaparse")
)

app = modal.App("deltaparse")
app.include(gpu.app)  # the H100 VllmParser class


@app.function(image=api_image, timeout=3600, retries=0)
def process_task(task_id: str, pdf: bytes, file_name: str, config: dict) -> None:
    """Parse one PDF on the H100 and store the Chunkr output in the task record."""
    from deltaparse.server import run_task

    def parse(data: bytes) -> list[dict]:
        return gpu.VllmParser().parse_pdfs.remote([data], mode="single", dpi=150)["pdfs"][0]

    run_task(modal.Dict.from_name(TASKS, create_if_missing=True), task_id, pdf, file_name, config,
             parse, usage=modal.Dict.from_name(USAGE, create_if_missing=True))


@app.function(image=api_image, schedule=modal.Period(hours=1))
def sweep() -> int:
    """Remove expired tasks even if nobody reads them again."""
    from deltaparse.server import sweep_expired

    return sweep_expired(modal.Dict.from_name(TASKS, create_if_missing=True))


class ModalRunner:
    def start(self, task_id: str, pdf: bytes, file_name: str, config: dict) -> str:
        return process_task.spawn(task_id, pdf, file_name, config).object_id

    def cancel(self, call_id: str) -> None:
        modal.FunctionCall.from_id(call_id).cancel()


@app.function(image=api_image, secrets=[modal.Secret.from_name("deltaparse-api-key")],
              scaledown_window=300)
@modal.concurrent(max_inputs=100)
@modal.asgi_app(label="deltaparse")
def web():
    from deltaparse.server import create_app

    store = modal.Dict.from_name(TASKS, create_if_missing=True)
    keys = modal.Dict.from_name(KEYS, create_if_missing=True)
    expires = int(os.environ.get("DELTAPARSE_DEFAULT_EXPIRES_IN") or 0) or None
    return create_app(store, ModalRunner(), api_key=os.environ["DELTAPARSE_API_KEY"], keys=keys,
                      default_expires_in=expires)
