"""Deploy the Chunkr-compatible API to your own Modal account (H100 + vLLM, scales to zero).

One-time setup, then deploy:
    pip install -e ".[gpu]"
    modal setup
    modal secret create deltaparse-api-key DELTAPARSE_API_KEY=<a long random string>
    modal deploy -m deltaparse.api        # prints your API URL

Then point Chunkr code at it: CHUNKR_URL=<url> (v1 client) or CHUNKR_BASE_URL=<url> (v2 client),
and CHUNKR_API_KEY=<the same random string>.
"""
import os
from datetime import datetime, timezone

import modal

from deltaparse import gpu

TASKS = "deltaparse-tasks"  # Modal Dict holding task status and output

api_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("fastapi[standard]", "httpx", "numpy", "pymupdf==1.27.2.3")  # pymupdf as on the GPU
    .add_local_python_source("deltaparse")
)

app = modal.App("deltaparse")
app.include(gpu.app)  # the H100 VllmParser class


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@app.function(image=api_image, timeout=3600, retries=0)
def process_task(task_id: str, pdf: bytes, file_name: str, config: dict) -> None:
    """Parse one PDF on the H100 and store the Chunkr output in the task record."""
    from deltaparse.chunkr import build_output

    tasks = modal.Dict.from_name(TASKS, create_if_missing=True)
    rec = tasks.get(task_id)
    if rec is None or rec["status"] == "Cancelled":
        return
    rec.update(status="Processing", message="Task is being processed", started_at=_now())
    tasks[task_id] = rec
    try:
        res = gpu.VllmParser().parse_pdfs.remote([pdf], mode="single", dpi=150)
        pages = [{k: v for k, v in p.items() if k != "jpg"} for p in res["pdfs"][0]]
        incomplete = sum(1 for p in pages if not p["complete"])
        output = build_output(pages, file_name, config)
        status, message = "Succeeded", "Task succeeded"
        if incomplete:
            message = f"Task succeeded; {incomplete} page(s) incomplete"
            if config.get("error_handling", "Fail") == "Fail" and incomplete == len(pages):
                status, message = "Failed", "no page could be parsed"
    except Exception as e:  # noqa: BLE001
        output, status, message, pages = None, "Failed", f"parse failed: {e}"[:500], []
    rec = tasks.get(task_id) or rec
    if rec["status"] == "Cancelled":
        return
    rec.update(status=status, message=message, output=output, page_count=len(pages) or None,
               finished_at=_now())
    tasks[task_id] = rec


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
    return create_app(store, ModalRunner(), api_key=os.environ["DELTAPARSE_API_KEY"])
