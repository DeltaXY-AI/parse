"""Chunkr-compatible HTTP API (FastAPI). Existing Chunkr code only changes its base URL and key.

Serves both Chunkr client generations (see docs/chunkr_api_reference.md):
  v1 client (chunkr-ai 0.3.x):   POST /task/parse,  GET /task/{id},  GET /task/{id}/cancel,  DELETE /task/{id}
  v2 client (chunkr-ai 0.1.0aN): POST /tasks/parse, GET /tasks/{id}[/parse], GET /tasks/{id}/cancel, DELETE /tasks/{id}
each with or without the /api/v1 prefix, plus GET /health.

create_app(store, runner, api_key, keys=None, default_expires_in=None):
  store:   a dict-like task store (a plain dict in tests, a modal.Dict on Modal)
  runner:  has start(task_id, pdf: bytes, file_name, config) -> call id, and cancel(call id)
  api_key: the self-host key; its tasks belong to keys.DEFAULT_OWNER
  keys:    optional per-customer keys (see deltaparse.keys); each customer sees only its own tasks
  default_expires_in: seconds to keep a task when the request does not set expires_in
"""
import base64
import binascii
import hmac
import uuid
from collections.abc import MutableMapping
from datetime import datetime, timedelta, timezone

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from deltaparse import __version__
from deltaparse.chunkr import for_api
from deltaparse.keys import DEFAULT_OWNER, customer_for

MAX_FILE_BYTES = 200 * 1024 * 1024
FINISHED = ("Succeeded", "Failed", "Cancelled")
IMAGE_MAGIC = (b"\x89PNG", b"\xff\xd8\xff", b"GIF8", b"II*\x00", b"MM\x00*", b"BM", b"RIFF")
CONFIG_KEYS = ("chunk_processing", "segment_processing", "segmentation_strategy", "ocr_strategy",
               "error_handling", "expires_in", "high_resolution", "pipeline", "llm_processing")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def decode_file(file: str) -> bytes:
    """Chunkr `file` field: http(s) URL, data URL or raw base64 -> bytes."""
    if file.startswith(("http://", "https://")):
        try:
            r = httpx.get(file, follow_redirects=True, timeout=120)
            r.raise_for_status()
        except httpx.HTTPError as e:
            raise HTTPException(400, f"could not fetch file URL: {e}") from e
        return r.content
    if file.startswith("ch://"):
        raise HTTPException(400, "the Files API (ch://files/...) is not supported; send base64 or a URL")
    if file.startswith("data:"):
        file = file.split(",", 1)[-1]
    try:
        return base64.b64decode(file, validate=False)
    except (binascii.Error, ValueError) as e:
        raise HTTPException(400, "file is not a URL or valid base64") from e


def to_pdf(data: bytes) -> bytes:
    """PDFs pass through; images become a one-page PDF; anything else is rejected."""
    if data[:1024].lstrip().startswith(b"%PDF") or b"%PDF" in data[:1024]:
        return data
    if data.startswith(IMAGE_MAGIC):
        import pymupdf

        with pymupdf.open(stream=data) as img:
            return img.convert_to_pdf()
    raise HTTPException(400, "unsupported file type: send a PDF or an image (PNG, JPEG, TIFF, GIF, BMP, WebP)")


def sweep_expired(store: MutableMapping, at: datetime | None = None) -> int:
    """Remove tasks past their expires_at; returns how many were removed."""
    at = at or datetime.now(timezone.utc)
    gone = [tid for tid, rec in list(store.items())
            if rec.get("expires_at") and datetime.fromisoformat(rec["expires_at"]) < at]
    for tid in gone:
        store.pop(tid, None)
    return len(gone)


def create_app(store: MutableMapping, runner, api_key: str | None, keys=None,
               default_expires_in: int | None = None) -> FastAPI:
    app = FastAPI(title="Parse (Chunkr-compatible API)", version=__version__)

    def check_auth(request: Request) -> str:
        """Return the caller's owner name, or raise 401."""
        if not api_key and keys is None:
            return DEFAULT_OWNER
        got = request.headers.get("authorization", "").strip()
        if got.lower().startswith("bearer "):
            got = got[7:].strip()
        if api_key and hmac.compare_digest(got, api_key):
            return DEFAULT_OWNER
        customer = customer_for(keys, got) if keys is not None else None
        if customer is None:
            raise HTTPException(401, "invalid or missing API key (Authorization header)")
        return customer

    def load(task_id: str, owner: str) -> dict:
        rec = store.get(task_id)
        if rec is None or rec.get("owner", DEFAULT_OWNER) != owner:  # never reveal others' tasks
            raise HTTPException(404, "task not found")
        if rec.get("expires_at") and datetime.fromisoformat(rec["expires_at"]) < datetime.now(timezone.utc):
            store.pop(task_id, None)
            raise HTTPException(404, "task expired")
        return rec

    def task_url(request: Request, prefix: str, api: str, task_id: str) -> str:
        base = str(request.base_url).rstrip("/")
        if request.headers.get("x-forwarded-proto") == "https" and base.startswith("http://"):
            base = "https://" + base[7:]
        return f"{base}{prefix}/{'task' if api == 'v1' else 'tasks'}/{task_id}"

    def render(rec: dict, request: Request, prefix: str, api: str, include_chunks: bool = True) -> dict:
        url = task_url(request, prefix, api, rec["task_id"])
        out = {
            "task_id": rec["task_id"], "status": rec["status"], "message": rec.get("message") or "",
            "created_at": rec["created_at"], "started_at": rec.get("started_at"),
            "finished_at": rec.get("finished_at"), "expires_at": rec.get("expires_at"),
            "configuration": {**rec["configuration"], "input_file_url": rec.get("file_url")},
            "output": for_api(rec.get("output"), api, rec["configuration"], include_chunks),
            "task_url": url,
        }
        if api == "v2":
            out.update(
                completed=rec["status"] in FINISHED, task_type="Parse",
                input_file_url=rec.get("file_url"),
                file_info={"url": rec.get("file_url") or url, "name": rec.get("file_name"),
                           "mime_type": "application/pdf", "page_count": rec.get("page_count")},
                version_info={"server_version": f"deltaparse-{__version__}",
                              "client_version": "Unspecified"},
            )
        return out

    async def create(request: Request, prefix: str, api: str):
        owner = check_auth(request)
        body = await request.json()
        if not isinstance(body, dict) or not isinstance(body.get("file"), str) or not body["file"]:
            raise HTTPException(422, "`file` (base64, data URL or http(s) URL) is required")
        data = decode_file(body["file"])
        if len(data) > MAX_FILE_BYTES:
            raise HTTPException(413, f"file larger than {MAX_FILE_BYTES // 2**20} MB")
        pdf = to_pdf(data)
        config = {k: body[k] for k in CONFIG_KEYS if body.get(k) is not None}
        if default_expires_in and not config.get("expires_in"):
            config["expires_in"] = int(default_expires_in)
        task_id = str(uuid.uuid4())
        created = datetime.now(timezone.utc)
        rec = {
            "task_id": task_id, "owner": owner, "status": "Starting", "message": "Task queued",
            "created_at": created.isoformat(), "configuration": config,
            "file_name": body.get("file_name") or f"{task_id}.pdf",
            "file_url": body["file"] if body["file"].startswith(("http://", "https://")) else None,
        }
        if config.get("expires_in"):
            rec["expires_at"] = (created + timedelta(seconds=int(config["expires_in"]))).isoformat()
        store[task_id] = rec
        try:
            call_id = runner.start(task_id, pdf, rec["file_name"], config)
        except Exception as e:  # noqa: BLE001
            rec.update(status="Failed", message=f"could not start the task: {e}", finished_at=now())
            store[task_id] = rec
        else:
            rec = store.get(task_id) or rec  # the runner may already have updated it
            rec["call_id"] = call_id
            store[task_id] = rec
        return render(rec, request, prefix, api)

    def get(request: Request, task_id: str, prefix: str, api: str):
        owner = check_auth(request)
        include = request.query_params.get("include_chunks", "true").lower() != "false"
        return render(load(task_id, owner), request, prefix, api, include)

    def cancel(request: Request, task_id: str, prefix: str, api: str):
        rec = load(task_id, check_auth(request))
        if rec["status"] in FINISHED:
            raise HTTPException(400, f"task cannot be cancelled: status is {rec['status']}")
        if rec.get("call_id"):
            runner.cancel(rec["call_id"])
        rec.update(status="Cancelled", message="Task cancelled", finished_at=now())
        store[task_id] = rec
        return render(rec, request, prefix, api)

    def delete(request: Request, task_id: str):
        rec = load(task_id, check_auth(request))
        if rec["status"] not in FINISHED:
            raise HTTPException(400, f"task must be finished to delete it: status is {rec['status']}")
        store.pop(task_id, None)
        return JSONResponse(None)

    def update(request: Request, task_id: str):
        check_auth(request)
        raise HTTPException(501, "updating a task is not supported: create a new task instead")

    for prefix in ("", "/api/v1"):
        for api, word in (("v1", "task"), ("v2", "tasks")):
            def bind(prefix=prefix, api=api, word=word):
                async def _create(request: Request):
                    return await create(request, prefix, api)

                def _get(request: Request, task_id: str):
                    return get(request, task_id, prefix, api)

                def _cancel(request: Request, task_id: str):
                    return cancel(request, task_id, prefix, api)

                app.post(f"{prefix}/{word}/parse")(_create)
                app.get(f"{prefix}/{word}/{{task_id}}")(_get)
                if api == "v2":
                    app.get(f"{prefix}/{word}/{{task_id}}/parse")(_get)
                app.get(f"{prefix}/{word}/{{task_id}}/cancel")(_cancel)
                app.delete(f"{prefix}/{word}/{{task_id}}")(delete)
                app.patch(f"{prefix}/{word}/{{task_id}}/parse")(update)
            bind()
        app.get(f"{prefix}/health")(lambda: "OK")

    return app
