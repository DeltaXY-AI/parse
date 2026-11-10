"""Chunkr-compatible API, tested with an in-process server and a fake GPU runner."""
import base64

import pymupdf
import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from deltaparse.chunkr import build_output  # noqa: E402
from deltaparse.pdf import render_pdf  # noqa: E402
from deltaparse.server import create_app  # noqa: E402

KEY = "test-key"


def fake_parse(pdf: bytes) -> list[dict]:
    """Stand-in for the H100: PyMuPDF text blocks as Nemotron-style blocks."""
    pages = render_pdf(pdf)
    with pymupdf.open(stream=pdf, filetype="pdf") as doc:
        for page, out in zip(doc, pages):
            w, h = page.rect.width, page.rect.height
            out["blocks"] = [{"class": "Text", "bbox": [x0 / w, y0 / h, x1 / w, y1 / h], "content": t.strip()}
                             for x0, y0, x1, y1, t, *_ in page.get_text("blocks") if t.strip()]
            out["markdown"] = "\n\n".join(b["content"] for b in out["blocks"])
    return pages


class FakeRunner:
    """Runs the 'parse' at once (or never, if hold=True) and records cancels."""

    def __init__(self, store, hold=False):
        self.store, self.hold, self.cancelled = store, hold, []

    def start(self, task_id, pdf, file_name, config):
        if not self.hold:
            pages = fake_parse(pdf)
            rec = self.store[task_id]
            rec.update(status="Succeeded", message="Task succeeded", page_count=len(pages),
                       output=build_output(pages, file_name, config), finished_at="2026-11-10T00:00:00+00:00")
            self.store[task_id] = rec
        return f"call-{task_id}"

    def cancel(self, call_id):
        self.cancelled.append(call_id)


@pytest.fixture
def server():
    store = {}
    runner = FakeRunner(store)
    return TestClient(create_app(store, runner, api_key=KEY)), store, runner


def b64(path):
    return base64.b64encode(path.read_bytes()).decode()


AUTH = {"Authorization": KEY}


@pytest.mark.parametrize("prefix", ["", "/api/v1"])
@pytest.mark.parametrize("word", ["task", "tasks"])
def test_create_then_get_on_every_route(server, sample_pdf, prefix, word):
    client, _, _ = server
    r = client.post(f"{prefix}/{word}/parse", headers=AUTH, json={"file": b64(sample_pdf), "file_name": "s.pdf"})
    assert r.status_code == 200, r.text
    task = r.json()
    assert task["status"] == "Succeeded"
    assert task["task_url"] == f"http://testserver{prefix}/{word}/{task['task_id']}"

    got = client.get(task["task_url"], headers=AUTH).json()
    out = got["output"]
    assert out["file_name"] == "s.pdf" and out["page_count"] == 1
    text = " ".join(s["text"] for c in out["chunks"] for s in c["segments"])
    assert "Quarterly Portfolio Summary" in text and "HDFC Bank" in text
    seg = out["chunks"][0]["segments"][0]
    assert (seg["page_width"], seg["page_height"]) == (1190.0, 1684.0)  # A4 at 144 dpi
    if word == "tasks":  # v2 shape
        assert got["completed"] is True and got["task_type"] == "Parse"
        assert got["file_info"]["name"] == "s.pdf" and got["version_info"]["server_version"]
    else:
        assert "completed" not in got


def test_v2_get_parse_route_and_include_chunks_false(server, sample_pdf):
    client, _, _ = server
    task = client.post("/tasks/parse", headers=AUTH, json={"file": b64(sample_pdf)}).json()
    r = client.get(f"/tasks/{task['task_id']}/parse", headers=AUTH, params={"include_chunks": "false"})
    assert r.status_code == 200 and r.json()["output"]["chunks"] == []


def test_auth_bare_key_or_bearer(server, sample_pdf):
    client, _, _ = server
    body = {"file": b64(sample_pdf)}
    assert client.post("/task/parse", json=body).status_code == 401
    assert client.post("/task/parse", headers={"Authorization": "wrong"}, json=body).status_code == 401
    assert client.post("/task/parse", headers={"Authorization": f"Bearer {KEY}"}, json=body).status_code == 200
    assert client.get("/health").json() == "OK"


def test_data_url_and_image_input(server, tmp_path, sample_pdf):
    client, _, _ = server
    r = client.post("/tasks/parse", headers=AUTH,
                    json={"file": "data:application/pdf;base64," + b64(sample_pdf)})
    assert r.json()["status"] == "Succeeded"

    png = tmp_path / "page.png"
    with pymupdf.open(str(sample_pdf)) as doc:
        doc[0].get_pixmap(dpi=72).save(str(png))
    r = client.post("/tasks/parse", headers=AUTH, json={"file": b64(png), "file_name": "page.png"})
    assert r.status_code == 200 and r.json()["status"] == "Succeeded"


@pytest.mark.parametrize("body, code", [
    ({}, 422),
    ({"file": base64.b64encode(b"PK\x03\x04 a docx").decode()}, 400),
    ({"file": "ch://files/abc"}, 400),
])
def test_bad_input(server, body, code):
    client, _, _ = server
    r = client.post("/task/parse", headers=AUTH, json=body)
    assert r.status_code == code, r.text


def test_cancel_only_while_running_and_delete_only_when_finished(sample_pdf):
    store = {}
    runner = FakeRunner(store, hold=True)
    client = TestClient(create_app(store, runner, api_key=KEY))
    task = client.post("/tasks/parse", headers=AUTH, json={"file": b64(sample_pdf)}).json()
    tid = task["task_id"]
    assert task["status"] == "Starting" and task["completed"] is False

    assert client.delete(f"/tasks/{tid}", headers=AUTH).status_code == 400  # still running
    assert client.get(f"/tasks/{tid}/cancel", headers=AUTH).json()["status"] == "Cancelled"
    assert runner.cancelled == [f"call-{tid}"]
    assert client.get(f"/tasks/{tid}/cancel", headers=AUTH).status_code == 400  # already finished
    assert client.delete(f"/tasks/{tid}", headers=AUTH).status_code == 200
    assert client.get(f"/tasks/{tid}", headers=AUTH).status_code == 404


def test_expired_task_is_gone(server, sample_pdf):
    client, store, _ = server
    task = client.post("/task/parse", headers=AUTH, json={"file": b64(sample_pdf), "expires_in": 60}).json()
    assert task["expires_at"]
    store[task["task_id"]]["expires_at"] = "2000-01-01T00:00:00+00:00"
    assert client.get(f"/task/{task['task_id']}", headers=AUTH).status_code == 404
    assert task["task_id"] not in store


def test_runner_failure_marks_task_failed(sample_pdf):
    class Broken:
        def start(self, *a):
            raise RuntimeError("GPU unavailable")

    client = TestClient(create_app({}, Broken(), api_key=KEY))
    task = client.post("/tasks/parse", headers=AUTH, json={"file": b64(sample_pdf)}).json()
    assert task["status"] == "Failed" and "GPU unavailable" in task["message"] and task["completed"]


def test_update_is_not_supported(server):
    client, _, _ = server
    assert client.patch("/task/abc/parse", headers=AUTH, json={}).status_code == 501
