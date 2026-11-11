"""Self-hosted API (#48, the Docker image): disk task store, thread runner, readiness, restarts."""
import base64
import threading
import time

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from deltaparse.selfhost import DirStore, build, fail_interrupted, serve  # noqa: E402
from test_server import fake_parse  # noqa: E402

KEY = "docker-key"
AUTH = {"Authorization": KEY}


def b64(path):
    return base64.b64encode(path.read_bytes()).decode()


def wait_for(client, tid, status, word="task", timeout=20):
    deadline = time.time() + timeout
    while time.time() < deadline:
        r = client.get(f"/api/v1/{word}/{tid}", headers=AUTH)
        assert r.status_code == 200, r.text
        if r.json()["status"] == status:
            return r.json()
        time.sleep(0.05)
    raise AssertionError(f"task {tid} never reached {status}: {r.json()['status']}")


def test_dir_store_round_trip_and_rejects_unsafe_ids(tmp_path):
    store = DirStore(tmp_path)
    store["abc-1"] = {"status": "Starting"}
    assert store["abc-1"] == {"status": "Starting"}
    assert list(store) == ["abc-1"] and len(store) == 1
    assert store.get("missing") is None
    for bad in ("../etc/passwd", "a/b", "..", ""):
        assert store.get(bad) is None
        with pytest.raises(KeyError):
            store[bad] = {}
    assert store.pop("abc-1") == {"status": "Starting"}
    assert store.pop("abc-1", None) is None
    assert not list(tmp_path.glob("*.json"))


def test_parse_task_end_to_end_with_both_client_generations(tmp_path, sample_pdf):
    app, _, _ = build(str(tmp_path), KEY, fake_parse)
    client = TestClient(app)
    assert client.get("/health").text == '"OK"'
    assert client.get("/ready").json() == {"ready": True}
    for word in ("task", "tasks"):
        r = client.post(f"/api/v1/{word}/parse", headers=AUTH, json={"file": b64(sample_pdf)})
        assert r.status_code == 200, r.text
        done = wait_for(client, r.json()["task_id"], "Succeeded", word)
        assert done["output"]["chunks"], done
        assert done["output"]["page_count"] == 1
    assert client.post("/api/v1/task/parse", json={"file": b64(sample_pdf)}).status_code == 401


def test_tasks_wait_for_vllm_and_can_be_cancelled_while_waiting(tmp_path, sample_pdf):
    ready = threading.Event()
    app, _, _ = build(str(tmp_path), KEY, fake_parse, ready=ready)
    client = TestClient(app)
    try:
        assert client.get("/ready").status_code == 503
        keep = client.post("/api/v1/task/parse", headers=AUTH, json={"file": b64(sample_pdf)}).json()["task_id"]
        drop = client.post("/api/v1/task/parse", headers=AUTH, json={"file": b64(sample_pdf)}).json()["task_id"]
        time.sleep(0.2)
        waiting = client.get(f"/api/v1/task/{keep}", headers=AUTH).json()
        assert waiting["status"] == "Starting" and waiting["message"].startswith("Starting GPU")
        assert client.get(f"/api/v1/task/{drop}/cancel", headers=AUTH).json()["status"] == "Cancelled"
        ready.set()
        assert client.get("/ready").json() == {"ready": True}
        wait_for(client, keep, "Succeeded")
        time.sleep(0.2)
        rec = client.get(f"/api/v1/task/{drop}", headers=AUTH).json()
        assert rec["status"] == "Cancelled" and rec["output"] is None
    finally:
        ready.set()  # never leave worker threads waiting


def test_backend_start_failure_fails_tasks_and_ready(tmp_path, sample_pdf):
    ready = threading.Event()
    app, runner, _ = build(str(tmp_path), KEY, fake_parse, ready=ready)
    client = TestClient(app)
    runner.error = "RuntimeError('vllm serve exited with 1')"
    ready.set()
    assert client.get("/ready").status_code == 500
    tid = client.post("/api/v1/task/parse", headers=AUTH, json={"file": b64(sample_pdf)}).json()["task_id"]
    failed = wait_for(client, tid, "Failed")
    assert "GPU backend failed to start" in failed["message"]


def test_tasks_survive_a_restart_and_interrupted_ones_fail(tmp_path, sample_pdf):
    app, _, store = build(str(tmp_path), KEY, fake_parse)
    client = TestClient(app)
    tid = client.post("/api/v1/task/parse", headers=AUTH, json={"file": b64(sample_pdf)}).json()["task_id"]
    wait_for(client, tid, "Succeeded")
    store["stuck"] = {**store[tid], "task_id": "stuck", "status": "Processing", "output": None}

    app2, _, _ = build(str(tmp_path), KEY, fake_parse)  # the container restarted
    client2 = TestClient(app2)
    assert client2.get(f"/api/v1/task/{tid}", headers=AUTH).json()["status"] == "Succeeded"
    stuck = client2.get("/api/v1/task/stuck", headers=AUTH).json()
    assert stuck["status"] == "Failed" and "restarted" in stuck["message"]
    assert fail_interrupted(DirStore(tmp_path)) == 0


def test_default_expiry_applies(tmp_path, sample_pdf):
    app, _, store = build(str(tmp_path), KEY, fake_parse, default_expires_in=60)
    client = TestClient(app)
    tid = client.post("/api/v1/task/parse", headers=AUTH, json={"file": b64(sample_pdf)}).json()["task_id"]
    assert store[tid]["expires_at"]


def test_serve_refuses_to_start_without_an_api_key(monkeypatch):
    monkeypatch.delenv("DELTAPARSE_API_KEY", raising=False)
    assert serve() == 2
