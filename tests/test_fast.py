"""Fast first result (#74): honest progress messages, per-step timings, the cold-start hint, /warm."""
import base64
import time
from datetime import datetime, timedelta, timezone

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from deltaparse.server import FINISHED, WARM_EVERY, create_app, prune_timings, run_task  # noqa: E402
from test_server import fake_parse  # noqa: E402

KEY = "test-key"
AUTH = {"Authorization": KEY}


class RecordingStore(dict):
    """A task store that remembers every message written, in order."""

    def __init__(self):
        super().__init__()
        self.messages = []

    def __setitem__(self, key, value):
        if not self.messages or self.messages[-1] != value.get("message"):
            self.messages.append(value.get("message"))
        super().__setitem__(key, dict(value))


def multi_page_parse(pdf, progress):
    """Like the GPU: pages finish one by one, then the result is returned."""
    pages = fake_parse(pdf)
    total = 12
    for done in range(total + 1):
        progress.pages(done, total)
    progress.timing(render_s=0.1, parse_s=0.5)
    return pages


def test_a_task_says_what_it_is_doing_in_order_and_keeps_chunkr_statuses(sample_pdf):
    store, timings = RecordingStore(), {}
    store["t"] = {"task_id": "t", "status": "Starting", "message": "Starting GPU (about 15 s)",
                  "created_at": datetime.now(timezone.utc).isoformat(), "configuration": {}}
    run_task(store, "t", sample_pdf.read_bytes(), "s.pdf", {}, multi_page_parse, timings=timings)
    msgs = store.messages
    assert msgs[0] == "Starting GPU (about 15 s)"
    assert msgs[1] == "Rendering pages"
    assert msgs[2] == "Parsing page 1 of 12"  # later pages are throttled to one write a second
    assert msgs[-2] == "Saving result"
    assert msgs[-1] == "Task succeeded"
    assert store["t"]["status"] == "Succeeded" and store["t"]["status"] in FINISHED


def test_progress_writes_are_throttled_but_each_second_shows_a_new_page(sample_pdf):
    store = RecordingStore()
    store["t"] = {"task_id": "t", "status": "Starting", "message": "", "created_at": "2026-11-11T00:00:00+00:00",
                  "configuration": {}}

    def slow(pdf, progress):
        for done in range(4):
            progress.pages(done, 4)
            time.sleep(0.6)
        return fake_parse(pdf)

    run_task(store, "t", sample_pdf.read_bytes(), "s.pdf", {}, slow)
    parsing = [m for m in store.messages if m.startswith("Parsing")]
    assert parsing[0] == "Parsing page 1 of 4" and len(parsing) in (2, 3)


def test_timings_are_recorded_per_task_and_survive_deleting_the_task(sample_pdf):
    store, timings = {}, {}
    client = TestClient(create_app(store, ThreadlessRunner(store, timings), api_key=KEY))
    pdf = base64.b64encode(sample_pdf.read_bytes()).decode()
    tid = client.post("/api/v1/task/parse", headers=AUTH, json={"file": pdf}).json()["task_id"]
    assert client.get(f"/api/v1/task/{tid}", headers=AUTH).json()["status"] == "Succeeded"
    assert client.delete(f"/api/v1/task/{tid}", headers=AUTH).status_code == 200
    t = timings[tid]
    assert t["pages"] == 1 and t["status"] == "Succeeded" and t["owner"] == "default"
    for k in ("queued_s", "gpu_s", "render_s", "parse_s", "save_s", "total_s"):
        assert isinstance(t[k], (int, float)) and t[k] >= 0, k
    assert t["total_s"] >= t["queued_s"]


class ThreadlessRunner:
    def __init__(self, store, timings):
        self.store, self.timings = store, timings

    def start(self, task_id, pdf, file_name, config):
        run_task(self.store, task_id, pdf, file_name, config, multi_page_parse, timings=self.timings)

    def cancel(self, task_id):
        pass


def test_new_tasks_carry_the_queued_hint_and_a_broken_hint_never_fails_a_task(sample_pdf):
    pdf = base64.b64encode(sample_pdf.read_bytes()).decode()

    class Hold:
        def start(self, *a):
            pass

        def cancel(self, task_id):
            pass

    store = {}
    client = TestClient(create_app(store, Hold(), api_key=KEY, queued_message=lambda: "Starting GPU (about 15 s)"))
    r = client.post("/task/parse", headers=AUTH, json={"file": pdf}).json()
    assert (r["status"], r["message"]) == ("Starting", "Starting GPU (about 15 s)")

    def broken():
        raise RuntimeError("state store down")

    client = TestClient(create_app({}, Hold(), api_key=KEY, queued_message=broken))
    r = client.post("/task/parse", headers=AUTH, json={"file": pdf})
    assert r.status_code == 200 and r.json()["message"] == "Task queued"


def test_gpu_hint_uses_measured_cold_starts_and_is_silent_when_warm():
    pytest.importorskip("modal")
    from deltaparse import gpu

    assert gpu.queued_message({}) == f"Starting GPU (about {gpu.DEFAULT_COLD_START_S} s)"
    assert gpu.queued_message({"cold_starts": [14.2, 18.9, 12.0]}) == "Starting GPU (about 14 s)"
    warm = {"gpu": {"up": True, "last_used": time.time() - 30}, "cold_starts": [14.2]}
    assert gpu.queued_message(warm) == "Task queued"
    idle = {"gpu": {"up": True, "last_used": time.time() - gpu.SCALEDOWN}, "cold_starts": [14.2]}
    assert gpu.queued_message(idle) == "Starting GPU (about 14 s)"  # it is scaling down
    assert gpu.queued_message({"gpu": {"up": False}, "cold_starts": [14.2]}) == "Starting GPU (about 14 s)"


def test_warm_starts_the_gpu_at_most_once_per_customer_every_five_minutes():
    from deltaparse import keys as K

    calls, log, keys = [], {}, {}
    client = TestClient(create_app({}, None, api_key=KEY, keys=keys, warmer=lambda: calls.append(1), warm_log=log))
    acme = {"Authorization": K.add_key(keys, "acme")}
    assert client.post("/warm").status_code == 401
    assert client.post("/api/v1/warm", headers=acme).status_code == 202
    r = client.post("/warm", headers=acme)
    assert r.status_code == 429 and 0 < int(r.headers["Retry-After"]) <= WARM_EVERY + 1
    assert client.post("/warm", headers=AUTH).status_code == 202  # another customer
    assert len(calls) == 2
    log["warm:acme"] -= WARM_EVERY + 1  # five minutes later
    assert client.post("/warm", headers=acme).status_code == 202 and len(calls) == 3


def test_warm_is_not_offered_without_a_warmer():
    client = TestClient(create_app({}, None, api_key=KEY))
    assert client.post("/warm", headers=AUTH).status_code in (404, 405)


def test_old_timings_are_pruned():
    now = datetime(2026, 11, 11, tzinfo=timezone.utc)
    timings = {"old": {"created_at": (now - timedelta(days=31)).isoformat()},
               "new": {"created_at": (now - timedelta(days=1)).isoformat()}}
    assert prune_timings(timings, 30, now) == 1 and set(timings) == {"new"}


def test_selfhost_hint_counts_down_until_vllm_is_ready(tmp_path, sample_pdf):
    import threading

    from deltaparse.selfhost import build

    ready = threading.Event()
    app, runner, _ = build(str(tmp_path), KEY, fake_parse, ready=ready)
    try:
        client = TestClient(app)
        pdf = base64.b64encode(sample_pdf.read_bytes()).decode()
        r = client.post("/task/parse", headers=AUTH, json={"file": pdf}).json()
        assert r["message"].startswith("Starting GPU (about ")
        ready.set()
        assert runner.queued_message() == "Task queued"
    finally:
        ready.set()


def _held_app(store, hold):
    class Hold:
        def start(self, *a):
            pass

        def cancel(self, task_id):
            pass

    return TestClient(create_app(store, Hold(), api_key=KEY, poll_hold=hold))


def test_a_poll_waits_for_the_task_to_change_and_answers_as_soon_as_it_does(sample_pdf):
    import threading

    store = {}
    client = _held_app(store, 3.0)
    pdf = base64.b64encode(sample_pdf.read_bytes()).decode()
    tid = client.post("/task/parse", headers=AUTH, json={"file": pdf}).json()["task_id"]

    def finish():
        time.sleep(0.6)
        rec = dict(store[tid])
        rec.update(status="Processing", message="Parsing page 1 of 1")
        store[tid] = rec

    threading.Thread(target=finish).start()
    t = time.monotonic()
    r = client.get(f"/task/{tid}", headers=AUTH).json()
    took = time.monotonic() - t
    assert r["message"] == "Parsing page 1 of 1"
    assert 0.5 < took < 1.5  # answered on the change, not after the full 3 s


def test_a_poll_gives_up_after_the_hold_and_finished_tasks_answer_at_once(sample_pdf):
    store = {}
    client = _held_app(store, 0.8)
    pdf = base64.b64encode(sample_pdf.read_bytes()).decode()
    tid = client.post("/task/parse", headers=AUTH, json={"file": pdf}).json()["task_id"]
    t = time.monotonic()
    assert client.get(f"/tasks/{tid}", headers=AUTH).json()["status"] == "Starting"
    assert 0.7 < time.monotonic() - t < 2.0
    rec = dict(store[tid])
    rec.update(status="Succeeded", message="Task succeeded")
    store[tid] = rec
    t = time.monotonic()
    assert client.get(f"/tasks/{tid}", headers=AUTH).json()["status"] == "Succeeded"
    assert time.monotonic() - t < 0.5


def test_modal_runner_answers_before_the_spawn_and_marks_a_failed_spawn(monkeypatch):
    pytest.importorskip("modal")
    from deltaparse import api

    class Boom:
        class run_task:  # noqa: N801
            @staticmethod
            def spawn(*a):
                time.sleep(0.3)
                raise RuntimeError("no GPU quota")

    monkeypatch.setattr(api.gpu, "VllmParser", lambda: Boom)
    store, state = {"t": {"task_id": "t", "status": "Starting", "message": "Task queued"}}, {}
    runner = api.ModalRunner(store, state)
    t = time.monotonic()
    runner.start("t", b"%PDF", "a.pdf", {})
    assert time.monotonic() - t < 0.1
    runner.pool.shutdown(wait=True)
    assert store["t"]["status"] == "Failed" and "no GPU quota" in store["t"]["message"]
