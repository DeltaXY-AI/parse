"""Hosted beta (#53): per-customer keys, task ownership, usage and retention."""
import base64
from datetime import datetime, timedelta, timezone

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from deltaparse import keys as K  # noqa: E402
from deltaparse.server import create_app, sweep_expired  # noqa: E402
from test_server import FakeRunner  # noqa: E402

SELF_HOST_KEY = "self-host-key"


def b64(path):
    return base64.b64encode(path.read_bytes()).decode()


@pytest.fixture
def hosted():
    store, keys = {}, {}
    runner = FakeRunner(store, hold=True)  # tasks stay running, so cancel can be tried
    client = TestClient(create_app(store, runner, api_key=SELF_HOST_KEY, keys=keys))
    a, b = K.add_key(keys, "acme"), K.add_key(keys, "globex")
    return client, store, keys, {"Authorization": a}, {"Authorization": b}


def create(client, headers, pdf, **extra):
    r = client.post("/api/v1/task/parse", headers=headers, json={"file": b64(pdf), **extra})
    assert r.status_code == 200, r.text
    return r.json()


def test_customer_cannot_see_cancel_or_delete_another_customers_task(hosted, sample_pdf):
    client, store, _, a, b = hosted
    tid = create(client, a, sample_pdf)["task_id"]
    assert store[tid]["owner"] == "acme"
    for word in ("task", "tasks"):
        assert client.get(f"/{word}/{tid}", headers=b).status_code == 404
        assert client.get(f"/{word}/{tid}/cancel", headers=b).status_code == 404
        assert client.delete(f"/{word}/{tid}", headers=b).status_code == 404
    assert client.get(f"/task/{tid}", headers=a).status_code == 200
    assert client.get(f"/task/{tid}/cancel", headers=a).json()["status"] == "Cancelled"
    assert client.delete(f"/task/{tid}", headers=a).status_code == 200


def test_keys_work_with_bearer_and_can_be_added_and_revoked_without_restart(hosted, sample_pdf):
    client, _, keys, a, _ = hosted
    assert client.post("/task/parse", headers={"Authorization": f"Bearer {a['Authorization']}"},
                       json={"file": b64(sample_pdf)}).status_code == 200
    late = K.add_key(keys, "initech")  # added after the app was built
    assert client.post("/task/parse", headers={"Authorization": late},
                       json={"file": b64(sample_pdf)}).status_code == 200
    assert K.revoke(keys, "acme") == 1
    assert client.post("/task/parse", headers=a, json={"file": b64(sample_pdf)}).status_code == 401
    assert client.get("/task/x", headers={"Authorization": "wrong"}).status_code == 401
    assert client.get("/task/x").status_code == 401


def test_self_host_key_still_works_and_owns_its_own_tasks(hosted, sample_pdf):
    client, store, _, a, _ = hosted
    me = {"Authorization": SELF_HOST_KEY}
    tid = create(client, me, sample_pdf)["task_id"]
    assert store[tid]["owner"] == K.DEFAULT_OWNER
    assert client.get(f"/task/{tid}", headers=me).status_code == 200
    assert client.get(f"/task/{tid}", headers=a).status_code == 404


def test_tasks_from_before_ownership_belong_to_the_self_host_key(hosted):
    client, store, _, a, _ = hosted
    store["old"] = {"task_id": "old", "status": "Succeeded", "created_at": "2026-11-01T00:00:00+00:00",
                    "configuration": {}}
    assert client.get("/task/old", headers={"Authorization": SELF_HOST_KEY}).status_code == 200
    assert client.get("/task/old", headers=a).status_code == 404


def test_keys_are_stored_hashed_and_listed_by_customer():
    keys = {}
    raw = K.add_key(keys, "acme")
    K.add_key(keys, "acme")  # a second key for the same customer
    assert len(raw) >= 32
    flat = " ".join(f"{k} {v}" for k, v in keys.items())
    assert raw not in flat
    assert K.customer_for(keys, raw) == "acme" and K.customer_for(keys, "nope") is None
    assert K.list_customers(keys) == {"acme": 2}
    assert K.revoke(keys, "acme") == 2 and keys == {}
    with pytest.raises(ValueError):
        K.add_key(keys, "bad name!")
    with pytest.raises(ValueError):
        K.add_key(keys, K.DEFAULT_OWNER)


def test_usage_is_counted_once_per_finished_task():
    usage = {}
    done = {"task_id": "t1", "owner": "acme", "status": "Succeeded", "page_count": 160,
            "finished_at": "2026-11-10T05:00:00+00:00"}
    K.record_usage(usage, done)
    K.record_usage(usage, done)  # a retry must not double count
    K.record_usage(usage, {**done, "task_id": "t2", "page_count": 252})
    K.record_usage(usage, {**done, "task_id": "t3", "owner": "globex", "page_count": 1})
    K.record_usage(usage, {**done, "task_id": "t4", "status": "Failed", "page_count": None})
    assert K.usage_report(usage) == {"acme": {"tasks": 2, "pages": 412}, "globex": {"tasks": 1, "pages": 1}}


def test_default_retention_applies_unless_the_request_sets_expires_in(sample_pdf):
    store = {}
    client = TestClient(create_app(store, FakeRunner(store), api_key=SELF_HOST_KEY, default_expires_in=3600))
    me = {"Authorization": SELF_HOST_KEY}
    t1 = create(client, me, sample_pdf)
    t2 = create(client, me, sample_pdf, expires_in=60)
    created = datetime.fromisoformat(t1["created_at"])
    assert datetime.fromisoformat(t1["expires_at"]) - created == timedelta(seconds=3600)
    assert datetime.fromisoformat(t2["expires_at"]) - datetime.fromisoformat(t2["created_at"]) == timedelta(seconds=60)


def test_sweep_removes_only_expired_tasks():
    now = datetime(2026, 11, 10, tzinfo=timezone.utc)
    store = {
        "old": {"task_id": "old", "expires_at": (now - timedelta(seconds=1)).isoformat()},
        "new": {"task_id": "new", "expires_at": (now + timedelta(hours=1)).isoformat()},
        "keep": {"task_id": "keep"},
    }
    assert sweep_expired(store, now) == 1
    assert set(store) == {"new", "keep"}
