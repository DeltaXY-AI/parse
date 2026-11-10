"""Chunkr's own Python client (whichever generation is installed) against a real local server.

The server runs with uvicorn on localhost and the client uses its own HTTP stack, unchanged.
Skipped unless chunkr-ai is installed:
    pip install "chunkr-ai==0.3.7"      then pytest tests/test_chunkr_sdk.py   (v1 client)
    pip install "chunkr-ai==0.1.0a15"   then pytest tests/test_chunkr_sdk.py   (v2 client)
"""
import importlib.util
import socket
import sys
import threading
import time
from pathlib import Path

import pytest

pytest.importorskip("chunkr_ai")
uvicorn = pytest.importorskip("uvicorn")

from deltaparse.server import create_app  # noqa: E402
from test_server import KEY, FakeRunner  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "chunkr_client_check", Path(__file__).parents[1] / "examples" / "chunkr_client_check.py")
check = importlib.util.module_from_spec(_spec)
sys.modules["chunkr_client_check"] = check
_spec.loader.exec_module(check)


@pytest.fixture
def live_server():
    store = {}
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(create_app(store, FakeRunner(store), api_key=KEY),
                                           host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    for _ in range(100):
        if server.started:
            break
        time.sleep(0.05)
    yield f"http://127.0.0.1:{port}", store
    server.should_exit = True
    thread.join(5)


def test_chunkr_client_unchanged(live_server, sample_pdf):
    url, store = live_server
    if check.sdk_generation() == "v2":
        res = check.run_v2(f"{url}/api/v1", KEY, sample_pdf)
    else:
        res = check.run_v1(url, KEY, sample_pdf)
        assert "Quarterly Portfolio Summary" in res["content"]
    assert res["status"] == "Succeeded"
    assert "Quarterly Portfolio Summary" in res["markdown"] and "HDFC Bank" in res["markdown"]
    assert set(res["segments"]) == {"Text"}
    assert res["task_id"] not in store  # the client deleted it
