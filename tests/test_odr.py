"""OpenDocRouter backend, tested against a fake API (no network, no key, no cost)."""
import json

import httpx
import pytest

from deltaparse import odr


def test_page_spec_compresses_runs():
    assert odr.page_spec([1, 2, 3, 7]) == "1-3,7"
    assert odr.page_spec([2]) == "2"
    assert odr.page_spec([1, 3, 4, 5, 9, 10]) == "1,3-5,9-10"
    assert odr.page_spec([]) == ""


def test_to_blocks_merges_boxes_and_maps_classes():
    page = {
        "markdown": "# Title\nline one\nline two",
        "layout": {"elements": [
            {"type": "title", "lines": [0, 0], "confidence": 0.9,
             "boxes": [{"x": 0.1, "y": 0.1, "w": 0.3, "h": 0.05}]},
            {"type": "text", "lines": [1, 2],
             "boxes": [{"x": 0.1, "y": 0.2, "w": 0.2, "h": 0.02}, {"x": 0.15, "y": 0.22, "w": 0.4, "h": 0.03}]},
            {"type": "picture", "boxes": []},  # no boxes: dropped
        ]},
    }
    title, text = odr.to_blocks(page)
    assert title == {"class": "Title", "bbox": [0.1, 0.1, pytest.approx(0.4), pytest.approx(0.15)],
                     "content": "# Title", "confidence": 0.9}
    assert text["class"] == "Text"
    assert text["bbox"] == [0.1, 0.2, pytest.approx(0.55), pytest.approx(0.25)]
    assert text["content"] == "line one\nline two"


def test_api_key_from_env_or_dotenv(tmp_path, monkeypatch):
    monkeypatch.setenv("OPEN_DOC_ROUTER_API_KEY", "from-env")
    assert odr.api_key(tmp_path / ".env") == "from-env"

    monkeypatch.delenv("OPEN_DOC_ROUTER_API_KEY")
    env = tmp_path / ".env"
    env.write_text("OTHER=1\nOPEN_DOC_ROUTER_API_KEY= from-file \n")
    assert odr.api_key(env) == "from-file"

    env.unlink()
    with pytest.raises(SystemExit, match="not set"):
        odr.api_key(env)


def test_request_retries_on_429_then_succeeds(monkeypatch):
    calls = []

    def handler(req):
        calls.append(req)
        if len(calls) == 1:
            return httpx.Response(429, headers={"Retry-After": "3"}, json={"error": {"code": "rate_limited"}})
        return httpx.Response(200, json={"ok": True})

    waits = []
    monkeypatch.setattr(odr.time, "sleep", waits.append)
    client = httpx.Client(transport=httpx.MockTransport(handler))
    assert odr.request(client, "GET", "https://api.test/credits") == {"ok": True}
    assert len(calls) == 2 and waits == [3.0]


def test_request_raises_on_client_error():
    client = httpx.Client(transport=httpx.MockTransport(lambda req: httpx.Response(400, text="bad body")))
    with pytest.raises(RuntimeError, match="400: bad body"):
        odr.request(client, "POST", "https://api.test/parse")


def test_parse_one_sends_only_non_blank_pages_and_records_failures(blank_then_text_pdf):
    sent = {}

    def handler(req):
        sent.update(json.loads(req.content))
        return httpx.Response(200, json={"pages": [
            {"page": 2, "status": "ok", "charge_usd": 0.001, "markdown": "Hello from page two",
             "layout": {"elements": [{"type": "text", "lines": [0, 0],
                                      "boxes": [{"x": 0.1, "y": 0.1, "w": 0.5, "h": 0.05}]}]}},
        ]})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    pages, cost, stats = odr.parse_one(client, blank_then_text_pdf, "nvidia/nemotron-3.5-parse")

    assert sent["pages"] == "2"  # the blank first page is never sent (or billed)
    assert sent["layout"] is True and sent["document"]["mime_type"] == "application/pdf"
    assert cost == pytest.approx(0.001)
    assert stats == {"pages": 2, "blank": 1, "failed": 0, "errors": {}}
    assert pages[0]["blank"] and pages[0]["markdown"] == ""
    assert pages[1]["markdown"] == "Hello from page two"
    assert pages[1]["blocks"][0]["class"] == "Text"
    assert all("png" not in p for p in pages)


def test_parse_one_marks_failed_pages(sample_pdf):
    def handler(req):
        return httpx.Response(200, json={"pages": [{"page": 1, "status": "failed", "error": {"code": "timeout"}}]})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    pages, cost, stats = odr.parse_one(client, sample_pdf, "m")
    assert stats["failed"] == 1 and stats["errors"] == {"timeout": 1}
    assert pages[0]["complete"] is False and pages[0]["error"] == "timeout"
    assert cost == 0
