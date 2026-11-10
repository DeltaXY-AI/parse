import pymupdf
import pytest

from deltaparse import cli


def test_cli_sample_writes_a_pdf(tmp_path, capsys):
    pdf = tmp_path / "docs" / "s.pdf"
    pdf.parent.mkdir()
    assert cli.main(["sample", str(pdf)]) == 0
    with pymupdf.open(str(pdf)) as doc:
        assert doc.page_count == 1
        assert "Quarterly Portfolio Summary" in doc[0].get_text()
    assert f"wrote {pdf}" in capsys.readouterr().out


def test_cli_gpu_builds_modal_command(monkeypatch):
    seen = {}
    monkeypatch.setattr(cli.subprocess, "call", lambda cmd: seen.setdefault("cmd", cmd) and 0)
    assert cli.main(["gpu", "x.pdf", "--limit", "3"]) == 0
    assert seen["cmd"][1:] == ["-m", "modal", "run", "-m", "deltaparse.gpu", "--path", "x.pdf",
                               "--out-dir", "out_vllm", "--mode", "single", "--limit", "3"]


def test_cli_requires_a_command(capsys):
    with pytest.raises(SystemExit):
        cli.main([])
    err = capsys.readouterr().err
    assert "<command>" in err


def test_cli_lists_only_current_commands():
    sub = next(a for a in cli.build_parser()._actions if a.dest == "command")
    assert sorted(sub.choices) == ["deploy", "gpu", "keys", "odr", "sample", "usage", "view"]


def test_cli_keys_and_usage_use_the_hosted_stores(monkeypatch, capsys):
    from deltaparse import keys as K

    stores = {"deltaparse-keys": {}, "deltaparse-usage": {}}
    monkeypatch.setattr(cli, "_modal_dict", lambda name: stores[name])
    assert cli.main(["keys", "add", "acme"]) == 0
    key = capsys.readouterr().out.strip().splitlines()[-1]
    assert key.startswith("dp_") and K.customer_for(stores["deltaparse-keys"], key) == "acme"

    assert cli.main(["keys", "list"]) == 0
    assert "acme" in capsys.readouterr().out

    K.record_usage(stores["deltaparse-usage"], {"task_id": "t", "owner": "acme", "status": "Succeeded",
                                                "page_count": 7})
    assert cli.main(["usage"]) == 0
    out = capsys.readouterr().out
    assert "acme" in out and "7" in out

    assert cli.main(["keys", "revoke", "acme"]) == 0
    assert stores["deltaparse-keys"] == {}
    assert cli.main(["keys", "revoke", "acme"]) == 1  # nothing to revoke
    assert cli.main(["keys", "add", "Bad Name"]) == 2
