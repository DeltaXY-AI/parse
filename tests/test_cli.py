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
    assert sorted(sub.choices) == ["deploy", "gpu", "odr", "sample", "view"]
