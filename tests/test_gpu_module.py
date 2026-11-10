"""The Modal app must import cleanly locally (no GPU is started by importing it)."""
import pytest


def test_gpu_module_imports_and_defines_app():
    pytest.importorskip("modal")
    from deltaparse import gpu

    assert gpu.app.name == "nemotron-parse-vllm"
    assert callable(gpu.main)
