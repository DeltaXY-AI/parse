from pathlib import Path

import pymupdf
import pytest

from deltaparse.sample import make_sample


@pytest.fixture
def sample_pdf(tmp_path: Path) -> Path:
    """The one-page sample document (heading, paragraph, table)."""
    return make_sample(tmp_path / "sample.pdf")


@pytest.fixture
def blank_then_text_pdf(tmp_path: Path) -> Path:
    """Two pages: page 1 is blank, page 2 has text."""
    doc = pymupdf.open()
    doc.new_page()
    doc.new_page().insert_text((72, 72), "Hello from page two", fontsize=14)
    path = tmp_path / "blank_then_text.pdf"
    doc.save(str(path))
    return path
