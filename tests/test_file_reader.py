"""Tests for FileReader — on-demand text extraction via Docling."""
import tempfile
from pathlib import Path

from file_reader import FileReader


def test_read_text_file():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write("Hello world, this is a test document.")
        f.flush()
        reader = FileReader()
        text = reader.read(f.name)
    assert "Hello world" in text


def test_read_markdown_file():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
        f.write("# Heading\n\nSome markdown content here.")
        f.flush()
        reader = FileReader()
        text = reader.read(f.name)
    assert "Heading" in text
    assert "markdown content" in text


def test_read_nonexistent_file():
    reader = FileReader()
    text = reader.read("/tmp/does_not_exist_12345.txt")
    assert "error" in text.lower() or "not found" in text.lower() or "failed" in text.lower()


def test_truncation():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write("x" * 10000)
        f.flush()
        reader = FileReader(max_chars=100)
        text = reader.read(f.name)
    assert len(text) <= 100


def test_lazy_converter_init():
    """Converter should not be loaded until first read()."""
    reader = FileReader()
    assert reader._converter is None
