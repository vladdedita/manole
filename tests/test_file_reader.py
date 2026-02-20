"""Tests for FileReader — on-demand text extraction via Docling."""
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from file_reader import DoclingExtractor, FileReader, TextExtractor


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


# --- TextExtractor protocol and DoclingExtractor tests ---


def test_docling_extractor_satisfies_text_extractor_protocol():
    """DoclingExtractor must be a structural subtype of TextExtractor."""
    extractor = DoclingExtractor()
    assert isinstance(extractor, TextExtractor)


def test_docling_extractor_lazy_loads_converter():
    """Converter is None until first extract() call."""
    extractor = DoclingExtractor()
    assert extractor._converter is None


@patch("file_reader.DocumentConverter")
def test_docling_extractor_extract_returns_markdown(mock_converter_cls):
    """extract() converts a file and returns markdown text."""
    mock_converter = MagicMock()
    mock_result = MagicMock()
    mock_result.document.export_to_markdown.return_value = "# Extracted Content"
    mock_converter.convert.return_value = mock_result
    mock_converter_cls.return_value = mock_converter

    extractor = DoclingExtractor()
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        f.write(b"fake pdf content")
        f.flush()
        result = extractor.extract(Path(f.name))

    assert result == "# Extracted Content"
    mock_converter.convert.assert_called_once()


@patch("file_reader.DocumentConverter")
def test_docling_extractor_raises_on_conversion_failure(mock_converter_cls):
    """extract() raises an exception when Docling conversion fails."""
    mock_converter = MagicMock()
    mock_converter.convert.side_effect = RuntimeError("conversion failed")
    mock_converter_cls.return_value = mock_converter

    extractor = DoclingExtractor()
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        f.write(b"fake pdf content")
        f.flush()
        with pytest.raises(RuntimeError, match="conversion failed"):
            extractor.extract(Path(f.name))
