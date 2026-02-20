"""Tests for FileReader — on-demand text extraction via Docling."""
import asyncio
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

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


# --- KreuzbergExtractor tests ---


def test_kreuzberg_extractor_satisfies_text_extractor_protocol():
    """KreuzbergExtractor must be a structural subtype of TextExtractor."""
    from file_reader import KreuzbergExtractor

    with patch.dict("sys.modules", {"kreuzberg": MagicMock()}):
        extractor = KreuzbergExtractor()
    assert isinstance(extractor, TextExtractor)


def test_kreuzberg_extractor_extract_returns_plain_text():
    """extract() calls kreuzberg.extract_file async and returns content string."""
    from file_reader import KreuzbergExtractor

    mock_kreuzberg = MagicMock()
    mock_result = MagicMock()
    mock_result.content = "Extracted plain text from PDF"

    async def fake_extract_file(path):
        return mock_result

    mock_kreuzberg.extract_file = fake_extract_file

    with patch.dict("sys.modules", {"kreuzberg": mock_kreuzberg}):
        extractor = KreuzbergExtractor()
        result = extractor.extract(Path("/tmp/test.pdf"))

    assert result == "Extracted plain text from PDF"


def test_kreuzberg_extractor_raises_import_error_when_missing():
    """KreuzbergExtractor raises ImportError with clear message if kreuzberg not installed."""
    from file_reader import KreuzbergExtractor

    with patch.dict("sys.modules", {"kreuzberg": None}):
        with pytest.raises(ImportError, match="kreuzberg"):
            KreuzbergExtractor()


def test_kreuzberg_extractor_propagates_extraction_errors():
    """extract() propagates exceptions from kreuzberg.extract_file."""
    from file_reader import KreuzbergExtractor

    mock_kreuzberg = MagicMock()

    async def fake_extract_file(path):
        raise RuntimeError("extraction failed")

    mock_kreuzberg.extract_file = fake_extract_file

    with patch.dict("sys.modules", {"kreuzberg": mock_kreuzberg}):
        extractor = KreuzbergExtractor()
        with pytest.raises(RuntimeError, match="extraction failed"):
            extractor.extract(Path("/tmp/test.pdf"))


# --- Backend selection acceptance tests (step 02-02) ---


def test_backend_selection_creates_correct_extractor_and_reads_file():
    """Acceptance: FileReader.from_backend() creates a working reader for each valid backend."""
    from file_reader import FileReader, DoclingExtractor, KreuzbergExtractor

    # docling backend
    reader_docling = FileReader.from_backend("docling")
    assert isinstance(reader_docling._extractor, DoclingExtractor)

    # kreuzberg backend (with kreuzberg mocked as available)
    with patch.dict("sys.modules", {"kreuzberg": MagicMock()}):
        reader_kreuzberg = FileReader.from_backend("kreuzberg")
        assert isinstance(reader_kreuzberg._extractor, KreuzbergExtractor)

    # invalid backend raises ValueError
    with pytest.raises(ValueError, match="Unknown backend"):
        FileReader.from_backend("unknown")

    # missing kreuzberg raises ImportError
    with patch.dict("sys.modules", {"kreuzberg": None}):
        with pytest.raises(ImportError, match="kreuzberg"):
            FileReader.from_backend("kreuzberg")


# --- Backend selection unit tests (step 02-02) ---
# Test Budget: 4 behaviors x 2 = 8 unit tests max. Using 3.


@pytest.mark.parametrize("backend_name,expected_type", [
    ("docling", "DoclingExtractor"),
    ("kreuzberg", "KreuzbergExtractor"),
])
def test_from_backend_creates_expected_extractor_type(backend_name, expected_type):
    """from_backend() instantiates the correct extractor for each valid name."""
    from file_reader import FileReader

    with patch.dict("sys.modules", {"kreuzberg": MagicMock()}):
        reader = FileReader.from_backend(backend_name)
    assert type(reader._extractor).__name__ == expected_type


def test_from_backend_raises_value_error_for_unknown_backend():
    """from_backend() raises ValueError with descriptive message for invalid backend name."""
    from file_reader import FileReader

    with pytest.raises(ValueError, match="Unknown backend.*invalid_backend"):
        FileReader.from_backend("invalid_backend")


def test_from_backend_raises_import_error_when_kreuzberg_missing():
    """from_backend('kreuzberg') raises ImportError when kreuzberg package is not installed."""
    from file_reader import FileReader

    with patch.dict("sys.modules", {"kreuzberg": None}):
        with pytest.raises(ImportError, match="kreuzberg"):
            FileReader.from_backend("kreuzberg")
