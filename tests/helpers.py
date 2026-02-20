"""Shared test fixtures and helpers for indexer tests."""
import tempfile
from pathlib import Path
from unittest.mock import MagicMock


def make_data_dir(files=None):
    """Create a temp directory with specified files."""
    tmp = tempfile.mkdtemp()
    for name in (files or []):
        p = Path(tmp) / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"fake-content")
    return Path(tmp)


def make_mock_chunk(content, metadata=None):
    """Create a mock chunk with content and metadata."""
    chunk = MagicMock()
    chunk.content = content
    chunk.metadata = metadata or {}
    return chunk


def make_mock_result(chunks=None, elements=None):
    """Create a mock ExtractionResult with chunks and elements."""
    result = MagicMock()
    result.chunks = chunks or []
    result.elements = elements or []
    return result
