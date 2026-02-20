"""Tests for KreuzbergIndexer: walk, extract, chunk, index.

Step 01-01: kreuzberg-integration
Test Budget: 4 behaviors x 2 = 8 max unit tests

Behaviors:
1. Walk directory + produce HNSW index via LeannBuilder
2. Chunks carry correct metadata (source, page_number, element_type, chunk_index)
3. Unsupported/corrupt files skipped with logged warning
4. Uses element-based extraction with 512-char chunks, 50-char overlap
"""
import logging
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest


def _make_data_dir(files=None):
    """Create a temp directory with specified files."""
    tmp = tempfile.mkdtemp()
    for name in (files or []):
        p = Path(tmp) / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"fake-content")
    return Path(tmp)


def _make_mock_chunk(content, metadata=None):
    """Create a mock chunk with content and metadata."""
    chunk = MagicMock()
    chunk.content = content
    chunk.metadata = metadata or {}
    return chunk


def _make_mock_result(chunks=None, elements=None):
    """Create a mock ExtractionResult with chunks and elements."""
    result = MagicMock()
    result.chunks = chunks or []
    result.elements = elements or []
    return result


# --- Acceptance test: full pipeline ---

@patch("indexer.LeannBuilder")
@patch("indexer.extract_file_sync")
def test_build_walks_directory_and_produces_index(mock_extract, MockBuilder):
    """Given a directory with supported files,
    when build() is called,
    then indexer extracts each file, feeds chunks to LeannBuilder, and builds an HNSW index."""
    from indexer import KreuzbergIndexer

    data_dir = _make_data_dir(["report.pdf", "slides.pptx", "notes.txt"])

    # Two supported files produce chunks; .txt is unsupported and should be skipped
    mock_extract.return_value = _make_mock_result(
        chunks=[
            _make_mock_chunk("Chunk one content", {"chunk_index": 0}),
            _make_mock_chunk("Chunk two content", {"chunk_index": 1}),
        ],
        elements=[
            MagicMock(metadata={"page_number": 1, "element_type": "paragraph"}),
            MagicMock(metadata={"page_number": 2, "element_type": "heading"}),
        ],
    )

    indexer = KreuzbergIndexer()
    result_path = indexer.build(data_dir, "test-index")

    # extract_file_sync called for each supported file (pdf, pptx), not for .txt
    assert mock_extract.call_count == 2

    # LeannBuilder received chunks from both files (2 chunks x 2 files = 4)
    builder_instance = MockBuilder.return_value
    assert builder_instance.add_text.call_count == 4
    assert builder_instance.build_index.call_count == 1

    # Returns the index path
    assert "test-index" in result_path
    assert result_path.endswith("documents.leann")


# --- Unit tests: metadata correctness ---

@patch("indexer.LeannBuilder")
@patch("indexer.extract_file_sync")
def test_chunk_metadata_includes_required_keys(mock_extract, MockBuilder):
    """Given extraction produces chunks with element metadata,
    when chunks are fed to LeannBuilder,
    then each chunk carries source, file_name, file_type, page_number, element_type, chunk_index."""
    from indexer import KreuzbergIndexer

    data_dir = _make_data_dir(["doc.pdf"])
    mock_extract.return_value = _make_mock_result(
        chunks=[_make_mock_chunk("Some text", {"chunk_index": 0})],
        elements=[MagicMock(metadata={"page_number": 3, "element_type": "paragraph"})],
    )

    indexer = KreuzbergIndexer()
    indexer.build(data_dir, "meta-test")

    add_call = MockBuilder.return_value.add_text.call_args
    metadata = add_call.kwargs["metadata"]

    assert metadata["source"] == "doc.pdf"
    assert metadata["file_name"] == "doc.pdf"
    assert metadata["file_type"] == "pdf"
    assert metadata["page_number"] == 3
    assert metadata["element_type"] == "paragraph"
    assert metadata["chunk_index"] == 0


# --- Unit tests: error handling ---

@patch("indexer.LeannBuilder")
@patch("indexer.extract_file_sync")
def test_corrupt_file_skipped_with_warning(mock_extract, MockBuilder, caplog):
    """Given a directory with a corrupt file,
    when build() encounters an extraction error,
    then it logs a warning and continues processing remaining files."""
    from indexer import KreuzbergIndexer

    data_dir = _make_data_dir(["good.pdf", "bad.docx", "also_good.html"])

    call_count = [0]
    def extract_side_effect(path, *args, **kwargs):
        call_count[0] += 1
        if "bad.docx" in str(path):
            raise RuntimeError("Corrupted file")
        return _make_mock_result(
            chunks=[_make_mock_chunk("Content", {"chunk_index": 0})],
            elements=[MagicMock(metadata={"page_number": 1, "element_type": "text"})],
        )

    mock_extract.side_effect = extract_side_effect

    indexer = KreuzbergIndexer()
    with caplog.at_level(logging.WARNING):
        indexer.build(data_dir, "error-test")

    # All 3 supported files attempted
    assert mock_extract.call_count == 3
    # Only 2 good files produced chunks
    assert MockBuilder.return_value.add_text.call_count == 2
    # Warning logged for the corrupt file
    assert any("bad.docx" in msg for msg in caplog.messages)


@patch("indexer.LeannBuilder")
@patch("indexer.extract_file_sync")
def test_empty_extraction_result_skipped(mock_extract, MockBuilder):
    """Given extraction produces no chunks for a file,
    when build() processes it,
    then it skips that file without error."""
    from indexer import KreuzbergIndexer

    data_dir = _make_data_dir(["empty.pdf", "full.docx"])

    def extract_side_effect(path, *args, **kwargs):
        if "empty.pdf" in str(path):
            return _make_mock_result(chunks=[], elements=[])
        return _make_mock_result(
            chunks=[_make_mock_chunk("Real content", {"chunk_index": 0})],
            elements=[MagicMock(metadata={"page_number": 1, "element_type": "text"})],
        )

    mock_extract.side_effect = extract_side_effect

    indexer = KreuzbergIndexer()
    indexer.build(data_dir, "empty-test")

    # Only the non-empty file's chunk is added
    assert MockBuilder.return_value.add_text.call_count == 1


@patch("indexer.LeannBuilder")
@patch("indexer.extract_file_sync")
def test_no_chunks_at_all_raises_error(mock_extract, MockBuilder):
    """Given no files produce any chunks,
    when build() finishes walking,
    then it raises an error (cannot build empty index)."""
    from indexer import KreuzbergIndexer

    data_dir = _make_data_dir(["empty.pdf"])
    mock_extract.return_value = _make_mock_result(chunks=[], elements=[])

    indexer = KreuzbergIndexer()
    with pytest.raises(RuntimeError, match="No chunks"):
        indexer.build(data_dir, "empty-index")


# --- Unit tests: extraction config ---

@patch("indexer.LeannBuilder")
@patch("indexer.extract_file_sync")
def test_extraction_uses_element_based_format_with_correct_chunking(mock_extract, MockBuilder):
    """Given build() is called,
    when it extracts files,
    then it uses element-based result format with 512-char chunks and 50-char overlap."""
    from indexer import KreuzbergIndexer
    from kreuzberg import ResultFormat, OutputFormat

    data_dir = _make_data_dir(["doc.pdf"])
    mock_extract.return_value = _make_mock_result(
        chunks=[_make_mock_chunk("Content", {"chunk_index": 0})],
        elements=[MagicMock(metadata={"page_number": 1, "element_type": "text"})],
    )

    indexer = KreuzbergIndexer()
    indexer.build(data_dir, "config-test")

    config = mock_extract.call_args.kwargs.get("config") or mock_extract.call_args[1].get("config")
    assert config.result_format == ResultFormat.ELEMENT_BASED
    assert config.output_format == OutputFormat.MARKDOWN
    assert config.chunking.max_chars == 512
    assert config.chunking.max_overlap == 50


# --- Unit test: unsupported file extensions filtered ---

@pytest.mark.parametrize("filename,should_extract", [
    ("report.pdf", True),
    ("slides.pptx", True),
    ("data.xlsx", True),
    ("page.html", True),
    ("page.htm", True),
    ("book.epub", True),
    ("letter.doc", True),
    ("letter.docx", True),
    ("readme.txt", False),
    ("script.py", False),
    ("image.png", False),
    ("data.csv", False),
])
@patch("indexer.LeannBuilder")
@patch("indexer.extract_file_sync")
def test_only_supported_extensions_are_extracted(mock_extract, MockBuilder, filename, should_extract):
    """Given files with various extensions,
    when build() walks the directory,
    then only supported document formats are extracted."""
    from indexer import KreuzbergIndexer

    data_dir = _make_data_dir([filename])
    mock_extract.return_value = _make_mock_result(
        chunks=[_make_mock_chunk("Content", {"chunk_index": 0})],
        elements=[MagicMock(metadata={"page_number": 1, "element_type": "text"})],
    )

    indexer = KreuzbergIndexer()
    try:
        indexer.build(data_dir, "ext-test")
    except RuntimeError:
        pass  # Expected when no chunks (unsupported file)

    if should_extract:
        assert mock_extract.call_count == 1, f"{filename} should be extracted"
    else:
        assert mock_extract.call_count == 0, f"{filename} should NOT be extracted"
