"""Tests for KreuzbergIndexer: walk, extract, chunk, index.

Step 01-01: kreuzberg-integration
Test Budget: 4 behaviors x 2 = 8 max unit tests

Step 01-01: incremental-reindexing / manifest
Test Budget: 2 behaviors x 2 = 4 max unit tests (using 2)

Behaviors (manifest):
1. build() writes manifest.json with version, files dict (mtime + chunks)
2. _read_manifest() returns None when manifest file is missing

Step 01-02: incremental-reindexing / incremental_update
Test Budget: 3 behaviors x 2 = 6 max unit tests

Behaviors (incremental_update):
1. New files (not in manifest) are extracted and indexed via update_index()
2. Modified files (mtime differs) are re-extracted and indexed
3. Unchanged files (mtime matches) are skipped

Step 01-03: incremental-reindexing / build() delegates to incremental_update
Test Budget: 3 behaviors x 2 = 6 max unit tests

Behaviors:
1. build() with existing index + manifest calls incremental_update (uses update_index, not build_index)
2. build() with existing index but no manifest skips (no extraction)
3. build() with force=True always does full rebuild regardless of manifest
"""
import json
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


# --- Acceptance test: manifest written after build ---

@patch("indexer.LeannBuilder")
@patch("indexer.extract_file_sync")
def test_build_writes_manifest_with_file_mtimes_and_chunk_counts(mock_extract, MockBuilder):
    """Given a directory with supported files,
    when build() completes successfully,
    then it writes manifest.json with version=1 and per-file mtime + chunk count."""
    from indexer import KreuzbergIndexer

    data_dir = _make_data_dir(["report.pdf", "slides.pptx"])

    mock_extract.return_value = _make_mock_result(
        chunks=[
            _make_mock_chunk("Chunk one", {"chunk_index": 0}),
            _make_mock_chunk("Chunk two", {"chunk_index": 1}),
        ],
        elements=[
            MagicMock(metadata={"page_number": 1, "element_type": "paragraph"}),
            MagicMock(metadata={"page_number": 2, "element_type": "heading"}),
        ],
    )

    indexer = KreuzbergIndexer()
    indexer.build(data_dir, "manifest-test")

    # Manifest should exist in the index directory
    manifest_path = Path(".leann") / "indexes" / "manifest-test" / "manifest.json"
    assert manifest_path.exists(), "manifest.json should be written after build"

    manifest = json.loads(manifest_path.read_text())

    # Version field
    assert manifest["version"] == 1

    # Files dict keyed by relative path
    assert "files" in manifest
    files = manifest["files"]
    assert "report.pdf" in files
    assert "slides.pptx" in files

    # Each entry has mtime (float) and chunks (int)
    for rel_path in ["report.pdf", "slides.pptx"]:
        entry = files[rel_path]
        assert isinstance(entry["mtime"], float)
        assert isinstance(entry["chunks"], int)
        assert entry["chunks"] == 2  # Each file produced 2 chunks
        # mtime should be a real file mtime (positive number)
        assert entry["mtime"] > 0


# --- Unit test: _read_manifest returns None when missing ---

def test_read_manifest_returns_none_when_missing():
    """Given no manifest.json exists,
    when _read_manifest() is called,
    then it returns None."""
    from indexer import KreuzbergIndexer

    indexer = KreuzbergIndexer()
    result = indexer._read_manifest("nonexistent-index-name")
    assert result is None


# --- Acceptance test: incremental_update detects new files ---

@patch("indexer.LeannBuilder")
@patch("indexer.extract_file_sync")
def test_incremental_update_extracts_only_new_files(mock_extract, MockBuilder):
    """Given a manifest with 1 file and a directory with 2 files,
    when incremental_update() is called,
    then it extracts only the new file and uses update_index() (not build_index())."""
    from indexer import KreuzbergIndexer

    data_dir = _make_data_dir(["existing.pdf", "new_file.docx"])
    existing_mtime = (data_dir / "existing.pdf").stat().st_mtime

    indexer = KreuzbergIndexer()
    # Pre-seed manifest with the existing file
    indexer._write_manifest("incr-test", {
        "existing.pdf": {"mtime": existing_mtime, "chunks": 2},
    })

    mock_extract.return_value = _make_mock_result(
        chunks=[_make_mock_chunk("New content", {"chunk_index": 0})],
        elements=[MagicMock(metadata={"page_number": 1, "element_type": "text"})],
    )

    index_path = str(Path(".leann") / "indexes" / "incr-test" / "documents.leann")
    indexer.incremental_update(data_dir, "incr-test")

    # Only the new file should be extracted (existing.pdf skipped)
    assert mock_extract.call_count == 1
    extracted_path = mock_extract.call_args[0][0]
    assert "new_file.docx" in extracted_path

    # update_index() called, NOT build_index()
    builder_instance = MockBuilder.return_value
    assert builder_instance.update_index.call_count == 1
    assert builder_instance.build_index.call_count == 0

    # Manifest updated with both files
    manifest = indexer._read_manifest("incr-test")
    assert "existing.pdf" in manifest["files"]
    assert "new_file.docx" in manifest["files"]


# --- Unit tests: incremental_update behaviors ---

@patch("indexer.LeannBuilder")
@patch("indexer.extract_file_sync")
def test_incremental_update_reextracts_modified_files(mock_extract, MockBuilder):
    """Given a manifest with a file whose mtime is stale,
    when incremental_update() is called,
    then it re-extracts that file."""
    from indexer import KreuzbergIndexer
    import time

    data_dir = _make_data_dir(["report.pdf"])
    # Record an old mtime in manifest (different from actual file mtime)
    indexer = KreuzbergIndexer()
    actual_mtime = (data_dir / "report.pdf").stat().st_mtime
    indexer._write_manifest("mod-test", {
        "report.pdf": {"mtime": actual_mtime - 100.0, "chunks": 1},
    })

    mock_extract.return_value = _make_mock_result(
        chunks=[_make_mock_chunk("Updated content", {"chunk_index": 0})],
        elements=[MagicMock(metadata={"page_number": 1, "element_type": "text"})],
    )

    indexer.incremental_update(data_dir, "mod-test")

    # Modified file should be extracted
    assert mock_extract.call_count == 1

    # Manifest mtime should be updated to current
    manifest = indexer._read_manifest("mod-test")
    assert manifest["files"]["report.pdf"]["mtime"] == actual_mtime


@patch("indexer.LeannBuilder")
@patch("indexer.extract_file_sync")
def test_incremental_update_skips_unchanged_files(mock_extract, MockBuilder):
    """Given all files in directory match manifest mtimes,
    when incremental_update() is called,
    then no files are extracted and no index update occurs."""
    from indexer import KreuzbergIndexer

    data_dir = _make_data_dir(["report.pdf", "slides.pptx"])

    indexer = KreuzbergIndexer()
    # Manifest mtimes match actual file mtimes exactly
    indexer._write_manifest("noop-test", {
        "report.pdf": {"mtime": (data_dir / "report.pdf").stat().st_mtime, "chunks": 2},
        "slides.pptx": {"mtime": (data_dir / "slides.pptx").stat().st_mtime, "chunks": 3},
    })

    indexer.incremental_update(data_dir, "noop-test")

    # No files extracted
    assert mock_extract.call_count == 0

    # No index update
    builder_instance = MockBuilder.return_value
    assert builder_instance.update_index.call_count == 0


# --- Acceptance test: build() delegates to incremental_update when manifest exists ---

@patch("indexer.LeannBuilder")
@patch("indexer.extract_file_sync")
def test_build_with_existing_index_and_manifest_runs_incremental_update(mock_extract, MockBuilder):
    """Given an existing index with manifest and a new file in data_dir,
    when build() is called (without force),
    then it delegates to incremental_update() using update_index (not build_index)."""
    from indexer import KreuzbergIndexer

    data_dir = _make_data_dir(["existing.pdf", "new_file.docx"])
    existing_mtime = (data_dir / "existing.pdf").stat().st_mtime

    indexer = KreuzbergIndexer()

    # Create existing index meta file (simulates a built index)
    index_name = "incr-build-test"
    meta_path = Path(".leann") / "indexes" / index_name / "documents.leann.meta.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text("{}")

    # Create manifest (simulates a post-01-01 index with manifest)
    indexer._write_manifest(index_name, {
        "existing.pdf": {"mtime": existing_mtime, "chunks": 2},
    })

    mock_extract.return_value = _make_mock_result(
        chunks=[_make_mock_chunk("New content", {"chunk_index": 0})],
        elements=[MagicMock(metadata={"page_number": 1, "element_type": "text"})],
    )

    result_path = indexer.build(data_dir, index_name)

    # Should use update_index (incremental), NOT build_index (full rebuild)
    builder_instance = MockBuilder.return_value
    assert builder_instance.update_index.call_count == 1, "Should call update_index for incremental"
    assert builder_instance.build_index.call_count == 0, "Should NOT call build_index"

    # Only the new file should be extracted (existing.pdf skipped by manifest mtime match)
    assert mock_extract.call_count == 1
    extracted_path = mock_extract.call_args[0][0]
    assert "new_file.docx" in extracted_path

    # Returns valid index path
    assert index_name in result_path
    assert result_path.endswith("documents.leann")


# --- Unit tests: build() skip vs incremental delegation ---

@patch("indexer.LeannBuilder")
@patch("indexer.extract_file_sync")
def test_build_with_existing_index_without_manifest_skips(mock_extract, MockBuilder):
    """Given an existing index but no manifest (pre-feature index),
    when build() is called without force,
    then it skips as before (no extraction, no build)."""
    from indexer import KreuzbergIndexer

    data_dir = _make_data_dir(["report.pdf"])

    indexer = KreuzbergIndexer()

    # Create existing index meta file but NO manifest
    index_name = "legacy-index"
    meta_path = Path(".leann") / "indexes" / index_name / "documents.leann.meta.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text("{}")

    result_path = indexer.build(data_dir, index_name)

    # No extraction, no building -- just skip
    assert mock_extract.call_count == 0
    builder_instance = MockBuilder.return_value
    assert builder_instance.build_index.call_count == 0
    assert builder_instance.update_index.call_count == 0

    # Returns the index path
    assert result_path.endswith("documents.leann")


@patch("indexer.LeannBuilder")
@patch("indexer.extract_file_sync")
def test_build_with_force_rebuilds_even_when_manifest_exists(mock_extract, MockBuilder):
    """Given an existing index with manifest,
    when build() is called with force=True,
    then it does a full rebuild using build_index (not update_index)."""
    from indexer import KreuzbergIndexer

    data_dir = _make_data_dir(["report.pdf"])
    file_mtime = (data_dir / "report.pdf").stat().st_mtime

    indexer = KreuzbergIndexer()

    # Create existing index meta file AND manifest
    index_name = "force-rebuild"
    meta_path = Path(".leann") / "indexes" / index_name / "documents.leann.meta.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text("{}")
    indexer._write_manifest(index_name, {
        "report.pdf": {"mtime": file_mtime, "chunks": 2},
    })

    mock_extract.return_value = _make_mock_result(
        chunks=[_make_mock_chunk("Content", {"chunk_index": 0})],
        elements=[MagicMock(metadata={"page_number": 1, "element_type": "text"})],
    )

    result_path = indexer.build(data_dir, index_name, force=True)

    # Full rebuild: build_index called, NOT update_index
    builder_instance = MockBuilder.return_value
    assert builder_instance.build_index.call_count == 1, "force=True should trigger full build_index"
    assert builder_instance.update_index.call_count == 0, "force=True should NOT use update_index"

    # File should be extracted (full rebuild processes all files)
    assert mock_extract.call_count == 1
