"""Tests for KreuzbergIndexer.start_watcher() and full incremental reindexing flow."""

import json
import shutil
import tempfile
import threading
import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from indexer import KreuzbergIndexer


class TestStartWatcher:
    """Test start_watcher() through the KreuzbergIndexer driving port."""

    def _make_indexer(self):
        """Create a KreuzbergIndexer with mocked extract_and_append_file."""
        indexer = KreuzbergIndexer.__new__(KreuzbergIndexer)
        indexer.embedding_model = "facebook/contriever"
        indexer.extract_and_append_file = MagicMock()
        indexer.SKIP_MIME_PREFIXES = ("image/",)
        return indexer

    def test_new_file_triggers_extract_and_append(self):
        """A new supported file triggers extract_and_append_file."""
        indexer = self._make_indexer()
        data_dir = Path("/tmp/test-watch")
        stop_event = threading.Event()

        fake_changes = [
            {(1, str(data_dir / "report.pdf"))},
        ]

        with (
            patch("indexer.watch", side_effect=lambda *a, **kw: iter(fake_changes)),
            patch("indexer.detect_mime_type_from_path", return_value="application/pdf"),
            patch.object(Path, "is_file", return_value=True),
        ):
            thread = indexer.start_watcher(data_dir, "test-index", stop_event)
            thread.join(timeout=2)

        indexer.extract_and_append_file.assert_called_once_with(
            Path(str(data_dir / "report.pdf")), data_dir, "test-index"
        )

    def test_image_files_are_skipped(self):
        """Files with image/ MIME type are not indexed."""
        indexer = self._make_indexer()
        data_dir = Path("/tmp/test-watch")
        stop_event = threading.Event()

        fake_changes = [
            {(1, str(data_dir / "photo.png"))},
        ]

        with (
            patch("indexer.watch", side_effect=lambda *a, **kw: iter(fake_changes)),
            patch("indexer.detect_mime_type_from_path", return_value="image/png"),
            patch.object(Path, "is_file", return_value=True),
        ):
            thread = indexer.start_watcher(data_dir, "test-index", stop_event)
            thread.join(timeout=2)

        indexer.extract_and_append_file.assert_not_called()

    def test_stop_event_causes_thread_exit(self):
        """Setting the stop_event causes the watcher thread to exit cleanly."""
        indexer = self._make_indexer()
        data_dir = Path("/tmp/test-watch")
        stop_event = threading.Event()

        def blocking_watch(*args, **kwargs):
            """Simulate a watch that blocks until stop_event is set."""
            while not stop_event.is_set():
                stop_event.wait(timeout=0.05)
            return
            yield  # make it a generator

        with patch("indexer.watch", side_effect=blocking_watch):
            thread = indexer.start_watcher(data_dir, "test-index", stop_event)
            assert thread.is_alive()
            stop_event.set()
            thread.join(timeout=2)
            assert not thread.is_alive(), "Thread should have exited after stop_event was set"


# --- Helpers for integration test ---

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


# --- Integration test: full incremental reindexing flow ---

class TestIncrementalReindexingFlow:
    """Integration test: initial build creates index + manifest,
    second build after adding a file triggers incremental update."""

    @patch("indexer.LeannBuilder")
    @patch("indexer.extract_file_sync")
    def test_second_build_uses_incremental_update_not_full_rebuild(self, mock_extract, MockBuilder):
        """Given an initial build() that indexed 1 file,
        when a second file is added and build() is called again,
        then only the new file is extracted via update_index() (not build_index()),
        and the manifest contains both files."""
        data_dir = Path(tempfile.mkdtemp())
        (data_dir / "report.pdf").write_bytes(b"fake-pdf-content")

        index_name = f"incr-flow-{uuid.uuid4().hex[:8]}"

        # --- First build: full index creation ---
        mock_extract.return_value = _make_mock_result(
            chunks=[_make_mock_chunk("Report chunk 1", {"chunk_index": 0})],
            elements=[MagicMock(metadata={"page_number": 1, "element_type": "paragraph"})],
        )

        indexer = KreuzbergIndexer()
        result_path = indexer.build(data_dir, index_name)

        # Simulate that build_index created the meta file (mocked build_index doesn't write it)
        meta_path = Path(".leann") / "indexes" / index_name / "documents.leann.meta.json"
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        meta_path.write_text("{}")

        # First build uses build_index (full)
        builder_instance = MockBuilder.return_value
        assert builder_instance.build_index.call_count == 1, "First build should use build_index"
        assert builder_instance.update_index.call_count == 0, "First build should NOT use update_index"
        assert mock_extract.call_count == 1, "First build should extract 1 file"

        # Manifest written with 1 file
        manifest = indexer._read_manifest(index_name)
        assert manifest is not None, "Manifest should exist after first build"
        assert len(manifest["files"]) == 1
        assert "report.pdf" in manifest["files"]
        assert manifest["files"]["report.pdf"]["chunks"] == 1

        # --- Reset mocks for second build ---
        mock_extract.reset_mock()
        MockBuilder.reset_mock()

        # --- Add a second file ---
        (data_dir / "slides.docx").write_bytes(b"fake-docx-content")

        mock_extract.return_value = _make_mock_result(
            chunks=[
                _make_mock_chunk("Slides chunk 1", {"chunk_index": 0}),
                _make_mock_chunk("Slides chunk 2", {"chunk_index": 1}),
            ],
            elements=[
                MagicMock(metadata={"page_number": 1, "element_type": "heading"}),
                MagicMock(metadata={"page_number": 2, "element_type": "paragraph"}),
            ],
        )

        # --- Second build: should be incremental ---
        result_path_2 = indexer.build(data_dir, index_name)

        # update_index called (incremental), NOT build_index
        builder_instance_2 = MockBuilder.return_value
        assert builder_instance_2.update_index.call_count == 1, "Second build should use update_index"
        assert builder_instance_2.build_index.call_count == 0, "Second build should NOT use build_index"

        # Only the new file extracted (report.pdf skipped because mtime unchanged)
        assert mock_extract.call_count == 1, "Only the new file should be extracted"
        extracted_path = mock_extract.call_args[0][0]
        assert "slides.docx" in extracted_path, "Only slides.docx should be extracted"

        # Manifest contains both files
        manifest = indexer._read_manifest(index_name)
        assert len(manifest["files"]) == 2, "Manifest should have both files"
        assert "report.pdf" in manifest["files"]
        assert "slides.docx" in manifest["files"]
        assert manifest["files"]["report.pdf"]["chunks"] == 1
        assert manifest["files"]["slides.docx"]["chunks"] == 2

        # Both builds return valid index path
        assert result_path == result_path_2
        assert result_path.endswith("documents.leann")

        # Cleanup
        index_dir = Path(".leann") / "indexes" / index_name
        if index_dir.exists():
            shutil.rmtree(index_dir)
