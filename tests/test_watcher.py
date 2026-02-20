"""Tests for KreuzbergIndexer.start_watcher() file watching capability."""

import threading
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
