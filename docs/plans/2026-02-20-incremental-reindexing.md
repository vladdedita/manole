# Incremental Reindexing Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Detect new and modified files in indexed directories via filesystem watching and mtime-based manifests, and incrementally append them to the existing LEANN index without full rebuilds.

**Architecture:** A JSON manifest tracks indexed files and their mtimes. On startup, a catch-up scan diffs the manifest against the filesystem and appends only changed files via `LeannBuilder.update_index()`. While the app runs, `watchfiles` monitors directories in a background thread and triggers the same incremental append logic.

**Tech Stack:** watchfiles (filesystem watcher), LeannBuilder (LEANN index), kreuzberg (document extraction)

---

### Task 1: Manifest Read/Write in KreuzbergIndexer

**Files:**
- Test: `tests/test_indexer.py`
- Modify: `indexer.py`

**Step 1: Write the failing test for manifest creation on build**

Add to `tests/test_indexer.py`:

```python
@patch("indexer.LeannBuilder")
@patch("indexer.extract_file_sync")
def test_build_writes_manifest_with_indexed_files(mock_extract, MockBuilder):
    """Given a successful build,
    when build() completes,
    then it writes a manifest.json with file paths and mtimes."""
    import json
    from indexer import KreuzbergIndexer

    data_dir = _make_data_dir(["report.pdf", "slides.pptx"])
    mock_extract.return_value = _make_mock_result(
        chunks=[_make_mock_chunk("Content", {"chunk_index": 0})],
        elements=[MagicMock(metadata={"page_number": 1, "element_type": "text"})],
    )

    indexer = KreuzbergIndexer()
    result_path = indexer.build(data_dir, "manifest-test")

    manifest_path = Path(result_path).parent / "manifest.json"
    assert manifest_path.exists()

    manifest = json.loads(manifest_path.read_text())
    assert manifest["version"] == 1
    assert "report.pdf" in manifest["files"]
    assert "slides.pptx" in manifest["files"]
    # Each entry has mtime and chunks count
    entry = manifest["files"]["report.pdf"]
    assert "mtime" in entry
    assert entry["chunks"] == 1
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_indexer.py::test_build_writes_manifest_with_indexed_files -v`
Expected: FAIL — no `manifest.json` written

**Step 3: Implement manifest writing in build()**

In `indexer.py`, add a `_write_manifest` method and call it at the end of `build()`. Track per-file chunk counts during the walk loop.

Add at the top of `indexer.py`:
```python
import json
```

Add method to `KreuzbergIndexer`:
```python
def _manifest_path(self, index_path: str) -> Path:
    return Path(index_path).parent / "manifest.json"

def _write_manifest(self, index_path: str, file_records: dict) -> None:
    manifest = {"version": 1, "files": file_records}
    self._manifest_path(index_path).write_text(json.dumps(manifest, indent=2))

def _read_manifest(self, index_path: str) -> dict | None:
    path = self._manifest_path(index_path)
    if not path.exists():
        return None
    return json.loads(path.read_text())
```

In `build()`, track file records:
- Before the walk loop: `file_records = {}`
- Inside the loop, after processing each file's chunks:
  ```python
  rel = str(file_path.relative_to(data_dir))
  file_records[rel] = {
      "mtime": file_path.stat().st_mtime,
      "chunks": len(result.chunks),
  }
  ```
- After `builder.build_index(index_path)`:
  ```python
  self._write_manifest(index_path, file_records)
  ```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_indexer.py::test_build_writes_manifest_with_indexed_files -v`
Expected: PASS

**Step 5: Commit**

```bash
git add tests/test_indexer.py indexer.py
git commit -m "feat(indexer): write manifest.json after build with file mtimes and chunk counts"
```

---

### Task 2: Catch-up Scan — Incremental Update on Startup

**Files:**
- Test: `tests/test_indexer.py`
- Modify: `indexer.py`

**Step 1: Write the failing test for incremental update detecting new files**

```python
@patch("indexer.LeannBuilder")
@patch("indexer.extract_file_sync")
def test_incremental_update_indexes_new_files_only(mock_extract, MockBuilder):
    """Given an existing index with manifest,
    when a new file is added and incremental_update() is called,
    then only the new file is extracted and appended via update_index()."""
    import json
    from indexer import KreuzbergIndexer

    data_dir = _make_data_dir(["existing.pdf", "new_file.docx"])

    # Manifest only knows about existing.pdf
    index_dir = Path(".leann") / "indexes" / "inc-test"
    index_dir.mkdir(parents=True, exist_ok=True)
    index_path = str(index_dir / "documents.leann")
    Path(f"{index_path}.meta.json").write_text("{}")  # fake existing index

    manifest = {
        "version": 1,
        "files": {
            "existing.pdf": {"mtime": (data_dir / "existing.pdf").stat().st_mtime, "chunks": 2}
        }
    }
    (index_dir / "manifest.json").write_text(json.dumps(manifest))

    mock_extract.return_value = _make_mock_result(
        chunks=[_make_mock_chunk("New content", {"chunk_index": 0})],
        elements=[MagicMock(metadata={"page_number": 1, "element_type": "text"})],
    )

    indexer = KreuzbergIndexer()
    indexer.incremental_update(data_dir, index_path)

    # Only new_file.docx extracted (existing.pdf skipped)
    assert mock_extract.call_count == 1
    extracted_path = mock_extract.call_args[0][0]
    assert "new_file.docx" in str(extracted_path)

    # update_index called (not build_index)
    builder = MockBuilder.return_value
    assert builder.update_index.call_count == 1
    assert builder.build_index.call_count == 0
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_indexer.py::test_incremental_update_indexes_new_files_only -v`
Expected: FAIL — `incremental_update` does not exist

**Step 3: Write the failing test for detecting modified files**

```python
@patch("indexer.LeannBuilder")
@patch("indexer.extract_file_sync")
def test_incremental_update_reindexes_modified_files(mock_extract, MockBuilder):
    """Given an existing index with manifest,
    when a file's mtime has changed and incremental_update() is called,
    then the modified file is re-extracted and appended."""
    import json
    from indexer import KreuzbergIndexer

    data_dir = _make_data_dir(["changed.pdf"])

    index_dir = Path(".leann") / "indexes" / "mod-test"
    index_dir.mkdir(parents=True, exist_ok=True)
    index_path = str(index_dir / "documents.leann")
    Path(f"{index_path}.meta.json").write_text("{}")

    # Manifest has old mtime (0.0) — file has been modified
    manifest = {
        "version": 1,
        "files": {"changed.pdf": {"mtime": 0.0, "chunks": 1}}
    }
    (index_dir / "manifest.json").write_text(json.dumps(manifest))

    mock_extract.return_value = _make_mock_result(
        chunks=[_make_mock_chunk("Updated content", {"chunk_index": 0})],
        elements=[MagicMock(metadata={"page_number": 1, "element_type": "text"})],
    )

    indexer = KreuzbergIndexer()
    indexer.incremental_update(data_dir, index_path)

    assert mock_extract.call_count == 1
    assert MockBuilder.return_value.update_index.call_count == 1
```

**Step 4: Write test for no-op when nothing changed**

```python
@patch("indexer.LeannBuilder")
@patch("indexer.extract_file_sync")
def test_incremental_update_skips_when_nothing_changed(mock_extract, MockBuilder):
    """Given all files match the manifest,
    when incremental_update() is called,
    then no extraction or index update occurs."""
    import json
    from indexer import KreuzbergIndexer

    data_dir = _make_data_dir(["same.pdf"])

    index_dir = Path(".leann") / "indexes" / "noop-test"
    index_dir.mkdir(parents=True, exist_ok=True)
    index_path = str(index_dir / "documents.leann")
    Path(f"{index_path}.meta.json").write_text("{}")

    manifest = {
        "version": 1,
        "files": {
            "same.pdf": {"mtime": (data_dir / "same.pdf").stat().st_mtime, "chunks": 1}
        }
    }
    (index_dir / "manifest.json").write_text(json.dumps(manifest))

    indexer = KreuzbergIndexer()
    indexer.incremental_update(data_dir, index_path)

    assert mock_extract.call_count == 0
    assert MockBuilder.return_value.update_index.call_count == 0
```

**Step 5: Run all three tests to verify they fail**

Run: `python -m pytest tests/test_indexer.py -k "incremental" -v`
Expected: FAIL — `incremental_update` does not exist

**Step 6: Implement incremental_update()**

Add to `KreuzbergIndexer` in `indexer.py`:

```python
def incremental_update(self, data_dir: Path, index_path: str) -> None:
    """Scan for new/modified files and append them to the existing index."""
    data_dir = Path(data_dir)
    manifest = self._read_manifest(index_path)
    if manifest is None:
        print("No manifest found, skipping incremental update")
        return

    known_files = manifest.get("files", {})
    to_index = []

    for file_path in sorted(data_dir.rglob("*")):
        if not file_path.is_file():
            continue
        try:
            mime = detect_mime_type_from_path(str(file_path))
        except Exception:
            continue
        if any(mime.startswith(p) for p in self.SKIP_MIME_PREFIXES):
            continue

        rel = str(file_path.relative_to(data_dir))
        current_mtime = file_path.stat().st_mtime

        if rel in known_files and known_files[rel]["mtime"] == current_mtime:
            continue  # unchanged
        to_index.append(file_path)

    if not to_index:
        print("No new or modified files found")
        return

    print(f"Incremental update: {len(to_index)} file(s) to index")
    self._extract_and_append(to_index, data_dir, index_path, known_files)
    self._write_manifest(index_path, known_files)
```

Also add the shared extraction + append helper:

```python
def _extract_and_append(self, files: list[Path], data_dir: Path, index_path: str, file_records: dict) -> None:
    """Extract files and append chunks to existing index."""
    config = ExtractionConfig(
        output_format=OutputFormat.MARKDOWN,
        result_format=ResultFormat.ELEMENT_BASED,
        include_document_structure=True,
        chunking=ChunkingConfig(max_chars=512, max_overlap=50),
    )

    builder = LeannBuilder(
        backend_name="hnsw",
        embedding_model=self.embedding_model,
        is_recompute=False,
    )

    total_chunks = 0
    for file_path in files:
        print(f"  Extracting: {file_path.name}")
        try:
            result = extract_file_sync(str(file_path), config=config)
        except Exception as exc:
            print(f"  FAILED: {file_path.name}: {exc}")
            continue

        if not result.chunks:
            continue

        elements = result.elements or []
        for i, chunk in enumerate(result.chunks):
            chunk_index = chunk.metadata.get("chunk_index", i)
            page_number = None
            element_type = None
            if i < len(elements):
                elem = elements[i]
                if isinstance(elem, dict):
                    elem_meta = elem.get("metadata") or {}
                    element_type = elem.get("element_type") or elem_meta.get("element_type")
                else:
                    elem_meta = elem.metadata or {}
                    element_type = elem_meta.get("element_type")
                page_number = elem_meta.get("page_number")

            builder.add_text(
                text=chunk.content,
                metadata={
                    "source": str(file_path.relative_to(data_dir)),
                    "file_name": file_path.name,
                    "file_type": file_path.suffix.lstrip("."),
                    "page_number": page_number,
                    "element_type": element_type,
                    "chunk_index": chunk_index,
                },
            )
            total_chunks += 1

        rel = str(file_path.relative_to(data_dir))
        file_records[rel] = {
            "mtime": file_path.stat().st_mtime,
            "chunks": len(result.chunks),
        }

    if total_chunks > 0:
        builder.update_index(index_path)
        print(f"  Appended {total_chunks} chunks to index")
```

**Step 7: Run tests to verify they pass**

Run: `python -m pytest tests/test_indexer.py -k "incremental" -v`
Expected: PASS (all 3)

**Step 8: Commit**

```bash
git add tests/test_indexer.py indexer.py
git commit -m "feat(indexer): incremental_update detects new/modified files via manifest diffing"
```

---

### Task 3: Wire Catch-up Scan into build() and build_index()

**Files:**
- Test: `tests/test_indexer.py`
- Modify: `indexer.py`
- Modify: `chat.py`

**Step 1: Write the failing test for build() doing incremental update when index exists**

```python
@patch("indexer.LeannBuilder")
@patch("indexer.extract_file_sync")
def test_build_runs_incremental_update_when_index_exists_with_manifest(mock_extract, MockBuilder):
    """Given an existing index with manifest and a new file added,
    when build() is called without force,
    then it runs incremental_update instead of skipping."""
    import json
    from indexer import KreuzbergIndexer

    data_dir = _make_data_dir(["old.pdf", "new.docx"])

    index_dir = Path(".leann") / "indexes" / "auto-inc"
    index_dir.mkdir(parents=True, exist_ok=True)
    index_path = str(index_dir / "documents.leann")
    Path(f"{index_path}.meta.json").write_text("{}")

    manifest = {
        "version": 1,
        "files": {
            "old.pdf": {"mtime": (data_dir / "old.pdf").stat().st_mtime, "chunks": 2}
        }
    }
    (index_dir / "manifest.json").write_text(json.dumps(manifest))

    mock_extract.return_value = _make_mock_result(
        chunks=[_make_mock_chunk("New content", {"chunk_index": 0})],
        elements=[MagicMock(metadata={"page_number": 1, "element_type": "text"})],
    )

    indexer = KreuzbergIndexer()
    result_path = indexer.build(data_dir, "auto-inc")

    # Should extract only the new file
    assert mock_extract.call_count == 1
    # Should use update_index, not build_index
    builder = MockBuilder.return_value
    assert builder.update_index.call_count == 1
    assert builder.build_index.call_count == 0
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_indexer.py::test_build_runs_incremental_update_when_index_exists_with_manifest -v`
Expected: FAIL — current `build()` skips entirely when index exists

**Step 3: Modify build() to call incremental_update when manifest exists**

Replace the skip-if-exists block in `build()`:

```python
# Incremental update if index + manifest exist
if not force and Path(f"{index_path}.meta.json").exists():
    manifest = self._read_manifest(index_path)
    if manifest is not None:
        self.incremental_update(data_dir, index_path)
        return index_path
    print(f"Index '{index_name}' already exists, skipping build (use force=True to rebuild)")
    return index_path
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_indexer.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add tests/test_indexer.py indexer.py
git commit -m "feat(indexer): build() runs incremental update when manifest exists"
```

---

### Task 4: Single-File Extract and Append

**Files:**
- Test: `tests/test_indexer.py`
- Modify: `indexer.py`

**Step 1: Write the failing test for extract_and_append_file()**

```python
@patch("indexer.LeannBuilder")
@patch("indexer.extract_file_sync")
def test_extract_and_append_file_indexes_single_file(mock_extract, MockBuilder):
    """Given an existing index,
    when extract_and_append_file() is called for one file,
    then it extracts the file and appends chunks via update_index()."""
    import json
    from indexer import KreuzbergIndexer

    data_dir = _make_data_dir(["new_report.pdf"])

    index_dir = Path(".leann") / "indexes" / "single-test"
    index_dir.mkdir(parents=True, exist_ok=True)
    index_path = str(index_dir / "documents.leann")
    Path(f"{index_path}.meta.json").write_text("{}")

    manifest = {"version": 1, "files": {}}
    (index_dir / "manifest.json").write_text(json.dumps(manifest))

    mock_extract.return_value = _make_mock_result(
        chunks=[
            _make_mock_chunk("Chunk 1", {"chunk_index": 0}),
            _make_mock_chunk("Chunk 2", {"chunk_index": 1}),
        ],
        elements=[
            MagicMock(metadata={"page_number": 1, "element_type": "text"}),
            MagicMock(metadata={"page_number": 2, "element_type": "text"}),
        ],
    )

    indexer = KreuzbergIndexer()
    indexer.extract_and_append_file(data_dir / "new_report.pdf", data_dir, index_path)

    assert mock_extract.call_count == 1
    builder = MockBuilder.return_value
    assert builder.add_text.call_count == 2
    assert builder.update_index.call_count == 1

    # Manifest updated
    updated = json.loads((index_dir / "manifest.json").read_text())
    assert "new_report.pdf" in updated["files"]
    assert updated["files"]["new_report.pdf"]["chunks"] == 2
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_indexer.py::test_extract_and_append_file_indexes_single_file -v`
Expected: FAIL — `extract_and_append_file` does not exist

**Step 3: Implement extract_and_append_file()**

Add to `KreuzbergIndexer`:

```python
def extract_and_append_file(self, file_path: Path, data_dir: Path, index_path: str) -> None:
    """Extract a single file and append its chunks to the existing index."""
    manifest = self._read_manifest(index_path) or {"version": 1, "files": {}}
    file_records = manifest.get("files", {})
    self._extract_and_append([Path(file_path)], Path(data_dir), index_path, file_records)
    self._write_manifest(index_path, file_records)
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_indexer.py::test_extract_and_append_file_indexes_single_file -v`
Expected: PASS

**Step 5: Commit**

```bash
git add tests/test_indexer.py indexer.py
git commit -m "feat(indexer): extract_and_append_file for single-file incremental indexing"
```

---

### Task 5: File Watcher in Server

**Files:**
- Test: `tests/test_server.py` (or a new `tests/test_watcher.py` if test_server is too large)
- Modify: `server.py`
- Modify: `pyproject.toml` (add `watchfiles` dependency)

**Step 1: Add watchfiles dependency**

In `pyproject.toml`, add `watchfiles` to the dependencies list (same section where `kreuzberg` is).

Run: `uv add watchfiles`

**Step 2: Write the failing test for watcher starting on init**

Create `tests/test_watcher.py`:

```python
"""Tests for file watching and incremental reindexing."""
import json
import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


def _make_indexed_dir():
    """Create a temp directory with a fake existing index and manifest."""
    tmp = tempfile.mkdtemp()
    data_dir = Path(tmp)

    # Create a fake file
    (data_dir / "existing.pdf").write_bytes(b"fake-content")

    # Create fake index
    index_dir = Path(".leann") / "indexes" / "watch-test"
    index_dir.mkdir(parents=True, exist_ok=True)
    index_path = str(index_dir / "documents.leann")
    Path(f"{index_path}.meta.json").write_text("{}")

    manifest = {
        "version": 1,
        "files": {
            "existing.pdf": {"mtime": (data_dir / "existing.pdf").stat().st_mtime, "chunks": 1}
        }
    }
    (index_dir / "manifest.json").write_text(json.dumps(manifest))

    return data_dir, index_path


@patch("indexer.LeannBuilder")
@patch("indexer.extract_file_sync")
def test_watcher_detects_new_file_and_triggers_append(mock_extract, MockBuilder):
    """Given a watched directory with an existing index,
    when a new file is created,
    then the watcher triggers extract_and_append_file for that file."""
    from indexer import KreuzbergIndexer
    from unittest.mock import MagicMock as MM

    mock_extract.return_value = MM()
    mock_extract.return_value.chunks = [MM(content="New", metadata={"chunk_index": 0})]
    mock_extract.return_value.elements = [MM(metadata={"page_number": 1, "element_type": "text"})]

    data_dir, index_path = _make_indexed_dir()
    indexer = KreuzbergIndexer()

    # Use a callback to track what the watcher processes
    processed = []
    original = indexer.extract_and_append_file
    def track_call(*args, **kwargs):
        processed.append(args[0])
        return original(*args, **kwargs)
    indexer.extract_and_append_file = track_call

    # Start watcher in background
    stop_event = threading.Event()
    watcher_thread = indexer.start_watcher(data_dir, index_path, stop_event)

    try:
        # Create a new file
        time.sleep(0.5)  # let watcher initialize
        (data_dir / "new_doc.pdf").write_bytes(b"new-content")
        time.sleep(2)  # let watcher detect
    finally:
        stop_event.set()
        watcher_thread.join(timeout=5)

    assert any("new_doc.pdf" in str(p) for p in processed)
```

**Step 3: Run test to verify it fails**

Run: `python -m pytest tests/test_watcher.py::test_watcher_detects_new_file_and_triggers_append -v`
Expected: FAIL — `start_watcher` does not exist

**Step 4: Implement start_watcher()**

Add to `KreuzbergIndexer` in `indexer.py`:

```python
def start_watcher(self, data_dir: Path, index_path: str, stop_event: "threading.Event") -> "threading.Thread":
    """Start a background thread that watches data_dir for changes."""
    import threading
    from watchfiles import watch

    data_dir = Path(data_dir)

    def _watch_loop():
        for changes in watch(data_dir, stop_event=stop_event, debounce=500):
            for change_type, path_str in changes:
                file_path = Path(path_str)
                if not file_path.is_file():
                    continue
                try:
                    mime = detect_mime_type_from_path(str(file_path))
                except Exception:
                    continue
                if any(mime.startswith(p) for p in self.SKIP_MIME_PREFIXES):
                    continue
                print(f"  Watcher: {file_path.name} changed, reindexing...")
                try:
                    self.extract_and_append_file(file_path, data_dir, index_path)
                except Exception as exc:
                    print(f"  Watcher: failed to index {file_path.name}: {exc}")

    thread = threading.Thread(target=_watch_loop, daemon=True, name="file-watcher")
    thread.start()
    return thread
```

Add `import threading` at the top of `indexer.py`.

**Step 5: Run test to verify it passes**

Run: `python -m pytest tests/test_watcher.py -v`
Expected: PASS

**Step 6: Commit**

```bash
git add tests/test_watcher.py indexer.py pyproject.toml uv.lock
git commit -m "feat(indexer): file watcher with watchfiles for real-time incremental indexing"
```

---

### Task 6: Wire Watcher into Server Lifecycle

**Files:**
- Modify: `server.py`

**Step 1: Add watcher start after index build in handle_init**

In `server.py`, after the index is built and components are wired up, start the watcher. Store the stop event and thread in the directory entry so we can clean up on shutdown or directory removal.

In `handle_init`, after `index_name = build_index(...)` and the component wiring block, add:

```python
# Start file watcher for incremental indexing
if pipeline == "kreuzberg":
    import threading
    from indexer import KreuzbergIndexer
    stop_event = threading.Event()
    indexer = KreuzbergIndexer()
    watcher_thread = indexer.start_watcher(data_dir_path, index_path, stop_event)
    entry["watcher_stop"] = stop_event
    entry["watcher_thread"] = watcher_thread
    self._log(f"File watcher started for {data_dir_path}")
```

Where `entry` is `self.directories[dir_id]`, and `index_path` is the resolved path from `find_index_path(index_name)`.

**Step 2: Stop watchers on directory removal and shutdown**

In `_delete_index_files` or `handle_remove_directory`, add:

```python
stop_event = entry.get("watcher_stop")
if stop_event:
    stop_event.set()
    thread = entry.get("watcher_thread")
    if thread:
        thread.join(timeout=3)
```

In `handle_shutdown`, stop all watchers:

```python
for entry in self.directories.values():
    stop_event = entry.get("watcher_stop")
    if stop_event:
        stop_event.set()
```

**Step 3: Run existing server tests to verify nothing breaks**

Run: `python -m pytest tests/test_server.py -v`
Expected: PASS

**Step 4: Commit**

```bash
git add server.py
git commit -m "feat(server): start/stop file watcher on directory init/remove lifecycle"
```

---

### Task 7: Integration Test

**Files:**
- Test: `tests/test_watcher.py`

**Step 1: Write integration test for the full flow**

```python
@patch("indexer.LeannBuilder")
@patch("indexer.extract_file_sync")
def test_full_incremental_flow_build_then_update(mock_extract, MockBuilder):
    """Given a directory is built with kreuzberg,
    when a new file is added and build() is called again,
    then only the new file is processed incrementally."""
    import json
    from indexer import KreuzbergIndexer
    from unittest.mock import MagicMock as MM

    data_dir = Path(tempfile.mkdtemp())
    (data_dir / "first.pdf").write_bytes(b"content-1")

    mock_extract.return_value = MM()
    mock_extract.return_value.chunks = [MM(content="First", metadata={"chunk_index": 0})]
    mock_extract.return_value.elements = [MM(metadata={"page_number": 1, "element_type": "text"})]

    indexer = KreuzbergIndexer()

    # First build: full index
    index_path = indexer.build(data_dir, "flow-test")
    assert mock_extract.call_count == 1
    assert MockBuilder.return_value.build_index.call_count == 1

    mock_extract.reset_mock()
    MockBuilder.reset_mock()

    # Add a new file
    (data_dir / "second.docx").write_bytes(b"content-2")

    mock_extract.return_value = MM()
    mock_extract.return_value.chunks = [MM(content="Second", metadata={"chunk_index": 0})]
    mock_extract.return_value.elements = [MM(metadata={"page_number": 1, "element_type": "text"})]

    # Second build: should do incremental update
    index_path = indexer.build(data_dir, "flow-test")
    assert mock_extract.call_count == 1  # only the new file
    assert MockBuilder.return_value.update_index.call_count == 1
    assert MockBuilder.return_value.build_index.call_count == 0

    # Manifest has both files
    manifest = json.loads((Path(index_path).parent / "manifest.json").read_text())
    assert "first.pdf" in manifest["files"]
    assert "second.docx" in manifest["files"]
```

**Step 2: Run test**

Run: `python -m pytest tests/test_watcher.py::test_full_incremental_flow_build_then_update -v`
Expected: PASS (if tasks 1-3 are done correctly)

**Step 3: Run full test suite**

Run: `python -m pytest tests/ -v --timeout=30`
Expected: All PASS

**Step 4: Commit**

```bash
git add tests/test_watcher.py
git commit -m "test: integration test for full incremental reindexing flow"
```
