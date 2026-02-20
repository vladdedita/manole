"""Ingestion pipeline: kreuzberg extract -> chunk -> LeannBuilder index."""
import json
import threading
from pathlib import Path

from kreuzberg import (
    ExtractionConfig,
    ChunkingConfig,
    OutputFormat,
    ResultFormat,
    detect_mime_type_from_path,
    extract_file_sync,
)
from leann import LeannBuilder
from watchfiles import watch



class KreuzbergIndexer:
    """Ingestion pipeline: kreuzberg extract -> chunk -> LeannBuilder index."""

    SKIP_MIME_PREFIXES = ("image/",)
    CHUNK_MAX_CHARS = 512
    CHUNK_MAX_OVERLAP = 50
    WATCHER_DEBOUNCE_MS = 500

    def __init__(self, embedding_model: str = "facebook/contriever"):
        self.embedding_model = embedding_model

    def _make_extraction_config(self) -> ExtractionConfig:
        """Create the standard extraction config for all pipelines."""
        return ExtractionConfig(
            output_format=OutputFormat.MARKDOWN,
            result_format=ResultFormat.ELEMENT_BASED,
            include_document_structure=True,
            chunking=ChunkingConfig(
                max_chars=self.CHUNK_MAX_CHARS,
                max_overlap=self.CHUNK_MAX_OVERLAP,
            ),
        )

    def _make_builder(self) -> LeannBuilder:
        """Create a LeannBuilder with standard settings."""
        return LeannBuilder(
            backend_name="hnsw",
            embedding_model=self.embedding_model,
            is_recompute=False,
        )

    @staticmethod
    def _extract_element_metadata(elements: list, index: int) -> tuple:
        """Extract page_number and element_type from an element at the given index.

        Returns (page_number, element_type) tuple. Both may be None.
        """
        if index >= len(elements):
            return None, None
        elem = elements[index]
        if isinstance(elem, dict):
            elem_meta = elem.get("metadata") or {}
            element_type = elem.get("element_type") or elem_meta.get("element_type")
        else:
            elem_meta = elem.metadata or {}
            element_type = elem_meta.get("element_type")
        page_number = elem_meta.get("page_number")
        return page_number, element_type

    def _index_path(self, index_name: str) -> str:
        """Return the standard index file path for a given index name."""
        return str(Path(".leann") / "indexes" / index_name / "documents.leann")

    def _manifest_path(self, index_name: str) -> Path:
        """Return the path to the manifest.json for a given index."""
        return Path(".leann") / "indexes" / index_name / "manifest.json"

    def _write_manifest(self, index_name: str, file_records: dict) -> None:
        """Write manifest.json with version and file records."""
        manifest = {"version": 1, "files": file_records}
        path = self._manifest_path(index_name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(manifest, indent=2))

    def _read_manifest(self, index_name: str) -> dict | None:
        """Read manifest.json for an index. Returns None if missing."""
        path = self._manifest_path(index_name)
        if not path.exists():
            return None
        return json.loads(path.read_text())

    def build(self, data_dir: Path, index_name: str, force: bool = False) -> str:
        """Build a LEANN index from all supported files in data_dir.

        Returns the index path.
        """
        data_dir = Path(data_dir)
        index_path = self._index_path(index_name)

        # Skip rebuild if index already exists
        if not force and Path(f"{index_path}.meta.json").exists():
            manifest = self._read_manifest(index_name)
            if manifest is not None:
                self.incremental_update(data_dir, index_name)
                return index_path
            print(f"Index '{index_name}' already exists, skipping build (use force=True to rebuild)")
            return index_path

        config = self._make_extraction_config()
        builder = self._make_builder()

        total_chunks = 0
        files_processed = 0
        files_skipped = 0
        file_records = {}

        for file_path in sorted(data_dir.rglob("*")):
            if not file_path.is_file():
                continue
            try:
                mime = detect_mime_type_from_path(str(file_path))
            except Exception:
                files_skipped += 1
                continue
            if any(mime.startswith(p) for p in self.SKIP_MIME_PREFIXES):
                files_skipped += 1
                continue

            print(f"  Extracting: {file_path.name} ({mime})")
            try:
                result = extract_file_sync(str(file_path), config=config)
            except Exception as exc:
                print(f"  FAILED: {file_path.name}: {exc}")
                files_skipped += 1
                continue

            if not result.chunks:
                print(f"  No chunks: {file_path.name}")
                continue

            files_processed += 1
            file_chunk_count = 0
            elements = result.elements or []
            for i, chunk in enumerate(result.chunks):
                chunk_index = chunk.metadata.get("chunk_index", i)
                page_number, element_type = self._extract_element_metadata(elements, i)

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
                file_chunk_count += 1

            rel_path = str(file_path.relative_to(data_dir))
            file_records[rel_path] = {
                "mtime": file_path.stat().st_mtime,
                "chunks": file_chunk_count,
            }

        if total_chunks == 0:
            raise RuntimeError(
                "No chunks extracted from any file. Cannot build empty index."
            )

        print(f"  {files_processed} files -> {total_chunks} chunks ({files_skipped} skipped)")
        builder.build_index(index_path)
        self._write_manifest(index_name, file_records)
        return index_path

    def _extract_and_append(
        self, data_dir: Path, files_to_process: list[Path], index_path: str, file_records: dict
    ) -> int:
        """Extract files and append chunks to an existing index via update_index().

        Updates file_records in-place with mtime and chunk count for each processed file.
        Returns total number of chunks appended.
        """
        config = self._make_extraction_config()
        builder = self._make_builder()

        total_chunks = 0

        for file_path in files_to_process:
            print(f"  Extracting: {file_path.name}")
            try:
                result = extract_file_sync(str(file_path), config=config)
            except Exception as exc:
                print(f"  FAILED: {file_path.name}: {exc}")
                continue

            if not result.chunks:
                print(f"  No chunks: {file_path.name}")
                continue

            file_chunk_count = 0
            elements = result.elements or []
            for i, chunk in enumerate(result.chunks):
                chunk_index = chunk.metadata.get("chunk_index", i)
                page_number, element_type = self._extract_element_metadata(elements, i)

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
                file_chunk_count += 1

            rel_path = str(file_path.relative_to(data_dir))
            file_records[rel_path] = {
                "mtime": file_path.stat().st_mtime,
                "chunks": file_chunk_count,
            }

        if total_chunks > 0:
            builder.update_index(index_path)

        return total_chunks

    def extract_and_append_file(self, file_path: Path, data_dir: Path, index_name: str) -> None:
        """Extract a single file and append its chunks to an existing index.

        Reads the current manifest (or creates a new one), delegates to
        _extract_and_append() for extraction and update_index(), then
        writes the updated manifest. This is the entry point the file
        watcher will call.
        """
        manifest = self._read_manifest(index_name) or {"version": 1, "files": {}}
        file_records = manifest.get("files", {})
        index_path = self._index_path(index_name)
        self._extract_and_append(Path(data_dir), [Path(file_path)], index_path, file_records)
        self._write_manifest(index_name, file_records)

    def incremental_update(self, data_dir: Path, index_name: str) -> str:
        """Incrementally update an existing index by processing only new/modified files.

        Compares file mtimes against the manifest to detect changes.
        Returns the index path.
        """
        data_dir = Path(data_dir)
        index_path = self._index_path(index_name)

        manifest = self._read_manifest(index_name)
        existing_files = manifest["files"] if manifest else {}

        # Walk directory and find files that need processing
        files_to_process = []
        for file_path in sorted(data_dir.rglob("*")):
            if not file_path.is_file():
                continue
            try:
                mime = detect_mime_type_from_path(str(file_path))
            except Exception:
                continue
            if any(mime.startswith(p) for p in self.SKIP_MIME_PREFIXES):
                continue

            rel_path = str(file_path.relative_to(data_dir))
            current_mtime = file_path.stat().st_mtime

            if rel_path in existing_files and existing_files[rel_path]["mtime"] == current_mtime:
                continue  # Unchanged

            files_to_process.append(file_path)

        # Copy existing file records to preserve unchanged entries
        file_records = dict(existing_files)

        if files_to_process:
            self._extract_and_append(data_dir, files_to_process, index_path, file_records)
            self._write_manifest(index_name, file_records)

        return index_path

    def start_watcher(
        self, data_dir: Path, index_name: str, stop_event: threading.Event
    ) -> threading.Thread:
        """Start a daemon thread that watches data_dir for file changes.

        On file creation/modification, calls extract_and_append_file() for
        supported file types. Image files are skipped. Setting stop_event
        causes the watcher thread to exit cleanly.

        Returns the daemon thread (already started).
        """
        data_dir = Path(data_dir)

        def _watch_loop():
            for changes in watch(data_dir, stop_event=stop_event, debounce=self.WATCHER_DEBOUNCE_MS):
                for _change_type, path_str in changes:
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
                        self.extract_and_append_file(file_path, data_dir, index_name)
                    except Exception as exc:
                        print(f"  Watcher: failed to index {file_path.name}: {exc}")

        thread = threading.Thread(target=_watch_loop, daemon=True, name="file-watcher")
        thread.start()
        return thread
