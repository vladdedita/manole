"""FileReader — on-demand text extraction via Docling."""
from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from docling.document_converter import DocumentConverter


@runtime_checkable
class TextExtractor(Protocol):
    """Port: extracts text content from a file path."""

    def extract(self, path: Path) -> str: ...


class DoclingExtractor:
    """Adapter: extracts text via Docling's DocumentConverter."""

    def __init__(self) -> None:
        self._converter: DocumentConverter | None = None

    def extract(self, path: Path) -> str:
        """Convert a document to markdown text. Raises on failure."""
        converter = self._get_converter()
        result = converter.convert(str(path))
        return result.document.export_to_markdown()

    def _get_converter(self) -> DocumentConverter:
        """Lazy-load Docling converter on first use."""
        if self._converter is None:
            self._converter = DocumentConverter()
        return self._converter


class KreuzbergExtractor:
    """Adapter: extracts text via kreuzberg's async extract_file."""

    def __init__(self) -> None:
        import importlib

        try:
            self._kreuzberg = importlib.import_module("kreuzberg")
        except ImportError:
            raise ImportError(
                "kreuzberg is not installed. Install it with: pip install kreuzberg"
            )

    def extract(self, path: Path) -> str:
        """Extract text from a file using kreuzberg. Raises on failure."""
        import asyncio

        result = asyncio.run(self._kreuzberg.extract_file(str(path)))
        return result.content


class FileReader:
    """On-demand text extraction with pluggable extractor backend."""

    # Extensions handled as plain text — no extractor needed.
    _PLAIN_TEXT_SUFFIXES = {".txt", ".log", ".cfg", ".ini", ".toml", ".yaml", ".yml", ".json", ".py", ".sh"}

    _BACKENDS: dict[str, type] = {}  # populated after class body

    def __init__(self, max_chars: int = 4000, extractor: TextExtractor | None = None):
        self.max_chars = max_chars
        self._extractor = extractor if extractor is not None else DoclingExtractor()

    @classmethod
    def from_backend(cls, backend: str, *, max_chars: int = 4000) -> "FileReader":
        """Create a FileReader with the named backend extractor.

        Valid backends: "docling", "kreuzberg".
        Raises ValueError for unknown backends.
        Raises ImportError if kreuzberg is not installed.
        """
        extractor_cls = cls._BACKENDS.get(backend)
        if extractor_cls is None:
            raise ValueError(
                f"Unknown backend '{backend}'. Choose from: {', '.join(sorted(cls._BACKENDS))}"
            )
        return cls(max_chars=max_chars, extractor=extractor_cls())

    @property
    def _converter(self):
        """Backward-compat: expose underlying converter from DoclingExtractor."""
        if isinstance(self._extractor, DoclingExtractor):
            return self._extractor._converter
        return None

    def read(self, path: str) -> str:
        """Extract text from any file. Returns text or error message."""
        file_path = Path(path)
        if not file_path.exists():
            return f"File not found: {path}"

        try:
            if file_path.suffix.lower() in self._PLAIN_TEXT_SUFFIXES:
                text = file_path.read_text(errors="replace")
            else:
                text = self._extractor.extract(file_path)
            if not text or not text.strip():
                return f"No text content extracted from {file_path.name}"
            return text[:self.max_chars]
        except Exception as e:
            return f"Failed to read {file_path.name}: {e}"


# Register available backends
FileReader._BACKENDS = {
    "docling": DoclingExtractor,
    "kreuzberg": KreuzbergExtractor,
}
