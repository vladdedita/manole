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


class FileReader:
    """Wraps Docling's DocumentConverter for on-demand text extraction."""

    # Extensions Docling cannot handle — read as plain text instead.
    _PLAIN_TEXT_SUFFIXES = {".txt", ".log", ".cfg", ".ini", ".toml", ".yaml", ".yml", ".json", ".py", ".sh"}

    def __init__(self, max_chars: int = 4000):
        self.max_chars = max_chars
        self._converter = None

    def read(self, path: str) -> str:
        """Extract text from any file. Returns text or error message."""
        file_path = Path(path)
        if not file_path.exists():
            return f"File not found: {path}"

        try:
            if file_path.suffix.lower() in self._PLAIN_TEXT_SUFFIXES:
                text = file_path.read_text(errors="replace")
            else:
                converter = self._get_converter()
                result = converter.convert(str(file_path))
                text = result.document.export_to_markdown()
            if not text or not text.strip():
                return f"No text content extracted from {file_path.name}"
            return text[:self.max_chars]
        except Exception as e:
            return f"Failed to read {file_path.name}: {e}"

    def _get_converter(self):
        """Lazy-load Docling converter on first use."""
        if self._converter is None:
            from docling.document_converter import DocumentConverter
            self._converter = DocumentConverter()
        return self._converter
