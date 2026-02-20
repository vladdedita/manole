"""Ingestion pipeline: kreuzberg extract -> chunk -> LeannBuilder index."""
import logging
from pathlib import Path

from kreuzberg import (
    ExtractionConfig,
    ChunkingConfig,
    OutputFormat,
    ResultFormat,
    extract_file_sync,
)
from leann import LeannBuilder

log = logging.getLogger(__name__)


class KreuzbergIndexer:
    """Ingestion pipeline: kreuzberg extract -> chunk -> LeannBuilder index."""

    SUPPORTED_EXTENSIONS = {
        ".pdf", ".docx", ".doc", ".pptx", ".xlsx",
        ".html", ".htm", ".epub",
    }

    def __init__(self, embedding_model: str = "facebook/contriever"):
        self.embedding_model = embedding_model

    def build(self, data_dir: Path, index_name: str, force: bool = False) -> str:
        """Build a LEANN index from all supported files in data_dir.

        Returns the index path.
        """
        data_dir = Path(data_dir)
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

        for file_path in sorted(data_dir.rglob("*")):
            if not file_path.is_file():
                continue
            if file_path.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
                continue

            try:
                result = extract_file_sync(str(file_path), config=config)
            except Exception as exc:
                log.warning("Failed to extract %s: %s", file_path.name, exc)
                continue

            if not result.chunks:
                continue

            # Build element metadata lookup by index
            elements = result.elements or []
            for i, chunk in enumerate(result.chunks):
                chunk_index = chunk.metadata.get("chunk_index", i)

                # Get element metadata if available
                page_number = None
                element_type = None
                if i < len(elements):
                    elem = elements[i]
                    # kreuzberg returns elements as dicts (not objects)
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

        if total_chunks == 0:
            raise RuntimeError(
                "No chunks extracted from any file. Cannot build empty index."
            )

        index_path = str(
            Path(".leann") / "indexes" / index_name / "documents.leann"
        )
        builder.build_index(index_path)

        log.info("Built index %s with %d chunks", index_name, total_chunks)
        return index_path
