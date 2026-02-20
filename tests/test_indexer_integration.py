"""Integration test: kreuzberg pipeline build and search.

Step 03-01: kreuzberg-integration
Uses REAL kreuzberg extraction, REAL LeannBuilder, and REAL LeannSearcher.
No mocks — verifies the full pipeline from HTML files to search results.

Acceptance criteria:
- Test indexes a small fixture directory (2-3 files) via kreuzberg pipeline
- Test searches the resulting index and retrieves a relevant chunk
- Returned search result contains expected metadata fields
"""
import shutil
import tempfile
from pathlib import Path

import pytest

from indexer import KreuzbergIndexer
from leann import LeannSearcher

# Index is created relative to CWD; use a unique name to avoid collisions
_INDEX_NAME = "test-integration-kreuzberg"
_INDEX_DIR = Path(".leann") / "indexes" / _INDEX_NAME


@pytest.mark.integration
def test_kreuzberg_pipeline_indexes_html_files_and_returns_search_results_with_metadata():
    """Given a directory with 2 HTML fixture files,
    when the kreuzberg pipeline builds an index and a search is performed,
    then at least one relevant result is returned with source, file_name,
    file_type, and chunk_index metadata fields."""
    # Clean up any leftover index from a previous run
    if _INDEX_DIR.exists():
        shutil.rmtree(_INDEX_DIR)

    with tempfile.TemporaryDirectory() as tmpdir:
        # --- Arrange: create fixture files with distinct content ---
        fixture_dir = Path(tmpdir) / "docs"
        fixture_dir.mkdir()

        (fixture_dir / "report.html").write_text(
            "<h1>Quarterly Report</h1>"
            "<p>Revenue grew 25% in Q3 2025. Total revenue reached 50 million dollars. "
            "The growth was driven by strong performance in the enterprise segment.</p>"
        )
        (fixture_dir / "meeting.html").write_text(
            "<h1>Meeting Notes</h1>"
            "<p>The engineering team discussed the new kreuzberg integration for document processing. "
            "Migration from the old parser is scheduled for next sprint.</p>"
        )

        try:
            # --- Act: build index ---
            indexer = KreuzbergIndexer()
            index_path = indexer.build(fixture_dir, _INDEX_NAME)

            # --- Assert: index was created on disk (leann uses prefix-based files) ---
            meta_file = Path(f"{index_path}.meta.json")
            assert meta_file.exists(), f"Index meta file not found at {meta_file}"

            # --- Act: search the index ---
            searcher = LeannSearcher(index_path, enable_warmup=False)
            try:
                results = searcher.search("revenue growth quarterly report", top_k=3)

                # --- Assert: search returns results ---
                assert len(results) > 0, "Search returned no results"

                # --- Assert: first result has expected structure ---
                result = results[0]
                assert result.score > 0, "SearchResult score should be positive"
                assert len(result.text) > 0, "SearchResult text is empty"

                # --- Assert: metadata contains expected fields from the indexer ---
                metadata = result.metadata
                expected_keys = {"source", "file_name", "file_type", "chunk_index"}
                missing_keys = expected_keys - set(metadata.keys())
                assert not missing_keys, (
                    f"Metadata missing keys: {missing_keys}. "
                    f"Present keys: {list(metadata.keys())}"
                )

                # --- Assert: metadata values are sensible ---
                assert metadata["file_type"] in ("html", "htm")
                assert metadata["file_name"].endswith(".html")

            finally:
                searcher.cleanup()

        finally:
            # Clean up index directory created in CWD
            if _INDEX_DIR.exists():
                shutil.rmtree(_INDEX_DIR)
