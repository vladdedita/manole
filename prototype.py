"""
Prototype: LEANN indexing and search exploration.

Step 1: Index file contents, search by semantic query.
Step 2: Add metadata enrichment, compare results.
"""

import os
import time
from pathlib import Path
from leann import LeannBuilder, LeannSearcher


DATA_DIR = Path("./test_data")
INDEX_DIR = Path("./indexes")
CONTENT_INDEX = str(INDEX_DIR / "content_only.leann")
METADATA_INDEX = str(INDEX_DIR / "with_metadata.leann")

# File types we can read as text
TEXT_EXTENSIONS = {".txt", ".md", ".py", ".json", ".yaml", ".yml", ".csv", ".log"}


def discover_files(root: Path) -> list[Path]:
    """Walk directory tree and collect readable files."""
    files = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in TEXT_EXTENSIONS:
            files.append(path)
    return files


def read_file_content(path: Path) -> str:
    """Read file content, return empty string on failure."""
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        print(f"  [WARN] Could not read {path}: {e}")
        return ""


def get_file_metadata(path: Path) -> dict:
    """Extract file metadata."""
    stat = path.stat()
    return {
        "name": path.name,
        "path": str(path),
        "extension": path.suffix,
        "size_bytes": stat.st_size,
        "modified": time.ctime(stat.st_mtime),
        "parent_dir": path.parent.name,
    }


# ── Step 1: Content-only indexing ────────────────────────────────────

def build_content_index():
    """Index files using only their text content."""
    print("=" * 60)
    print("STEP 1: Content-only indexing")
    print("=" * 60)

    files = discover_files(DATA_DIR)
    print(f"\nFound {len(files)} files to index:")
    for f in files:
        print(f"  - {f}")

    INDEX_DIR.mkdir(exist_ok=True)

    builder = LeannBuilder(backend_name="hnsw")

    print("\nIndexing file contents...")
    for f in files:
        content = read_file_content(f)
        if content.strip():
            builder.add_text(content)
            print(f"  [OK] Indexed: {f.name} ({len(content)} chars)")

    print("\nBuilding index...")
    t0 = time.time()
    builder.build_index(CONTENT_INDEX)
    print(f"Index built in {time.time() - t0:.2f}s")
    print(f"Index saved to: {CONTENT_INDEX}")


def search_content_index(queries: list[str]):
    """Run queries against the content-only index."""
    print("\n" + "-" * 60)
    print("Searching content-only index")
    print("-" * 60)

    searcher = LeannSearcher(CONTENT_INDEX)

    for query in queries:
        print(f"\nQuery: \"{query}\"")
        results = searcher.search(query, top_k=3)
        print(f"Results ({len(results)} matches):")
        for i, result in enumerate(results):
            # Print what we get back — exploring the result structure
            print(f"  [{i+1}] {result}")
        print()


# ── Step 2: Metadata-enriched indexing ───────────────────────────────

def build_metadata_index():
    """Index files with metadata prepended to content."""
    print("=" * 60)
    print("STEP 2: Metadata-enriched indexing")
    print("=" * 60)

    files = discover_files(DATA_DIR)

    INDEX_DIR.mkdir(exist_ok=True)

    builder = LeannBuilder(backend_name="hnsw")

    print("\nIndexing with metadata enrichment...")
    for f in files:
        content = read_file_content(f)
        if not content.strip():
            continue

        meta = get_file_metadata(f)
        # Prepend metadata as structured context
        enriched = (
            f"File: {meta['name']}\n"
            f"Path: {meta['path']}\n"
            f"Type: {meta['extension']}\n"
            f"Folder: {meta['parent_dir']}\n"
            f"Size: {meta['size_bytes']} bytes\n"
            f"Modified: {meta['modified']}\n"
            f"---\n"
            f"{content}"
        )
        builder.add_text(enriched)
        print(f"  [OK] Indexed: {f.name} (content + metadata)")

    print("\nBuilding index...")
    t0 = time.time()
    builder.build_index(METADATA_INDEX)
    print(f"Index built in {time.time() - t0:.2f}s")
    print(f"Index saved to: {METADATA_INDEX}")


def search_metadata_index(queries: list[str]):
    """Run queries against the metadata-enriched index."""
    print("\n" + "-" * 60)
    print("Searching metadata-enriched index")
    print("-" * 60)

    searcher = LeannSearcher(METADATA_INDEX)

    for query in queries:
        print(f"\nQuery: \"{query}\"")
        results = searcher.search(query, top_k=3)
        print(f"Results ({len(results)} matches):")
        for i, result in enumerate(results):
            print(f"  [{i+1}] {result}")
        print()


# ── Main ─────────────────────────────────────────────────────────────

TEST_QUERIES = [
    "Where is the documentation about Kubernetes?",
    "Find files with financial data",
    "What are the key topics in machine learning research?",
    "Show me design diagrams for microservices",
    "Where did I save the AI agent project ideas?",
    "Find documents with client email addresses",
    "Summarize notes about the Manole project",
    "recipe",
]


if __name__ == "__main__":
    # Step 1
    build_content_index()
    search_content_index(TEST_QUERIES)

    # Step 2
    build_metadata_index()
    search_metadata_index(TEST_QUERIES)

    # Compare
    print("=" * 60)
    print("COMPARISON COMPLETE")
    print("Check the results above to see how metadata enrichment")
    print("affects search relevance and ranking.")
    print("=" * 60)
