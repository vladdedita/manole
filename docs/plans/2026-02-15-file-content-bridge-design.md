# File-Content Bridge Design

## Problem

When a user asks "any macbook invoice?", the system fails to find `macbook_ssd.pdf` because:

1. **Vector search misses it** — the OCR'd chunk text is Romanian invoice boilerplate ("Seria: ROWHP, Nr anexa document...") that doesn't semantically match "macbook invoice"
2. **Filename carries the meaning** — `macbook_ssd.pdf` in `Cheltuieli/2025/apr_2025/` clearly relates to the query, but semantic search only looks at chunk text
3. **No file reading capability** — even if the agent found the file by name (via `grep_files`), it can't read what's inside

## Solution: Filename Fallback Inside `semantic_search`

**One new capability, zero new agent tools.**

When chunk-based search finds no relevant results, `semantic_search` automatically:

1. Extracts keywords from the query
2. Greps filenames for those keywords
3. Extracts text from matching files using Docling
4. Runs the same map-filter on extracted text
5. Returns facts from file reading

The agent sees no change — `semantic_search` just returns better results.

## Components

### 1. `FileReader` (new, `file_reader.py`)

Wraps Docling's `DocumentConverter` for on-demand text extraction.

```python
class FileReader:
    def __init__(self, max_chars=4000):
        self.max_chars = max_chars
        self._converter = None  # lazy-loaded

    def read(self, path: str) -> str:
        """Extract text from any file. Returns text or error."""
        # Uses docling.DocumentConverter
        # Handles: PDF, DOCX, images (OCR), markdown, plain text
        # Truncates to max_chars
```

- Lazy-loads Docling converter on first use (avoids import cost when not needed)
- Truncates extracted text to 4000 chars (enough for invoice data, fits in context)
- Handles all file types in the test_data: PDF, HEIC, PNG, JPEG, MD, TXT

### 2. `Searcher` Changes (`searcher.py`)

Constructor gains optional `FileReader` and `ToolBox`:

```python
class Searcher:
    def __init__(self, leann_searcher, model, file_reader=None, toolbox=None, debug=False):
```

`search_and_extract()` gains a filename fallback after map-filter returns no results:

```python
def search_and_extract(self, query, top_k=5):
    # ... existing vector search + map-filter ...

    if not facts_by_source and self.file_reader and self.toolbox:
        return self._filename_fallback(query)

    # ... existing formatting ...
```

`_filename_fallback(query)`:
1. Extract keywords from query (split, filter stopwords, words > 2 chars)
2. For each keyword, grep filenames via `self.toolbox.grep(keyword)`
3. For matching files (max 3), extract text via `self.file_reader.read(path)`
4. Run map-filter (`_extract_facts`) on extracted text
5. Return formatted facts or "no relevant results"

### 3. Dependency: `docling`

Add to `pyproject.toml`:
```toml
dependencies = [
    "docling>=2.70",
    ...
]
```

Docling handles: PDF, DOCX, PPTX, XLSX, HTML, images (OCR), markdown, plain text.

## Data Flow

```
semantic_search("macbook invoice")
  │
  ├── 1. Vector search (LEANN) → chunks
  ├── 2. Score pre-filter (0.85 threshold)
  ├── 3. Map-filter (LLM) → relevant facts
  │
  ├── If facts found → return them
  │
  ├── 4. FILENAME FALLBACK (new)
  │   ├── Extract keywords: ["macbook", "invoice"]
  │   ├── Grep filenames → macbook_ssd.pdf
  │   ├── Docling extract → full text
  │   ├── Map-filter with query → facts
  │   └── Return facts from file reading
  │
  └── If still nothing → "no relevant results"
```

## What Doesn't Change

- Agent loop (`agent.py`) — no changes
- Tool definitions (`tools.py`) — no new tools
- Router (`router.py`) — no changes
- Rewriter (`rewriter.py`) — no changes
- All 91 existing tests stay passing

## Query Flow Examples

### "any macbook invoice?"
1. Vector search → 5 chunks → map-filter → 0 relevant (meeting_notes.txt etc.)
2. Filename fallback: keywords ["macbook", "invoice"]
3. Grep "macbook" → `Cheltuieli/2025/apr_2025/macbook_ssd.pdf`
4. Docling extracts PDF text
5. Map-filter finds: Invoice #269100348828, Dante International S.A., etc.
6. Returns formatted facts

### "what's in the budget report?"
1. Vector search → relevant chunks found → facts extracted → returned
2. Filename fallback never triggered

### "any photos of the whiteboard?"
1. Vector search → nothing relevant
2. Filename fallback: grep "whiteboard" → `images/whiteboard_notes.png`
3. Docling OCR extracts text from image
4. Map-filter extracts facts → returned
