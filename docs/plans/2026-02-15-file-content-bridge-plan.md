# File-Content Bridge Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** When semantic search finds no relevant results, automatically grep filenames for query keywords, extract text from matching files via Docling, and run map-filter to return facts.

**Architecture:** `FileReader` wraps Docling for on-demand text extraction. `Searcher` gains a `_filename_fallback()` method called when chunk-based search returns nothing relevant. `ToolBox` gains a `grep_paths()` method returning actual `Path` objects (the existing `grep()` returns formatted strings). No new agent tools — the fallback is transparent inside `semantic_search`.

**Tech Stack:** Python 3.13, Docling (document extraction), pytest, existing llama-cpp-python model for map-filter.

---

### Task 1: Add `docling` Dependency

**Files:**
- Modify: `pyproject.toml`

**Step 1: Add docling to dependencies**

In `pyproject.toml`, add `"docling>=2.70"` to the `dependencies` list:

```toml
dependencies = [
    "docling>=2.70",
    "leann>=0.3.6",
    "llama-cpp-python>=0.3.0",
    "torchvision>=0.25.0",
    "transformers>=4.55",
]
```

**Step 2: Install**

Run: `uv sync`
Expected: docling installs successfully

**Step 3: Verify import works**

Run: `uv run python -c "from docling.document_converter import DocumentConverter; print('ok')"`
Expected: `ok`

**Step 4: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "feat: add docling dependency for file text extraction"
```

---

### Task 2: Create `FileReader` with Tests

**Files:**
- Create: `file_reader.py`
- Create: `tests/test_file_reader.py`

**Step 1: Write the failing tests**

Create `tests/test_file_reader.py`:

```python
"""Tests for FileReader — on-demand text extraction via Docling."""
import tempfile
from pathlib import Path

from file_reader import FileReader


def test_read_text_file():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write("Hello world, this is a test document.")
        f.flush()
        reader = FileReader()
        text = reader.read(f.name)
    assert "Hello world" in text


def test_read_markdown_file():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
        f.write("# Heading\n\nSome markdown content here.")
        f.flush()
        reader = FileReader()
        text = reader.read(f.name)
    assert "Heading" in text
    assert "markdown content" in text


def test_read_nonexistent_file():
    reader = FileReader()
    text = reader.read("/tmp/does_not_exist_12345.txt")
    assert "error" in text.lower() or "not found" in text.lower() or "failed" in text.lower()


def test_truncation():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write("x" * 10000)
        f.flush()
        reader = FileReader(max_chars=100)
        text = reader.read(f.name)
    assert len(text) <= 100


def test_lazy_converter_init():
    """Converter should not be loaded until first read()."""
    reader = FileReader()
    assert reader._converter is None
```

**Step 2: Run tests to verify they fail**

Run: `uv run python -m pytest tests/test_file_reader.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'file_reader'`

**Step 3: Write minimal implementation**

Create `file_reader.py`:

```python
"""FileReader — on-demand text extraction via Docling."""
from pathlib import Path


class FileReader:
    """Wraps Docling's DocumentConverter for on-demand text extraction."""

    def __init__(self, max_chars: int = 4000):
        self.max_chars = max_chars
        self._converter = None

    def read(self, path: str) -> str:
        """Extract text from any file. Returns text or error message."""
        file_path = Path(path)
        if not file_path.exists():
            return f"File not found: {path}"

        try:
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
```

**Step 4: Run tests to verify they pass**

Run: `uv run python -m pytest tests/test_file_reader.py -v`
Expected: 5 passed

**Step 5: Commit**

```bash
git add file_reader.py tests/test_file_reader.py
git commit -m "feat: add FileReader for on-demand text extraction via Docling"
```

---

### Task 3: Add `grep_paths()` to ToolBox

**Files:**
- Modify: `toolbox.py:82-88`
- Modify: `tests/test_toolbox.py`

The existing `toolbox.grep()` returns a formatted string like `"Files matching 'macbook':\n  - path/to/file"`. The Searcher needs actual `Path` objects. Add a `grep_paths()` method.

**Step 1: Write the failing test**

Add to `tests/test_toolbox.py`:

```python
def test_grep_paths_returns_path_objects():
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "invoice_001.pdf").write_text("inv1")
        (Path(tmp) / "invoice_002.pdf").write_text("inv2")
        (Path(tmp) / "readme.txt").write_text("readme")
        tb = ToolBox(tmp)
        paths = tb.grep_paths("invoice")
    assert len(paths) == 2
    assert all(isinstance(p, Path) for p in paths)
    names = {p.name for p in paths}
    assert "invoice_001.pdf" in names
    assert "invoice_002.pdf" in names


def test_grep_paths_no_match():
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "readme.txt").write_text("hello")
        tb = ToolBox(tmp)
        paths = tb.grep_paths("nonexistent")
    assert paths == []


def test_grep_paths_limit():
    with tempfile.TemporaryDirectory() as tmp:
        for i in range(10):
            (Path(tmp) / f"doc_{i}.txt").write_text(f"doc {i}")
        tb = ToolBox(tmp)
        paths = tb.grep_paths("doc", limit=3)
    assert len(paths) == 3
```

**Step 2: Run tests to verify they fail**

Run: `uv run python -m pytest tests/test_toolbox.py::test_grep_paths_returns_path_objects -v`
Expected: FAIL with `AttributeError: 'ToolBox' object has no attribute 'grep_paths'`

**Step 3: Write minimal implementation**

Add to `toolbox.py` after the `grep()` method (after line 88):

```python
    def grep_paths(self, pattern: str, limit: int = 20) -> list[Path]:
        """Find files by name pattern. Returns Path objects."""
        files = [f for f in self.root.rglob("*") if f.is_file() and not f.name.startswith(".")]
        matches = [f for f in files if pattern.lower() in f.name.lower()]
        return matches[:limit]
```

**Step 4: Run tests to verify they pass**

Run: `uv run python -m pytest tests/test_toolbox.py -v`
Expected: All tests pass (10 existing + 3 new = 13 passed)

**Step 5: Commit**

```bash
git add toolbox.py tests/test_toolbox.py
git commit -m "feat: add grep_paths() to ToolBox returning Path objects"
```

---

### Task 4: Add Keyword Extraction Helper

**Files:**
- Modify: `searcher.py`
- Modify: `tests/test_searcher.py`

The filename fallback needs to extract keywords from the user's query. Simple approach: split, filter stopwords, keep words > 2 chars.

**Step 1: Write the failing tests**

Add to `tests/test_searcher.py`:

```python
from searcher import Searcher, MAP_SYSTEM, extract_keywords


def test_extract_keywords_basic():
    assert extract_keywords("any macbook invoice?") == ["macbook", "invoice"]


def test_extract_keywords_filters_stopwords():
    result = extract_keywords("what is the file size")
    assert "what" not in result
    assert "the" not in result
    assert "file" in result
    assert "size" in result


def test_extract_keywords_lowercase():
    assert extract_keywords("MacBook PDF") == ["macbook", "pdf"]


def test_extract_keywords_short_words_removed():
    result = extract_keywords("is it an ok file")
    assert "is" not in result
    assert "it" not in result
    assert "an" not in result
    assert "ok" not in result
    assert "file" in result
```

Also update the import at the top of the test file from:
```python
from searcher import Searcher, MAP_SYSTEM
```
to:
```python
from searcher import Searcher, MAP_SYSTEM, extract_keywords
```

**Step 2: Run tests to verify they fail**

Run: `uv run python -m pytest tests/test_searcher.py::test_extract_keywords_basic -v`
Expected: FAIL with `ImportError: cannot import name 'extract_keywords'`

**Step 3: Write minimal implementation**

Add to `searcher.py` before the `Searcher` class (after line 24, after `MAX_FACTS_PER_CHUNK`):

```python
_STOPWORDS = frozenset({
    "a", "an", "the", "is", "was", "are", "were", "be", "been",
    "do", "does", "did", "has", "have", "had", "it", "its",
    "of", "for", "in", "on", "to", "at", "by", "my", "me",
    "what", "when", "where", "how", "who", "which", "any",
    "and", "or", "not", "no", "but", "if", "so", "can",
    "all", "each", "every", "this", "that", "there", "here",
    "from", "with", "about", "into", "over", "after", "before",
    "show", "find", "get", "tell", "give", "list",
})


def extract_keywords(query: str) -> list[str]:
    """Extract searchable keywords from a query string."""
    words = query.lower().replace("?", "").replace("!", "").replace(".", "").split()
    return [w for w in words if w not in _STOPWORDS and len(w) > 2]
```

**Step 4: Run tests to verify they pass**

Run: `uv run python -m pytest tests/test_searcher.py -v`
Expected: All tests pass (11 existing + 4 new = 15 passed)

**Step 5: Commit**

```bash
git add searcher.py tests/test_searcher.py
git commit -m "feat: add extract_keywords helper for filename fallback"
```

---

### Task 5: Add Filename Fallback to Searcher

**Files:**
- Modify: `searcher.py:27-66`
- Modify: `tests/test_searcher.py`

This is the core change: when `search_and_extract()` finds no relevant facts from chunks, it falls back to grepping filenames, reading matching files, and running map-filter on the extracted text.

**Step 1: Write the failing tests**

Add to `tests/test_searcher.py`:

```python
from pathlib import Path
from file_reader import FileReader


class FakeFileReader:
    """Fake FileReader that returns predefined text for any path."""
    def __init__(self, text="Invoice #999 from Dante International, Amount: $500"):
        self.text = text
        self.read_calls = []

    def read(self, path):
        self.read_calls.append(path)
        return self.text


class FakeToolBox:
    """Fake ToolBox with grep_paths support."""
    def __init__(self, paths=None):
        self.paths = paths or []

    def grep_paths(self, pattern, limit=3):
        return [p for p in self.paths if pattern.lower() in p.name.lower()][:limit]


def test_filename_fallback_when_chunks_irrelevant():
    """When all chunks are irrelevant, fallback greps filenames and reads files."""
    results = _make_results("unrelated meeting notes")
    model = _make_model([
        # First call: map-filter on chunk → irrelevant
        json.dumps({"relevant": False, "facts": []}),
        # Second call: map-filter on file text → relevant
        json.dumps({"relevant": True, "facts": ["Invoice #999", "Dante International"]}),
    ])
    leann = FakeLeann(results)
    file_reader = FakeFileReader()
    toolbox = FakeToolBox(paths=[Path("/data/macbook_ssd.pdf")])

    searcher = Searcher(leann, model, file_reader=file_reader, toolbox=toolbox)
    output = searcher.search_and_extract("macbook invoice")

    assert "Invoice #999" in output
    assert "macbook_ssd.pdf" in output
    assert len(file_reader.read_calls) == 1


def test_filename_fallback_no_matching_files():
    """When no filenames match keywords, returns standard no-results message."""
    results = _make_results("unrelated text")
    model = _make_model([
        json.dumps({"relevant": False, "facts": []}),
    ])
    leann = FakeLeann(results)
    file_reader = FakeFileReader()
    toolbox = FakeToolBox(paths=[])  # no files match

    searcher = Searcher(leann, model, file_reader=file_reader, toolbox=toolbox)
    output = searcher.search_and_extract("macbook invoice")

    assert "none were relevant" in output.lower()


def test_filename_fallback_not_triggered_when_chunks_relevant():
    """Filename fallback should NOT run when chunk search finds results."""
    results = _make_results("Invoice #123 for MacBook Pro")
    model = _make_model([
        json.dumps({"relevant": True, "facts": ["Invoice #123", "MacBook Pro"]}),
    ])
    leann = FakeLeann(results)
    file_reader = FakeFileReader()
    toolbox = FakeToolBox(paths=[Path("/data/macbook_ssd.pdf")])

    searcher = Searcher(leann, model, file_reader=file_reader, toolbox=toolbox)
    output = searcher.search_and_extract("macbook invoice")

    assert "Invoice #123" in output
    assert len(file_reader.read_calls) == 0  # should NOT have read any files


def test_filename_fallback_without_file_reader():
    """Without file_reader, fallback is skipped gracefully."""
    results = _make_results("unrelated text")
    model = _make_model([
        json.dumps({"relevant": False, "facts": []}),
    ])
    leann = FakeLeann(results)

    searcher = Searcher(leann, model)  # no file_reader, no toolbox
    output = searcher.search_and_extract("macbook invoice")

    assert "none were relevant" in output.lower()


def test_filename_fallback_caps_at_3_files():
    """Fallback reads at most 3 matching files."""
    results = _make_results("unrelated")
    model = _make_model([
        json.dumps({"relevant": False, "facts": []}),
        # 3 file reads, each returns relevant facts
        json.dumps({"relevant": True, "facts": ["fact1"]}),
        json.dumps({"relevant": True, "facts": ["fact2"]}),
        json.dumps({"relevant": True, "facts": ["fact3"]}),
    ])
    leann = FakeLeann(results)
    file_reader = FakeFileReader()
    toolbox = FakeToolBox(paths=[
        Path(f"/data/macbook_{i}.pdf") for i in range(5)
    ])

    searcher = Searcher(leann, model, file_reader=file_reader, toolbox=toolbox)
    output = searcher.search_and_extract("macbook")

    assert len(file_reader.read_calls) == 3
```

**Step 2: Run tests to verify they fail**

Run: `uv run python -m pytest tests/test_searcher.py::test_filename_fallback_when_chunks_irrelevant -v`
Expected: FAIL with `TypeError: Searcher.__init__() got an unexpected keyword argument 'file_reader'`

**Step 3: Write implementation**

Modify `searcher.py`. Change the `Searcher.__init__` signature and add `_filename_fallback`:

Replace the `__init__` method (line 30-33):

```python
    def __init__(self, leann_searcher, model, file_reader=None, toolbox=None, debug: bool = False):
        self.leann = leann_searcher
        self.model = model
        self.file_reader = file_reader
        self.toolbox = toolbox
        self.debug = debug
```

In `search_and_extract`, replace the "no facts" block (lines 57-58):

```python
        if not facts_by_source:
            if self.file_reader and self.toolbox:
                return self._filename_fallback(query)
            return "Search returned results but none were relevant to the query."
```

Add the `_filename_fallback` method after `_extract_facts` (after line 105):

```python
    def _filename_fallback(self, query: str) -> str:
        """Grep filenames for query keywords, read matches, extract facts."""
        keywords = extract_keywords(query)
        if not keywords:
            return "Search returned results but none were relevant to the query."

        if self.debug:
            print(f"  [SEARCH] Filename fallback: keywords={keywords}")

        # Collect unique matching file paths across all keywords
        seen = set()
        matching_paths = []
        for keyword in keywords:
            for path in self.toolbox.grep_paths(keyword, limit=3):
                path_str = str(path)
                if path_str not in seen:
                    seen.add(path_str)
                    matching_paths.append(path)

        matching_paths = matching_paths[:3]  # cap total files

        if not matching_paths:
            if self.debug:
                print("  [SEARCH] Filename fallback: no matching files")
            return "Search returned results but none were relevant to the query."

        if self.debug:
            print(f"  [SEARCH] Filename fallback: reading {[p.name for p in matching_paths]}")

        # Read each file and run map-filter
        facts_by_source = {}
        for path in matching_paths:
            text = self.file_reader.read(str(path))
            if text.startswith("File not found") or text.startswith("Failed to read") or text.startswith("No text content"):
                if self.debug:
                    print(f"  [SEARCH] Filename fallback: {path.name}: {text[:80]}")
                continue

            # Create a fake chunk-like object for _extract_facts
            fake_chunk = type("Chunk", (), {
                "id": path.name,
                "text": text,
                "score": 0.0,
                "metadata": {"file_name": path.name, "file_path": str(path)},
            })()

            extracted = self._extract_facts(query, fake_chunk)
            if extracted["relevant"] and extracted["facts"]:
                facts_by_source.setdefault(path.name, []).extend(extracted["facts"])

        if not facts_by_source:
            return "Search returned results but none were relevant to the query."

        lines = []
        for source, facts in facts_by_source.items():
            lines.append(f"From {source}:")
            for fact in facts:
                lines.append(f"  - {fact}")
        return "\n".join(lines)
```

**Step 4: Run tests to verify they pass**

Run: `uv run python -m pytest tests/test_searcher.py -v`
Expected: All tests pass (15 from Task 4 + 5 new = 20 passed)

**Step 5: Run full test suite to check nothing broke**

Run: `uv run python -m pytest tests/ -v`
Expected: All tests pass (existing 91 + new tests from Tasks 2-5)

**Step 6: Commit**

```bash
git add searcher.py tests/test_searcher.py
git commit -m "feat: add filename fallback to Searcher for content bridge"
```

---

### Task 6: Wire FileReader into Chat Loop

**Files:**
- Modify: `chat.py:75-99`

**Step 1: Update `chat_loop` to create `FileReader` and pass it to `Searcher`**

In `chat.py`, update the `chat_loop` function. Add the import and pass `file_reader` and `toolbox` to `Searcher`:

Change the imports block (lines 76-83) to include `FileReader`:

```python
    from leann import LeannSearcher
    from models import ModelManager
    from searcher import Searcher
    from file_reader import FileReader
    from toolbox import ToolBox
    from tools import ToolRegistry
    from router import route
    from rewriter import QueryRewriter
    from agent import Agent
```

Change the Searcher construction (line 97) to pass file_reader and toolbox:

```python
    # Create search and tools
    leann_searcher = LeannSearcher(index_path, enable_warmup=True)
    file_reader = FileReader()
    toolbox = ToolBox(data_dir)
    searcher = Searcher(leann_searcher, model, file_reader=file_reader, toolbox=toolbox, debug=True)
    tool_registry = ToolRegistry(searcher, toolbox)
```

**Step 2: Run full test suite**

Run: `uv run python -m pytest tests/ -v`
Expected: All tests pass

**Step 3: Commit**

```bash
git add chat.py
git commit -m "feat: wire FileReader into chat loop for filename fallback"
```

---

### Task 7: Integration Test for Filename Fallback

**Files:**
- Modify: `tests/test_integration.py`

**Step 1: Write the integration test**

Add to `tests/test_integration.py`:

```python
def test_filename_fallback_finds_file_by_name():
    """When semantic search finds nothing, filename fallback reads matching files.

    Call order with shared model.generate mock:
    1. Agent step 0: response[0] (no tool call → fallback router → semantic_search)
    2. semantic_search → searcher → _extract_facts for chunk → response[1] (irrelevant)
    3. Filename fallback: grep "invoice" finds invoice_macbook.txt, reads it,
       _extract_facts on file text → response[2] (relevant facts)
    4. Agent step 1: response[3] (final answer using facts)
    """
    from file_reader import FileReader

    results = [
        FakeSearchResult(
            id="1", text="Team meeting notes for March sprint planning",
            score=0.85, metadata={"file_name": "meeting_notes.txt"},
        )
    ]

    agent, model = _setup(
        model_responses=[
            # 1. Agent step 0 → no tool call → fallback router → semantic_search
            "Let me search for invoice information.",
            # 2. Searcher: map-filter on chunk (meeting notes) → irrelevant
            json.dumps({"relevant": False, "facts": []}),
            # 3. Filename fallback: map-filter on file text → relevant
            json.dumps({"relevant": True, "facts": ["Invoice #456", "MacBook Pro M4"]}),
            # 4. Agent step 1 → final answer
            "Found an invoice: Invoice #456 for a MacBook Pro M4.",
        ],
        search_results=results,
        files={"invoice_macbook.txt": "Invoice #456\nItem: MacBook Pro M4\nAmount: $2499"},
    )

    # Patch the searcher with file_reader and toolbox
    # The _setup function creates searcher without these, so we add them
    searcher = agent.tools.searcher
    searcher.file_reader = FileReader()
    searcher.toolbox = agent.tools.toolbox

    answer = agent.run("any macbook invoice?")
    assert "456" in answer or "MacBook" in answer
```

Note: This test uses a real `.txt` file (not PDF) to avoid needing Docling's full PDF pipeline in unit tests. The important thing being tested is the fallback chain: chunks irrelevant → grep filename → read file → extract facts.

**Step 2: Run tests to verify they pass**

Run: `uv run python -m pytest tests/test_integration.py -v`
Expected: All 7 tests pass (6 existing + 1 new)

**Step 3: Run full test suite**

Run: `uv run python -m pytest tests/ -v`
Expected: All tests pass

**Step 4: Commit**

```bash
git add tests/test_integration.py
git commit -m "test: add integration test for filename fallback in semantic search"
```

---

### Task 8: Smoke Test with Real Data

**Files:** None (manual test)

**Step 1: Run the chat loop against real test_data**

Run: `uv run python chat.py --reuse test_data test_data`

Test these queries and verify behavior:

1. `any macbook invoice?` — Should trigger filename fallback, find `macbook_ssd.pdf`, extract invoice facts via Docling
2. `what is the budget?` — Should work via normal chunk search (no fallback needed)
3. `how many PDF files?` — Should route to count_files (filesystem tool, no search)
4. `what invoices do I have?` — Should find invoices via chunk search first

**Step 2: Verify debug output shows fallback**

For query 1, you should see:
```
[SEARCH] Filename fallback: keywords=['macbook', 'invoice']
[SEARCH] Filename fallback: reading ['macbook_ssd.pdf']
[SEARCH] macbook_ssd.pdf: relevant=True, facts=N
```

**Step 3: If all queries work, the feature is complete**

No commit needed — this is a verification step.
