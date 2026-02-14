# Dual-Model Agentic RAG Pipeline Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the single-model AgenticRAG pipeline with a dual-model architecture using LFM2-350M-Extract (planner) and LFM2-1.2B-RAG (map/reduce), plus a smart ToolBox for filesystem queries.

**Architecture:** Two GGUF models loaded concurrently via llama-cpp-python (~2.2GB total). The 350M model handles structured JSON extraction (planning), while the 1.2B RAG model handles grounded fact extraction and synthesis. LeannSearcher stays for vector search. A pure Python ToolBox handles filesystem queries. Pipeline routes queries through three paths: semantic_search, filesystem, or hybrid (ToolBox narrows scope, then semantic search).

**Tech Stack:** llama-cpp-python (GGUF model loading), leann (vector search), pytest (testing)

**Design doc:** `docs/plans/2026-02-14-dual-model-pipeline-design.md`

---

### Task 1: Add llama-cpp-python dependency

**Files:**
- Modify: `pyproject.toml`

**Step 1: Add dependency**

Add `llama-cpp-python` to the dependencies list in `pyproject.toml`:

```toml
dependencies = [
    "leann>=0.3.6",
    "llama-cpp-python>=0.3.0",
    "torchvision>=0.25.0",
    "transformers>=4.55",
]
```

**Step 2: Install**

Run: `cd /Users/ded/Projects/assist/manole && uv sync`
Expected: Successful install, no errors.

**Step 3: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "feat: add llama-cpp-python dependency for dual-model pipeline"
```

---

### Task 2: Extract parser.py from chat.py

**Files:**
- Create: `parser.py`
- Create: `tests/test_parser.py`
- Modify: `chat.py` (remove `parse_json`, import from `parser`)

**Step 1: Write tests for parser**

Create `tests/test_parser.py`. These are migrated from `tests/test_agentic_rag.py` with additions:

```python
"""Tests for JSON parsing with regex fallback."""
from parser import parse_json


def test_parse_json_valid():
    result = parse_json('{"relevant": true, "facts": ["Invoice #123"]}')
    assert result == {"relevant": True, "facts": ["Invoice #123"]}


def test_parse_json_with_surrounding_text():
    result = parse_json('Here is the JSON:\n{"relevant": false, "facts": []}\nDone.')
    assert result == {"relevant": False, "facts": []}


def test_parse_json_malformed_fallback_relevant():
    result = parse_json('"relevant": true, some garbage')
    assert result["relevant"] is True


def test_parse_json_malformed_fallback_not_relevant():
    result = parse_json('The answer is "relevant": false and nothing else')
    assert result["relevant"] is False


def test_parse_json_total_garbage():
    result = parse_json("I don't understand the question")
    assert result is None


def test_parse_json_planner_output():
    """Planner returns keywords/filters JSON."""
    raw = '{"keywords": ["invoice", "Anthropic"], "file_filter": "pdf", "source_hint": "Invoice"}'
    result = parse_json(raw)
    assert result["keywords"] == ["invoice", "Anthropic"]
    assert result["file_filter"] == "pdf"


def test_parse_json_planner_with_preamble():
    """350M model might add text before JSON."""
    raw = 'Here is the extracted JSON:\n{"keywords": ["invoice"], "file_filter": null, "source_hint": null}'
    result = parse_json(raw)
    assert result["keywords"] == ["invoice"]


def test_parse_json_nested_braces():
    """Handle JSON with nested structures."""
    raw = '{"keywords": ["test"], "tool_actions": ["count"]}'
    result = parse_json(raw)
    assert result["keywords"] == ["test"]
    assert result["tool_actions"] == ["count"]
```

**Step 2: Run tests to verify they fail**

Run: `cd /Users/ded/Projects/assist/manole && uv run pytest tests/test_parser.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'parser'` (or similar)

**Step 3: Create parser.py**

Create `parser.py`:

```python
"""JSON parsing with multi-level regex fallback for small LLM output."""
import json
import re


def parse_json(text: str) -> dict | None:
    """Parse JSON from LLM output with fallback regex extraction.

    Tries three strategies:
    1. Direct JSON parse
    2. Extract first {...} block from surrounding text
    3. Regex extraction of 'relevant' and 'facts' fields
    """
    # Try direct parse
    try:
        return json.loads(text.strip())
    except (json.JSONDecodeError, ValueError):
        pass

    # Try to find a JSON object in the text
    match = re.search(r'\{[^{}]+\}', text)
    if match:
        try:
            return json.loads(match.group())
        except (json.JSONDecodeError, ValueError):
            pass

    # Fallback: extract "relevant" field via regex
    rel_match = re.search(r'"relevant"\s*:\s*(true|false)', text, re.IGNORECASE)
    if rel_match:
        relevant = rel_match.group(1).lower() == "true"
        facts_match = re.search(r'"facts"\s*:\s*\[([^\]]*)\]', text)
        facts = []
        if facts_match:
            facts = [f.strip().strip('"') for f in facts_match.group(1).split(",") if f.strip()]
        return {"relevant": relevant, "facts": facts}

    return None
```

**Step 4: Run tests to verify they pass**

Run: `cd /Users/ded/Projects/assist/manole && uv run pytest tests/test_parser.py -v`
Expected: All 9 tests PASS.

**Step 5: Update chat.py to import from parser**

In `chat.py`, remove the `parse_json` function body and replace with:

```python
from parser import parse_json
```

Remove the `import json` and `import re` lines from chat.py (if no longer needed by remaining code). Keep them if `AgenticRAG` still uses `json` directly.

**Step 6: Verify existing tests still pass**

Run: `cd /Users/ded/Projects/assist/manole && uv run pytest tests/ -v`
Expected: All existing tests PASS (they import `parse_json` from `chat` which now re-exports from `parser`).

**Step 7: Commit**

```bash
git add parser.py tests/test_parser.py chat.py
git commit -m "refactor: extract parse_json to parser.py module"
```

---

### Task 3: Build ModelManager (models.py)

**Files:**
- Create: `models.py`
- Create: `tests/test_models.py`

**Step 1: Write tests for ModelManager**

Create `tests/test_models.py`. Since we can't load real GGUF models in tests, we test the interface contract with mocks:

```python
"""Tests for ModelManager dual-model loading."""
from unittest.mock import patch, MagicMock
from models import ModelManager


def test_model_manager_has_plan_method():
    mgr = ModelManager.__new__(ModelManager)
    mgr.planner_model = None
    mgr.rag_model = None
    assert hasattr(mgr, "plan")
    assert hasattr(mgr, "extract")
    assert hasattr(mgr, "synthesize")


def test_plan_calls_planner_model():
    mgr = ModelManager.__new__(ModelManager)
    mock_model = MagicMock()
    mock_model.return_value = {"choices": [{"text": '{"keywords": ["test"]}'}]}
    mgr.planner_model = mock_model
    mgr.rag_model = None

    result = mgr.plan("test prompt")
    mock_model.assert_called_once()
    assert '{"keywords": ["test"]}' in result


def test_extract_calls_rag_model():
    mgr = ModelManager.__new__(ModelManager)
    mgr.planner_model = None
    mock_model = MagicMock()
    mock_model.return_value = {"choices": [{"text": '{"relevant": true}'}]}
    mgr.rag_model = mock_model

    result = mgr.extract("test prompt")
    mock_model.assert_called_once()


def test_synthesize_calls_rag_model():
    mgr = ModelManager.__new__(ModelManager)
    mgr.planner_model = None
    mock_model = MagicMock()
    mock_model.return_value = {"choices": [{"text": "The answer is 42."}]}
    mgr.rag_model = mock_model

    result = mgr.synthesize("test prompt")
    mock_model.assert_called_once()


def test_plan_max_tokens_is_256():
    """Planner should use small max_tokens for JSON output."""
    mgr = ModelManager.__new__(ModelManager)
    mock_model = MagicMock()
    mock_model.return_value = {"choices": [{"text": "{}"}]}
    mgr.planner_model = mock_model
    mgr.rag_model = None

    mgr.plan("test")
    call_kwargs = mock_model.call_args
    assert call_kwargs[1].get("max_tokens") == 256 or call_kwargs.kwargs.get("max_tokens") == 256
```

**Step 2: Run tests to verify they fail**

Run: `cd /Users/ded/Projects/assist/manole && uv run pytest tests/test_models.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'models'`

**Step 3: Implement ModelManager**

Create `models.py`:

```python
"""ModelManager: loads LFM2-350M-Extract and LFM2-1.2B-RAG concurrently."""
from pathlib import Path


class ModelManager:
    """Manages two GGUF models loaded via llama-cpp-python."""

    DEFAULT_PLANNER_PATH = "models/LFM2-350M-Extract-Q4_0.gguf"
    DEFAULT_RAG_PATH = "models/LFM2-1.2B-RAG-Q4_0.gguf"

    def __init__(
        self,
        planner_path: str | None = None,
        rag_path: str | None = None,
        n_threads: int = 4,
    ):
        self.planner_path = planner_path or self.DEFAULT_PLANNER_PATH
        self.rag_path = rag_path or self.DEFAULT_RAG_PATH
        self.n_threads = n_threads
        self.planner_model = None
        self.rag_model = None

    def load(self):
        """Load both models. Call once at startup."""
        from llama_cpp import Llama

        self.planner_model = Llama(
            model_path=self.planner_path,
            n_ctx=2048,
            n_threads=self.n_threads,
            verbose=False,
        )
        self.rag_model = Llama(
            model_path=self.rag_path,
            n_ctx=4096,
            n_threads=self.n_threads,
            verbose=False,
        )

    def plan(self, prompt: str) -> str:
        """Run prompt through 350M-Extract model. For structured JSON extraction."""
        response = self.planner_model(prompt, max_tokens=256, temperature=0.0)
        return response["choices"][0]["text"]

    def extract(self, prompt: str) -> str:
        """Run prompt through 1.2B-RAG model. For per-chunk fact extraction."""
        response = self.rag_model(prompt, max_tokens=512, temperature=0.0)
        return response["choices"][0]["text"]

    def synthesize(self, prompt: str) -> str:
        """Run prompt through 1.2B-RAG model. For answer synthesis."""
        response = self.rag_model(prompt, max_tokens=1024, temperature=0.1)
        return response["choices"][0]["text"]
```

**Step 4: Run tests to verify they pass**

Run: `cd /Users/ded/Projects/assist/manole && uv run pytest tests/test_models.py -v`
Expected: All 5 tests PASS.

**Step 5: Commit**

```bash
git add models.py tests/test_models.py
git commit -m "feat: add ModelManager for dual GGUF model loading"
```

---

### Task 4: Build Planner (planner.py)

**Files:**
- Create: `planner.py`
- Create: `tests/test_planner.py`

**Step 1: Write tests**

Create `tests/test_planner.py`:

```python
"""Tests for Planner stage — query to structured JSON plan."""
import json
from planner import Planner, PLANNER_PROMPT


class FakeModelManager:
    """Returns canned responses for plan() calls."""
    def __init__(self, responses: list[str]):
        self.responses = list(responses)
        self.prompts = []

    def plan(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.responses.pop(0) if self.responses else ""


def test_planner_prompt_contains_examples():
    """Prompt must have few-shot examples for all three tool types."""
    assert "semantic_search" in PLANNER_PROMPT
    assert "filesystem" in PLANNER_PROMPT
    assert "hybrid" in PLANNER_PROMPT


def test_planner_extracts_semantic_search_plan():
    response = json.dumps({
        "keywords": ["invoice", "Anthropic"],
        "file_filter": "pdf",
        "source_hint": "Invoice",
        "tool": "semantic_search",
        "time_filter": None,
        "tool_actions": [],
    })
    models = FakeModelManager([response])
    planner = Planner(models)
    plan = planner.plan("find my Anthropic invoices")

    assert plan["tool"] == "semantic_search"
    assert plan["keywords"] == ["invoice", "Anthropic"]
    assert plan["file_filter"] == "pdf"
    assert plan["source_hint"] == "Invoice"


def test_planner_extracts_filesystem_plan():
    response = json.dumps({
        "keywords": ["PDF", "files", "count"],
        "file_filter": "pdf",
        "source_hint": None,
        "tool": "filesystem",
        "time_filter": None,
        "tool_actions": ["count"],
    })
    models = FakeModelManager([response])
    planner = Planner(models)
    plan = planner.plan("how many PDF files do I have?")

    assert plan["tool"] == "filesystem"
    assert "count" in plan["tool_actions"]


def test_planner_extracts_hybrid_plan():
    response = json.dumps({
        "keywords": ["modified", "today", "summary"],
        "file_filter": None,
        "source_hint": None,
        "tool": "hybrid",
        "time_filter": "today",
        "tool_actions": ["list_recent"],
    })
    models = FakeModelManager([response])
    planner = Planner(models)
    plan = planner.plan("summarize files I modified today")

    assert plan["tool"] == "hybrid"
    assert plan["time_filter"] == "today"


def test_planner_garbage_output_returns_safe_defaults():
    models = FakeModelManager(["I don't understand the question"])
    planner = Planner(models)
    plan = planner.plan("find invoices")

    assert plan["tool"] == "semantic_search"
    assert plan["keywords"] == []
    assert plan["file_filter"] is None
    assert plan["source_hint"] is None
    assert plan["time_filter"] is None
    assert plan["tool_actions"] == []


def test_planner_partial_json_extracts_what_it_can():
    """350M model outputs JSON with missing fields."""
    response = '{"keywords": ["invoice"], "tool": "semantic_search"}'
    models = FakeModelManager([response])
    planner = Planner(models)
    plan = planner.plan("invoices")

    assert plan["keywords"] == ["invoice"]
    assert plan["tool"] == "semantic_search"
    assert plan["file_filter"] is None  # missing field gets default
```

**Step 2: Run tests to verify they fail**

Run: `cd /Users/ded/Projects/assist/manole && uv run pytest tests/test_planner.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'planner'`

**Step 3: Implement Planner**

Create `planner.py`:

```python
"""Stage 1: Planner — extract structured search parameters from user queries."""
from parser import parse_json

PLANNER_PROMPT = (
    "Extract search parameters from the user's question as JSON.\n\n"
    "Output fields:\n"
    '- "keywords": list of 2-4 search terms\n'
    '- "file_filter": file extension like "pdf", "txt", "py", or null\n'
    '- "source_hint": filename substring to filter by, or null\n'
    '- "tool": "semantic_search", "filesystem", or "hybrid"\n'
    '- "time_filter": "today", "this_week", "this_month", or null\n'
    '- "tool_actions": list from ["count", "list_recent", "metadata", "tree", "grep"], or []\n'
    "\n"
    "Examples:\n\n"
    'Question: "find my Anthropic invoices"\n'
    'JSON: {"keywords": ["invoice", "Anthropic"], "file_filter": "pdf", '
    '"source_hint": "Invoice", "tool": "semantic_search", '
    '"time_filter": null, "tool_actions": []}\n\n'
    'Question: "how many PDF files do I have?"\n'
    'JSON: {"keywords": ["PDF", "files", "count"], "file_filter": "pdf", '
    '"source_hint": null, "tool": "filesystem", '
    '"time_filter": null, "tool_actions": ["count"]}\n\n'
    'Question: "summarize files I modified today"\n'
    'JSON: {"keywords": ["modified", "today", "summary"], "file_filter": null, '
    '"source_hint": null, "tool": "hybrid", '
    '"time_filter": "today", "tool_actions": ["list_recent"]}\n\n'
    'Question: "what is my folder structure?"\n'
    'JSON: {"keywords": ["folder", "structure", "directory"], "file_filter": null, '
    '"source_hint": null, "tool": "filesystem", '
    '"time_filter": null, "tool_actions": ["tree"]}\n\n'
    'Question: "notes about machine learning"\n'
    'JSON: {"keywords": ["machine learning", "notes", "AI"], "file_filter": null, '
    '"source_hint": null, "tool": "semantic_search", '
    '"time_filter": null, "tool_actions": []}\n\n'
    "Question: {query}\n"
    "JSON:"
)

_DEFAULT_PLAN = {
    "keywords": [],
    "file_filter": None,
    "source_hint": None,
    "tool": "semantic_search",
    "time_filter": None,
    "tool_actions": [],
}


class Planner:
    """Extracts structured search plan from user queries using 350M-Extract model."""

    def __init__(self, models, debug: bool = False):
        self.models = models
        self.debug = debug

    def plan(self, query: str) -> dict:
        prompt = PLANNER_PROMPT.format(query=query)
        raw = self.models.plan(prompt)

        if self.debug:
            print(f"  [PLAN] Raw: {raw}")

        parsed = parse_json(raw)
        if parsed is None:
            if self.debug:
                print("  [PLAN] Parse failed, using defaults")
            return dict(_DEFAULT_PLAN)

        # Fill in missing fields with defaults
        result = {}
        for key, default in _DEFAULT_PLAN.items():
            result[key] = parsed.get(key, default)

        # Validate tool field
        if result["tool"] not in ("semantic_search", "filesystem", "hybrid"):
            result["tool"] = "semantic_search"

        # Validate time_filter
        if result["time_filter"] not in ("today", "this_week", "this_month", None):
            result["time_filter"] = None

        if self.debug:
            print(f"  [PLAN] {result}")

        return result
```

**Step 4: Run tests to verify they pass**

Run: `cd /Users/ded/Projects/assist/manole && uv run pytest tests/test_planner.py -v`
Expected: All 6 tests PASS.

**Step 5: Commit**

```bash
git add planner.py tests/test_planner.py
git commit -m "feat: add Planner stage with few-shot prompt for 350M-Extract model"
```

---

### Task 5: Build Smart ToolBox (toolbox.py)

**Files:**
- Create: `toolbox.py`
- Create: `tests/test_toolbox.py`

**Step 1: Write tests**

Create `tests/test_toolbox.py`:

```python
"""Tests for Smart ToolBox — filesystem operations."""
import os
import tempfile
import time
from pathlib import Path
from toolbox import ToolBox


def _make_test_dir():
    """Create a temp directory with test files."""
    tmp = tempfile.mkdtemp()
    # Create files
    (Path(tmp) / "invoice.pdf").write_text("pdf content")
    (Path(tmp) / "notes.txt").write_text("some notes")
    (Path(tmp) / "code.py").write_text("print('hello')")
    # Create subdirectory
    sub = Path(tmp) / "subdir"
    sub.mkdir()
    (sub / "deep.pdf").write_text("deep pdf")
    return tmp


def test_count_all_files():
    tmp = _make_test_dir()
    tb = ToolBox(tmp)
    result = tb.count_files()
    assert "4" in result  # 4 files total


def test_count_with_extension_filter():
    tmp = _make_test_dir()
    tb = ToolBox(tmp)
    result = tb.count_files(ext_filter="pdf")
    assert "2" in result  # invoice.pdf + deep.pdf


def test_list_recent_files():
    tmp = _make_test_dir()
    tb = ToolBox(tmp)
    result = tb.list_recent_files()
    assert "invoice.pdf" in result
    assert "notes.txt" in result


def test_list_recent_with_limit():
    tmp = _make_test_dir()
    tb = ToolBox(tmp)
    result = tb.list_recent_files(limit=2)
    lines = [l for l in result.strip().split("\n") if l.strip().startswith("-")]
    # Could be header + 2 file lines, depends on format
    assert result.count(".") >= 2  # at least 2 filenames with extensions


def test_list_recent_with_extension_filter():
    tmp = _make_test_dir()
    tb = ToolBox(tmp)
    result = tb.list_recent_files(ext_filter="pdf")
    assert "invoice.pdf" in result
    assert "notes.txt" not in result


def test_metadata_for_file():
    tmp = _make_test_dir()
    tb = ToolBox(tmp)
    result = tb.get_file_metadata(name_hint="invoice")
    assert "invoice.pdf" in result
    assert "KB" in result or "bytes" in result.lower()


def test_tree_structure():
    tmp = _make_test_dir()
    tb = ToolBox(tmp)
    result = tb.tree()
    assert "subdir" in result
    assert "invoice.pdf" in result


def test_tree_with_depth():
    tmp = _make_test_dir()
    tb = ToolBox(tmp)
    result = tb.tree(max_depth=0)
    # At depth 0, should show top-level only (files + dirs listed)
    assert "subdir" in result
    assert "deep.pdf" not in result


def test_grep_filenames():
    tmp = _make_test_dir()
    tb = ToolBox(tmp)
    result = tb.grep("invoice")
    assert "invoice.pdf" in result


def test_grep_no_match():
    tmp = _make_test_dir()
    tb = ToolBox(tmp)
    result = tb.grep("nonexistent")
    assert "No files" in result or "0" in result


def test_execute_routes_count():
    tmp = _make_test_dir()
    tb = ToolBox(tmp)
    plan = {"tool_actions": ["count"], "file_filter": "pdf", "time_filter": None}
    result = tb.execute(plan)
    assert "2" in result


def test_execute_routes_list_recent():
    tmp = _make_test_dir()
    tb = ToolBox(tmp)
    plan = {"tool_actions": ["list_recent"], "file_filter": None, "time_filter": None}
    result = tb.execute(plan)
    assert "invoice.pdf" in result


def test_execute_routes_tree():
    tmp = _make_test_dir()
    tb = ToolBox(tmp)
    plan = {"tool_actions": ["tree"], "file_filter": None, "time_filter": None}
    result = tb.execute(plan)
    assert "subdir" in result


def test_get_matching_files_returns_paths():
    """For hybrid queries, ToolBox returns file paths."""
    tmp = _make_test_dir()
    tb = ToolBox(tmp)
    plan = {"tool_actions": ["list_recent"], "file_filter": "pdf", "time_filter": None}
    paths = tb.get_matching_files(plan)
    assert all(isinstance(p, str) for p in paths)
    assert any("invoice.pdf" in p for p in paths)
    assert not any("notes.txt" in p for p in paths)


def test_time_filter_today():
    """Files created just now should pass 'today' filter."""
    tmp = _make_test_dir()
    tb = ToolBox(tmp)
    plan = {"tool_actions": ["list_recent"], "file_filter": None, "time_filter": "today"}
    result = tb.execute(plan)
    # All files were just created, so all should appear
    assert "invoice.pdf" in result
```

**Step 2: Run tests to verify they fail**

Run: `cd /Users/ded/Projects/assist/manole && uv run pytest tests/test_toolbox.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'toolbox'`

**Step 3: Implement ToolBox**

Create `toolbox.py`:

```python
"""Smart ToolBox: LLM-routed filesystem operations with time awareness."""
from datetime import datetime, timedelta
from pathlib import Path


class ToolBox:
    """Pure Python filesystem tools operating on the indexed data directory."""

    def __init__(self, data_dir: str):
        self.root = Path(data_dir)

    def _time_cutoff(self, time_filter: str | None) -> float | None:
        """Convert time_filter string to a Unix timestamp cutoff."""
        if not time_filter:
            return None
        now = datetime.now()
        if time_filter == "today":
            cutoff = now - timedelta(hours=24)
        elif time_filter == "this_week":
            cutoff = now - timedelta(days=7)
        elif time_filter == "this_month":
            cutoff = now - timedelta(days=30)
        else:
            return None
        return cutoff.timestamp()

    def _list_files(self, ext_filter: str | None = None, time_filter: str | None = None) -> list[Path]:
        """Get all files matching filters."""
        pattern = f"**/*.{ext_filter}" if ext_filter else "**/*"
        files = [f for f in self.root.glob(pattern) if f.is_file() and not f.name.startswith(".")]

        cutoff = self._time_cutoff(time_filter)
        if cutoff:
            files = [f for f in files if f.stat().st_mtime >= cutoff]

        return files

    def count_files(self, ext_filter: str | None = None, time_filter: str | None = None) -> str:
        files = self._list_files(ext_filter, time_filter)
        label = f".{ext_filter} " if ext_filter else ""
        return f"Found {len(files)} {label}files."

    def list_recent_files(
        self,
        ext_filter: str | None = None,
        time_filter: str | None = None,
        limit: int = 10,
    ) -> str:
        files = self._list_files(ext_filter, time_filter)
        files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
        if not files:
            return "No matching files found."
        lines = []
        for f in files[:limit]:
            mtime = datetime.fromtimestamp(f.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
            rel = f.relative_to(self.root)
            lines.append(f"  - {rel} (modified: {mtime})")
        return "Recent files:\n" + "\n".join(lines)

    def get_file_metadata(self, name_hint: str | None = None) -> str:
        files = [f for f in self.root.rglob("*") if f.is_file() and not f.name.startswith(".")]
        if name_hint:
            files = [f for f in files if name_hint.lower() in f.name.lower()]
        if not files:
            return "No matching files found."
        lines = []
        for f in files[:10]:
            stat = f.stat()
            size_kb = stat.st_size / 1024
            mtime = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M")
            rel = f.relative_to(self.root)
            lines.append(f"  - {rel}: {size_kb:.1f}KB, modified {mtime}")
        return "File metadata:\n" + "\n".join(lines)

    def tree(self, max_depth: int | None = None) -> str:
        """Directory tree with optional depth limit."""
        lines = [f"{self.root.name}/"]
        self._tree_recurse(self.root, "", 0, max_depth, lines)
        return "\n".join(lines)

    def _tree_recurse(self, path: Path, prefix: str, depth: int, max_depth: int | None, lines: list):
        entries = sorted(path.iterdir(), key=lambda e: (not e.is_dir(), e.name.lower()))
        entries = [e for e in entries if not e.name.startswith(".")]
        for i, entry in enumerate(entries):
            is_last = i == len(entries) - 1
            connector = "└── " if is_last else "├── "
            lines.append(f"{prefix}{connector}{entry.name}{'/' if entry.is_dir() else ''}")
            if entry.is_dir() and (max_depth is None or depth < max_depth):
                extension = "    " if is_last else "│   "
                self._tree_recurse(entry, prefix + extension, depth + 1, max_depth, lines)

    def grep(self, pattern: str) -> str:
        """Search file names by pattern."""
        files = [f for f in self.root.rglob("*") if f.is_file() and not f.name.startswith(".")]
        matches = [f for f in files if pattern.lower() in f.name.lower()]
        if not matches:
            return f"No files matching '{pattern}'."
        lines = [f"  - {f.relative_to(self.root)}" for f in matches[:20]]
        return f"Files matching '{pattern}':\n" + "\n".join(lines)

    def execute(self, plan: dict) -> str:
        """Execute tool_actions from plan, return formatted results."""
        actions = plan.get("tool_actions", [])
        ext = plan.get("file_filter")
        time_f = plan.get("time_filter")
        source = plan.get("source_hint")

        results = []
        for action in actions:
            if action == "count":
                results.append(self.count_files(ext, time_f))
            elif action == "list_recent":
                results.append(self.list_recent_files(ext, time_f))
            elif action == "metadata":
                results.append(self.get_file_metadata(source))
            elif action == "tree":
                results.append(self.tree())
            elif action == "grep":
                if source:
                    results.append(self.grep(source))

        if not results:
            # Fallback: list recent files
            results.append(self.list_recent_files(ext, time_f))

        return "\n\n".join(results)

    def get_matching_files(self, plan: dict) -> list[str]:
        """For hybrid queries: return list of matching file paths (not formatted text)."""
        ext = plan.get("file_filter")
        time_f = plan.get("time_filter")
        files = self._list_files(ext, time_f)
        return [str(f) for f in files]
```

**Step 4: Run tests to verify they pass**

Run: `cd /Users/ded/Projects/assist/manole && uv run pytest tests/test_toolbox.py -v`
Expected: All 16 tests PASS.

**Step 5: Commit**

```bash
git add toolbox.py tests/test_toolbox.py
git commit -m "feat: add smart ToolBox with time-aware filesystem operations"
```

---

### Task 6: Build Searcher (searcher.py)

**Files:**
- Create: `searcher.py`
- Create: `tests/test_searcher.py`

**Step 1: Write tests**

Create `tests/test_searcher.py`:

```python
"""Tests for Searcher — leann vector search with metadata filters."""
from dataclasses import dataclass, field
from searcher import Searcher


@dataclass
class FakeSearchResult:
    id: str
    text: str
    score: float
    metadata: dict = field(default_factory=dict)


class FakeLeannSearcher:
    """Mimics LeannSearcher.search()."""
    def __init__(self, results: list[FakeSearchResult]):
        self.results = results
        self.last_kwargs = {}

    def search(self, query, top_k=5, metadata_filters=None, **kwargs):
        self.last_kwargs = {"query": query, "top_k": top_k, "metadata_filters": metadata_filters}
        if metadata_filters:
            # Simulate filtering by source contains
            source_filter = metadata_filters.get("source", {})
            contains = source_filter.get("contains", "")
            if contains:
                return [r for r in self.results if contains.lower() in r.metadata.get("source", "").lower()][:top_k]
        return self.results[:top_k]


def _make_results(*texts, sources=None):
    if sources is None:
        sources = [f"file{i}.pdf" for i in range(len(texts))]
    return [
        FakeSearchResult(id=str(i), text=t, score=0.9 - i * 0.1, metadata={"source": sources[i]})
        for i, t in enumerate(texts)
    ]


def test_search_basic():
    results = _make_results("chunk1", "chunk2", "chunk3")
    searcher = Searcher(FakeLeannSearcher(results))
    plan = {"keywords": ["test"], "source_hint": None, "file_filter": None}
    found = searcher.search(plan, top_k=5)
    assert len(found) == 3


def test_search_applies_source_hint():
    results = _make_results("invoice data", "recipe data", sources=["Invoice_001.pdf", "recipe.txt"])
    searcher = Searcher(FakeLeannSearcher(results))
    plan = {"keywords": ["invoice"], "source_hint": "Invoice", "file_filter": None}
    found = searcher.search(plan, top_k=5)
    assert len(found) == 1
    assert "Invoice" in found[0].metadata["source"]


def test_search_applies_file_filter():
    results = _make_results("data1", "data2", sources=["doc.pdf", "notes.txt"])
    searcher = Searcher(FakeLeannSearcher(results))
    plan = {"keywords": ["data"], "source_hint": None, "file_filter": "pdf"}
    found = searcher.search(plan, top_k=5)
    assert len(found) == 1
    assert found[0].metadata["source"].endswith(".pdf")


def test_search_source_hint_takes_precedence_over_file_filter():
    """source_hint is more specific, should be used when both present."""
    results = _make_results("data", sources=["Invoice_001.pdf"])
    searcher = Searcher(FakeLeannSearcher(results))
    plan = {"keywords": ["invoice"], "source_hint": "Invoice", "file_filter": "pdf"}
    found = searcher.search(plan, top_k=5)
    assert len(found) == 1


def test_search_unfiltered_fallback():
    results = _make_results("chunk1", "chunk2")
    searcher = Searcher(FakeLeannSearcher(results))
    plan = {"keywords": ["test"]}
    found = searcher.search_unfiltered(plan, top_k=5)
    assert len(found) == 2


def test_search_respects_top_k():
    results = _make_results("a", "b", "c", "d", "e")
    searcher = Searcher(FakeLeannSearcher(results))
    plan = {"keywords": ["test"], "source_hint": None, "file_filter": None}
    found = searcher.search(plan, top_k=2)
    assert len(found) == 2


def test_search_with_file_filter_paths():
    """Hybrid mode: only search within specified file paths."""
    results = _make_results("data1", "data2", "data3", sources=["a.pdf", "b.txt", "c.pdf"])
    leann = FakeLeannSearcher(results)
    searcher = Searcher(leann)
    plan = {"keywords": ["data"], "source_hint": None, "file_filter": None}
    found = searcher.search(plan, top_k=5, file_filter_paths=["/path/to/a.pdf", "/path/to/c.pdf"])
    # Should filter to only chunks from a.pdf and c.pdf
    assert all("pdf" in r.metadata["source"] for r in found)
```

**Step 2: Run tests to verify they fail**

Run: `cd /Users/ded/Projects/assist/manole && uv run pytest tests/test_searcher.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'searcher'`

**Step 3: Implement Searcher**

Create `searcher.py`:

```python
"""Stage 2: Searcher — vector search with metadata filters via LeannSearcher."""


class Searcher:
    """Wraps LeannSearcher with plan-based metadata filter construction."""

    def __init__(self, leann_searcher):
        self.leann = leann_searcher

    def _build_filters(self, plan: dict) -> dict | None:
        """Build metadata_filters dict from plan."""
        source_hint = plan.get("source_hint")
        file_filter = plan.get("file_filter")

        if source_hint:
            return {"source": {"contains": source_hint}}
        if file_filter:
            return {"source": {"contains": f".{file_filter}"}}
        return None

    def search(
        self,
        plan: dict,
        top_k: int = 5,
        file_filter_paths: list[str] | None = None,
    ) -> list:
        """Search with optional metadata filters from plan."""
        query = " ".join(plan.get("keywords", []))
        if not query:
            query = "document"  # safe fallback

        metadata_filters = self._build_filters(plan)
        results = self.leann.search(query, top_k=top_k, metadata_filters=metadata_filters)

        # For hybrid queries, filter results to only chunks from specified files
        if file_filter_paths:
            path_basenames = {p.rsplit("/", 1)[-1].lower() for p in file_filter_paths}
            results = [
                r for r in results
                if r.metadata.get("source", "").rsplit("/", 1)[-1].lower() in path_basenames
            ]

        return results

    def search_unfiltered(self, plan: dict, top_k: int = 5) -> list:
        """Fallback: search without any metadata filters."""
        query = " ".join(plan.get("keywords", []))
        if not query:
            query = "document"
        return self.leann.search(query, top_k=top_k)
```

**Step 4: Run tests to verify they pass**

Run: `cd /Users/ded/Projects/assist/manole && uv run pytest tests/test_searcher.py -v`
Expected: All 7 tests PASS.

**Step 5: Commit**

```bash
git add searcher.py tests/test_searcher.py
git commit -m "feat: add Searcher with metadata filter construction and hybrid support"
```

---

### Task 7: Build Mapper (mapper.py)

**Files:**
- Create: `mapper.py`
- Create: `tests/test_mapper.py`

**Step 1: Write tests**

Create `tests/test_mapper.py`:

```python
"""Tests for Mapper — per-chunk fact extraction using 1.2B-RAG model."""
import json
from dataclasses import dataclass, field
from mapper import Mapper, MAP_PROMPT


@dataclass
class FakeSearchResult:
    id: str
    text: str
    score: float
    metadata: dict = field(default_factory=dict)


class FakeModelManager:
    def __init__(self, responses: list[str]):
        self.responses = list(responses)
        self.prompts = []

    def extract(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.responses.pop(0) if self.responses else ""


def test_map_prompt_contains_placeholders():
    assert "{query}" in MAP_PROMPT
    assert "{chunk_text}" in MAP_PROMPT


def test_map_relevant_chunk():
    response = json.dumps({"relevant": True, "facts": ["Invoice #123", "Amount: $50"]})
    models = FakeModelManager([response])
    mapper = Mapper(models)
    chunk = FakeSearchResult(id="0", text="Invoice #123 for $50", score=0.9, metadata={"source": "a.pdf"})
    result = mapper.map_chunk("find invoices", chunk)
    assert result["relevant"] is True
    assert "Invoice #123" in result["facts"]
    assert result["source"] == "a.pdf"


def test_map_irrelevant_chunk():
    response = json.dumps({"relevant": False, "facts": []})
    models = FakeModelManager([response])
    mapper = Mapper(models)
    chunk = FakeSearchResult(id="0", text="Nice weather", score=0.3, metadata={"source": "b.txt"})
    result = mapper.map_chunk("find invoices", chunk)
    assert result["relevant"] is False
    assert result["facts"] == []


def test_map_garbage_defaults_to_relevant():
    models = FakeModelManager(["I have no idea"])
    mapper = Mapper(models)
    chunk = FakeSearchResult(id="0", text="Some text here", score=0.5, metadata={})
    result = mapper.map_chunk("query", chunk)
    assert result["relevant"] is True
    assert chunk.text[:200] in result["facts"]


def test_map_chunk_truncates_long_text():
    """Chunk text is truncated to 500 chars before sending to LLM."""
    models = FakeModelManager([json.dumps({"relevant": True, "facts": ["fact"]})])
    mapper = Mapper(models)
    long_text = "x" * 1000
    chunk = FakeSearchResult(id="0", text=long_text, score=0.9, metadata={})
    mapper.map_chunk("query", chunk)
    # Check the prompt was truncated
    assert len(models.prompts[0]) < len(long_text) + len(MAP_PROMPT)


def test_extract_facts_multiple_chunks():
    responses = [
        json.dumps({"relevant": True, "facts": ["fact1"]}),
        json.dumps({"relevant": False, "facts": []}),
        json.dumps({"relevant": True, "facts": ["fact2", "fact3"]}),
    ]
    models = FakeModelManager(responses)
    mapper = Mapper(models)
    chunks = [
        FakeSearchResult(id=str(i), text=f"text{i}", score=0.9, metadata={"source": f"f{i}.pdf"})
        for i in range(3)
    ]
    mapped = mapper.extract_facts("query", chunks)
    assert len(mapped) == 3
    assert mapped[0]["relevant"] is True
    assert mapped[1]["relevant"] is False
    assert mapped[2]["facts"] == ["fact2", "fact3"]
```

**Step 2: Run tests to verify they fail**

Run: `cd /Users/ded/Projects/assist/manole && uv run pytest tests/test_mapper.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'mapper'`

**Step 3: Implement Mapper**

Create `mapper.py`:

```python
"""Stage 3: Mapper — per-chunk fact extraction using 1.2B-RAG model."""
from parser import parse_json

MAP_PROMPT = (
    "Does this text answer the question? Extract facts or say not relevant.\n\n"
    "Question: {query}\n"
    "Text: {chunk_text}\n\n"
    'If relevant, output: {{"relevant": true, "facts": ["fact1", "fact2"]}}\n'
    'If NOT relevant, output: {{"relevant": false, "facts": []}}\n\n'
    "JSON:\n"
)


class Mapper:
    """Extracts facts from individual chunks using the 1.2B-RAG model."""

    def __init__(self, models, debug: bool = False):
        self.models = models
        self.debug = debug

    def map_chunk(self, query: str, chunk) -> dict:
        """Extract facts from a single search result chunk."""
        prompt = MAP_PROMPT.format(query=query, chunk_text=chunk.text[:500])
        raw = self.models.extract(prompt)
        parsed = parse_json(raw)

        source = chunk.metadata.get("source", chunk.id)

        if parsed is None:
            if self.debug:
                print(f"  [MAP] {source}: parse failed, treating as relevant")
            return {"relevant": True, "facts": [chunk.text[:200]], "source": source}

        result = {
            "relevant": parsed.get("relevant", True),
            "facts": parsed.get("facts", []),
            "source": source,
        }
        if self.debug:
            print(f"  [MAP] {source}: relevant={result['relevant']}, facts={len(result['facts'])}")
        return result

    def extract_facts(self, query: str, chunks: list) -> list[dict]:
        """Map all chunks, returning list of {relevant, facts, source} dicts."""
        if self.debug:
            print(f"  [MAP] Processing {len(chunks)} chunks...")
        return [self.map_chunk(query, chunk) for chunk in chunks]
```

**Step 4: Run tests to verify they pass**

Run: `cd /Users/ded/Projects/assist/manole && uv run pytest tests/test_mapper.py -v`
Expected: All 6 tests PASS.

**Step 5: Commit**

```bash
git add mapper.py tests/test_mapper.py
git commit -m "feat: add Mapper for per-chunk fact extraction with 1.2B-RAG model"
```

---

### Task 8: Build Reducer (reducer.py)

**Files:**
- Create: `reducer.py`
- Create: `tests/test_reducer.py`

**Step 1: Write tests**

Create `tests/test_reducer.py`:

```python
"""Tests for Reducer — answer synthesis and confidence checking."""
from reducer import Reducer, REDUCE_PROMPT, confidence_score


class FakeModelManager:
    def __init__(self, responses: list[str]):
        self.responses = list(responses)
        self.prompts = []

    def synthesize(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.responses.pop(0) if self.responses else ""


def test_reduce_prompt_contains_placeholders():
    assert "{facts_list}" in REDUCE_PROMPT
    assert "{query}" in REDUCE_PROMPT


def test_synthesize_from_facts():
    models = FakeModelManager(["Found 2 invoices: #123 for $50 and #456 for $75."])
    reducer = Reducer(models)
    relevant = [
        {"relevant": True, "facts": ["Invoice #123", "Amount: $50"], "source": "a.pdf"},
        {"relevant": True, "facts": ["Invoice #456", "Amount: $75"], "source": "b.pdf"},
    ]
    answer = reducer.synthesize("find invoices", relevant)
    assert "2 invoices" in answer


def test_synthesize_no_relevant_returns_message():
    models = FakeModelManager([])
    reducer = Reducer(models)
    answer = reducer.synthesize("find invoices", [])
    assert "No relevant" in answer


def test_synthesize_prompt_includes_sources():
    models = FakeModelManager(["answer"])
    reducer = Reducer(models)
    relevant = [
        {"relevant": True, "facts": ["fact1"], "source": "report.pdf"},
    ]
    reducer.synthesize("query", relevant)
    assert "report.pdf" in models.prompts[0]


def test_confidence_score_high_overlap():
    facts = ["Invoice #123", "Amount: $50", "Date: Dec 4"]
    answer = "Invoice #123 for $50 dated Dec 4"
    score = confidence_score(answer, facts)
    assert score >= 0.5


def test_confidence_score_low_overlap():
    facts = ["Invoice #123", "Amount: $50"]
    answer = "The weather is nice today and I like cats"
    score = confidence_score(answer, facts)
    assert score < 0.3


def test_confidence_score_empty_facts():
    assert confidence_score("some answer", []) == 0.0


def test_confidence_score_empty_answer():
    assert confidence_score("", ["fact1"]) == 0.0


def test_confidence_check_passes_high():
    models = FakeModelManager([])
    reducer = Reducer(models)
    relevant = [{"facts": ["Invoice #123", "Amount: $50"]}]
    answer = "Invoice #123 for $50"
    result = reducer.confidence_check(answer, relevant)
    assert "(low confidence)" not in result.lower()


def test_confidence_check_flags_low():
    models = FakeModelManager([])
    reducer = Reducer(models)
    relevant = [{"facts": ["Invoice #123", "Amount: $50"]}]
    answer = "The sky is blue and birds fly south"
    result = reducer.confidence_check(answer, relevant)
    assert "(low confidence)" in result.lower()


def test_format_filesystem_answer():
    models = FakeModelManager(["You have 5 PDF files in your collection."])
    reducer = Reducer(models)
    result = reducer.format_filesystem_answer("how many PDFs?", "Found 5 .pdf files.")
    assert "5" in result
```

**Step 2: Run tests to verify they fail**

Run: `cd /Users/ded/Projects/assist/manole && uv run pytest tests/test_reducer.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'reducer'`

**Step 3: Implement Reducer**

Create `reducer.py`:

```python
"""Stage 5: Reducer — answer synthesis and confidence checking."""

REDUCE_PROMPT = (
    "Here are facts extracted from the user's files:\n"
    "{facts_list}\n\n"
    "Question: {query}\n\n"
    "Using ONLY these facts, write a concise answer. "
    'If the facts don\'t answer the question, say "No relevant information found."\n\n'
    "Answer:\n"
)

FILESYSTEM_PROMPT = (
    "The user asked: {query}\n\n"
    "Here is the result from their filesystem:\n"
    "{result}\n\n"
    "Write a brief, natural answer based on this data.\n\n"
    "Answer:\n"
)


def confidence_score(answer: str, facts: list[str]) -> float:
    """Compute token overlap between answer and source facts. Returns 0.0-1.0."""
    if not facts:
        return 0.0
    answer_tokens = set(answer.lower().split())
    if not answer_tokens:
        return 0.0
    fact_tokens = set()
    for fact in facts:
        fact_tokens.update(fact.lower().split())
    if not fact_tokens:
        return 0.0
    overlap = answer_tokens & fact_tokens
    return len(overlap) / len(answer_tokens)


class Reducer:
    """Synthesizes answers from extracted facts using 1.2B-RAG model."""

    def __init__(self, models, debug: bool = False):
        self.models = models
        self.debug = debug

    def synthesize(self, query: str, relevant: list[dict]) -> str:
        """Synthesize answer from filtered relevant facts."""
        if not relevant:
            return "No relevant information found."

        facts_list = ""
        for item in relevant:
            source = item.get("source", "?")
            facts_list += f"\nFrom {source}:\n"
            for fact in item["facts"]:
                facts_list += f"  - {fact}\n"

        prompt = REDUCE_PROMPT.format(facts_list=facts_list, query=query)
        if self.debug:
            print(f"  [REDUCE] Synthesizing from {len(relevant)} sources")
        answer = self.models.synthesize(prompt)
        return answer.strip()

    def confidence_check(self, answer: str, relevant: list[dict]) -> str:
        """Python-based confidence check via token overlap. No LLM call."""
        all_facts = []
        for item in relevant:
            all_facts.extend(item.get("facts", []))

        score = confidence_score(answer, all_facts)
        if self.debug:
            print(f"  [CHECK] Confidence: {score:.2f}")

        if score < 0.2:
            if self.debug:
                print("  [CHECK] Low confidence — answer may not be grounded")
            return f"{answer}\n\n(Low confidence) Answer may not reflect source documents."

        return answer

    def format_filesystem_answer(self, query: str, result: str) -> str:
        """Format ToolBox output into natural language via LLM."""
        prompt = FILESYSTEM_PROMPT.format(query=query, result=result)
        answer = self.models.synthesize(prompt)
        return answer.strip()
```

**Step 4: Run tests to verify they pass**

Run: `cd /Users/ded/Projects/assist/manole && uv run pytest tests/test_reducer.py -v`
Expected: All 12 tests PASS.

**Step 5: Commit**

```bash
git add reducer.py tests/test_reducer.py
git commit -m "feat: add Reducer for answer synthesis and confidence checking"
```

---

### Task 9: Build Pipeline Orchestrator (pipeline.py)

**Files:**
- Create: `pipeline.py`
- Create: `tests/test_pipeline.py`

**Step 1: Write tests**

Create `tests/test_pipeline.py`:

```python
"""Integration tests for AgenticRAG pipeline orchestrator."""
import json
from dataclasses import dataclass, field
from pipeline import AgenticRAG


@dataclass
class FakeSearchResult:
    id: str
    text: str
    score: float
    metadata: dict = field(default_factory=dict)


class FakeModelManager:
    """Returns canned responses for plan/extract/synthesize calls."""
    def __init__(self, plan_responses=None, extract_responses=None, synth_responses=None):
        self.plan_responses = list(plan_responses or [])
        self.extract_responses = list(extract_responses or [])
        self.synth_responses = list(synth_responses or [])
        self.plan_prompts = []
        self.extract_prompts = []
        self.synth_prompts = []

    def plan(self, prompt):
        self.plan_prompts.append(prompt)
        return self.plan_responses.pop(0) if self.plan_responses else "{}"

    def extract(self, prompt):
        self.extract_prompts.append(prompt)
        return self.extract_responses.pop(0) if self.extract_responses else "{}"

    def synthesize(self, prompt):
        self.synth_prompts.append(prompt)
        return self.synth_responses.pop(0) if self.synth_responses else ""

    def load(self):
        pass


class FakeLeannSearcher:
    def __init__(self, results):
        self.results = results

    def search(self, query, top_k=5, metadata_filters=None, **kwargs):
        if metadata_filters:
            source_filter = metadata_filters.get("source", {})
            contains = source_filter.get("contains", "")
            if contains:
                return [r for r in self.results if contains.lower() in r.metadata.get("source", "").lower()][:top_k]
        return self.results[:top_k]


def _make_results(*texts, sources=None):
    if sources is None:
        sources = [f"file{i}.pdf" for i in range(len(texts))]
    return [
        FakeSearchResult(id=str(i), text=t, score=0.9 - i * 0.1, metadata={"source": sources[i]})
        for i, t in enumerate(texts)
    ]


def test_semantic_search_pipeline():
    """Full semantic search flow: plan → search → map → filter → reduce."""
    plan_json = json.dumps({
        "keywords": ["invoice"], "file_filter": "pdf", "source_hint": "Invoice",
        "tool": "semantic_search", "time_filter": None, "tool_actions": [],
    })
    map_responses = [
        json.dumps({"relevant": True, "facts": ["Invoice #123", "Amount: $50"]}),
        json.dumps({"relevant": False, "facts": []}),
    ]
    synth_response = "Found 1 invoice: #123 for $50."

    models = FakeModelManager(
        plan_responses=[plan_json],
        extract_responses=map_responses,
        synth_responses=[synth_response],
    )
    results = _make_results("Invoice #123 for $50", "Weather report", sources=["Invoice_001.pdf", "weather.txt"])
    leann = FakeLeannSearcher(results)

    rag = AgenticRAG(models=models, leann_searcher=leann, data_dir="/tmp/test")
    answer = rag.ask("find invoices")
    assert "#123" in answer
    assert "$50" in answer


def test_filesystem_pipeline():
    """Filesystem queries bypass search entirely."""
    import tempfile
    from pathlib import Path

    tmp = tempfile.mkdtemp()
    (Path(tmp) / "a.pdf").write_text("pdf1")
    (Path(tmp) / "b.pdf").write_text("pdf2")
    (Path(tmp) / "c.txt").write_text("txt1")

    plan_json = json.dumps({
        "keywords": ["PDF", "count"], "file_filter": "pdf", "source_hint": None,
        "tool": "filesystem", "time_filter": None, "tool_actions": ["count"],
    })
    synth_response = "You have 2 PDF files."

    models = FakeModelManager(
        plan_responses=[plan_json],
        synth_responses=[synth_response],
    )
    leann = FakeLeannSearcher([])

    rag = AgenticRAG(models=models, leann_searcher=leann, data_dir=tmp)
    answer = rag.ask("how many PDFs?")
    assert "2" in answer


def test_no_results_returns_message():
    plan_json = json.dumps({
        "keywords": ["quantum"], "file_filter": None, "source_hint": None,
        "tool": "semantic_search", "time_filter": None, "tool_actions": [],
    })
    models = FakeModelManager(plan_responses=[plan_json])
    leann = FakeLeannSearcher([])

    rag = AgenticRAG(models=models, leann_searcher=leann, data_dir="/tmp/test")
    answer = rag.ask("quantum physics?")
    assert "No relevant" in answer


def test_all_chunks_irrelevant_returns_message():
    plan_json = json.dumps({
        "keywords": ["quantum"], "file_filter": None, "source_hint": None,
        "tool": "semantic_search", "time_filter": None, "tool_actions": [],
    })
    map_response = json.dumps({"relevant": False, "facts": []})

    models = FakeModelManager(
        plan_responses=[plan_json],
        extract_responses=[map_response],
    )
    results = _make_results("Cooking recipe")
    leann = FakeLeannSearcher(results)

    rag = AgenticRAG(models=models, leann_searcher=leann, data_dir="/tmp/test")
    answer = rag.ask("quantum physics?")
    assert "No relevant" in answer


def test_search_fallback_when_filters_return_empty():
    """When filtered search returns 0 results, retry unfiltered."""
    plan_json = json.dumps({
        "keywords": ["invoice"], "file_filter": None, "source_hint": "nonexistent",
        "tool": "semantic_search", "time_filter": None, "tool_actions": [],
    })
    map_response = json.dumps({"relevant": True, "facts": ["some fact"]})
    synth_response = "Found something via fallback."

    models = FakeModelManager(
        plan_responses=[plan_json],
        extract_responses=[map_response],
        synth_responses=[synth_response],
    )
    results = _make_results("actual data")
    leann = FakeLeannSearcher(results)

    rag = AgenticRAG(models=models, leann_searcher=leann, data_dir="/tmp/test")
    answer = rag.ask("invoices")
    assert "fallback" in answer.lower() or "Found" in answer


def test_debug_mode_does_not_crash():
    plan_json = json.dumps({
        "keywords": ["test"], "file_filter": None, "source_hint": None,
        "tool": "semantic_search", "time_filter": None, "tool_actions": [],
    })
    map_response = json.dumps({"relevant": True, "facts": ["fact"]})
    synth_response = "answer"

    models = FakeModelManager(
        plan_responses=[plan_json],
        extract_responses=[map_response],
        synth_responses=[synth_response],
    )
    results = _make_results("text")
    leann = FakeLeannSearcher(results)

    rag = AgenticRAG(models=models, leann_searcher=leann, data_dir="/tmp/test", debug=True)
    answer = rag.ask("test query")
    assert answer  # just verify it doesn't crash
```

**Step 2: Run tests to verify they fail**

Run: `cd /Users/ded/Projects/assist/manole && uv run pytest tests/test_pipeline.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'pipeline'`

**Step 3: Implement pipeline.py**

Create `pipeline.py`:

```python
"""AgenticRAG: dual-model agentic RAG pipeline orchestrator."""
import time

from planner import Planner
from searcher import Searcher
from mapper import Mapper
from reducer import Reducer
from toolbox import ToolBox


class AgenticRAG:
    """Orchestrates the full Plan → Search → Map → Filter → Reduce pipeline."""

    def __init__(
        self,
        models,
        leann_searcher,
        data_dir: str,
        top_k: int = 5,
        debug: bool = False,
    ):
        self.planner = Planner(models, debug=debug)
        self.searcher = Searcher(leann_searcher)
        self.mapper = Mapper(models, debug=debug)
        self.reducer = Reducer(models, debug=debug)
        self.toolbox = ToolBox(data_dir)
        self.top_k = top_k
        self.debug = debug

    def _log(self, msg: str):
        if self.debug:
            print(f"  [PIPELINE] {msg}")

    def ask(self, query: str) -> str:
        """Run the full agentic RAG pipeline."""
        t0 = time.time()

        # Stage 1: Plan
        plan = self.planner.plan(query)
        self._log(f"Plan: tool={plan['tool']}")

        # Filesystem path: ToolBox → format → return
        if plan["tool"] == "filesystem":
            result = self.toolbox.execute(plan)
            answer = self.reducer.format_filesystem_answer(query, result)
            self._log(f"Filesystem answer in {time.time() - t0:.1f}s")
            return answer

        # Hybrid path: ToolBox narrows file scope → search within those
        if plan["tool"] == "hybrid":
            file_paths = self.toolbox.get_matching_files(plan)
            self._log(f"Hybrid: {len(file_paths)} files from ToolBox")
            chunks = self.searcher.search(plan, top_k=self.top_k, file_filter_paths=file_paths)
        else:
            # Semantic search path
            chunks = self.searcher.search(plan, top_k=self.top_k)

        # Fallback: retry without filters
        if not chunks:
            self._log("No results with filters, retrying unfiltered")
            chunks = self.searcher.search_unfiltered(plan, top_k=self.top_k)

        if not chunks:
            return "No relevant information found in your files."

        # Stage 3: Map
        mapped = self.mapper.extract_facts(query, chunks)

        # Stage 4: Filter
        relevant = [m for m in mapped if m["relevant"]]
        self._log(f"Filter: {len(relevant)}/{len(mapped)} chunks relevant")

        if not relevant:
            return "No relevant information found in your files."

        # Stage 5: Reduce
        answer = self.reducer.synthesize(query, relevant)

        # Stage 6: Confidence check
        answer = self.reducer.confidence_check(answer, relevant)

        self._log(f"Pipeline completed in {time.time() - t0:.1f}s")
        return answer
```

**Step 4: Run tests to verify they pass**

Run: `cd /Users/ded/Projects/assist/manole && uv run pytest tests/test_pipeline.py -v`
Expected: All 6 tests PASS.

**Step 5: Commit**

```bash
git add pipeline.py tests/test_pipeline.py
git commit -m "feat: add AgenticRAG pipeline orchestrator with three routing paths"
```

---

### Task 10: Update chat.py to use new pipeline

**Files:**
- Modify: `chat.py`

**Step 1: Write the updated chat.py**

Slim down `chat.py` to be a thin CLI shell:

- Remove: `parse_json`, `confidence_score`, `PLANNER_PROMPT`, `MAP_PROMPT`, `REDUCE_PROMPT`, `AgenticRAG` class, `TOP_K`, `DEBUG_SOURCES` constants
- Keep: `get_index_name`, `build_index`, `find_index_path`, `chat_loop`, `main`
- Change: `chat_loop` creates the new `AgenticRAG` from `pipeline.py`

The new `chat_loop` should look like:

```python
def chat_loop(index_name: str, data_dir: str):
    from leann import LeannSearcher
    from models import ModelManager
    from pipeline import AgenticRAG

    print("\nLoading models...")
    t0 = time.time()

    index_path = find_index_path(index_name)
    print(f"Using index: {index_path}")

    # Load both GGUF models
    models = ModelManager()
    models.load()

    # Create leann searcher for vector search
    leann_searcher = LeannSearcher(index_path)

    # Create pipeline
    rag = AgenticRAG(
        models=models,
        leann_searcher=leann_searcher,
        data_dir=data_dir,
        top_k=5,
        debug=True,
    )

    print(f"Ready in {time.time() - t0:.1f}s")
    print("=" * 50)
    print("Ask anything about your files. Type 'quit' to exit.")
    print("Type 'debug' to toggle pipeline trace.")
    print("=" * 50)

    while True:
        try:
            query = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not query:
            continue
        if query.lower() in ("quit", "exit", "q"):
            print("Bye!")
            break
        if query.lower() == "debug":
            rag.debug = not rag.debug
            print(f"Pipeline trace: {'ON' if rag.debug else 'OFF'}")
            continue

        t0 = time.time()
        response = rag.ask(query)
        elapsed = time.time() - t0
        print(f"\n{response}")
        print(f"\n({elapsed:.1f}s)")
```

Update `main()` to pass `data_dir` to `chat_loop`:

```python
def main():
    args = sys.argv[1:]

    if "--reuse" in args:
        idx = args.index("--reuse")
        if idx + 1 >= len(args):
            print("Error: --reuse requires an index name")
            sys.exit(1)
        index_name = args[idx + 1]
        data_dir = args[idx + 2] if idx + 2 < len(args) else "./test_data"
        chat_loop(index_name, str(Path(data_dir).resolve()))
        return

    force = "--force" in args
    args = [a for a in args if a != "--force"]

    if args:
        data_dir = Path(args[0]).resolve()
    else:
        data_dir = Path("./test_data").resolve()

    if not data_dir.is_dir():
        print(f"Error: {data_dir} is not a directory")
        sys.exit(1)

    index_name = build_index(data_dir, force=force)
    chat_loop(index_name, str(data_dir))
```

**Step 2: Run all tests**

Run: `cd /Users/ded/Projects/assist/manole && uv run pytest tests/ -v --ignore=tests/test_agentic_rag.py`
Expected: All tests from tasks 2-9 PASS. The old `test_agentic_rag.py` is ignored since it tests the old pipeline.

**Step 3: Commit**

```bash
git add chat.py
git commit -m "refactor: slim chat.py to thin CLI shell, delegate to pipeline.py"
```

---

### Task 11: Clean up old test file

**Files:**
- Delete or rename: `tests/test_agentic_rag.py`

**Step 1: Remove old test file**

The old `tests/test_agentic_rag.py` tests the monolithic `AgenticRAG` from `chat.py` which no longer exists. Remove it:

```bash
git rm tests/test_agentic_rag.py
```

**Step 2: Run full test suite**

Run: `cd /Users/ded/Projects/assist/manole && uv run pytest tests/ -v`
Expected: All tests PASS (from test_parser, test_models, test_planner, test_toolbox, test_searcher, test_mapper, test_reducer, test_pipeline).

**Step 3: Commit**

```bash
git commit -m "chore: remove old monolithic agentic RAG tests"
```

---

### Task 12: Download GGUF models and smoke test

**Files:**
- Create: `models/` directory
- No code changes

**Step 1: Create models directory**

```bash
mkdir -p /Users/ded/Projects/assist/manole/models
```

**Step 2: Download models from HuggingFace**

Download the Q4_0 quantized GGUF files for both models. Exact URLs depend on HuggingFace repo structure — check available quantizations:

```bash
# Check what's available (adjust repo names as needed)
huggingface-cli search LFM2-350M-Extract gguf
huggingface-cli search LFM2-1.2B-RAG gguf
```

Download to `models/` directory:

```bash
cd /Users/ded/Projects/assist/manole/models
# Download planner model
huggingface-cli download <repo>/LFM2-350M-Extract-GGUF LFM2-350M-Extract-Q4_0.gguf --local-dir .
# Download RAG model
huggingface-cli download <repo>/LFM2-1.2B-RAG-GGUF LFM2-1.2B-RAG-Q4_0.gguf --local-dir .
```

**Step 3: Verify models load**

```bash
cd /Users/ded/Projects/assist/manole
uv run python -c "
from models import ModelManager
m = ModelManager()
m.load()
print('Planner:', m.plan('test'))
print('RAG:', m.extract('test'))
print('Both models loaded successfully!')
"
```

Expected: Both models load and produce output (content doesn't matter, just no crashes).

**Step 4: Add models to .gitignore**

Append to `.gitignore`:

```
models/
```

**Step 5: End-to-end smoke test**

```bash
cd /Users/ded/Projects/assist/manole
uv run python chat.py --reuse test_data ./test_data
```

Try these queries:
1. `"any invoices?"` — should route to semantic_search
2. `"how many PDF files?"` — should route to filesystem
3. `"summarize files modified today"` — should route to hybrid

**Step 6: Commit**

```bash
git add .gitignore
git commit -m "chore: add models/ to gitignore for GGUF files"
```

---

## Summary

| Task | Description | New Files | Test Count |
|------|-------------|-----------|------------|
| 1 | Add llama-cpp-python dep | — | 0 |
| 2 | Extract parser.py | parser.py, tests/test_parser.py | 9 |
| 3 | Build ModelManager | models.py, tests/test_models.py | 5 |
| 4 | Build Planner | planner.py, tests/test_planner.py | 6 |
| 5 | Build Smart ToolBox | toolbox.py, tests/test_toolbox.py | 16 |
| 6 | Build Searcher | searcher.py, tests/test_searcher.py | 7 |
| 7 | Build Mapper | mapper.py, tests/test_mapper.py | 6 |
| 8 | Build Reducer | reducer.py, tests/test_reducer.py | 12 |
| 9 | Build Pipeline | pipeline.py, tests/test_pipeline.py | 6 |
| 10 | Update chat.py | — | 0 |
| 11 | Clean up old tests | — | 0 |
| 12 | Download models + smoke test | — | 0 |
| **Total** | | **16 new files** | **~67 tests** |
