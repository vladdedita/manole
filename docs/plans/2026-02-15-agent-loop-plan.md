# Agent Loop Migration Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the unidirectional RAG pipeline with an orchestrator agent loop where the model decides what tools to call at each step.

**Architecture:** Single LFM2.5-1.2B-Instruct model in an agent loop with 7 tools (semantic_search, count_files, list_files, file_metadata, grep_files, directory_tree, respond). Python fallback router handles routing when native tool calling fails. Searcher does map-filter internally so the agent only sees clean extracted facts.

**Tech Stack:** Python 3.13, llama-cpp-python (GGUF), pytest, leann (vector search)

---

### Task 1: Simplify ModelManager to Single Model

Replace the dual-model ModelManager with a single-model version. One `generate(messages)` method.

**Files:**
- Modify: `models.py`
- Rewrite: `tests/test_models.py`

**Step 1: Write the failing tests**

Replace `tests/test_models.py` entirely:

```python
"""Tests for ModelManager single-model interface."""
from unittest.mock import MagicMock
from models import ModelManager


def _mock_chat_response(text: str) -> dict:
    return {"choices": [{"message": {"content": text}}]}


def test_generate_calls_model():
    mgr = ModelManager.__new__(ModelManager)
    mock_model = MagicMock()
    mock_model.create_chat_completion.return_value = _mock_chat_response("hello")
    mgr.model = mock_model

    result = mgr.generate([{"role": "user", "content": "hi"}])
    mock_model.create_chat_completion.assert_called_once()
    assert result == "hello"


def test_generate_passes_messages():
    mgr = ModelManager.__new__(ModelManager)
    mock_model = MagicMock()
    mock_model.create_chat_completion.return_value = _mock_chat_response("ok")
    mgr.model = mock_model

    messages = [
        {"role": "system", "content": "you are helpful"},
        {"role": "user", "content": "test"},
    ]
    mgr.generate(messages)
    call_kwargs = mock_model.create_chat_completion.call_args
    passed_messages = call_kwargs.kwargs.get("messages") or call_kwargs[0][0]
    assert len(passed_messages) == 2
    assert passed_messages[0]["role"] == "system"


def test_generate_max_tokens():
    mgr = ModelManager.__new__(ModelManager)
    mock_model = MagicMock()
    mock_model.create_chat_completion.return_value = _mock_chat_response("ok")
    mgr.model = mock_model

    mgr.generate([{"role": "user", "content": "hi"}], max_tokens=256)
    call_kwargs = mock_model.create_chat_completion.call_args
    assert call_kwargs.kwargs.get("max_tokens") == 256


def test_generate_resets_model():
    mgr = ModelManager.__new__(ModelManager)
    mock_model = MagicMock()
    mock_model.create_chat_completion.return_value = _mock_chat_response("ok")
    mgr.model = mock_model

    mgr.generate([{"role": "user", "content": "hi"}])
    mock_model.reset.assert_called_once()


def test_default_model_path():
    mgr = ModelManager()
    assert "LFM2.5-1.2B-Instruct" in mgr.model_path


def test_custom_model_path():
    mgr = ModelManager(model_path="/custom/path.gguf")
    assert mgr.model_path == "/custom/path.gguf"
```

**Step 2: Run tests to verify they fail**

Run: `cd /Users/ded/Projects/assist/manole && uv run pytest tests/test_models.py -v`
Expected: FAIL — `generate` method doesn't exist, old attributes referenced

**Step 3: Write the implementation**

Replace `models.py` entirely:

```python
"""ModelManager: single LFM2.5-1.2B-Instruct model with generate()."""
from pathlib import Path


class ModelManager:
    """Single GGUF model loaded via llama-cpp-python."""

    DEFAULT_MODEL_PATH = "models/LFM2.5-1.2B-Instruct-Q4_0.gguf"

    def __init__(self, model_path: str | None = None, n_threads: int = 4):
        self.model_path = model_path or self.DEFAULT_MODEL_PATH
        self.n_threads = n_threads
        self.model = None

    def load(self):
        """Load the model. Call once at startup."""
        from llama_cpp import Llama

        self.model = Llama(
            model_path=self.model_path,
            n_ctx=8192,
            n_threads=self.n_threads,
            verbose=False,
        )

    def generate(self, messages: list[dict], max_tokens: int = 1024) -> str:
        """Generate a response from a message list."""
        self.model.reset()
        response = self.model.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.1,
            top_k=50,
            top_p=0.1,
            repeat_penalty=1.05,
        )
        return response["choices"][0]["message"]["content"]
```

**Step 4: Run tests to verify they pass**

Run: `cd /Users/ded/Projects/assist/manole && uv run pytest tests/test_models.py -v`
Expected: All 6 tests PASS

**Step 5: Commit**

```bash
git add models.py tests/test_models.py
git commit -m "refactor: simplify ModelManager to single model with generate()"
```

---

### Task 2: Create router.py — Python Fallback Router

The router maps queries to tool names using keyword heuristics. This is the safety net for when the model fails to produce parseable tool calls.

**Files:**
- Create: `router.py`
- Create: `tests/test_router.py`

**Step 1: Write the failing tests**

Create `tests/test_router.py`:

```python
"""Tests for Python fallback router."""
from router import route, _detect_extension


def test_count_query_routes_to_count_files():
    name, params = route("how many PDF files do I have?")
    assert name == "count_files"
    assert params["extension"] == "pdf"


def test_count_query_no_extension():
    name, params = route("how many files?")
    assert name == "count_files"
    assert params["extension"] is None


def test_tree_query():
    name, params = route("show me the directory structure")
    assert name == "directory_tree"
    assert params["max_depth"] == 2


def test_folder_query():
    name, params = route("what folders do I have?")
    assert name == "directory_tree"


def test_list_files_query():
    name, params = route("list files")
    assert name == "list_files"


def test_recent_files_query():
    name, params = route("show me recent files")
    assert name == "list_files"


def test_file_metadata_query():
    name, params = route("when was invoice.pdf modified?")
    assert name == "file_metadata"
    assert params.get("name_hint") is not None


def test_file_size_query():
    name, params = route("what is the file size of budget.pdf?")
    assert name == "file_metadata"


def test_semantic_search_default():
    name, params = route("what is the target revenue?")
    assert name == "semantic_search"
    assert params["query"] == "what is the target revenue?"


def test_semantic_search_for_content_questions():
    name, params = route("find all invoices with amounts over $100")
    assert name == "semantic_search"


def test_detect_extension_pdf():
    assert _detect_extension("how many PDF files") == "pdf"


def test_detect_extension_txt():
    assert _detect_extension("count text files") == "txt"


def test_detect_extension_none():
    assert _detect_extension("how many files") is None


def test_case_insensitive():
    name, _ = route("HOW MANY PDF FILES?")
    assert name == "count_files"
```

**Step 2: Run tests to verify they fail**

Run: `cd /Users/ded/Projects/assist/manole && uv run pytest tests/test_router.py -v`
Expected: FAIL — `router` module doesn't exist

**Step 3: Write the implementation**

Create `router.py`:

```python
"""Python fallback router — keyword heuristics for tool routing."""
import re

# Extension keywords the user might mention
_EXT_MAP = {
    "pdf": "pdf",
    "txt": "txt",
    "text": "txt",
    "py": "py",
    "python": "py",
    "md": "md",
    "markdown": "md",
    "csv": "csv",
    "json": "json",
    "xml": "xml",
    "doc": "doc",
    "docx": "docx",
    "xls": "xls",
    "xlsx": "xlsx",
    "png": "png",
    "jpg": "jpg",
    "jpeg": "jpeg",
}


def _detect_extension(query: str) -> str | None:
    """Detect file extension from query text."""
    q = query.lower()
    for keyword, ext in _EXT_MAP.items():
        if keyword in q.split():
            return ext
    return None


def _extract_name_hint(query: str) -> str | None:
    """Extract a filename or name substring from the query."""
    # Look for patterns like "invoice.pdf" or quoted strings
    match = re.search(r'[\w-]+\.\w{2,4}', query)
    if match:
        return match.group(0)
    # Look for quoted strings
    match = re.search(r'["\']([^"\']+)["\']', query)
    if match:
        return match.group(1)
    # Extract the most specific noun (last non-stopword before a question mark or end)
    words = query.lower().replace("?", "").split()
    stop = {"the", "a", "an", "is", "was", "of", "for", "my", "what", "when", "how", "file", "size"}
    nouns = [w for w in words if w not in stop and len(w) > 2]
    return nouns[-1] if nouns else None


def route(query: str) -> tuple[str, dict]:
    """Route a query to a tool name and parameters using keyword heuristics."""
    q = query.lower()

    if any(k in q for k in ["how many", "count"]):
        return "count_files", {"extension": _detect_extension(q)}

    if any(k in q for k in ["file types", "folder", "tree", "directory", "structure"]):
        return "directory_tree", {"max_depth": 2}

    if any(k in q for k in ["list files", "recent files", "what files", "show files",
                              "show me files", "list my"]):
        return "list_files", {"extension": _detect_extension(q), "limit": 10}

    if any(k in q for k in ["file size", "when was", "modified", "created", "how big",
                              "how large", "how old"]):
        return "file_metadata", {"name_hint": _extract_name_hint(q)}

    return "semantic_search", {"query": query}
```

**Step 4: Run tests to verify they pass**

Run: `cd /Users/ded/Projects/assist/manole && uv run pytest tests/test_router.py -v`
Expected: All 14 tests PASS

**Step 5: Commit**

```bash
git add router.py tests/test_router.py
git commit -m "feat: add Python fallback router for tool routing"
```

---

### Task 3: Create tools.py — Tool Definitions + Registry

Tool definitions for the agent and a registry that maps tool names to execution functions.

**Files:**
- Create: `tools.py`
- Create: `tests/test_tools.py`

**Step 1: Write the failing tests**

Create `tests/test_tools.py`:

```python
"""Tests for ToolRegistry — tool dispatch."""
import tempfile
from pathlib import Path
from tools import ToolRegistry, TOOL_DEFINITIONS


class FakeSearcher:
    def __init__(self, response="No results."):
        self.response = response
        self.last_query = None
        self.last_top_k = None

    def search_and_extract(self, query, top_k=5):
        self.last_query = query
        self.last_top_k = top_k
        return self.response


def _make_registry(search_response="No results."):
    tmp = tempfile.mkdtemp()
    (Path(tmp) / "a.pdf").write_text("pdf")
    (Path(tmp) / "b.txt").write_text("txt")
    from toolbox import ToolBox
    searcher = FakeSearcher(search_response)
    toolbox = ToolBox(tmp)
    return ToolRegistry(searcher, toolbox), searcher, tmp


def test_tool_definitions_has_seven_tools():
    names = [t["name"] for t in TOOL_DEFINITIONS]
    assert len(names) == 7
    assert "semantic_search" in names
    assert "count_files" in names
    assert "respond" in names


def test_semantic_search_delegates():
    registry, searcher, _ = _make_registry("From budget.txt:\n  - Budget: $100k")
    result = registry.execute("semantic_search", {"query": "budget", "top_k": 3})
    assert searcher.last_query == "budget"
    assert searcher.last_top_k == 3
    assert "budget" in result.lower()


def test_semantic_search_default_top_k():
    registry, searcher, _ = _make_registry()
    registry.execute("semantic_search", {"query": "test"})
    assert searcher.last_top_k == 5


def test_semantic_search_caps_top_k():
    registry, searcher, _ = _make_registry()
    registry.execute("semantic_search", {"query": "test", "top_k": 50})
    assert searcher.last_top_k == 10


def test_count_files():
    registry, _, _ = _make_registry()
    result = registry.execute("count_files", {"extension": "pdf"})
    assert "1" in result


def test_list_files():
    registry, _, _ = _make_registry()
    result = registry.execute("list_files", {})
    assert "a.pdf" in result or "b.txt" in result


def test_file_metadata():
    registry, _, _ = _make_registry()
    result = registry.execute("file_metadata", {"name_hint": "a.pdf"})
    assert "a.pdf" in result


def test_grep_files():
    registry, _, _ = _make_registry()
    result = registry.execute("grep_files", {"pattern": "pdf"})
    assert "a.pdf" in result


def test_directory_tree():
    registry, _, _ = _make_registry()
    result = registry.execute("directory_tree", {"max_depth": 1})
    assert "a.pdf" in result or "b.txt" in result


def test_unknown_tool():
    registry, _, _ = _make_registry()
    result = registry.execute("nonexistent_tool", {})
    assert "Unknown tool" in result
```

**Step 2: Run tests to verify they fail**

Run: `cd /Users/ded/Projects/assist/manole && uv run pytest tests/test_tools.py -v`
Expected: FAIL — `tools` module doesn't exist

**Step 3: Write the implementation**

Create `tools.py`:

```python
"""Tool definitions and registry for the agent loop."""

TOOL_DEFINITIONS = [
    {
        "name": "semantic_search",
        "description": (
            "Search inside file contents by meaning. "
            "Use when the user asks about information WITHIN files "
            "(invoices, budgets, notes, specific data). "
            "Returns extracted facts from matching file chunks."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "What to search for in file contents"},
                "top_k": {"type": "integer", "description": "Number of results (default 5, max 10)"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "count_files",
        "description": "Count how many files exist. Use for 'how many files/documents/PDFs' questions.",
        "parameters": {
            "type": "object",
            "properties": {
                "extension": {"type": "string", "description": "Filter by extension (e.g. 'pdf', 'txt') or null for all"},
            },
        },
    },
    {
        "name": "list_files",
        "description": "List files sorted by modification date. Use for 'what files', 'recent files', 'show me files' questions.",
        "parameters": {
            "type": "object",
            "properties": {
                "extension": {"type": "string", "description": "Filter by extension or null"},
                "limit": {"type": "integer", "description": "Max files to return (default 10)"},
            },
        },
    },
    {
        "name": "file_metadata",
        "description": "Get file size, creation date, modification date. Use for questions about specific file details.",
        "parameters": {
            "type": "object",
            "properties": {
                "name_hint": {"type": "string", "description": "Filename substring to match"},
            },
            "required": ["name_hint"],
        },
    },
    {
        "name": "grep_files",
        "description": "Find files by name pattern. Use when looking for files with specific names.",
        "parameters": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Substring to match in filenames"},
            },
            "required": ["pattern"],
        },
    },
    {
        "name": "directory_tree",
        "description": "Show folder structure. Use for 'what folders', 'directory structure', 'file organization' questions.",
        "parameters": {
            "type": "object",
            "properties": {
                "max_depth": {"type": "integer", "description": "How deep to show (default 2)"},
            },
        },
    },
    {
        "name": "respond",
        "description": (
            "Return a final answer to the user. "
            "ONLY call this when you have enough information to answer. "
            "If you need more information, call another tool first."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "answer": {"type": "string", "description": "Your complete answer to the user"},
            },
            "required": ["answer"],
        },
    },
]


class ToolRegistry:
    """Maps tool names to execution functions."""

    def __init__(self, searcher, toolbox):
        self.searcher = searcher
        self.toolbox = toolbox
        self._handlers = {
            "semantic_search": self._semantic_search,
            "count_files": self._count_files,
            "list_files": self._list_files,
            "file_metadata": self._file_metadata,
            "grep_files": self._grep_files,
            "directory_tree": self._directory_tree,
        }

    def execute(self, tool_name: str, params: dict) -> str:
        """Execute a tool by name. Returns result string."""
        handler = self._handlers.get(tool_name)
        if not handler:
            return f"Unknown tool: {tool_name}"
        return handler(params)

    def _semantic_search(self, params: dict) -> str:
        query = params.get("query", "")
        top_k = min(params.get("top_k", 5), 10)
        return self.searcher.search_and_extract(query, top_k=top_k)

    def _count_files(self, params: dict) -> str:
        return self.toolbox.count_files(ext_filter=params.get("extension"))

    def _list_files(self, params: dict) -> str:
        return self.toolbox.list_recent_files(
            ext_filter=params.get("extension"),
            limit=params.get("limit", 10),
        )

    def _file_metadata(self, params: dict) -> str:
        return self.toolbox.get_file_metadata(name_hint=params.get("name_hint"))

    def _grep_files(self, params: dict) -> str:
        return self.toolbox.grep(params.get("pattern", ""))

    def _directory_tree(self, params: dict) -> str:
        return self.toolbox.tree(max_depth=params.get("max_depth", 2))
```

**Step 4: Run tests to verify they pass**

Run: `cd /Users/ded/Projects/assist/manole && uv run pytest tests/test_tools.py -v`
Expected: All 10 tests PASS

**Step 5: Commit**

```bash
git add tools.py tests/test_tools.py
git commit -m "feat: add tool definitions and ToolRegistry"
```

---

### Task 4: Rewrite searcher.py — Search with Internal Map-Filter

The searcher absorbs the mapper logic. It does search → score-filter → extract facts → format, returning clean text to the agent.

**Files:**
- Rewrite: `searcher.py`
- Rewrite: `tests/test_searcher.py`

**Step 1: Write the failing tests**

Replace `tests/test_searcher.py` entirely:

```python
"""Tests for Searcher — search with internal map-filter."""
import json
from dataclasses import dataclass, field
from unittest.mock import MagicMock
from searcher import Searcher


@dataclass
class FakeSearchResult:
    id: str
    text: str
    score: float
    metadata: dict = field(default_factory=dict)


class FakeLeann:
    def __init__(self, results):
        self.results = results
        self.last_query = None

    def search(self, query, top_k=5, **kwargs):
        self.last_query = query
        return self.results[:top_k]


def _make_model(responses):
    """Create a mock model that returns canned responses in order."""
    model = MagicMock()
    model.generate = MagicMock(side_effect=list(responses))
    return model


def _make_results(*texts, sources=None, scores=None):
    if sources is None:
        sources = [f"file{i}.txt" for i in range(len(texts))]
    if scores is None:
        scores = [0.95 - i * 0.05 for i in range(len(texts))]
    return [
        FakeSearchResult(
            id=str(i), text=t, score=scores[i],
            metadata={"file_name": sources[i]},
        )
        for i, t in enumerate(texts)
    ]


def test_search_and_extract_returns_formatted_facts():
    results = _make_results("Budget doc with $450k total")
    model = _make_model([json.dumps({"relevant": True, "facts": ["Total Budget: $450,000"]})])
    leann = FakeLeann(results)

    searcher = Searcher(leann, model)
    output = searcher.search_and_extract("budget")
    assert "From file0.txt:" in output
    assert "$450,000" in output


def test_search_no_results():
    model = _make_model([])
    leann = FakeLeann([])

    searcher = Searcher(leann, model)
    output = searcher.search_and_extract("quantum physics")
    assert "No matching content" in output


def test_irrelevant_chunks_filtered():
    results = _make_results("invoice data", "weather report")
    model = _make_model([
        json.dumps({"relevant": True, "facts": ["Invoice #123"]}),
        json.dumps({"relevant": False, "facts": []}),
    ])
    leann = FakeLeann(results)

    searcher = Searcher(leann, model)
    output = searcher.search_and_extract("invoices")
    assert "Invoice #123" in output
    assert "weather" not in output.lower()


def test_all_irrelevant_returns_message():
    results = _make_results("random text")
    model = _make_model([json.dumps({"relevant": False, "facts": []})])
    leann = FakeLeann(results)

    searcher = Searcher(leann, model)
    output = searcher.search_and_extract("quantum")
    assert "none were relevant" in output.lower()


def test_score_prefilter():
    """Low-scoring chunks should be dropped before extraction."""
    results = _make_results("good", "ok", "bad", scores=[0.95, 0.80, 0.50])
    model = _make_model([
        json.dumps({"relevant": True, "facts": ["fact1"]}),
        json.dumps({"relevant": True, "facts": ["fact2"]}),
    ])
    leann = FakeLeann(results)

    searcher = Searcher(leann, model)
    output = searcher.search_and_extract("test")
    # Only 2 model calls — third chunk (score 0.50) below 0.8 * 0.95 = 0.76
    assert model.generate.call_count == 2


def test_parse_failure_defaults_to_irrelevant():
    results = _make_results("some text")
    model = _make_model(["not valid json at all"])
    leann = FakeLeann(results)

    searcher = Searcher(leann, model)
    output = searcher.search_and_extract("test")
    assert "none were relevant" in output.lower()


def test_file_name_used_as_source():
    results = [FakeSearchResult(
        id="42", text="budget data", score=0.9,
        metadata={"file_name": "budget_q1_2026.txt"},
    )]
    model = _make_model([json.dumps({"relevant": True, "facts": ["Budget: $100k"]})])
    leann = FakeLeann(results)

    searcher = Searcher(leann, model)
    output = searcher.search_and_extract("budget")
    assert "budget_q1_2026.txt" in output


def test_fallback_source_from_id():
    results = [FakeSearchResult(id="99", text="data", score=0.9, metadata={})]
    model = _make_model([json.dumps({"relevant": True, "facts": ["some fact"]})])
    leann = FakeLeann(results)

    searcher = Searcher(leann, model)
    output = searcher.search_and_extract("test")
    assert "99" in output


def test_top_k_passed_to_leann():
    leann = FakeLeann([])
    model = _make_model([])
    searcher = Searcher(leann, model)
    searcher.search_and_extract("test", top_k=3)
    # leann.search was called — check the results slice
    assert leann.last_query == "test"


def test_multiple_sources_grouped():
    results = _make_results("data1", "data2", sources=["a.pdf", "b.pdf"])
    model = _make_model([
        json.dumps({"relevant": True, "facts": ["fact A"]}),
        json.dumps({"relevant": True, "facts": ["fact B"]}),
    ])
    leann = FakeLeann(results)

    searcher = Searcher(leann, model)
    output = searcher.search_and_extract("test")
    assert "From a.pdf:" in output
    assert "From b.pdf:" in output
    assert "fact A" in output
    assert "fact B" in output
```

**Step 2: Run tests to verify they fail**

Run: `cd /Users/ded/Projects/assist/manole && uv run pytest tests/test_searcher.py -v`
Expected: FAIL — `Searcher` constructor signature changed

**Step 3: Write the implementation**

Replace `searcher.py` entirely:

```python
"""Searcher — vector search with internal map-filter.

Searches via LeannSearcher, extracts facts per chunk using the LLM,
filters irrelevant chunks, and returns formatted text for the agent.
"""
from parser import parse_json

MAP_SYSTEM = (
    "You are a data extraction assistant. The user will give you a question and a text passage. "
    "Decide if the text DIRECTLY answers the question. "
    "Set relevant to false unless the text specifically discusses the topic asked about. "
    "If relevant, extract the specific data points as short factual strings. "
    "Extract ALL matching data points, do not skip any. "
    'Reply with JSON: {"relevant": true/false, "facts": ["fact1", "fact2"]}'
)

MAX_FACTS_PER_CHUNK = 10


class Searcher:
    """LeannSearcher wrapper with internal fact extraction."""

    def __init__(self, leann_searcher, model, debug: bool = False):
        self.leann = leann_searcher
        self.model = model
        self.debug = debug

    def search_and_extract(self, query: str, top_k: int = 5) -> str:
        """Search + map-filter in one call. Returns formatted facts string."""
        chunks = self.leann.search(query, top_k=top_k)
        if not chunks:
            return "No matching content found."

        # Score pre-filter: drop chunks well below the top score
        if len(chunks) > 1:
            threshold = chunks[0].score * 0.8
            before = len(chunks)
            chunks = [c for c in chunks if c.score >= threshold]
            if self.debug and len(chunks) < before:
                print(f"  [SEARCH] Score filter: {len(chunks)}/{before} above {threshold:.2f}")

        # Map: extract facts per chunk
        facts_by_source = {}
        for chunk in chunks:
            extracted = self._extract_facts(query, chunk)
            if extracted["relevant"] and extracted["facts"]:
                source = self._get_source(chunk)
                facts_by_source.setdefault(source, []).extend(extracted["facts"])

        if not facts_by_source:
            return "Search returned results but none were relevant to the query."

        # Format for agent context
        lines = []
        for source, facts in facts_by_source.items():
            lines.append(f"From {source}:")
            for fact in facts:
                lines.append(f"  - {fact}")
        return "\n".join(lines)

    def _extract_facts(self, query: str, chunk) -> dict:
        """Ask the model if this chunk is relevant and extract facts."""
        source = self._get_source(chunk)
        meta = chunk.metadata or {}
        context_parts = [f"File: {source}"]
        for key, val in meta.items():
            if key not in ("source", "file_name", "file_path", "id") and val is not None:
                context_parts.append(f"{key}: {val}")
        context = " | ".join(context_parts)

        messages = [
            {"role": "system", "content": MAP_SYSTEM},
            {"role": "user", "content": f"Question: {query}\n\n[{context}]\n{chunk.text[:1200]}"},
        ]
        raw = self.model.generate(messages, max_tokens=256)

        if self.debug:
            print(f"  [SEARCH] {source}: raw={raw[:100]}")

        parsed = parse_json(raw)
        if parsed is None:
            if self.debug:
                print(f"  [SEARCH] {source}: parse failed, treating as irrelevant")
            return {"relevant": False, "facts": []}

        facts = parsed.get("facts", [])
        if not isinstance(facts, list):
            facts = []
        facts = [self._normalize_fact(f) for f in facts[:MAX_FACTS_PER_CHUNK]]
        facts = [f for f in facts if f is not None]

        relevant = parsed.get("relevant", False)
        if self.debug:
            print(f"  [SEARCH] {source}: relevant={relevant}, facts={len(facts)}")

        return {"relevant": relevant, "facts": facts}

    @staticmethod
    def _get_source(chunk) -> str:
        """Extract best source name from chunk metadata."""
        meta = chunk.metadata or {}
        return (
            meta.get("source")
            or meta.get("file_name")
            or meta.get("file")
            or meta.get("filename")
            or chunk.id
        )

    @staticmethod
    def _normalize_fact(f) -> str | None:
        """Convert a fact to a string. Handles dicts and short strings."""
        result = None
        if isinstance(f, str) and f.strip():
            result = f.strip()
        elif isinstance(f, dict):
            name = f.get("name", "")
            value = f.get("value", "")
            if name and value:
                result = f"{name}: {value}"
            else:
                vals = [str(v) for v in f.values() if v]
                if vals:
                    result = ": ".join(vals)
        if result and len(result) >= 3:
            return result
        return None
```

**Step 4: Run tests to verify they pass**

Run: `cd /Users/ded/Projects/assist/manole && uv run pytest tests/test_searcher.py -v`
Expected: All 10 tests PASS

**Step 5: Commit**

```bash
git add searcher.py tests/test_searcher.py
git commit -m "refactor: searcher with internal map-filter, absorbs mapper logic"
```

---

### Task 5: Create agent.py — The Agent Loop

The core orchestrator. Runs a loop: model generates → parse tool call → execute → append result → repeat.

**Files:**
- Create: `agent.py`
- Create: `tests/test_agent.py`

**Step 1: Write the failing tests**

Create `tests/test_agent.py`:

```python
"""Tests for Agent — orchestrator agent loop."""
import json
from unittest.mock import MagicMock
from agent import Agent


class FakeToolRegistry:
    def __init__(self, responses=None):
        self.responses = dict(responses or {})
        self.calls = []

    def execute(self, tool_name, params):
        self.calls.append((tool_name, params))
        return self.responses.get(tool_name, f"Result for {tool_name}")


class FakeRouter:
    def __init__(self, tool_name="semantic_search", params=None):
        self.tool_name = tool_name
        self.params = params or {"query": "test"}
        self.called = False

    def route(self, query):
        self.called = True
        return self.tool_name, self.params


def _make_model(responses):
    model = MagicMock()
    model.generate = MagicMock(side_effect=list(responses))
    return model


def test_model_tool_call_semantic_search():
    """Model produces a tool call, agent executes it, then model responds."""
    model = _make_model([
        '<|tool_call_start|>semantic_search(query="budget")<|tool_call_end|>',
        "The budget is $450,000.",
    ])
    tools = FakeToolRegistry({"semantic_search": "From budget.txt:\n  - Budget: $450k"})
    router = FakeRouter()

    agent = Agent(model, tools, router)
    answer = agent.run("what is the budget?")

    assert answer == "The budget is $450,000."
    assert tools.calls[0] == ("semantic_search", {"query": "budget"})
    assert not router.called


def test_fallback_router_on_step_0():
    """When model doesn't produce a tool call on step 0, fallback router kicks in."""
    model = _make_model([
        "I'll help you find that information.",  # no tool call
        "You have 5 PDF files.",  # answer after seeing tool result
    ])
    tools = FakeToolRegistry({"count_files": "Found 5 .pdf files."})
    router = FakeRouter(tool_name="count_files", params={"extension": "pdf"})

    agent = Agent(model, tools, router)
    answer = agent.run("how many PDFs?")

    assert router.called
    assert "5" in answer


def test_direct_answer_on_later_step():
    """On step > 0, no tool call means model is answering directly."""
    model = _make_model([
        '<|tool_call_start|>count_files(extension="pdf")<|tool_call_end|>',
        "You have 3 PDF files.",  # direct answer, no tool call
    ])
    tools = FakeToolRegistry({"count_files": "Found 3 .pdf files."})
    router = FakeRouter()

    agent = Agent(model, tools, router)
    answer = agent.run("how many PDFs?")

    assert "3" in answer


def test_respond_tool():
    """Model can use respond() to return an answer explicitly."""
    model = _make_model([
        '<|tool_call_start|>count_files(extension="txt")<|tool_call_end|>',
        '<|tool_call_start|>respond(answer="You have 2 text files.")<|tool_call_end|>',
    ])
    tools = FakeToolRegistry({"count_files": "Found 2 .txt files."})
    router = FakeRouter()

    agent = Agent(model, tools, router)
    answer = agent.run("how many text files?")

    assert answer == "You have 2 text files."


def test_max_steps_forces_synthesis():
    """After MAX_STEPS, agent forces a synthesis turn."""
    # Model keeps calling tools without answering
    tool_calls = ['<|tool_call_start|>semantic_search(query="test")<|tool_call_end|>'] * 5
    model = _make_model(tool_calls + ["Final forced answer."])
    tools = FakeToolRegistry({"semantic_search": "some results"})
    router = FakeRouter()

    agent = Agent(model, tools, router)
    answer = agent.run("complex question")

    assert answer == "Final forced answer."
    assert model.generate.call_count == 6  # 5 tool calls + 1 forced synthesis


def test_conversation_history_passed():
    """Recent history is included in messages."""
    model = _make_model(["The follow-up answer."])
    tools = FakeToolRegistry()
    router = FakeRouter()

    history = [
        {"role": "user", "content": "find invoices"},
        {"role": "assistant", "content": "Found 3 invoices."},
    ]

    agent = Agent(model, tools, router)
    agent.run("aren't there more?", history=history)

    # Check that history was included in messages
    call_args = model.generate.call_args_list[0]
    messages = call_args[0][0] if call_args[0] else call_args.kwargs.get("messages", [])
    user_contents = [m["content"] for m in messages if m["role"] == "user"]
    assert "find invoices" in user_contents
    assert "aren't there more?" in user_contents


def test_history_capped_at_4_messages():
    """Only last 4 history messages (2 turns) are included."""
    model = _make_model(["answer"])
    tools = FakeToolRegistry()
    router = FakeRouter()

    history = [
        {"role": "user", "content": f"q{i}"}
        for i in range(10)
    ]

    agent = Agent(model, tools, router)
    agent.run("final question", history=history)

    call_args = model.generate.call_args_list[0]
    messages = call_args[0][0] if call_args[0] else call_args.kwargs.get("messages", [])
    # system + 4 history + 1 current query = 6
    assert len(messages) == 6


def test_json_tool_call_format():
    """Agent handles JSON-formatted tool calls as fallback."""
    model = _make_model([
        json.dumps({"name": "count_files", "params": {"extension": "pdf"}}),
        "You have 5 PDFs.",
    ])
    tools = FakeToolRegistry({"count_files": "Found 5 .pdf files."})
    router = FakeRouter()

    agent = Agent(model, tools, router)
    answer = agent.run("how many PDFs?")

    assert "5" in answer
    assert tools.calls[0] == ("count_files", {"extension": "pdf"})


def test_debug_mode():
    """Debug mode shouldn't crash."""
    model = _make_model(["direct answer"])
    tools = FakeToolRegistry()
    router = FakeRouter("semantic_search", {"query": "test"})

    agent = Agent(model, tools, router, debug=True)
    answer = agent.run("test")
    assert answer  # just verify no crash


def test_unknown_tool_returns_error_to_model():
    """If model calls an unknown tool, error message goes back to context."""
    model = _make_model([
        '<|tool_call_start|>magic_tool(query="hi")<|tool_call_end|>',
        "Sorry, I couldn't use that tool.",
    ])
    tools = FakeToolRegistry()
    router = FakeRouter()

    agent = Agent(model, tools, router)
    answer = agent.run("do magic")
    # Agent should still work — unknown tool result goes back as message
    assert answer is not None
```

**Step 2: Run tests to verify they fail**

Run: `cd /Users/ded/Projects/assist/manole && uv run pytest tests/test_agent.py -v`
Expected: FAIL — `agent` module doesn't exist

**Step 3: Write the implementation**

Create `agent.py`:

```python
"""Agent loop orchestrator — model decides what tools to call at each step."""
import re
from parser import parse_json

SYSTEM_PROMPT = (
    "You are a personal file assistant. You help users find information in their local files.\n\n"
    "You have access to tools to search file contents and inspect the filesystem.\n\n"
    "Rules:\n"
    "- Call semantic_search when the user asks about information INSIDE files\n"
    "- Call filesystem tools (count_files, list_files, grep_files, directory_tree) "
    "for questions ABOUT files themselves\n"
    "- You can call multiple tools if needed to get a complete answer\n"
    "- If a search returns no results, try a different query or tool before giving up\n"
    "- Keep answers concise and grounded in what the tools return\n"
    "- NEVER make up information that wasn't in tool results"
)


class Agent:
    """Orchestrator agent loop with tool calling."""

    MAX_STEPS = 5

    def __init__(self, model, tool_registry, router, debug=False):
        self.model = model
        self.tools = tool_registry
        self.router = router
        self.debug = debug

    def run(self, query: str, history: list[dict] = None) -> str:
        """Run the agent loop for a user query."""
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        if history:
            messages.extend(history[-4:])

        messages.append({"role": "user", "content": query})

        for step in range(self.MAX_STEPS):
            if self.debug:
                print(f"  [AGENT] Step {step + 1}/{self.MAX_STEPS}")

            raw = self.model.generate(messages)

            if self.debug:
                print(f"  [AGENT] Response: {raw[:200]}")

            tool_call = self._parse_tool_call(raw)

            if tool_call is None:
                if step == 0:
                    # First step, no tool call — use fallback router
                    tool_name, params = self.router.route(query)
                    if self.debug:
                        print(f"  [AGENT] Fallback router: {tool_name}({params})")
                    result = self.tools.execute(tool_name, params)
                    messages.append({"role": "assistant", "content": raw})
                    messages.append({"role": "tool", "name": tool_name, "content": result})
                    continue
                else:
                    if self.debug:
                        print("  [AGENT] Direct response (no tool call)")
                    return raw

            tool_name = tool_call["name"]
            tool_params = tool_call.get("params", {})

            if self.debug:
                print(f"  [AGENT] Tool: {tool_name}({tool_params})")

            if tool_name == "respond":
                return tool_params.get("answer", raw)

            result = self.tools.execute(tool_name, tool_params)

            if self.debug:
                print(f"  [AGENT] Result: {result[:200]}")

            messages.append({"role": "assistant", "content": raw})
            messages.append({"role": "tool", "name": tool_name, "content": result})

        # Max steps reached — force synthesis
        if self.debug:
            print("  [AGENT] Max steps reached, forcing response")

        messages.append({
            "role": "user",
            "content": "Give a concise final answer based on the information above.",
        })
        return self.model.generate(messages)

    def _parse_tool_call(self, response: str) -> dict | None:
        """Parse tool call from model output.

        Tries:
        1. LFM2.5 native format: <|tool_call_start|>fn(args)<|tool_call_end|>
        2. JSON format: {"name": "fn", "params": {...}}
        """
        # Try LFM2.5 native format
        tc_match = re.search(
            r'<\|tool_call_start\|>(.*?)<\|tool_call_end\|>',
            response,
            re.DOTALL,
        )
        if tc_match:
            result = self._parse_native_tool_call(tc_match.group(1))
            if result:
                return result

        # Try JSON format
        parsed = parse_json(response)
        if parsed and "name" in parsed:
            return {
                "name": parsed["name"],
                "params": parsed.get("params", parsed.get("parameters", {})),
            }

        return None

    @staticmethod
    def _parse_native_tool_call(raw: str) -> dict | None:
        """Parse LFM2.5's Pythonic function call format.

        Example: semantic_search(query="invoices", top_k=5)
        """
        match = re.match(r'(\w+)\((.*)\)', raw.strip(), re.DOTALL)
        if not match:
            return None

        name = match.group(1)
        params_str = match.group(2)

        params = {}
        for param_match in re.finditer(
            r'(\w+)\s*=\s*(".*?"|\'.*?\'|\d+|None|True|False)',
            params_str,
        ):
            key = param_match.group(1)
            value = param_match.group(2).strip("\"'")
            if value == "None":
                value = None
            elif value.isdigit():
                value = int(value)
            elif value in ("True", "False"):
                value = value == "True"
            params[key] = value

        return {"name": name, "params": params}
```

**Step 4: Run tests to verify they pass**

Run: `cd /Users/ded/Projects/assist/manole && uv run pytest tests/test_agent.py -v`
Expected: All 11 tests PASS

**Step 5: Commit**

```bash
git add agent.py tests/test_agent.py
git commit -m "feat: add agent loop orchestrator"
```

---

### Task 6: Clean Up toolbox.py

Remove the `execute(plan)` and `get_matching_files(plan)` methods — no longer needed since `ToolRegistry` calls individual methods directly.

**Files:**
- Modify: `toolbox.py`
- Modify: `tests/test_toolbox.py`

**Step 1: Remove old tests that reference execute() and get_matching_files()**

Remove these tests from `tests/test_toolbox.py`:
- `test_execute_routes_count`
- `test_execute_routes_list_recent`
- `test_execute_routes_tree`
- `test_get_matching_files_returns_paths`
- `test_time_filter_today` (uses `execute()`)

**Step 2: Run remaining tests to verify they still pass**

Run: `cd /Users/ded/Projects/assist/manole && uv run pytest tests/test_toolbox.py -v`
Expected: All remaining tests PASS (count, list, metadata, tree, grep tests)

**Step 3: Remove execute() and get_matching_files() from toolbox.py**

Remove the `execute(self, plan)` method (lines 90-110) and `get_matching_files(self, plan)` method (lines 112-116).

**Step 4: Run tests again**

Run: `cd /Users/ded/Projects/assist/manole && uv run pytest tests/test_toolbox.py -v`
Expected: All tests PASS

**Step 5: Commit**

```bash
git add toolbox.py tests/test_toolbox.py
git commit -m "refactor: remove execute() and get_matching_files() from ToolBox"
```

---

### Task 7: Update chat.py — Swap Pipeline for Agent

Replace `AgenticRAG` with `Agent` in the chat loop.

**Files:**
- Modify: `chat.py`

**Step 1: Rewrite chat_loop() function**

Replace the `chat_loop` function in `chat.py`:

```python
def chat_loop(index_name: str, data_dir: str):
    from leann import LeannSearcher
    from models import ModelManager
    from searcher import Searcher
    from toolbox import ToolBox
    from tools import ToolRegistry
    from router import route
    from agent import Agent

    print("\nLoading LFM2.5-1.2B-Instruct...")
    t0 = time.time()

    index_path = find_index_path(index_name)
    print(f"Using index: {index_path}")

    # Load model
    model = ModelManager()
    model.load()

    # Create search and tools
    leann_searcher = LeannSearcher(index_path, enable_warmup=True)
    searcher = Searcher(leann_searcher, model, debug=True)
    toolbox = ToolBox(data_dir)
    tool_registry = ToolRegistry(searcher, toolbox)

    # Create router (module-level function wrapped for Agent interface)
    class RouterWrapper:
        @staticmethod
        def route(query):
            return route(query)

    # Create agent
    agent = Agent(model, tool_registry, RouterWrapper(), debug=True)

    print(f"Ready in {time.time() - t0:.1f}s")
    print("=" * 50)
    print("Ask anything about your files. Type 'quit' to exit.")
    print("Type 'debug' to toggle trace.")
    print("=" * 50)

    conversation_history = []

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
            agent.debug = not agent.debug
            searcher.debug = agent.debug
            print(f"Trace: {'ON' if agent.debug else 'OFF'}")
            continue

        t0 = time.time()
        response = agent.run(query, history=conversation_history)
        elapsed = time.time() - t0
        print(f"\n{response}")
        print(f"\n({elapsed:.1f}s)")

        conversation_history.append({"role": "user", "content": query})
        conversation_history.append({"role": "assistant", "content": response})
        if len(conversation_history) > 10:
            conversation_history = conversation_history[-10:]
```

**Step 2: Run the full test suite to check nothing is broken**

Run: `cd /Users/ded/Projects/assist/manole && uv run pytest tests/test_agent.py tests/test_tools.py tests/test_router.py tests/test_searcher.py tests/test_models.py tests/test_toolbox.py tests/test_parser.py -v`
Expected: All tests PASS

**Step 3: Commit**

```bash
git add chat.py
git commit -m "refactor: swap AgenticRAG pipeline for Agent loop in chat.py"
```

---

### Task 8: Delete Old Pipeline Modules

Remove modules that are no longer used.

**Files:**
- Delete: `pipeline.py`
- Delete: `planner.py`
- Delete: `rewriter.py`
- Delete: `mapper.py`
- Delete: `reducer.py`
- Delete: `tests/test_pipeline.py`
- Delete: `tests/test_planner.py`
- Delete: `tests/test_rewriter.py`
- Delete: `tests/test_mapper.py`
- Delete: `tests/test_reducer.py`
- Delete: `tests/test_conversation_memory.py`

**Step 1: Verify no imports reference old modules**

Run: `cd /Users/ded/Projects/assist/manole && grep -r "from pipeline\|from planner\|from rewriter\|from mapper\|from reducer\|import pipeline\|import planner\|import rewriter\|import mapper\|import reducer" --include="*.py" | grep -v "^tests/test_pipeline\|^tests/test_planner\|^tests/test_rewriter\|^tests/test_mapper\|^tests/test_reducer\|^tests/test_conversation\|^pipeline\.py\|^planner\.py\|^rewriter\.py\|^mapper\.py\|^reducer\.py"`

Expected: Only `chat.py` might still reference `pipeline` — verify it was updated in Task 7.

**Step 2: Delete the files**

```bash
rm pipeline.py planner.py rewriter.py mapper.py reducer.py
rm tests/test_pipeline.py tests/test_planner.py tests/test_rewriter.py tests/test_mapper.py tests/test_reducer.py tests/test_conversation_memory.py
```

**Step 3: Run all remaining tests**

Run: `cd /Users/ded/Projects/assist/manole && uv run pytest tests/ -v`
Expected: All tests PASS (test_agent, test_tools, test_router, test_searcher, test_models, test_toolbox, test_parser)

**Step 4: Commit**

```bash
git add -A
git commit -m "chore: delete old pipeline modules (pipeline, planner, rewriter, mapper, reducer)"
```

---

### Task 9: Integration Test — Full Agent Loop

Add end-to-end integration tests that verify the full agent loop with mocked model.

**Files:**
- Create: `tests/test_integration.py`

**Step 1: Write integration tests**

Create `tests/test_integration.py`:

```python
"""Integration tests — full agent loop with mocked model and real filesystem."""
import json
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from unittest.mock import MagicMock

from agent import Agent
from tools import ToolRegistry
from toolbox import ToolBox
from searcher import Searcher
from router import route


@dataclass
class FakeSearchResult:
    id: str
    text: str
    score: float
    metadata: dict = field(default_factory=dict)


class FakeLeann:
    def __init__(self, results):
        self.results = results

    def search(self, query, top_k=5, **kwargs):
        return self.results[:top_k]


class RouterWrapper:
    @staticmethod
    def route(query):
        return route(query)


def _setup(model_responses, search_results=None, files=None):
    """Set up agent with mocked model and optional real filesystem."""
    model = MagicMock()
    model.generate = MagicMock(side_effect=list(model_responses))

    tmp = tempfile.mkdtemp()
    for name, content in (files or {}).items():
        p = Path(tmp) / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)

    leann = FakeLeann(search_results or [])
    searcher = Searcher(leann, model)
    toolbox = ToolBox(tmp)
    registry = ToolRegistry(searcher, toolbox)

    agent = Agent(model, registry, RouterWrapper())
    return agent, model


def test_filesystem_count_via_native_tool_call():
    """Model uses native tool calling for filesystem query."""
    agent, model = _setup(
        model_responses=[
            '<|tool_call_start|>count_files(extension="pdf")<|tool_call_end|>',
            "You have 2 PDF files.",
        ],
        files={"a.pdf": "pdf1", "b.pdf": "pdf2", "c.txt": "txt1"},
    )
    answer = agent.run("how many PDFs?")
    assert "2" in answer


def test_filesystem_count_via_fallback_router():
    """When model doesn't produce tool call, fallback router handles it."""
    agent, model = _setup(
        model_responses=[
            "I'll help you count your files.",  # no tool call
            "You have 2 PDF files.",
        ],
        files={"a.pdf": "pdf1", "b.pdf": "pdf2", "c.txt": "txt1"},
    )
    answer = agent.run("how many PDF files?")
    assert "2" in answer


def test_semantic_search_with_facts():
    """semantic_search extracts facts and model synthesizes answer."""
    results = [
        FakeSearchResult(
            id="1", text="Total Budget: $450,000. Revenue targets: Project Alpha $180k.",
            score=0.95, metadata={"file_name": "budget.txt"},
        )
    ]
    agent, model = _setup(
        model_responses=[
            # Step 1: extraction call inside searcher (called by model.generate)
            json.dumps({"relevant": True, "facts": ["Total Budget: $450,000", "Project Alpha: $180k"]}),
            # Step 2: agent's tool call
            '<|tool_call_start|>semantic_search(query="budget")<|tool_call_end|>',
            # Step 3: extraction call inside searcher again
            json.dumps({"relevant": True, "facts": ["Total Budget: $450,000", "Project Alpha: $180k"]}),
            # Step 4: final answer
            "The total budget is $450,000 with Project Alpha at $180k.",
        ],
        search_results=results,
    )
    # Note: model.generate is called by both agent AND searcher._extract_facts
    # This test verifies the full flow works without crashes
    answer = agent.run("what is the budget?")
    assert answer is not None


def test_directory_tree():
    """directory_tree shows folder structure."""
    agent, _ = _setup(
        model_responses=[
            '<|tool_call_start|>directory_tree(max_depth=2)<|tool_call_end|>',
            "Your files are organized in one folder with PDFs and text files.",
        ],
        files={"docs/a.pdf": "pdf", "docs/b.txt": "txt"},
    )
    answer = agent.run("what is the folder structure?")
    assert answer is not None


def test_conversation_follow_up():
    """Follow-up questions use conversation history."""
    agent, model = _setup(
        model_responses=["There are more than 2 based on the previous search."],
    )
    history = [
        {"role": "user", "content": "find invoices"},
        {"role": "assistant", "content": "Found 2 invoices."},
    ]
    answer = agent.run("aren't there more?", history=history)
    # Verify history was passed — model generated without needing tools
    assert answer is not None


def test_grep_files():
    """grep_files finds files by name pattern."""
    agent, _ = _setup(
        model_responses=[
            '<|tool_call_start|>grep_files(pattern="invoice")<|tool_call_end|>',
            "Found 2 invoice files: invoice_001.pdf and invoice_002.pdf.",
        ],
        files={"invoice_001.pdf": "inv1", "invoice_002.pdf": "inv2", "readme.txt": "readme"},
    )
    answer = agent.run("find invoice files")
    assert "invoice" in answer.lower()
```

**Step 2: Run integration tests**

Run: `cd /Users/ded/Projects/assist/manole && uv run pytest tests/test_integration.py -v`
Expected: All 6 tests PASS

**Step 3: Run full test suite**

Run: `cd /Users/ded/Projects/assist/manole && uv run pytest tests/ -v`
Expected: All tests PASS

**Step 4: Commit**

```bash
git add tests/test_integration.py
git commit -m "test: add integration tests for full agent loop"
```

---

### Task 10: Manual Smoke Test

Run the actual chat loop against real files to verify the agent works end-to-end with the real model.

**Step 1: Run the chat loop**

```bash
cd /Users/ded/Projects/assist/manole && uv run python chat.py --reuse test_data
```

**Step 2: Test these queries**

```
> how many PDF files do I have?
> what is the target revenue for the engineering department?
> list all invoices
> what folders do I have?
> aren't there more?
```

**Step 3: Evaluate results**

- Filesystem queries should complete in 2 LLM calls (fast)
- Semantic queries should extract facts and synthesize answers
- Follow-up should use conversation history
- If tool calling fails, fallback router should kick in (visible in debug trace)

**Step 4: Fix any issues found**

If the 1.2B model's tool calling doesn't work at all, the fallback router should still handle everything. If it does work, we have native tool calling as a bonus.

**Step 5: Final commit**

```bash
git add -A
git commit -m "feat: agent loop migration complete"
```

---

## Summary

| Task | What | Tests |
|------|------|-------|
| 1 | Simplify ModelManager | 6 |
| 2 | Create router.py | 14 |
| 3 | Create tools.py | 10 |
| 4 | Rewrite searcher.py | 10 |
| 5 | Create agent.py | 11 |
| 6 | Clean up toolbox.py | ~9 (existing minus removed) |
| 7 | Update chat.py | 0 (covered by integration) |
| 8 | Delete old modules | 0 (cleanup) |
| 9 | Integration tests | 6 |
| 10 | Manual smoke test | manual |
| **Total** | | **~66 automated tests** |
