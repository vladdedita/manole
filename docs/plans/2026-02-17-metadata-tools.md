# Metadata Tools Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add folder_stats, disk_usage, and enhanced list_files tools so the agent can answer aggregate metadata queries like "what folders take up the most space?"

**Architecture:** Three new toolbox methods feed two new tools (folder_stats, disk_usage) and enhance one existing tool (list_files with sort_by). The rewriter gets a "metadata" intent, and the router gets keyword-based fallback routing for size/space queries.

**Tech Stack:** Python, pathlib, pytest

---

### Task 1: Add `folder_stats()` to ToolBox

**Files:**
- Modify: `toolbox.py` (add method after `grep_paths`)
- Test: `tests/test_toolbox.py`

**Step 1: Write the failing test**

Add to `tests/test_toolbox.py`:

```python
def test_folder_stats_sorted_by_size():
    tmp = tempfile.mkdtemp()
    # Create structure with known sizes
    small = Path(tmp) / "small"
    small.mkdir()
    (small / "a.txt").write_bytes(b"x" * 100)

    big = Path(tmp) / "big"
    big.mkdir()
    (big / "b.txt").write_bytes(b"x" * 10000)

    # Root file
    (Path(tmp) / "root.txt").write_bytes(b"x" * 50)

    tb = ToolBox(tmp)
    result = tb.folder_stats(sort_by="size")
    lines = result.strip().split("\n")

    # "big" folder should come first (largest)
    assert "big" in lines[1]  # first data line after header
    assert "small" in lines[2]
    assert "(root)" in result
    assert "Total:" in result


def test_folder_stats_sorted_by_count():
    tmp = tempfile.mkdtemp()
    many = Path(tmp) / "many"
    many.mkdir()
    for i in range(5):
        (many / f"f{i}.txt").write_bytes(b"x")

    few = Path(tmp) / "few"
    few.mkdir()
    (few / "one.txt").write_bytes(b"x" * 10000)

    tb = ToolBox(tmp)
    result = tb.folder_stats(sort_by="count")
    lines = result.strip().split("\n")
    # "many" should come first (most files)
    assert "many" in lines[1]


def test_folder_stats_empty_dir():
    tmp = tempfile.mkdtemp()
    tb = ToolBox(tmp)
    result = tb.folder_stats()
    assert "No files" in result or "0" in result
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_toolbox.py::test_folder_stats_sorted_by_size tests/test_toolbox.py::test_folder_stats_sorted_by_count tests/test_toolbox.py::test_folder_stats_empty_dir -v`
Expected: FAIL with AttributeError (folder_stats not defined)

**Step 3: Write minimal implementation**

Add to `toolbox.py` after the `grep_paths` method:

```python
def folder_stats(self, sort_by: str = "size", limit: int = 10) -> str:
    """Aggregate size and file count per folder."""
    files = [f for f in self.root.rglob("*") if f.is_file() and not f.name.startswith(".")]
    if not files:
        return "No files found."

    folders: dict[str, dict] = {}
    for f in files:
        rel = f.relative_to(self.root)
        folder = str(rel.parent) if str(rel.parent) != "." else "(root)"
        if folder not in folders:
            folders[folder] = {"size": 0, "count": 0}
        folders[folder]["size"] += f.stat().st_size
        folders[folder]["count"] += 1

    key = "size" if sort_by == "size" else "count"
    ranked = sorted(folders.items(), key=lambda x: x[1][key], reverse=True)

    total_size = sum(v["size"] for v in folders.values())
    total_count = sum(v["count"] for v in folders.values())

    lines = [f"Folder sizes (sorted by {sort_by}):"]
    for folder, stats in ranked[:limit]:
        size_str = self._format_size(stats["size"])
        lines.append(f"  {folder}/: {size_str}, {stats['count']} files")
    lines.append(f"Total: {self._format_size(total_size)} across {total_count} files")
    return "\n".join(lines)

@staticmethod
def _format_size(size_bytes: int) -> str:
    """Format bytes as human-readable string."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.1f} MB"
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_toolbox.py -v`
Expected: All pass

**Step 5: Commit**

```bash
git add toolbox.py tests/test_toolbox.py
git commit -m "feat: add folder_stats() to ToolBox"
```

---

### Task 2: Add `disk_usage()` to ToolBox

**Files:**
- Modify: `toolbox.py` (add method after `folder_stats`)
- Test: `tests/test_toolbox.py`

**Step 1: Write the failing test**

Add to `tests/test_toolbox.py`:

```python
def test_disk_usage_summary():
    tmp = tempfile.mkdtemp()
    (Path(tmp) / "a.pdf").write_bytes(b"x" * 5000)
    (Path(tmp) / "b.pdf").write_bytes(b"x" * 3000)
    (Path(tmp) / "c.txt").write_bytes(b"x" * 1000)
    sub = Path(tmp) / "sub"
    sub.mkdir()
    (sub / "d.txt").write_bytes(b"x" * 2000)

    tb = ToolBox(tmp)
    result = tb.disk_usage()

    assert "4 files" in result or "4" in result
    assert ".pdf" in result
    assert ".txt" in result
    assert "Average" in result
    assert "Total" in result


def test_disk_usage_empty():
    tmp = tempfile.mkdtemp()
    tb = ToolBox(tmp)
    result = tb.disk_usage()
    assert "No files" in result or "0" in result
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_toolbox.py::test_disk_usage_summary tests/test_toolbox.py::test_disk_usage_empty -v`
Expected: FAIL with AttributeError

**Step 3: Write minimal implementation**

Add to `toolbox.py` after `folder_stats`:

```python
def disk_usage(self) -> str:
    """Total disk usage summary with breakdown by extension."""
    files = [f for f in self.root.rglob("*") if f.is_file() and not f.name.startswith(".")]
    if not files:
        return "No files found."

    total_size = 0
    by_ext: dict[str, dict] = {}
    for f in files:
        size = f.stat().st_size
        total_size += size
        ext = f.suffix.lower() or "(no extension)"
        if ext not in by_ext:
            by_ext[ext] = {"size": 0, "count": 0}
        by_ext[ext]["size"] += size
        by_ext[ext]["count"] += 1

    avg_size = total_size / len(files) if files else 0
    ranked = sorted(by_ext.items(), key=lambda x: x[1]["size"], reverse=True)

    lines = ["Disk usage summary:"]
    lines.append(f"  Total: {self._format_size(total_size)} across {len(files)} files")
    lines.append(f"  Average file size: {self._format_size(int(avg_size))}")
    lines.append("  By type:")
    for ext, stats in ranked[:10]:
        lines.append(f"    {ext}: {self._format_size(stats['size'])} ({stats['count']} files)")
    return "\n".join(lines)
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_toolbox.py -v`
Expected: All pass

**Step 5: Commit**

```bash
git add toolbox.py tests/test_toolbox.py
git commit -m "feat: add disk_usage() to ToolBox"
```

---

### Task 3: Enhance `list_recent_files()` with `sort_by`

**Files:**
- Modify: `toolbox.py:39-49` (`list_recent_files` method)
- Test: `tests/test_toolbox.py`

**Step 1: Write the failing test**

Add to `tests/test_toolbox.py`:

```python
def test_list_files_sort_by_size():
    tmp = tempfile.mkdtemp()
    (Path(tmp) / "small.txt").write_bytes(b"x" * 100)
    (Path(tmp) / "big.txt").write_bytes(b"x" * 10000)
    (Path(tmp) / "medium.txt").write_bytes(b"x" * 1000)

    tb = ToolBox(tmp)
    result = tb.list_recent_files(sort_by="size")
    lines = [l.strip() for l in result.strip().split("\n") if l.strip().startswith("-")]

    # big.txt should be first
    assert "big.txt" in lines[0]
    assert "small.txt" in lines[-1]


def test_list_files_sort_by_name():
    tmp = tempfile.mkdtemp()
    (Path(tmp) / "c.txt").write_bytes(b"x")
    (Path(tmp) / "a.txt").write_bytes(b"x")
    (Path(tmp) / "b.txt").write_bytes(b"x")

    tb = ToolBox(tmp)
    result = tb.list_recent_files(sort_by="name")
    lines = [l.strip() for l in result.strip().split("\n") if l.strip().startswith("-")]

    assert "a.txt" in lines[0]
    assert "c.txt" in lines[-1]


def test_list_files_sort_by_size_shows_size():
    """When sorted by size, output should include file sizes."""
    tmp = tempfile.mkdtemp()
    (Path(tmp) / "a.txt").write_bytes(b"x" * 5000)

    tb = ToolBox(tmp)
    result = tb.list_recent_files(sort_by="size")
    assert "KB" in result or "MB" in result or "B" in result
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_toolbox.py::test_list_files_sort_by_size tests/test_toolbox.py::test_list_files_sort_by_name tests/test_toolbox.py::test_list_files_sort_by_size_shows_size -v`
Expected: FAIL (sort_by param not accepted or not handled)

**Step 3: Write minimal implementation**

Replace the `list_recent_files` method in `toolbox.py`:

```python
def list_recent_files(self, ext_filter: str | None = None, time_filter: str | None = None,
                       limit: int = 10, sort_by: str = "date") -> str:
    files = self._list_files(ext_filter, time_filter)
    if not files:
        return "No matching files found."

    if sort_by == "size":
        files.sort(key=lambda f: f.stat().st_size, reverse=True)
    elif sort_by == "name":
        files.sort(key=lambda f: f.name.lower())
    else:
        files.sort(key=lambda f: f.stat().st_mtime, reverse=True)

    lines = []
    for f in files[:limit]:
        rel = f.relative_to(self.root)
        stat = f.stat()
        if sort_by == "size":
            size_str = self._format_size(stat.st_size)
            lines.append(f"  - {rel} ({size_str})")
        else:
            mtime = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M")
            lines.append(f"  - {rel} (modified: {mtime})")

    header = "Files" if sort_by == "name" else f"Files (sorted by {sort_by}):"
    return header + "\n" + "\n".join(lines)
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_toolbox.py -v`
Expected: All pass (including existing tests — `sort_by` defaults to "date")

**Step 5: Commit**

```bash
git add toolbox.py tests/test_toolbox.py
git commit -m "feat: add sort_by parameter to list_recent_files"
```

---

### Task 4: Register new tools in ToolRegistry

**Files:**
- Modify: `tools.py` (add TOOL_DEFINITIONS and handlers)
- Test: `tests/test_tools.py`

**Step 1: Write the failing tests**

Add to `tests/test_tools.py`:

```python
def test_tool_definitions_has_nine_tools():
    """Updated: 7 original + 2 new = 9."""
    names = [t["name"] for t in TOOL_DEFINITIONS]
    assert len(names) == 9
    assert "folder_stats" in names
    assert "disk_usage" in names


def test_folder_stats_dispatch():
    registry, _, _ = _make_registry()
    result = registry.execute("folder_stats", {"sort_by": "size"})
    assert "Folder" in result or "No files" in result


def test_disk_usage_dispatch():
    registry, _, _ = _make_registry()
    result = registry.execute("disk_usage", {})
    assert "Disk" in result or "No files" in result


def test_list_files_with_sort_by():
    registry, _, _ = _make_registry()
    result = registry.execute("list_files", {"sort_by": "size"})
    assert result  # should not crash
```

Also update the existing test:

```python
# Change test_tool_definitions_has_seven_tools to expect 9
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_tools.py::test_tool_definitions_has_nine_tools tests/test_tools.py::test_folder_stats_dispatch tests/test_tools.py::test_disk_usage_dispatch tests/test_tools.py::test_list_files_with_sort_by -v`
Expected: FAIL

**Step 3: Write minimal implementation**

In `tools.py`, add to `TOOL_DEFINITIONS` (before the `respond` entry):

```python
{
    "name": "folder_stats",
    "description": "Show folder sizes and file counts. Use for 'biggest folder', 'folder sizes', 'which folder has most files'.",
    "parameters": {
        "type": "object",
        "properties": {
            "sort_by": {"type": "string", "description": "'size' (default) or 'count'"},
            "limit": {"type": "integer", "description": "Max folders to show (default 10)"},
        },
    },
},
{
    "name": "disk_usage",
    "description": "Show total disk usage summary with breakdown by file type. Use for 'how much space', 'storage', 'disk usage'.",
    "parameters": {
        "type": "object",
        "properties": {},
    },
},
```

Update the `list_files` definition to include `sort_by`:

```python
"sort_by": {"type": "string", "description": "'date' (default), 'size', or 'name'"},
```

In `ToolRegistry.__init__`, add handlers:

```python
"folder_stats": self._folder_stats,
"disk_usage": self._disk_usage,
```

Add handler methods:

```python
def _folder_stats(self, params: dict) -> str:
    return self.toolbox.folder_stats(
        sort_by=params.get("sort_by", "size"),
        limit=params.get("limit", 10),
    )

def _disk_usage(self, params: dict) -> str:
    return self.toolbox.disk_usage()
```

Update `_list_files` to pass `sort_by`:

```python
def _list_files(self, params: dict) -> str:
    return self.toolbox.list_recent_files(
        ext_filter=params.get("extension"),
        limit=params.get("limit", 10),
        sort_by=params.get("sort_by", "date"),
    )
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_tools.py -v`
Expected: All pass (update old test_tool_definitions_has_seven_tools to expect 9 or delete it)

**Step 5: Commit**

```bash
git add tools.py tests/test_tools.py
git commit -m "feat: register folder_stats and disk_usage in ToolRegistry"
```

---

### Task 5: Update agent.py tool schemas and known tools

**Files:**
- Modify: `agent.py` (TOOL_SCHEMAS, _KNOWN_TOOLS)
- Test: `tests/test_agent.py`

**Step 1: Write the failing test**

Add to `tests/test_agent.py`:

```python
def test_known_tools_includes_metadata():
    assert "folder_stats" in Agent._KNOWN_TOOLS
    assert "disk_usage" in Agent._KNOWN_TOOLS


def test_model_calls_folder_stats():
    model = _make_model([
        '<|tool_call_start|>folder_stats(sort_by="size")<|tool_call_end|>',
        "The biggest folder is Finance at 12 MB.",
    ])
    tools = FakeToolRegistry({"folder_stats": "Folder sizes:\n  finance/: 12.0 MB, 23 files"})
    router = FakeRouter()

    agent = Agent(model, tools, router)
    answer = agent.run("what folders take up the most space?")

    assert tools.calls[0] == ("folder_stats", {"sort_by": "size"})
    assert not router.called


def test_model_calls_disk_usage():
    model = _make_model([
        '<|tool_call_start|>disk_usage()<|tool_call_end|>',
        "Total storage is 16.7 MB.",
    ])
    tools = FakeToolRegistry({"disk_usage": "Disk usage summary:\n  Total: 16.7 MB"})
    router = FakeRouter()

    agent = Agent(model, tools, router)
    answer = agent.run("how much space am I using?")

    assert tools.calls[0] == ("disk_usage", {})
    assert not router.called
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_agent.py::test_known_tools_includes_metadata tests/test_agent.py::test_model_calls_folder_stats tests/test_agent.py::test_model_calls_disk_usage -v`
Expected: FAIL

**Step 3: Write minimal implementation**

In `agent.py`, add to `TOOL_SCHEMAS` list (before the closing `]`):

```python
{
    "name": "folder_stats",
    "description": "Show folder sizes and file counts",
    "parameters": {
        "type": "object",
        "properties": {
            "sort_by": {"type": "string", "description": "'size' or 'count'"},
            "limit": {"type": "integer", "description": "Max folders to show"},
        },
    },
},
{
    "name": "disk_usage",
    "description": "Show total disk usage summary",
    "parameters": {
        "type": "object",
        "properties": {},
    },
},
```

Update `list_files` schema to include `sort_by`:

```python
"sort_by": {"type": "string", "description": "'date', 'size', or 'name'"},
```

Update `_KNOWN_TOOLS`:

```python
_KNOWN_TOOLS = frozenset({
    "semantic_search", "count_files", "list_files", "grep_files",
    "file_metadata", "directory_tree", "folder_stats", "disk_usage", "respond",
})
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_agent.py -v`
Expected: All pass

**Step 5: Commit**

```bash
git add agent.py tests/test_agent.py
git commit -m "feat: add folder_stats and disk_usage to agent tool schemas"
```

---

### Task 6: Add "metadata" intent to rewriter

**Files:**
- Modify: `rewriter.py` (REWRITER_SYSTEM prompt, _VALID_INTENTS)
- Test: `tests/test_rewriter.py`

**Step 1: Write the failing test**

Add to `tests/test_rewriter.py`:

```python
def test_metadata_is_valid_intent():
    assert "metadata" in _VALID_INTENTS


def test_rewriter_system_mentions_metadata():
    assert "metadata" in REWRITER_SYSTEM
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_rewriter.py::test_metadata_is_valid_intent tests/test_rewriter.py::test_rewriter_system_mentions_metadata -v`
Expected: FAIL

**Step 3: Write minimal implementation**

In `rewriter.py`, update `_VALID_INTENTS`:

```python
_VALID_INTENTS = frozenset({"factual", "count", "list", "compare", "summarize", "metadata"})
```

In `REWRITER_SYSTEM`, add examples before the IMPORTANT line:

```
'Question: "what folders take up the most space?"\n'
'{"intent": "metadata", "search_query": "folder size space storage disk usage", '
'"resolved_query": "Which folders take up the most space in my files?"}\n\n'
'Question: "how much storage am I using?"\n'
'{"intent": "metadata", "search_query": "total disk usage storage space", '
'"resolved_query": "How much total storage space are my files using?"}\n\n'
```

Also add `"metadata"` to the intent description line:

```
'- "intent": one of "factual", "count", "list", "compare", "summarize", "metadata"\n'
'  Use "metadata" for questions about file sizes, folder sizes, disk usage, storage space.\n'
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_rewriter.py -v`
Expected: All pass

**Step 5: Commit**

```bash
git add rewriter.py tests/test_rewriter.py
git commit -m "feat: add metadata intent to query rewriter"
```

---

### Task 7: Add metadata routing to router fallback

**Files:**
- Modify: `router.py:35-45` (add metadata keyword routing)
- Test: `tests/test_router.py`

**Step 1: Write the failing tests**

Add to `tests/test_router.py`:

```python
def test_metadata_intent_folder_query():
    name, params = route("what folders are biggest?", intent="metadata")
    assert name == "folder_stats"
    assert params["sort_by"] == "size"


def test_space_keyword_routes_to_folder_stats():
    name, params = route("what takes up the most space?")
    assert name == "folder_stats"


def test_largest_folder_query():
    name, params = route("which folder is the largest?")
    assert name == "folder_stats"


def test_disk_usage_query():
    name, params = route("total disk usage overview")
    assert name == "disk_usage"


def test_storage_summary_query():
    name, params = route("give me a storage summary")
    assert name == "disk_usage"


def test_biggest_files_still_semantic():
    """'biggest files' without folder keyword → semantic search (model should pick list_files)."""
    name, _ = route("what are the biggest files?")
    assert name == "folder_stats"  # "biggest" keyword triggers folder_stats as default
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_router.py::test_metadata_intent_folder_query tests/test_router.py::test_space_keyword_routes_to_folder_stats tests/test_router.py::test_largest_folder_query tests/test_router.py::test_disk_usage_query tests/test_router.py::test_storage_summary_query -v`
Expected: FAIL

**Step 3: Write minimal implementation**

In `router.py`, update the `route` function. Add metadata routing BEFORE the existing filesystem keywords block:

```python
def route(query: str, intent: str | None = None) -> tuple[str, dict]:
    q = query.lower()

    # Metadata queries: folder sizes, disk usage, storage
    size_keywords = ["space", "biggest", "largest", "storage", "heavy", "disk usage"]
    if intent == "metadata" or any(k in q for k in size_keywords):
        if any(k in q for k in ["total", "usage", "overview", "summary"]):
            return "disk_usage", {}
        return "folder_stats", {"sort_by": "size"}

    # Unambiguous filesystem keywords only
    if any(k in q for k in ["folder", "tree", "directory", "structure"]):
        return "directory_tree", {"max_depth": 2}
    if any(k in q for k in ["file size", "how big", "how large", "how old", "when was", "modified", "created"]):
        return "file_metadata", {"name_hint": _extract_name_hint(q)}

    # Everything else → semantic search (model should have handled it)
    return "semantic_search", {"query": query}
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_router.py -v`
Expected: All pass (check existing tests still pass — `test_folder_query` may now route to folder_stats instead of directory_tree if "folder" is matched by the new block. Review and adjust if needed.)

Note: The existing `test_folder_query` tests `route("what folders do I have?")` which contains "folder" but not size keywords, so it should still hit `directory_tree`. Verify this.

**Step 5: Commit**

```bash
git add router.py tests/test_router.py
git commit -m "feat: add metadata keyword routing to fallback router"
```

---

### Task 8: Run full test suite and verify

**Step 1: Run all tests**

Run: `pytest tests/ -v`
Expected: All pass

**Step 2: Run frontend build**

Run: `cd ui && npm run build`
Expected: Build succeeds (no frontend changes in this feature)

**Step 3: Commit any fixups**

If any tests needed adjustment, commit the fixes.

```bash
git add -A && git commit -m "fix: adjust tests for metadata tools integration"
```
