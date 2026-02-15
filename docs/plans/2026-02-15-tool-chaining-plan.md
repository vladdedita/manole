# Tool Chaining Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** When the 1.2B model stops after one tool call without addressing all query keywords, Python auto-injects follow-up tool calls.

**Architecture:** Add a `_needs_followup()` method to Agent that compares query keywords against accumulated tool results. When uncovered keywords remain, it picks the next logical tool (grep_files for name search, semantic_search for content). Only activates at step > 0 when the model gives a direct response instead of calling a tool.

**Tech Stack:** Python, existing `extract_keywords()` from searcher.py, existing Agent loop.

---

### Task 1: Test keyword-coverage followup for grep

**Files:**
- Modify: `tests/test_agent.py`
- Modify: `agent.py`

**Step 1: Write the failing test**

Add to `tests/test_agent.py`:

```python
def test_followup_grep_when_keyword_missing():
    """Python injects grep_files when model stops but query keyword not in results."""
    model = _make_model([
        # Step 1: model calls count_files
        '[count_files(extension="pdf")]',
        # Step 2: model gives direct answer (keyword "macbook" not covered)
        "There are 25 PDF files.",
        # Step 3: after Python injects grep_files, model synthesizes
        "Found macbook_ssd.pdf — a MacBook-related PDF.",
    ])
    tools = FakeToolRegistry({
        "count_files": "Found 25 .pdf files.",
        "grep_files": "macbook_ssd.pdf",
    })
    router = FakeRouter()

    agent = Agent(model, tools, router)
    answer = agent.run("any macbook pdfs")

    # grep_files should have been auto-injected
    tool_calls = [(name, params) for name, params in tools.calls]
    assert ("count_files", {"extension": "pdf"}) in tool_calls
    assert any(name == "grep_files" and "macbook" in params.get("pattern", "") for name, params in tool_calls)
    assert not router.called
```

**Step 2: Run test to verify it fails**

Run: `uv run python -m pytest tests/test_agent.py::test_followup_grep_when_keyword_missing -v`
Expected: FAIL — `_needs_followup` doesn't exist yet, so the agent returns "There are 25 PDF files." at step 2 without injecting grep.

**Step 3: Implement `_needs_followup()` and integrate into loop**

In `agent.py`, add import at top:

```python
from searcher import extract_keywords
```

Add method to Agent class:

```python
def _needs_followup(self, query: str, messages: list[dict]) -> dict | None:
    """Check if query keywords are covered by tool results. Return next tool call if not."""
    keywords = extract_keywords(query)
    if not keywords:
        return None

    # Collect all tool result text from messages
    result_text = ""
    tools_used = set()
    for msg in messages:
        if msg["role"] == "tool":
            result_text += msg["content"].lower() + " "
            try:
                parsed = json.loads(msg["content"])
                if isinstance(parsed, dict) and "tool" in parsed:
                    tools_used.add(parsed["tool"])
            except (json.JSONDecodeError, TypeError):
                pass

    # Check which keywords are missing from results
    missing = [kw for kw in keywords if kw not in result_text]
    if not missing:
        return None

    # Pick next tool based on what's missing and what we've already tried
    if "grep_files" not in tools_used:
        return {"name": "grep_files", "params": {"pattern": missing[0]}}

    if "semantic_search" not in tools_used:
        return {"name": "semantic_search", "params": {"query": " ".join(missing)}}

    return None
```

Modify the agent loop — replace the `step > 0, no tool call` branch in `run()`:

```python
                else:
                    followup = self._needs_followup(query, messages)
                    if followup:
                        tool_name = followup["name"]
                        tool_params = followup["params"]
                        if self.debug:
                            print(f"  [AGENT] Followup: {tool_name}({tool_params})")
                        result = self.tools.execute(tool_name, tool_params)
                        if self.debug:
                            print(f"  [AGENT] Followup result: {result[:200]}")
                        messages.append({"role": "assistant", "content": raw})
                        messages.append({
                            "role": "tool",
                            "content": json.dumps({"tool": tool_name, "result": result}),
                        })
                        continue
                    if self.debug:
                        print("  [AGENT] Direct response (no tool call)")
                    return raw
```

**Step 4: Run test to verify it passes**

Run: `uv run python -m pytest tests/test_agent.py::test_followup_grep_when_keyword_missing -v`
Expected: PASS

**Step 5: Commit**

```bash
git add agent.py tests/test_agent.py
git commit -m "feat: add Python-orchestrated tool chaining via _needs_followup"
```

---

### Task 2: Test followup falls back to semantic_search

**Files:**
- Modify: `tests/test_agent.py`

**Step 1: Write the failing test**

```python
def test_followup_semantic_search_after_grep():
    """When grep was already used, followup tries semantic_search."""
    model = _make_model([
        # Step 1: model calls grep_files
        '[grep_files(pattern="invoice")]',
        # Step 2: model gives answer but "total" not covered
        "Found 5 invoice files.",
        # Step 3: after Python injects semantic_search, model answers
        "The total across all invoices is $2,500.",
    ])
    tools = FakeToolRegistry({
        "grep_files": "invoice_001.pdf\ninvoice_002.pdf",
        "semantic_search": "From invoice_001.pdf: Total: $1,200\nFrom invoice_002.pdf: Total: $1,300",
    })
    router = FakeRouter()

    agent = Agent(model, tools, router)
    answer = agent.run("what is the total of my invoices")

    tool_calls = [(name, params) for name, params in tools.calls]
    assert any(name == "grep_files" for name, params in tool_calls)
    assert any(name == "semantic_search" for name, params in tool_calls)
```

**Step 2: Run test to verify it passes**

Run: `uv run python -m pytest tests/test_agent.py::test_followup_semantic_search_after_grep -v`
Expected: PASS (implementation from Task 1 handles this — grep already used, so semantic_search is next)

**Step 3: Commit**

```bash
git add tests/test_agent.py
git commit -m "test: add semantic_search followup after grep"
```

---

### Task 3: Test no followup when keywords covered

**Files:**
- Modify: `tests/test_agent.py`

**Step 1: Write the test**

```python
def test_no_followup_when_keywords_covered():
    """No followup when all query keywords appear in tool results."""
    model = _make_model([
        '[semantic_search(query="carbonara eggs")]',
        "The recipe calls for 4 eggs.",
    ])
    tools = FakeToolRegistry({
        "semantic_search": "From pasta_carbonara.txt: 4 eggs, pecorino romano, guanciale",
    })
    router = FakeRouter()

    agent = Agent(model, tools, router)
    answer = agent.run("how many eggs in carbonara")

    assert answer == "The recipe calls for 4 eggs."
    # Only one tool call — no followup injected
    assert len(tools.calls) == 1
```

**Step 2: Run test to verify it passes**

Run: `uv run python -m pytest tests/test_agent.py::test_no_followup_when_keywords_covered -v`
Expected: PASS — "eggs" and "carbonara" both appear in the semantic_search result

**Step 3: Commit**

```bash
git add tests/test_agent.py
git commit -m "test: verify no followup when keywords are covered"
```

---

### Task 4: Test no infinite loop on repeated followup

**Files:**
- Modify: `tests/test_agent.py`

**Step 1: Write the test**

```python
def test_followup_stops_after_both_tools_tried():
    """Followup doesn't loop forever — stops when grep and semantic_search both tried."""
    model = _make_model([
        '[count_files(extension="pdf")]',
        "There are 25 PDFs.",       # keyword "macbook" missing, grep injected
        "Still 25 PDFs.",            # keyword still missing, semantic_search injected
        "I found some results.",     # keyword still missing, but both tools tried — stop
    ])
    tools = FakeToolRegistry({
        "count_files": "Found 25 .pdf files.",
        "grep_files": "No files matching 'macbook'.",
        "semantic_search": "No results found.",
    })
    router = FakeRouter()

    agent = Agent(model, tools, router)
    answer = agent.run("any macbook pdfs")

    # Should stop after trying grep + semantic_search, not loop
    assert answer == "I found some results."
    tool_names = [name for name, _ in tools.calls]
    assert tool_names.count("grep_files") == 1
    assert tool_names.count("semantic_search") == 1
```

**Step 2: Run test to verify it passes**

Run: `uv run python -m pytest tests/test_agent.py::test_followup_stops_after_both_tools_tried -v`
Expected: PASS — `_needs_followup` returns None when both grep_files and semantic_search have been used

**Step 3: Commit**

```bash
git add tests/test_agent.py
git commit -m "test: verify followup stops after both tools exhausted"
```

---

### Task 5: Run full test suite and verify

**Step 1: Run all tests**

Run: `uv run python -m pytest tests/ -v`
Expected: All tests pass (111 existing + 4 new = 115)

**Step 2: Final commit if any adjustments needed**

```bash
git add -A
git commit -m "fix: adjustments from full test run"
```
