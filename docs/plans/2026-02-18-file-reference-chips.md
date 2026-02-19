# File Reference Chips Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Show clickable filename chips below assistant answers, listing files that contributed facts to the response, with click-to-open-in-OS behavior.

**Architecture:** Thread a structured `sources` list through the pipeline: `Searcher` returns filenames alongside text, `ToolRegistry.execute()` returns a `(text, sources)` tuple, `Agent.run()` accumulates sources across tool calls and returns them, `Server` includes them in the NDJSON `result` message, and the frontend renders them as chips in `MessageBubble`.

**Tech Stack:** Python (backend), TypeScript/React (frontend), Electron IPC (file open)

---

### Task 1: Searcher returns structured sources alongside text

**Files:**
- Modify: `searcher.py:52-94` (`search_and_extract`)
- Modify: `searcher.py:138-201` (`_filename_fallback`)
- Test: `tests/test_searcher.py`

**Step 1: Write failing tests**

Add two tests to `tests/test_searcher.py`:

```python
def test_search_and_extract_returns_sources():
    """search_and_extract returns (text, sources) tuple with source filenames."""
    results = _make_results("Budget doc", sources=["budget.pdf"])
    model = _make_model([json.dumps({"relevant": True, "facts": ["Budget: $450k"]})])
    leann = FakeLeann(results)
    searcher = Searcher(leann, model)

    text, sources = searcher.search_and_extract("budget")

    assert isinstance(sources, list)
    assert "budget.pdf" in sources
    assert "budget.pdf" in text  # still has "From budget.pdf:"


def test_search_and_extract_no_results_returns_empty_sources():
    """When no chunks match, sources list is empty."""
    leann = FakeLeann([])
    model = _make_model([])
    searcher = Searcher(leann, model)

    text, sources = searcher.search_and_extract("anything")

    assert sources == []
    assert "No matching" in text
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_searcher.py::test_search_and_extract_returns_sources tests/test_searcher.py::test_search_and_extract_no_results_returns_empty_sources -v`
Expected: FAIL — `search_and_extract` returns `str`, not a tuple

**Step 3: Update `search_and_extract` to return `tuple[str, list[str]]`**

In `searcher.py`, change `search_and_extract` to return `(text, list_of_source_names)`:

- Early returns (`"No matching content found."`) become `("No matching content found.", [])`
- The `_filename_fallback` call: change `_filename_fallback` to also return `(text, sources)`
- The final formatting block: collect `list(facts_by_source.keys())` as sources
- Return `("\n".join(lines), sources)` at the end

Similarly update `_filename_fallback` to return `(text, list[str])`:
- All early returns become `("...", [])`
- Final return becomes `("\n".join(lines), list(facts_by_source.keys()))`

**Step 4: Fix existing tests that destructure the return value**

All existing tests that do `result = searcher.search_and_extract(...)` and then `assert "From" in result` need to be updated to either:
- `text, sources = searcher.search_and_extract(...)` then `assert "From" in text`
- Or `result, _ = searcher.search_and_extract(...)`

Scan all usages in `tests/test_searcher.py` and update them.

**Step 5: Run full test suite**

Run: `python -m pytest tests/test_searcher.py -v`
Expected: All tests PASS

**Step 6: Commit**

```bash
git add searcher.py tests/test_searcher.py
git commit -m "feat(searcher): return structured sources alongside text"
```

---

### Task 2: ToolRegistry propagates sources from semantic_search

**Files:**
- Modify: `tools.py:131-147` (`execute`, `_semantic_search`)
- Test: `tests/test_tools.py`

**Step 1: Write failing test**

```python
def test_semantic_search_returns_sources():
    """execute() returns (text, sources) when semantic_search produces sources."""
    searcher = MagicMock()
    searcher.search_and_extract.return_value = ("From doc.pdf:\n  - fact", ["doc.pdf"])
    toolbox = MagicMock()
    registry = ToolRegistry(searcher, toolbox)

    text, sources = registry.execute("semantic_search", {"query": "test"})

    assert sources == ["doc.pdf"]
    assert "doc.pdf" in text
```

**Step 2: Run to verify failure**

Run: `python -m pytest tests/test_tools.py::test_semantic_search_returns_sources -v`
Expected: FAIL — `execute` returns `str`, not tuple

**Step 3: Update `execute()` and `_semantic_search()`**

Change `execute()` return type to `tuple[str, list[str]]`:
- `_semantic_search` returns the tuple directly from `searcher.search_and_extract()`
- All other handlers (`_grep_files`, `_count_files`, etc.) return `(text, [])` — no sources
- Unknown tool returns `(f"Unknown tool: {tool_name}", [])`

**Step 4: Fix existing test_tools.py tests**

Update all assertions on `registry.execute(...)` to destructure `text, sources = ...`.

**Step 5: Run full tools tests**

Run: `python -m pytest tests/test_tools.py -v`
Expected: All PASS

**Step 6: Commit**

```bash
git add tools.py tests/test_tools.py
git commit -m "feat(tools): propagate sources from execute()"
```

---

### Task 3: Agent.run() accumulates and returns sources

**Files:**
- Modify: `agent.py:104-226` (`run`)
- Test: `tests/test_agent.py`

**Step 1: Write failing test**

```python
def test_run_returns_sources():
    """Agent.run() returns (answer, sources) tuple."""
    model = _make_model([
        '<|tool_call_start|>semantic_search(query="budget")<|tool_call_end|>',
        "The budget is $450k.",
    ])
    tools = FakeToolRegistry({"semantic_search": "From budget.pdf:\n  - Budget: $450k"})
    router = FakeRouter()

    agent = Agent(model, tools, router)
    answer, sources = agent.run("what is the budget?")

    assert answer == "The budget is $450k."
    assert "budget.pdf" in sources
```

**Step 2: Run to verify failure**

Run: `python -m pytest tests/test_agent.py::test_run_returns_sources -v`
Expected: FAIL

**Step 3: Update `FakeToolRegistry.execute` in test_agent.py**

Update `FakeToolRegistry.execute` to return `(text, [])` so it matches the new `ToolRegistry` signature. For the test above, make it return `("From budget.pdf:\n  - Budget: $450k", ["budget.pdf"])`.

**Step 4: Update `Agent.run()` to accumulate sources**

In `agent.py`:
- Add `all_sources: list[str] = []` at the top of `run()`
- Every `self.tools.execute(...)` call: destructure `result, sources = self.tools.execute(...)`, then `all_sources.extend(sources)`
- All `return raw` statements become `return raw, list(dict.fromkeys(all_sources))` (deduplicated, order-preserved)
- The `return tool_params.get("answer", raw)` (respond tool) also returns sources

**Step 5: Fix all existing agent tests**

Update all `answer = agent.run(...)` to `answer, sources = agent.run(...)` (or `answer, _ = ...`).

**Step 6: Run full agent tests**

Run: `python -m pytest tests/test_agent.py -v`
Expected: All PASS

**Step 7: Commit**

```bash
git add agent.py tests/test_agent.py
git commit -m "feat(agent): accumulate and return sources from tool calls"
```

---

### Task 4: Server includes sources in NDJSON result

**Files:**
- Modify: `server.py:304-362` (`handle_query`)
- Modify: `server.py:364-385` (`_query_all`)
- Test: `tests/test_server.py`

**Step 1: Write failing test**

```python
def test_query_result_includes_sources(server_with_directory):
    """The result message from handle_query includes a sources array."""
    server, dir_id = server_with_directory
    # Mock agent.run to return (text, sources)
    entry = list(server.directories.values())[0]
    entry["agent"].run = lambda q, history=None, on_token=None, on_step=None: (
        "The budget is $450k.", ["budget.pdf"]
    )

    result = server.handle_query(1, {"text": "budget", "directoryId": dir_id})

    assert result["data"]["sources"] == ["budget.pdf"]
    assert result["data"]["text"] == "The budget is $450k."
```

**Step 2: Run to verify failure**

Run: `python -m pytest tests/test_server.py::test_query_result_includes_sources -v`
Expected: FAIL — `data` has no `sources` key

**Step 3: Update `handle_query` and `_query_all`**

In `handle_query`:
- `response, sources = agent.run(...)` instead of `response = agent.run(...)`
- Final return: `{"data": {"text": response, "sources": sources}}`
- Store only `response` (text) in `conversation_history`

In `_query_all`:
- `response, sources = agent.run(...)`
- `results.append({"directoryId": ..., "text": response, "sources": sources})`

**Step 4: Fix existing server tests**

Any tests that mock or check `agent.run` return values need updating to return tuples.

**Step 5: Run server tests**

Run: `python -m pytest tests/test_server.py -v`
Expected: All PASS

**Step 6: Commit**

```bash
git add server.py tests/test_server.py
git commit -m "feat(server): include sources array in query result"
```

---

### Task 5: Frontend ChatMessage model gains sources field

**Files:**
- Modify: `ui/src/hooks/useChat.ts`
- Modify: `ui/src/lib/protocol.ts`

**Step 1: Add `sources` to `ResultData` in protocol.ts**

```typescript
export interface ResultData {
  text?: string;
  sources?: string[];  // <-- add this
  status?: string;
  indexName?: string;
  indexes?: string[];
  debug?: boolean;
}
```

**Step 2: Add `sources` to `ChatMessage` in useChat.ts**

```typescript
export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  text: string;
  isStreaming: boolean;
  agentSteps: AgentStep[];
  sources: string[];  // <-- add this
}
```

**Step 3: Update the reducer**

- `"user_message"` case: new assistant message gets `sources: []`
- `"user_message"` case: new user message gets `sources: []`
- New action type: `| { type: "response_complete"; text: string; sources: string[] }`
- `"response_complete"` case: set `sources: action.sources`

**Step 4: Update `sendMessage` to pass sources**

In the `sendMessage` callback where it dispatches `response_complete`:
```typescript
if (result.type === "result") {
  const data = result.data as { text: string; sources?: string[] };
  dispatch({
    type: "response_complete",
    text: data.text,
    sources: data.sources || [],
  });
}
```

**Step 5: Commit**

```bash
git add ui/src/hooks/useChat.ts ui/src/lib/protocol.ts
git commit -m "feat(ui): add sources field to ChatMessage and protocol"
```

---

### Task 6: MessageBubble renders source chips

**Files:**
- Modify: `ui/src/components/MessageBubble.tsx`

**Step 1: Add SourceChips component and rendering**

Below the answer text `<p>` and before the `AgentSteps`, render source chips:

```tsx
{!isUser && !message.isStreaming && message.sources.length > 0 && (
  <div className="flex flex-wrap gap-1.5 mt-2 pt-2 border-t border-border/50">
    {message.sources.map((source) => (
      <button
        key={source}
        onClick={() => window.electron?.openFile(source)}
        title={source}
        className="inline-flex items-center gap-1 px-2 py-0.5 text-xs font-medium
                   bg-bg-secondary border border-border rounded-full
                   hover:bg-accent-muted hover:border-accent/30
                   transition-colors cursor-pointer text-text-secondary hover:text-text-primary"
      >
        <svg className="w-3 h-3 shrink-0" viewBox="0 0 16 16" fill="currentColor">
          <path d="M3.5 2A1.5 1.5 0 002 3.5v9A1.5 1.5 0 003.5 14h9a1.5 1.5 0 001.5-1.5V6.621a1.5 1.5 0 00-.44-1.06l-3.12-3.122A1.5 1.5 0 009.378 2H3.5z"/>
        </svg>
        {source}
      </button>
    ))}
  </div>
)}
```

**Step 2: Commit**

```bash
git add ui/src/components/MessageBubble.tsx
git commit -m "feat(ui): render source file chips in message bubbles"
```

---

### Task 7: Electron IPC to open files in OS default app

**Files:**
- Modify: `ui/electron/python.ts` or `ui/electron/main.ts` — whichever handles IPC
- Modify: `ui/src/preload.ts` or equivalent — expose `openFile` to renderer

**Step 1: Find the Electron main/preload entry points**

Look for `main.ts`, `preload.ts`, or `index.ts` in `ui/electron/`. The bridge needs:
1. An IPC handler in the main process that calls `shell.openPath(filePath)`
2. A preload exposure so `window.electron.openFile(name)` works in the renderer

**Step 2: Add IPC handler in main process**

```typescript
import { shell, ipcMain } from "electron";

ipcMain.handle("open-file", async (_event, filePath: string) => {
  await shell.openPath(filePath);
});
```

**Step 3: Expose in preload**

```typescript
contextBridge.exposeInMainWorld("electron", {
  openFile: (filePath: string) => ipcRenderer.invoke("open-file", filePath),
});
```

**Step 4: Add TypeScript declaration**

In a `global.d.ts` or at the top of `MessageBubble.tsx`:
```typescript
declare global {
  interface Window {
    electron?: {
      openFile: (path: string) => Promise<void>;
    };
  }
}
```

**Step 5: Commit**

```bash
git add ui/electron/ ui/src/
git commit -m "feat(electron): IPC handler for opening files in OS default app"
```

---

### Task 8: Pass full file paths (not just names) as sources

**Files:**
- Modify: `searcher.py` (use file paths instead of just names where available)
- Modify: `server.py` (resolve relative paths to absolute using `entry["path"]`)

**Step 1: Assess current state**

After Task 1, `sources` contains filenames from `_get_source()` — these are just names like `"budget.pdf"`, not full paths. For `shell.openPath` to work, the frontend needs absolute paths.

**Step 2: Option A — resolve in server (preferred, minimal change)**

In `handle_query`, after getting `response, sources = agent.run(...)`:

```python
import os
base_dir = entry["path"]
full_sources = []
for s in sources:
    full = os.path.join(base_dir, s)
    if os.path.exists(full):
        full_sources.append(full)
    else:
        # Try to find it recursively
        for root, dirs, files in os.walk(base_dir):
            if s in files:
                full_sources.append(os.path.join(root, s))
                break
        else:
            full_sources.append(s)  # fallback to name
```

Send `full_sources` in the result. The chips display `os.path.basename(source)` as label, full path as tooltip and for opening.

**Step 3: Update MessageBubble to show basename but pass full path**

```tsx
{message.sources.map((source) => {
  const basename = source.split("/").pop() || source;
  return (
    <button key={source} onClick={() => window.electron?.openFile(source)} title={source}>
      {basename}
    </button>
  );
})}
```

**Step 4: Run full test suite**

Run: `python -m pytest tests/ -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add searcher.py server.py ui/src/components/MessageBubble.tsx
git commit -m "feat: resolve source filenames to full paths for OS open"
```

---

### Task 9: End-to-end manual test

**Steps:**
1. Start the app: `cd ui && npm run dev`
2. Open a directory with mixed file types (PDF, text, images)
3. Ask a question that requires searching files (e.g., "what invoices are there?")
4. Verify: answer appears with filename chips below it
5. Click a chip — verify the file opens in the OS default application
6. Ask a conversational follow-up that doesn't use tools — verify no chips appear
7. Test the "search all directories" mode — verify chips appear for cross-directory results

**Expected behavior:**
- Chips only appear after streaming completes
- Chips show short filenames, full paths in tooltip
- Clicking opens the correct file
- No chips for pure conversation (no tool calls)
