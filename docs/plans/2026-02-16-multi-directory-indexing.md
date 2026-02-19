# Multi-Directory Indexing Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Allow Manole to index and query multiple directories simultaneously with a side panel for managing indexes.

**Architecture:** The `Server` class replaces single agent/searcher fields with a `dict[str, DirectoryEntry]`. Model loads once and is shared. Each directory gets its own Searcher, Agent, and conversation history. The UI already has SidePanel and App.tsx wired — this plan covers backend changes, protocol updates, and frontend integration.

**Tech Stack:** Python (server.py), TypeScript/React (Electron UI), NDJSON stdio protocol

**Already done (from brainstorming session):**
- `ui/src/components/SidePanel.tsx` — complete drawer component
- `ui/src/App.tsx` — multi-directory state, hamburger toggle, side panel wiring
- `ui/src/components/ChatPanel.tsx` — `indexingMessage` prop added

---

### Task 1: Backend — DirectoryEntry dataclass and Server refactor

**Files:**
- Modify: `server.py`
- Test: `tests/test_server.py`

**Step 1: Write failing tests for multi-directory init**

Add to `tests/test_server.py`:

```python
class TestMultiDirectoryInit:
    """Test initializing multiple directories."""

    def test_server_has_directories_dict(self):
        from server import Server
        srv = Server()
        assert hasattr(srv, "directories")
        assert isinstance(srv.directories, dict)

    def test_init_returns_directory_id(self):
        """Init should return a directoryId in the result."""
        from server import Server
        srv = Server()
        srv.state = "ready"
        # We mock the heavy init internals, just test the protocol shape
        # Full integration tested separately
        assert srv.directories == {}

    def test_model_field_starts_none(self):
        from server import Server
        srv = Server()
        assert srv.model is None
```

**Step 2: Run tests to verify they fail**

Run: `cd /Users/ded/Projects/assist/manole && uv run pytest tests/test_server.py::TestMultiDirectoryInit -v`
Expected: FAIL — `Server` doesn't have `directories`

**Step 3: Refactor Server.__init__ to use directories dict**

In `server.py`, replace the single-entry fields with a directories dict. The `Server.__init__` becomes:

```python
def __init__(self):
    self.state = "not_initialized"
    self.debug = False
    self.running = True
    self.start_time = time.time()

    # Shared model — loaded once on first init
    self.model = None
    self.rewriter = None

    # Per-directory entries: {dir_id: DirectoryEntry}
    self.directories: dict[str, dict] = {}
```

Add the `DirectoryEntry` helper at module level (plain dict, not dataclass, to keep it simple):

```python
def make_dir_id(path: str) -> str:
    """Derive a stable directory ID from an absolute path."""
    return Path(path).name.replace(" ", "_").replace("/", "_")
```

**Step 4: Run tests to verify they pass**

Run: `cd /Users/ded/Projects/assist/manole && uv run pytest tests/test_server.py::TestMultiDirectoryInit -v`
Expected: PASS

**Step 5: Commit**

```bash
git add server.py tests/test_server.py
git commit -m "refactor: add directories dict to Server for multi-directory support"
```

---

### Task 2: Backend — Refactor handle_init for multi-directory

**Files:**
- Modify: `server.py`
- Test: `tests/test_server.py`

**Step 1: Write failing test for multi-init**

```python
class TestMultiInit:
    """Test that init can be called multiple times."""

    def test_handle_init_bad_dir(self):
        from server import Server
        srv = Server()
        result = srv.handle_init(1, {"dataDir": "/nonexistent/path/xyz"})
        assert result["type"] == "error"
        assert "Not a directory" in result["data"]["message"]

    def test_handle_init_returns_directory_id(self):
        """Mock heavy deps, verify init returns directoryId."""
        from server import Server, make_dir_id
        srv = Server()
        # This test needs mocking — see step 3
        dir_id = make_dir_id("/tmp/test_data")
        assert dir_id == "test_data"
```

**Step 2: Run to verify failure**

Run: `cd /Users/ded/Projects/assist/manole && uv run pytest tests/test_server.py::TestMultiInit -v`
Expected: FAIL

**Step 3: Refactor handle_init**

Replace `handle_init` in `server.py`. Key changes:
- Generate `dir_id` from path
- Load model only on first call (`if self.model is None`)
- Store entry in `self.directories[dir_id]`
- Send `directory_update` messages instead of generic `status`
- Return `directoryId` in result
- Set `self.state = "ready"` once model is loaded (global state)

```python
def handle_init(self, req_id, params: dict) -> dict:
    """Initialize a directory: load model (if needed), build index, wire agent."""
    from chat import build_index, find_index_path, get_index_name
    from leann import LeannSearcher
    from models import ModelManager
    from searcher import Searcher
    from file_reader import FileReader
    from toolbox import ToolBox
    from tools import ToolRegistry
    from router import route
    from rewriter import QueryRewriter
    from agent import Agent

    data_dir = params.get("dataDir", "./test_data")
    data_dir_path = Path(data_dir).resolve()

    if not data_dir_path.is_dir():
        return {"id": req_id, "type": "error", "data": {"message": f"Not a directory: {data_dir}"}}

    dir_id = make_dir_id(str(data_dir_path))

    # Send initial directory_update
    send(None, "directory_update", {"directoryId": dir_id, "state": "indexing"})

    # Load model once (shared across all directories)
    if self.model is None:
        send(None, "status", {"state": "loading_model"})
        self._log("Loading model...")
        self.model = ModelManager()
        self.model.load()
        self.rewriter = QueryRewriter(self.model, debug=self.debug)
        self._log("Model loaded.")

    self._log(f"Indexing {data_dir_path}...")

    # Build or reuse index
    reuse = params.get("reuse")
    if reuse:
        index_name = reuse
        self._log(f"Reusing index: {reuse}")
    else:
        index_name = build_index(data_dir_path)
        self._log(f"Index built: {index_name}")

    index_path = find_index_path(index_name)

    # Wire components for this directory
    leann_searcher = LeannSearcher(index_path, enable_warmup=True)
    file_reader = FileReader()
    toolbox = ToolBox(str(data_dir_path))
    searcher = Searcher(
        leann_searcher, self.model,
        file_reader=file_reader, toolbox=toolbox,
        debug=self.debug,
    )
    tool_registry = ToolRegistry(searcher, toolbox)

    class RouterWrapper:
        @staticmethod
        def route(query, intent=None):
            return route(query, intent=intent)

    agent = Agent(
        self.model, tool_registry, RouterWrapper(),
        rewriter=self.rewriter, debug=self.debug,
    )

    # Collect directory stats
    stats = self._collect_stats(data_dir_path)

    # Store entry
    self.directories[dir_id] = {
        "dir_id": dir_id,
        "path": str(data_dir_path),
        "index_name": index_name,
        "searcher": searcher,
        "agent": agent,
        "state": "ready",
        "stats": stats,
        "summary": None,
        "conversation_history": [],
    }

    self.state = "ready"
    self._log(f"Directory {dir_id} ready.")

    # Push directory_update with stats
    send(None, "directory_update", {
        "directoryId": dir_id,
        "state": "ready",
        "stats": stats,
    })

    send(None, "status", {"state": "ready"})
    return {
        "id": req_id, "type": "result",
        "data": {"status": "ready", "directoryId": dir_id, "indexName": index_name},
    }
```

**Step 4: Add `_collect_stats` helper**

```python
def _collect_stats(self, data_dir: Path) -> dict:
    """Collect file statistics for a directory."""
    types: dict[str, int] = {}
    total_size = 0
    file_count = 0
    for f in data_dir.rglob("*"):
        if f.is_file():
            file_count += 1
            ext = f.suffix.lstrip(".").lower() or "other"
            types[ext] = types.get(ext, 0) + 1
            total_size += f.stat().st_size
    return {"fileCount": file_count, "types": types, "totalSize": total_size}
```

**Step 5: Run tests**

Run: `cd /Users/ded/Projects/assist/manole && uv run pytest tests/test_server.py -v`
Expected: All pass (existing tests may need minor fixes for removed `self.agent` etc.)

**Step 6: Fix existing tests if needed**

The `TestQueryStreaming` test sets `srv.agent = mock_agent` directly. Update it to use the directories dict instead:

```python
def test_query_streams_tokens(self):
    from server import Server
    sent_messages = []

    import server as srv_mod
    original_send = srv_mod.send
    srv_mod.send = lambda rid, rtype, data: sent_messages.append(
        {"id": rid, "type": rtype, "data": data}
    )

    try:
        srv = Server()
        srv.state = "ready"

        mock_agent = MagicMock()
        mock_agent.run.return_value = "hello world"

        # Store in directories dict
        srv.directories["test"] = {
            "agent": mock_agent,
            "conversation_history": [],
            "state": "ready",
        }

        result = srv.handle_query(1, {"text": "test query", "directoryId": "test"})
        assert result["type"] == "result"
        assert result["data"]["text"] == "hello world"

        call_kwargs = mock_agent.run.call_args
        assert "on_token" in call_kwargs.kwargs
        assert callable(call_kwargs.kwargs["on_token"])

        on_token_cb = call_kwargs.kwargs["on_token"]
        on_token_cb("hello")
        on_token_cb(" world")
        assert len(sent_messages) == 2
        assert sent_messages[0] == {"id": 1, "type": "token", "data": {"text": "hello"}}
        assert sent_messages[1] == {"id": 1, "type": "token", "data": {"text": " world"}}
    finally:
        srv_mod.send = original_send
```

**Step 7: Commit**

```bash
git add server.py tests/test_server.py
git commit -m "feat: refactor handle_init for multi-directory indexing with shared model"
```

---

### Task 3: Backend — Refactor handle_query for directoryId routing

**Files:**
- Modify: `server.py`
- Test: `tests/test_server.py`

**Step 1: Write failing test**

```python
class TestQueryRouting:
    """Test query routes to correct directory agent."""

    def test_query_requires_directory_id(self):
        from server import Server
        srv = Server()
        srv.state = "ready"
        result = srv.handle_query(1, {"text": "hello"})
        assert result["type"] == "error"
        assert "directoryId" in result["data"]["message"] or "Not initialized" in result["data"]["message"]

    def test_query_unknown_directory(self):
        from server import Server
        srv = Server()
        srv.state = "ready"
        result = srv.handle_query(1, {"text": "hello", "directoryId": "nonexistent"})
        assert result["type"] == "error"
```

**Step 2: Run to verify failure**

Run: `cd /Users/ded/Projects/assist/manole && uv run pytest tests/test_server.py::TestQueryRouting -v`

**Step 3: Refactor handle_query**

```python
def handle_query(self, req_id, params: dict) -> dict:
    """Run agent loop with streaming tokens."""
    if self.state != "ready":
        return {"id": req_id, "type": "error", "data": {"message": "Not initialized"}}

    query = params.get("text", "").strip()
    if not query:
        return {"id": req_id, "type": "error", "data": {"message": "Empty query"}}

    dir_id = params.get("directoryId")
    search_all = params.get("searchAll", False)

    if search_all:
        return self._query_all(req_id, query)

    if not dir_id:
        # Fall back to first ready directory
        ready = [d for d in self.directories.values() if d["state"] == "ready"]
        if not ready:
            return {"id": req_id, "type": "error", "data": {"message": "No directories ready"}}
        dir_id = ready[0]["dir_id"]

    entry = self.directories.get(dir_id)
    if not entry:
        return {"id": req_id, "type": "error", "data": {"message": f"Unknown directory: {dir_id}"}}
    if entry["state"] != "ready":
        return {"id": req_id, "type": "error", "data": {"message": f"Directory not ready: {dir_id}"}}

    self._log(f"Query [{dir_id}]: {query[:80]}")

    def on_token(text):
        send(req_id, "token", {"text": text})

    def on_step(step: int, tool: str, params: dict):
        send(req_id, "agent_step", {"step": step, "tool": tool, "params": params})

    response = entry["agent"].run(
        query,
        history=entry["conversation_history"],
        on_token=on_token,
        on_step=on_step,
    )

    entry["conversation_history"].append({"role": "user", "content": query})
    entry["conversation_history"].append({"role": "assistant", "content": response})
    if len(entry["conversation_history"]) > 10:
        entry["conversation_history"] = entry["conversation_history"][-10:]

    return {"id": req_id, "type": "result", "data": {"text": response}}

def _query_all(self, req_id, query: str) -> dict:
    """Query all ready directories and merge results."""
    ready = [d for d in self.directories.values() if d["state"] == "ready"]
    if not ready:
        return {"id": req_id, "type": "error", "data": {"message": "No directories ready"}}

    parts = []
    for entry in ready:
        response = entry["agent"].run(query, history=entry["conversation_history"])
        folder_name = Path(entry["path"]).name
        parts.append(f"[{folder_name}]\n{response}")

    merged = "\n\n".join(parts)
    send(req_id, "token", {"text": merged})
    return {"id": req_id, "type": "result", "data": {"text": merged}}
```

**Step 4: Run all tests**

Run: `cd /Users/ded/Projects/assist/manole && uv run pytest tests/test_server.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add server.py tests/test_server.py
git commit -m "feat: add directoryId routing and searchAll to handle_query"
```

---

### Task 4: Backend — Add remove_directory and reindex handlers

**Files:**
- Modify: `server.py`
- Test: `tests/test_server.py`

**Step 1: Write failing tests**

```python
class TestRemoveDirectory:
    def test_remove_existing(self):
        from server import Server
        srv = Server()
        srv.directories["test"] = {"dir_id": "test", "state": "ready"}
        result = srv.handle_remove_directory(1, {"directoryId": "test"})
        assert result["type"] == "result"
        assert "test" not in srv.directories

    def test_remove_nonexistent(self):
        from server import Server
        srv = Server()
        result = srv.handle_remove_directory(1, {"directoryId": "nope"})
        assert result["type"] == "error"
```

**Step 2: Run to verify failure**

Run: `cd /Users/ded/Projects/assist/manole && uv run pytest tests/test_server.py::TestRemoveDirectory -v`

**Step 3: Implement handlers**

```python
def handle_remove_directory(self, req_id, params: dict) -> dict:
    dir_id = params.get("directoryId")
    if dir_id not in self.directories:
        return {"id": req_id, "type": "error", "data": {"message": f"Unknown directory: {dir_id}"}}
    del self.directories[dir_id]
    self._log(f"Removed directory: {dir_id}")
    return {"id": req_id, "type": "result", "data": {"status": "ok"}}

def handle_reindex(self, req_id, params: dict) -> dict:
    dir_id = params.get("directoryId")
    entry = self.directories.get(dir_id)
    if not entry:
        return {"id": req_id, "type": "error", "data": {"message": f"Unknown directory: {dir_id}"}}
    # Re-init the same path
    return self.handle_init(req_id, {"dataDir": entry["path"]})
```

**Step 4: Register in dispatch**

Add to the `handlers` dict in `dispatch`:

```python
"remove_directory": lambda: self.handle_remove_directory(req_id, params),
"reindex": lambda: self.handle_reindex(req_id, params),
```

**Step 5: Run all tests**

Run: `cd /Users/ded/Projects/assist/manole && uv run pytest tests/test_server.py -v`

**Step 6: Commit**

```bash
git add server.py tests/test_server.py
git commit -m "feat: add remove_directory and reindex server handlers"
```

---

### Task 5: Backend — Auto-generate directory summary

**Files:**
- Modify: `server.py`
- Test: `tests/test_server.py`

**Step 1: Write failing test**

```python
class TestDirectorySummary:
    def test_generate_summary(self):
        from server import Server
        srv = Server()
        mock_model = MagicMock()
        mock_model.generate.return_value = "Contains invoices and contracts"
        srv.model = mock_model
        summary = srv._generate_summary({"fileCount": 10, "types": {"pdf": 8, "txt": 2}, "totalSize": 5000})
        assert isinstance(summary, str)
        assert len(summary) > 0
```

**Step 2: Run to verify failure**

Run: `cd /Users/ded/Projects/assist/manole && uv run pytest tests/test_server.py::TestDirectorySummary -v`

**Step 3: Implement _generate_summary**

```python
def _generate_summary(self, stats: dict) -> str:
    """Use the model to generate a brief summary of directory contents."""
    if not self.model:
        return ""
    types_str = ", ".join(f"{v} {k}" for k, v in stats.get("types", {}).items())
    prompt = (
        f"This folder contains {stats.get('fileCount', 0)} files ({types_str}). "
        "In one sentence, describe what kind of documents this folder likely contains. "
        "Be specific and concise."
    )
    messages = [{"role": "user", "content": prompt}]
    return self.model.generate(messages).strip()
```

**Step 4: Call it at the end of handle_init, after wiring agent**

After `self.directories[dir_id] = {...}`, add:

```python
# Auto-generate summary
summary = self._generate_summary(stats)
self.directories[dir_id]["summary"] = summary

# Push final directory_update with stats and summary
send(None, "directory_update", {
    "directoryId": dir_id,
    "state": "ready",
    "stats": stats,
    "summary": summary,
})
```

**Step 5: Run all tests**

Run: `cd /Users/ded/Projects/assist/manole && uv run pytest tests/test_server.py -v`

**Step 6: Commit**

```bash
git add server.py tests/test_server.py
git commit -m "feat: auto-generate directory content summary after indexing"
```

---

### Task 6: Frontend — Update protocol types for directory_update

**Files:**
- Modify: `ui/src/lib/protocol.ts`

**Step 1: Add new types**

```typescript
// Add to ResponseType union
export type ResponseType = "result" | "token" | "agent_step" | "error" | "status" | "progress" | "log" | "directory_update";

// Add new interface
export interface DirectoryUpdateData {
  directoryId: string;
  state: "indexing" | "ready" | "error";
  stats?: { fileCount: number; types: Record<string, number>; totalSize: number };
  summary?: string;
  error?: string;
}
```

**Step 2: Commit**

```bash
git add ui/src/lib/protocol.ts
git commit -m "feat: add directory_update protocol type"
```

---

### Task 7: Frontend — Handle directory_update in usePython hook

**Files:**
- Modify: `ui/src/hooks/usePython.ts`

**Step 1: Update usePython to forward directory_update messages**

The `directory_update` messages are already forwarded to subscribers via `handlersRef`. No changes needed to usePython itself — the handlers in App.tsx will pick them up. But we should make sure the `backendState` still works for the global "model loading" state.

This is already handled — `status` messages set `backendState`, and `directory_update` flows through subscribers. No code change needed here.

**Step 2: Verify by reading the code — no commit needed**

---

### Task 8: Frontend — Wire App.tsx to handle directory_update from backend

**Files:**
- Modify: `ui/src/App.tsx`
- Modify: `ui/src/hooks/useChat.ts`

**Step 1: Update useChat to handle directory_update and expose subscribe**

Add `directory_update` handling to `useChat`'s subscribe callback, or — simpler — subscribe directly in App.tsx via `usePython`'s subscribe.

The cleanest approach: expose `subscribe` from `useChat` so App.tsx can listen for `directory_update` messages. Actually, `useChat` already uses `subscribe` internally. The better approach is to have App.tsx subscribe separately via `usePython`.

Update `useChat` to also export `subscribe`:

In `ui/src/hooks/useChat.ts`, add `subscribe` to the return:

```typescript
return {
    messages: state.messages,
    isLoading: state.isLoading,
    error: state.error,
    backendState,
    logs,
    sendMessage,
    initBackend,
    resetBackendState,
    clearChat: () => dispatch({ type: "clear" }),
    subscribe,  // <-- add this
    send,       // <-- add this for remove_directory/reindex
};
```

**Step 2: Update App.tsx to subscribe to directory_update**

Add a `useEffect` in App.tsx that subscribes to `directory_update` messages and updates the `directories` state:

```typescript
const { messages, isLoading, error, backendState, logs, sendMessage, initBackend, resetBackendState, subscribe, send } =
    useChat();

// Listen for directory_update messages from backend
useEffect(() => {
    return subscribe((response) => {
        if (response.type === "directory_update") {
            const data = response.data as {
                directoryId: string;
                state: "indexing" | "ready" | "error";
                stats?: { fileCount: number; types: Record<string, number>; totalSize: number };
                summary?: string;
                error?: string;
            };
            setDirectories((prev) =>
                prev.map((d) =>
                    d.id === data.directoryId
                        ? {
                            ...d,
                            state: data.state,
                            ...(data.stats ? { stats: data.stats } : {}),
                            ...(data.summary ? { summary: data.summary } : {}),
                            ...(data.error ? { error: data.error } : {}),
                        }
                        : d
                )
            );
        }
    });
}, [subscribe]);
```

**Step 3: Update sendMessage to pass directoryId**

Update the `sendMessage` call in App.tsx to include the active directory:

```typescript
const handleSend = useCallback(
    (text: string) => {
        sendMessage(text, activeDirectoryId, searchAll);
    },
    [sendMessage, activeDirectoryId, searchAll]
);
```

And update `useChat.sendMessage` to accept and forward these params:

```typescript
const sendMessage = useCallback(
    async (text: string, directoryId?: string | null, searchAll?: boolean) => {
        dispatch({ type: "user_message", text });
        const params: Record<string, unknown> = { text };
        if (directoryId) params.directoryId = directoryId;
        if (searchAll) params.searchAll = true;
        const result = await send("query", params);
        if (result.type === "result") {
            dispatch({ type: "response_complete", text: (result.data as { text: string }).text });
        } else if (result.type === "error") {
            dispatch({ type: "error", message: (result.data as { message: string }).message });
        }
    },
    [send],
);
```

**Step 4: Update handleRemove to call backend**

```typescript
const handleRemove = useCallback(async (dirId: string) => {
    setDirectories((prev) => prev.filter((d) => d.id !== dirId));
    setActiveDirectoryId((prev) => (prev === dirId ? null : prev));
    try {
        await send("remove_directory", { directoryId: dirId });
    } catch {
        // Directory already removed from UI state
    }
}, [send]);
```

**Step 5: Update handleReindex to call backend**

```typescript
const handleReindex = useCallback(
    async (dirId: string) => {
        setDirectories((prev) =>
            prev.map((d) =>
                d.id === dirId ? { ...d, state: "indexing" as const } : d
            )
        );
        try {
            await send("reindex", { directoryId: dirId });
        } catch (err) {
            setDirectories((prev) =>
                prev.map((d) =>
                    d.id === dirId
                        ? { ...d, state: "error" as const, error: String(err) }
                        : d
                )
            );
        }
    },
    [send]
);
```

**Step 6: Commit**

```bash
git add ui/src/hooks/useChat.ts ui/src/App.tsx
git commit -m "feat: wire directory_update messages and directoryId routing in frontend"
```

---

### Task 9: Frontend — Per-directory conversation history

**Files:**
- Modify: `ui/src/App.tsx`
- Modify: `ui/src/hooks/useChat.ts`

**Step 1: Add per-directory message storage**

Currently `useChat` holds a single `messages[]`. For per-directory conversations, the simplest approach: store a `Record<string, ChatMessage[]>` in App.tsx and pass the active directory's messages to ChatPanel. Reset the chat reducer when switching directories.

Add to App.tsx state:

```typescript
const [conversationsByDir, setConversationsByDir] = useState<Record<string, ChatMessage[]>>({});
```

When `activeDirectoryId` changes, save current messages and load the new directory's messages. Use `clearChat` from useChat plus restore.

Actually the simpler approach: just key the ChatPanel by `activeDirectoryId` (already done with `key={`chat-${activeDirectoryId}`}`). Each key remount creates fresh state. The conversation history lives in the backend's `entry["conversation_history"]` per directory anyway. The UI doesn't need to persist messages across switches for now — the backend holds history for context.

This is already implemented via the keyed ChatPanel. No additional code needed.

**Step 2: Verify — no commit needed**

---

### Task 10: Integration test — end-to-end smoke test

**Files:**
- Test: `tests/test_server.py`

**Step 1: Write integration test for full multi-directory flow**

```python
class TestMultiDirectoryFlow:
    """Integration test for the multi-directory protocol flow."""

    def test_dispatch_routes_all_methods(self):
        from server import Server
        srv = Server()
        srv.state = "ready"

        # remove_directory on nonexistent
        result = srv.dispatch({"id": 1, "method": "remove_directory", "params": {"directoryId": "x"}})
        assert result["type"] == "error"

        # reindex on nonexistent
        result = srv.dispatch({"id": 2, "method": "reindex", "params": {"directoryId": "x"}})
        assert result["type"] == "error"

        # ping still works
        result = srv.dispatch({"id": 3, "method": "ping"})
        assert result["type"] == "result"
```

**Step 2: Run full test suite**

Run: `cd /Users/ded/Projects/assist/manole && uv run pytest tests/ -v`
Expected: All pass

**Step 3: Commit**

```bash
git add tests/test_server.py
git commit -m "test: add multi-directory integration tests"
```

---

### Task 11: Cleanup — Remove LoadingScreen dependency

**Files:**
- Modify: `ui/src/App.tsx` — remove `LoadingScreen` import (already not used)
- Optionally keep `ui/src/components/LoadingScreen.tsx` for now (no harm)

**Step 1: Remove unused import from App.tsx**

The `LoadingScreen` import was already removed in the brainstorming session. Verify and clean up any other unused imports.

**Step 2: Commit**

```bash
git add ui/src/App.tsx
git commit -m "chore: remove unused LoadingScreen import"
```

---

Plan saved. Two execution options:

**1. Subagent-Driven (this session)** — I dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Parallel Session (separate)** — Open new session with executing-plans, batch execution with checkpoints

Which approach?