# Electron React UI Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Wire a polished Electron + React + shadcn UI to the existing Python backend via stdio NDJSON, with streaming token support.

**Architecture:** Python backend runs as a child process of Electron. Communication is NDJSON over stdin/stdout. A thin `server.py` adapter wraps the existing Agent/Searcher/Model modules. The only modification to existing code is adding streaming support to `ModelManager.generate()`.

**Tech Stack:** electron-vite, React 18, shadcn/ui, Tailwind CSS, TypeScript, Motion (framer-motion), PyInstaller

**Visual Design:** `docs/plans/2026-02-16-electron-ui-visual-design.md` — Warm Brutalism aesthetic with Cormorant Garamond + DM Sans + IBM Plex Mono typography, dark amber palette, film grain texture, Motion animations for agent step reveals and streaming.

**Design doc:** `docs/plans/2026-02-16-electron-ui-design.md`

---

### Task 1: Add streaming support to ModelManager

The only change to existing code. Add a `stream=True` parameter and `on_token` callback to `ModelManager.generate()`. Backward-compatible — existing callers are unaffected.

**Files:**
- Modify: `models.py`
- Test: `tests/test_models.py`

**Step 1: Write the failing test**

Add to `tests/test_models.py`:

```python
def test_generate_stream_calls_on_token(mock_model):
    """Streaming generate calls on_token for each chunk and returns full text."""
    tokens = []
    result = mock_model.generate(
        [{"role": "user", "content": "hi"}],
        stream=True,
        on_token=lambda t: tokens.append(t),
    )
    # The mock should produce at least one token
    assert isinstance(result, str)
    assert len(result) > 0
    assert len(tokens) > 0
    assert "".join(tokens) == result


def test_generate_stream_without_callback(mock_model):
    """Streaming without on_token still returns full text."""
    result = mock_model.generate(
        [{"role": "user", "content": "hi"}],
        stream=True,
    )
    assert isinstance(result, str)
    assert len(result) > 0


def test_generate_non_stream_unchanged(mock_model):
    """Default non-streaming behavior is unchanged."""
    result = mock_model.generate(
        [{"role": "user", "content": "hi"}],
    )
    assert isinstance(result, str)
    assert len(result) > 0
```

Note: `mock_model` is an existing fixture or we create one. Check the existing test file first — if there's already a fixture, use it. If not, create a `ModelManager` with a mock `Llama` that supports `create_chat_completion(stream=True)` returning an iterator of chunk dicts.

**Step 2: Run test to verify it fails**

Run: `cd /Users/ded/Projects/assist/manole && uv run pytest tests/test_models.py -v -k "stream"`
Expected: FAIL — `generate()` doesn't accept `stream` parameter

**Step 3: Implement streaming in ModelManager.generate()**

Modify `models.py` — the `generate` method:

```python
def generate(self, messages: list[dict], max_tokens: int = 1024,
             stream: bool = False, on_token=None) -> str:
    self.model.reset()
    if stream:
        chunks = self.model.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.1,
            top_k=50,
            top_p=0.1,
            repeat_penalty=1.05,
            stream=True,
        )
        parts = []
        for chunk in chunks:
            delta = chunk["choices"][0].get("delta", {})
            text = delta.get("content", "")
            if text:
                parts.append(text)
                if on_token:
                    on_token(text)
        return "".join(parts)

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

**Step 4: Run test to verify it passes**

Run: `cd /Users/ded/Projects/assist/manole && uv run pytest tests/test_models.py -v`
Expected: ALL PASS

**Step 5: Run full test suite to verify no regressions**

Run: `cd /Users/ded/Projects/assist/manole && uv run pytest tests/ -v`
Expected: ALL PASS — existing callers don't pass `stream=True`, so nothing changes

**Step 6: Commit**

```bash
git add models.py tests/test_models.py
git commit -m "feat: add streaming support to ModelManager.generate()"
```

---

### Task 2: Create server.py NDJSON adapter

The thin adapter that reads NDJSON from stdin, dispatches to existing modules, writes NDJSON to stdout. This is the bridge between Electron and Python.

**Files:**
- Create: `server.py` (note: this replaces the concept — the current `server.py` doesn't exist, there's only `chat.py`)
- Test: `tests/test_server.py`

**Step 1: Write the failing tests**

Create `tests/test_server.py`:

```python
"""Tests for the NDJSON server protocol layer."""
import json
import io
import sys
import pytest
from unittest.mock import MagicMock, patch


def make_request(method, params=None, req_id=1):
    return json.dumps({"id": req_id, "method": method, "params": params or {}})


class TestProtocol:
    """Test the NDJSON read/write protocol."""

    def test_parse_valid_request(self):
        from server import parse_request
        req = parse_request('{"id": 1, "method": "ping", "params": {}}')
        assert req["id"] == 1
        assert req["method"] == "ping"

    def test_parse_invalid_json(self):
        from server import parse_request
        req = parse_request("not json")
        assert req is None

    def test_parse_missing_method(self):
        from server import parse_request
        req = parse_request('{"id": 1}')
        assert req is None

    def test_format_result(self):
        from server import format_response
        line = format_response(1, "result", {"status": "ok"})
        parsed = json.loads(line)
        assert parsed == {"id": 1, "type": "result", "data": {"status": "ok"}}

    def test_format_token(self):
        from server import format_response
        line = format_response(2, "token", {"text": "hello"})
        parsed = json.loads(line)
        assert parsed["id"] == 2
        assert parsed["type"] == "token"
        assert parsed["data"]["text"] == "hello"

    def test_format_error(self):
        from server import format_response
        line = format_response(None, "error", {"message": "boom"})
        parsed = json.loads(line)
        assert parsed["id"] is None
        assert parsed["type"] == "error"


class TestPing:
    """Test the ping/health check method."""

    def test_ping_before_init(self):
        from server import Server
        srv = Server()
        result = srv.handle_ping(1)
        assert result["data"]["state"] == "not_initialized"

    def test_ping_after_ready(self):
        from server import Server
        srv = Server()
        srv.state = "ready"
        result = srv.handle_ping(1)
        assert result["data"]["state"] == "ready"
        assert "uptime" in result["data"]


class TestToggleDebug:
    """Test debug toggle."""

    def test_toggle_debug(self):
        from server import Server
        srv = Server()
        assert srv.debug is False
        result = srv.handle_toggle_debug(1)
        assert srv.debug is True
        assert result["data"]["debug"] is True


class TestShutdown:
    """Test clean shutdown."""

    def test_shutdown_sets_running_false(self):
        from server import Server
        srv = Server()
        srv.running = True
        result = srv.handle_shutdown(1)
        assert srv.running is False
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/ded/Projects/assist/manole && uv run pytest tests/test_server.py -v`
Expected: FAIL — `server` module doesn't exist

**Step 3: Implement server.py**

Create `server.py`:

```python
"""NDJSON stdio server — thin adapter between Electron and the Python backend.

Reads JSON requests from stdin (one per line), dispatches to existing modules,
writes JSON responses to stdout. The core modules (Agent, Searcher, ModelManager)
are untouched — this is purely a protocol adapter.

Protocol:
    Request:  {"id": int, "method": str, "params": dict}
    Response: {"id": int|null, "type": str, "data": dict}
"""
import json
import sys
import time
from pathlib import Path


def parse_request(line: str) -> dict | None:
    try:
        req = json.loads(line.strip())
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(req, dict) or "method" not in req:
        return None
    return req


def format_response(req_id, resp_type: str, data: dict) -> str:
    return json.dumps({"id": req_id, "type": resp_type, "data": data})


def send(req_id, resp_type: str, data: dict):
    """Write a single NDJSON line to stdout."""
    print(format_response(req_id, resp_type, data), flush=True)


class Server:
    """NDJSON protocol server wrapping the existing backend."""

    def __init__(self):
        self.state = "not_initialized"
        self.debug = False
        self.running = True
        self.start_time = time.time()

        # Initialized by handle_init
        self.model = None
        self.agent = None
        self.searcher = None
        self.rewriter = None
        self.conversation_history = []
        self.data_dir = None
        self.index_name = None

    def handle_ping(self, req_id) -> dict:
        data = {
            "state": self.state,
            "uptime": round(time.time() - self.start_time, 1),
        }
        return {"id": req_id, "type": "result", "data": data}

    def handle_toggle_debug(self, req_id) -> dict:
        self.debug = not self.debug
        if self.agent:
            self.agent.debug = self.debug
        if self.searcher:
            self.searcher.debug = self.debug
        if self.rewriter:
            self.rewriter.debug = self.debug
        return {"id": req_id, "type": "result", "data": {"debug": self.debug}}

    def handle_shutdown(self, req_id) -> dict:
        self.running = False
        return {"id": req_id, "type": "result", "data": {"status": "shutting_down"}}

    def handle_init(self, req_id, params: dict) -> dict:
        """Initialize the backend: load model, build/reuse index, wire agent."""
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

        send(None, "status", {"state": "loading_model"})

        # Load model
        self.model = ModelManager()
        self.model.load()

        send(None, "status", {"state": "indexing"})

        # Build or reuse index
        reuse = params.get("reuse")
        if reuse:
            self.index_name = reuse
        else:
            self.index_name = build_index(data_dir_path)

        index_path = find_index_path(self.index_name)

        # Wire components
        leann_searcher = LeannSearcher(index_path, enable_warmup=True)
        file_reader = FileReader()
        toolbox = ToolBox(str(data_dir_path))
        self.searcher = Searcher(
            leann_searcher, self.model,
            file_reader=file_reader, toolbox=toolbox,
            debug=self.debug,
        )
        tool_registry = ToolRegistry(self.searcher, toolbox)

        class RouterWrapper:
            @staticmethod
            def route(query, intent=None):
                return route(query, intent=intent)

        self.rewriter = QueryRewriter(self.model, debug=self.debug)
        self.agent = Agent(
            self.model, tool_registry, RouterWrapper(),
            rewriter=self.rewriter, debug=self.debug,
        )
        self.data_dir = str(data_dir_path)
        self.conversation_history = []
        self.state = "ready"

        send(None, "status", {"state": "ready"})
        return {
            "id": req_id, "type": "result",
            "data": {"status": "ready", "indexName": self.index_name},
        }

    def handle_query(self, req_id, params: dict) -> dict:
        """Run agent loop with streaming tokens."""
        if self.state != "ready":
            return {"id": req_id, "type": "error", "data": {"message": "Not initialized"}}

        query = params.get("text", "").strip()
        if not query:
            return {"id": req_id, "type": "error", "data": {"message": "Empty query"}}

        # Run agent — for now, non-streaming (streaming wired in Task 3)
        response = self.agent.run(query, history=self.conversation_history)

        self.conversation_history.append({"role": "user", "content": query})
        self.conversation_history.append({"role": "assistant", "content": response})
        if len(self.conversation_history) > 10:
            self.conversation_history = self.conversation_history[-10:]

        return {"id": req_id, "type": "result", "data": {"text": response}}

    def handle_list_indexes(self, req_id) -> dict:
        """List available LEANN indexes."""
        indexes = []
        for base in [Path(".leann/indexes"), Path.home() / ".leann/indexes"]:
            if base.is_dir():
                for d in sorted(base.iterdir()):
                    if d.is_dir():
                        indexes.append(d.name)
        return {"id": req_id, "type": "result", "data": {"indexes": indexes}}

    def dispatch(self, req: dict):
        """Route a parsed request to the appropriate handler."""
        req_id = req.get("id")
        method = req["method"]
        params = req.get("params", {})

        handlers = {
            "ping": lambda: self.handle_ping(req_id),
            "init": lambda: self.handle_init(req_id, params),
            "query": lambda: self.handle_query(req_id, params),
            "toggle_debug": lambda: self.handle_toggle_debug(req_id),
            "list_indexes": lambda: self.handle_list_indexes(req_id),
            "shutdown": lambda: self.handle_shutdown(req_id),
        }

        handler = handlers.get(method)
        if not handler:
            return {"id": req_id, "type": "error", "data": {"message": f"Unknown method: {method}"}}

        try:
            return handler()
        except Exception as e:
            return {"id": req_id, "type": "error", "data": {"message": str(e)}}

    def run(self, input_stream=None):
        """Main loop: read stdin, dispatch, write stdout."""
        stream = input_stream or sys.stdin
        for line in stream:
            line = line.strip()
            if not line:
                continue

            req = parse_request(line)
            if req is None:
                send(None, "error", {"message": "Invalid JSON"})
                continue

            result = self.dispatch(req)
            if result:
                print(format_response(result["id"], result["type"], result["data"]), flush=True)

            if not self.running:
                break


if __name__ == "__main__":
    # Redirect all print output from core modules to stderr so only
    # NDJSON protocol messages go to stdout
    import io
    import os

    # Save real stdout for NDJSON
    _real_stdout = sys.stdout

    # Redirect Python's stdout to stderr for all non-protocol prints
    # (build_index, model loading, debug traces, etc.)
    sys.stdout = sys.stderr

    # Override send() to use the real stdout
    import builtins
    _original_print = builtins.print

    def _ndjson_print(*args, **kwargs):
        """Print to real stdout only when flush=True (our protocol sends)."""
        if kwargs.get("flush"):
            kwargs["file"] = _real_stdout
        _original_print(*args, **kwargs)

    builtins.print = _ndjson_print

    server = Server()
    server.run(sys.stdin)
```

**Step 4: Run tests to verify they pass**

Run: `cd /Users/ded/Projects/assist/manole && uv run pytest tests/test_server.py -v`
Expected: ALL PASS

**Step 5: Run full test suite**

Run: `cd /Users/ded/Projects/assist/manole && uv run pytest tests/ -v`
Expected: ALL PASS

**Step 6: Commit**

```bash
git add server.py tests/test_server.py
git commit -m "feat: add NDJSON stdio server adapter for Electron IPC"
```

---

### Task 3: Wire streaming tokens through server.py query handler

Connect the streaming `ModelManager.generate(stream=True)` to the NDJSON protocol so tokens flow to Electron in real-time.

**Files:**
- Modify: `server.py` (handle_query method)
- Modify: `agent.py` (pass stream/on_token through Agent.run)
- Test: `tests/test_server.py` (add streaming test)

**Step 1: Write the failing test**

Add to `tests/test_server.py`:

```python
class TestQueryStreaming:
    """Test that query sends token events before result."""

    def test_query_streams_tokens(self):
        """Query handler should send token events via send() during generation."""
        from server import Server
        sent_messages = []

        # Patch send to capture output
        import server as srv_mod
        original_send = srv_mod.send
        srv_mod.send = lambda rid, rtype, data: sent_messages.append(
            {"id": rid, "type": rtype, "data": data}
        )

        try:
            srv = Server()
            srv.state = "ready"

            # Mock agent that simulates streaming by calling the on_token callback
            mock_agent = MagicMock()
            mock_agent.run.return_value = "hello world"
            srv.agent = mock_agent

            # Mock model with streaming support
            mock_model = MagicMock()
            srv.model = mock_model

            result = srv.handle_query(1, {"text": "test query"})
            assert result["type"] == "result"
            assert result["data"]["text"] == "hello world"
        finally:
            srv_mod.send = original_send
```

Note: Full streaming integration requires modifying `Agent.run()` to accept and propagate `on_token`. This test verifies the server-side wiring. The actual token streaming through the agent loop is a deeper change — we'll do it in two parts: first wire the callback through `server.py → Agent.run()`, then verify tokens reach the protocol.

**Step 2: Run test to verify behavior**

Run: `cd /Users/ded/Projects/assist/manole && uv run pytest tests/test_server.py::TestQueryStreaming -v`

**Step 3: Add on_token propagation to Agent.run()**

Modify `agent.py` — the `run` method signature:

```python
def run(self, query: str, history: list[dict] = None, on_token=None) -> str:
```

Then in every `self.model.generate(messages)` call inside `run()`, pass through:

```python
raw = self.model.generate(messages, stream=bool(on_token), on_token=on_token)
```

There are 3 `self.model.generate()` calls in `Agent.run()`:
1. Line 112: main loop generation
2. Line 185: forced synthesis at max steps

All should pass `stream` and `on_token` through.

**Step 4: Update server.py handle_query to use streaming**

```python
def handle_query(self, req_id, params: dict) -> dict:
    if self.state != "ready":
        return {"id": req_id, "type": "error", "data": {"message": "Not initialized"}}

    query = params.get("text", "").strip()
    if not query:
        return {"id": req_id, "type": "error", "data": {"message": "Empty query"}}

    def on_token(text):
        send(req_id, "token", {"text": text})

    response = self.agent.run(
        query,
        history=self.conversation_history,
        on_token=on_token,
    )

    self.conversation_history.append({"role": "user", "content": query})
    self.conversation_history.append({"role": "assistant", "content": response})
    if len(self.conversation_history) > 10:
        self.conversation_history = self.conversation_history[-10:]

    return {"id": req_id, "type": "result", "data": {"text": response}}
```

**Step 5: Run all tests**

Run: `cd /Users/ded/Projects/assist/manole && uv run pytest tests/ -v`
Expected: ALL PASS — existing callers of `Agent.run()` don't pass `on_token`, so `on_token=None` means no streaming, backward-compatible.

**Step 6: Commit**

```bash
git add server.py agent.py tests/test_server.py
git commit -m "feat: wire streaming tokens through agent to NDJSON protocol"
```

---

### Task 4: Scaffold electron-vite project

Create the `ui/` directory with electron-vite, React, TypeScript, Tailwind, shadcn.

**Files:**
- Create: `ui/` directory (electron-vite scaffold)
- Modify: `ui/package.json` (add shadcn deps)

**Step 1: Scaffold electron-vite project**

```bash
cd /Users/ded/Projects/assist/manole
npm create @electron-vite@latest ui -- --template react-ts
```

Follow prompts: project name `neurofind`, React + TypeScript template.

**Step 2: Install dependencies**

```bash
cd /Users/ded/Projects/assist/manole/ui
npm install
```

**Step 3: Add Tailwind CSS**

```bash
cd /Users/ded/Projects/assist/manole/ui
npm install -D tailwindcss @tailwindcss/vite
```

Configure Tailwind in the renderer vite config and add `@import "tailwindcss"` to the main CSS file.

**Step 4: Initialize shadcn**

```bash
cd /Users/ded/Projects/assist/manole/ui
npx shadcn@latest init
```

Choose: TypeScript, default style, CSS variables for colors.

**Step 4b: Install Motion library**

```bash
cd /Users/ded/Projects/assist/manole/ui
npm i motion
```

**Step 4c: Add Google Fonts**

Add to `ui/src/assets/main.css` (or the renderer's root CSS):

```css
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@600;700&family=DM+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');
```

**Step 4d: Apply visual design theme**

Override shadcn CSS variables and add Tailwind font families per `docs/plans/2026-02-16-electron-ui-visual-design.md` — the "shadcn Theme Overrides" and "Tailwind Config Extensions" sections. Add the film grain overlay CSS to the root stylesheet.

**Step 5: Verify it runs**

```bash
cd /Users/ded/Projects/assist/manole/ui
npm run dev
```

Expected: Electron window opens with the default React template.

**Step 6: Commit**

```bash
cd /Users/ded/Projects/assist/manole
git add ui/
git commit -m "feat: scaffold electron-vite React app with Tailwind and shadcn"
```

---

### Task 5: Create TypeScript protocol types and Python bridge

Define the shared protocol types and the Electron main process module that spawns Python and speaks NDJSON.

**Files:**
- Create: `ui/src/lib/protocol.ts`
- Create: `ui/electron/python.ts`
- Modify: `ui/electron/main.ts`
- Create: `ui/electron/preload.ts` (modify the scaffolded one)

**Step 1: Create protocol types**

Create `ui/src/lib/protocol.ts`:

```typescript
// NDJSON protocol types matching server.py

export interface Request {
  id: number;
  method: string;
  params: Record<string, unknown>;
}

export type ResponseType = "result" | "token" | "agent_step" | "error" | "status" | "progress";

export interface Response {
  id: number | null;
  type: ResponseType;
  data: Record<string, unknown>;
}

export interface TokenData {
  text: string;
}

export interface AgentStepData {
  step: number;
  tool: string;
  params: Record<string, unknown>;
}

export interface StatusData {
  state: "loading_model" | "indexing" | "ready" | "not_initialized";
}

export interface ProgressData {
  stage: string;
  percent: number;
}

export interface ResultData {
  text?: string;
  status?: string;
  indexName?: string;
  indexes?: string[];
  debug?: boolean;
}

export interface ErrorData {
  message: string;
}
```

**Step 2: Create Python bridge**

Create `ui/electron/python.ts`:

```typescript
import { ChildProcess, spawn } from "child_process";
import { app } from "electron";
import { join } from "path";
import { createInterface } from "readline";
import type { Request, Response } from "../src/lib/protocol";

export type ResponseHandler = (response: Response) => void;

export class PythonBridge {
  private process: ChildProcess | null = null;
  private nextId = 1;
  private handlers: Map<number, ResponseHandler> = new Map();
  private globalHandler: ResponseHandler | null = null;

  /** Get the path to the Python executable or PyInstaller binary. */
  private getPythonCommand(): { command: string; args: string[] } {
    if (app.isPackaged) {
      const binary = join(process.resourcesPath, "manole-server");
      return { command: binary, args: [] };
    }
    // Dev mode: run server.py with Python from the venv
    const projectRoot = join(__dirname, "..", "..");
    const python = join(projectRoot, ".venv", "bin", "python");
    const serverPy = join(projectRoot, "server.py");
    return { command: python, args: [serverPy] };
  }

  /** Spawn the Python process and start reading NDJSON from stdout. */
  spawn(onMessage: ResponseHandler): void {
    const { command, args } = this.getPythonCommand();
    this.globalHandler = onMessage;

    this.process = spawn(command, args, {
      stdio: ["pipe", "pipe", "pipe"],
    });

    // Read stdout line-by-line
    const rl = createInterface({ input: this.process.stdout! });
    rl.on("line", (line: string) => {
      try {
        const response: Response = JSON.parse(line);
        // Route to request-specific handler or global handler
        if (response.id !== null && this.handlers.has(response.id)) {
          if (response.type === "result" || response.type === "error") {
            const handler = this.handlers.get(response.id)!;
            handler(response);
            this.handlers.delete(response.id);
          } else {
            // Streaming: token, agent_step — send to global
            onMessage(response);
          }
        } else {
          onMessage(response);
        }
      } catch {
        // Ignore non-JSON lines (debug output on stderr should prevent this)
      }
    });

    // Forward stderr for debugging
    this.process.stderr?.on("data", (data: Buffer) => {
      console.error("[python]", data.toString());
    });

    this.process.on("exit", (code) => {
      console.error(`[python] exited with code ${code}`);
      onMessage({
        id: null,
        type: "error",
        data: { message: `Python process exited (code ${code})` },
      });
    });
  }

  /** Send a request and return a promise that resolves with the result. */
  send(method: string, params: Record<string, unknown> = {}): Promise<Response> {
    return new Promise((resolve, reject) => {
      if (!this.process?.stdin?.writable) {
        reject(new Error("Python process not running"));
        return;
      }

      const id = this.nextId++;
      const request: Request = { id, method, params };

      this.handlers.set(id, resolve);

      const line = JSON.stringify(request) + "\n";
      this.process.stdin.write(line);
    });
  }

  /** Kill the Python process. */
  kill(): void {
    if (this.process) {
      this.send("shutdown").catch(() => {});
      setTimeout(() => this.process?.kill(), 2000);
    }
  }
}
```

**Step 3: Update preload.ts to expose IPC**

Modify `ui/electron/preload.ts`:

```typescript
import { contextBridge, ipcRenderer } from "electron";

contextBridge.exposeInMainWorld("api", {
  send: (method: string, params?: Record<string, unknown>) =>
    ipcRenderer.invoke("python:send", method, params),
  onMessage: (callback: (response: unknown) => void) => {
    const listener = (_event: unknown, response: unknown) => callback(response);
    ipcRenderer.on("python:message", listener);
    return () => ipcRenderer.removeListener("python:message", listener);
  },
});
```

**Step 4: Update main.ts to wire PythonBridge with IPC**

Modify `ui/electron/main.ts` — add after window creation:

```typescript
import { PythonBridge } from "./python";

const python = new PythonBridge();

// Spawn Python and forward all messages to renderer
python.spawn((response) => {
  mainWindow?.webContents.send("python:message", response);
});

// Handle renderer → Python requests
ipcMain.handle("python:send", async (_event, method: string, params?: Record<string, unknown>) => {
  return python.send(method, params ?? {});
});

// Clean shutdown
app.on("before-quit", () => {
  python.kill();
});
```

**Step 5: Verify it compiles**

```bash
cd /Users/ded/Projects/assist/manole/ui
npm run build
```

Expected: Builds without TypeScript errors.

**Step 6: Commit**

```bash
cd /Users/ded/Projects/assist/manole
git add ui/src/lib/protocol.ts ui/electron/python.ts ui/electron/preload.ts ui/electron/main.ts
git commit -m "feat: add NDJSON Python bridge and IPC protocol types"
```

---

### Task 6: Create usePython and useChat hooks

React hooks that manage communication with the Python backend and conversation state.

**Files:**
- Create: `ui/src/hooks/usePython.ts`
- Create: `ui/src/hooks/useChat.ts`
- Create: `ui/src/lib/types.ts` (renderer-side window.api type)

**Step 1: Create window API type declaration**

Create `ui/src/lib/types.ts`:

```typescript
import type { Response } from "./protocol";

export interface PythonAPI {
  send: (method: string, params?: Record<string, unknown>) => Promise<Response>;
  onMessage: (callback: (response: Response) => void) => () => void;
}

declare global {
  interface Window {
    api: PythonAPI;
  }
}
```

**Step 2: Create usePython hook**

Create `ui/src/hooks/usePython.ts`:

```typescript
import { useEffect, useCallback, useState, useRef } from "react";
import type { Response, StatusData } from "../lib/protocol";

export type MessageHandler = (response: Response) => void;

export function usePython() {
  const [backendState, setBackendState] = useState<string>("not_initialized");
  const handlersRef = useRef<Set<MessageHandler>>(new Set());

  useEffect(() => {
    const cleanup = window.api.onMessage((response: Response) => {
      // Handle status updates
      if (response.type === "status") {
        const data = response.data as unknown as StatusData;
        setBackendState(data.state);
      }
      // Forward to all registered handlers
      for (const handler of handlersRef.current) {
        handler(response);
      }
    });
    return cleanup;
  }, []);

  const send = useCallback(
    (method: string, params?: Record<string, unknown>) =>
      window.api.send(method, params),
    [],
  );

  const subscribe = useCallback((handler: MessageHandler) => {
    handlersRef.current.add(handler);
    return () => {
      handlersRef.current.delete(handler);
    };
  }, []);

  return { send, subscribe, backendState };
}
```

**Step 3: Create useChat hook**

Create `ui/src/hooks/useChat.ts`:

```typescript
import { useReducer, useCallback, useEffect } from "react";
import { usePython } from "./usePython";
import type { Response } from "../lib/protocol";

export interface AgentStep {
  step: number;
  tool: string;
  params: Record<string, unknown>;
}

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  text: string;
  isStreaming: boolean;
  agentSteps: AgentStep[];
}

interface ChatState {
  messages: ChatMessage[];
  isLoading: boolean;
  error: string | null;
}

type ChatAction =
  | { type: "user_message"; text: string }
  | { type: "stream_token"; text: string }
  | { type: "agent_step"; step: AgentStep }
  | { type: "response_complete"; text: string }
  | { type: "error"; message: string }
  | { type: "clear" };

function chatReducer(state: ChatState, action: ChatAction): ChatState {
  switch (action.type) {
    case "user_message": {
      const userMsg: ChatMessage = {
        id: `user-${Date.now()}`,
        role: "user",
        text: action.text,
        isStreaming: false,
        agentSteps: [],
      };
      const assistantMsg: ChatMessage = {
        id: `assistant-${Date.now()}`,
        role: "assistant",
        text: "",
        isStreaming: true,
        agentSteps: [],
      };
      return {
        ...state,
        messages: [...state.messages, userMsg, assistantMsg],
        isLoading: true,
        error: null,
      };
    }
    case "stream_token": {
      const messages = [...state.messages];
      const last = messages[messages.length - 1];
      if (last?.role === "assistant" && last.isStreaming) {
        messages[messages.length - 1] = {
          ...last,
          text: last.text + action.text,
        };
      }
      return { ...state, messages };
    }
    case "agent_step": {
      const messages = [...state.messages];
      const last = messages[messages.length - 1];
      if (last?.role === "assistant" && last.isStreaming) {
        messages[messages.length - 1] = {
          ...last,
          agentSteps: [...last.agentSteps, action.step],
        };
      }
      return { ...state, messages };
    }
    case "response_complete": {
      const messages = [...state.messages];
      const last = messages[messages.length - 1];
      if (last?.role === "assistant") {
        messages[messages.length - 1] = {
          ...last,
          text: action.text,
          isStreaming: false,
        };
      }
      return { ...state, messages, isLoading: false };
    }
    case "error":
      return { ...state, isLoading: false, error: action.message };
    case "clear":
      return { messages: [], isLoading: false, error: null };
    default:
      return state;
  }
}

export function useChat() {
  const { send, subscribe, backendState } = usePython();
  const [state, dispatch] = useReducer(chatReducer, {
    messages: [],
    isLoading: false,
    error: null,
  });

  useEffect(() => {
    return subscribe((response: Response) => {
      switch (response.type) {
        case "token":
          dispatch({ type: "stream_token", text: (response.data as { text: string }).text });
          break;
        case "agent_step":
          dispatch({
            type: "agent_step",
            step: response.data as unknown as AgentStep,
          });
          break;
        case "error":
          if (response.id !== null) {
            dispatch({ type: "error", message: (response.data as { message: string }).message });
          }
          break;
      }
    });
  }, [subscribe]);

  const sendMessage = useCallback(
    async (text: string) => {
      dispatch({ type: "user_message", text });
      const result = await send("query", { text });
      if (result.type === "result") {
        dispatch({
          type: "response_complete",
          text: (result.data as { text: string }).text,
        });
      } else if (result.type === "error") {
        dispatch({
          type: "error",
          message: (result.data as { message: string }).message,
        });
      }
    },
    [send],
  );

  const initBackend = useCallback(
    (dataDir: string, reuse?: string) =>
      send("init", { dataDir, ...(reuse ? { reuse } : {}) }),
    [send],
  );

  return {
    messages: state.messages,
    isLoading: state.isLoading,
    error: state.error,
    backendState,
    sendMessage,
    initBackend,
    clearChat: () => dispatch({ type: "clear" }),
  };
}
```

**Step 4: Verify it compiles**

```bash
cd /Users/ded/Projects/assist/manole/ui
npm run build
```

Expected: No TypeScript errors.

**Step 5: Commit**

```bash
cd /Users/ded/Projects/assist/manole
git add ui/src/hooks/ ui/src/lib/types.ts
git commit -m "feat: add usePython and useChat hooks for backend communication"
```

---

### Task 7: Build the chat UI components

The main chat interface using shadcn components. **Follow the visual design system in `docs/plans/2026-02-16-electron-ui-visual-design.md` exactly** — it specifies colors, fonts, animations, component structure, and Motion usage for each component.

**Files:**
- Create: `ui/src/components/ChatPanel.tsx`
- Create: `ui/src/components/MessageBubble.tsx`
- Create: `ui/src/components/AgentSteps.tsx`
- Create: `ui/src/components/StatusBar.tsx`
- Modify: `ui/src/App.tsx`

**Step 1: Install required shadcn components**

```bash
cd /Users/ded/Projects/assist/manole/ui
npx shadcn@latest add button input scroll-area collapsible badge
```

**IMPORTANT:** All component styling below is placeholder. The **actual** styles, colors, animations, fonts, and Motion usage MUST follow `docs/plans/2026-02-16-electron-ui-visual-design.md`. Key points:
- Use `font-display` (Cormorant Garamond) for headings
- Use `font-sans` (DM Sans) for body text/chat messages
- Use `font-mono` (IBM Plex Mono) for file paths, tool names, agent traces
- Use warm dark palette (`--bg-primary`, `--accent`, etc.)
- Use Motion (`motion/react`) for message entry, agent step stagger, welcome screen entrance
- Add film grain overlay, typing indicator dots, streaming cursor
- User messages: right-aligned, amber accent bg, rounded-br-md
- Assistant messages: left-aligned, bg-tertiary, rounded-bl-md
- Agent steps: vertical amber timeline with staggered slide-in animation

**Step 2: Create MessageBubble component**

Create `ui/src/components/MessageBubble.tsx`:

```tsx
import { AgentSteps } from "./AgentSteps";
import type { ChatMessage } from "../hooks/useChat";

interface Props {
  message: ChatMessage;
}

export function MessageBubble({ message }: Props) {
  const isUser = message.role === "user";

  return (
    <div className={`flex ${isUser ? "justify-end" : "justify-start"} mb-4`}>
      <div
        className={`max-w-[80%] rounded-2xl px-4 py-3 ${
          isUser
            ? "bg-primary text-primary-foreground"
            : "bg-muted text-foreground"
        }`}
      >
        <p className="whitespace-pre-wrap text-sm">{message.text}</p>
        {message.isStreaming && !message.text && (
          <span className="inline-block h-4 w-1 animate-pulse bg-current" />
        )}
        {message.agentSteps.length > 0 && (
          <AgentSteps steps={message.agentSteps} />
        )}
      </div>
    </div>
  );
}
```

**Step 3: Create AgentSteps component**

Create `ui/src/components/AgentSteps.tsx`:

```tsx
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import { Badge } from "@/components/ui/badge";
import type { AgentStep } from "../hooks/useChat";

interface Props {
  steps: AgentStep[];
}

export function AgentSteps({ steps }: Props) {
  return (
    <Collapsible className="mt-2">
      <CollapsibleTrigger className="flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground">
        <span>{steps.length} tool call{steps.length !== 1 ? "s" : ""}</span>
      </CollapsibleTrigger>
      <CollapsibleContent className="mt-1 space-y-1">
        {steps.map((step, i) => (
          <div key={i} className="flex items-center gap-2 text-xs text-muted-foreground">
            <Badge variant="outline" className="text-[10px] px-1 py-0">
              {step.tool}
            </Badge>
            <span className="truncate">
              {JSON.stringify(step.params)}
            </span>
          </div>
        ))}
      </CollapsibleContent>
    </Collapsible>
  );
}
```

**Step 4: Create ChatPanel component**

Create `ui/src/components/ChatPanel.tsx`:

```tsx
import { useState, useRef, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { ScrollArea } from "@/components/ui/scroll-area";
import { MessageBubble } from "./MessageBubble";
import type { ChatMessage } from "../hooks/useChat";

interface Props {
  messages: ChatMessage[];
  isLoading: boolean;
  onSend: (text: string) => void;
}

export function ChatPanel({ messages, isLoading, onSend }: Props) {
  const [input, setInput] = useState("");
  const scrollRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom on new messages/tokens
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    const text = input.trim();
    if (!text || isLoading) return;
    setInput("");
    onSend(text);
  };

  return (
    <div className="flex h-full flex-col">
      <ScrollArea ref={scrollRef} className="flex-1 p-4">
        {messages.length === 0 && (
          <div className="flex h-full items-center justify-center text-muted-foreground">
            <p>Ask anything about your files.</p>
          </div>
        )}
        {messages.map((msg) => (
          <MessageBubble key={msg.id} message={msg} />
        ))}
      </ScrollArea>

      <form onSubmit={handleSubmit} className="border-t p-4">
        <div className="flex gap-2">
          <Input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Ask a question..."
            disabled={isLoading}
            className="flex-1"
            autoFocus
          />
          <Button type="submit" disabled={isLoading || !input.trim()}>
            Send
          </Button>
        </div>
      </form>
    </div>
  );
}
```

**Step 5: Create StatusBar component**

Create `ui/src/components/StatusBar.tsx`:

```tsx
import { Badge } from "@/components/ui/badge";

interface Props {
  backendState: string;
  dataDir?: string;
}

export function StatusBar({ backendState, dataDir }: Props) {
  const stateColors: Record<string, string> = {
    ready: "bg-green-500",
    loading_model: "bg-yellow-500",
    indexing: "bg-yellow-500",
    not_initialized: "bg-gray-500",
  };

  const stateLabels: Record<string, string> = {
    ready: "Ready",
    loading_model: "Loading model...",
    indexing: "Indexing...",
    not_initialized: "Not initialized",
  };

  return (
    <div className="flex items-center gap-3 border-t px-4 py-2 text-xs text-muted-foreground">
      <div className="flex items-center gap-1.5">
        <span className={`h-2 w-2 rounded-full ${stateColors[backendState] || "bg-gray-500"}`} />
        <span>{stateLabels[backendState] || backendState}</span>
      </div>
      {dataDir && <span className="truncate">{dataDir}</span>}
    </div>
  );
}
```

**Step 6: Wire everything in App.tsx**

Replace `ui/src/App.tsx`:

```tsx
import { useEffect, useState } from "react";
import { ChatPanel } from "./components/ChatPanel";
import { StatusBar } from "./components/StatusBar";
import { useChat } from "./hooks/useChat";

export default function App() {
  const { messages, isLoading, error, backendState, sendMessage, initBackend } =
    useChat();
  const [dataDir, setDataDir] = useState<string | null>(null);

  // Auto-init with default directory on startup
  useEffect(() => {
    if (backendState === "not_initialized" && !dataDir) {
      // In production, show a directory picker. For now, init with test_data.
      const defaultDir = "./test_data";
      setDataDir(defaultDir);
      initBackend(defaultDir);
    }
  }, [backendState, dataDir, initBackend]);

  return (
    <div className="flex h-screen flex-col bg-background text-foreground">
      <header className="border-b px-4 py-3">
        <h1 className="text-lg font-semibold">NeuroFind</h1>
      </header>

      <main className="flex-1 overflow-hidden">
        <ChatPanel messages={messages} isLoading={isLoading} onSend={sendMessage} />
      </main>

      {error && (
        <div className="border-t bg-destructive/10 px-4 py-2 text-sm text-destructive">
          {error}
        </div>
      )}

      <StatusBar backendState={backendState} dataDir={dataDir ?? undefined} />
    </div>
  );
}
```

**Step 7: Verify it compiles**

```bash
cd /Users/ded/Projects/assist/manole/ui
npm run build
```

Expected: Compiles without errors.

**Step 8: Commit**

```bash
cd /Users/ded/Projects/assist/manole
git add ui/src/
git commit -m "feat: add chat UI components with streaming and agent step display"
```

---

### Task 8: End-to-end smoke test

Verify the full pipeline works: Electron spawns Python, sends init + query, tokens stream to the chat UI.

**Files:**
- No new files — this is a manual integration test

**Step 1: Start the Electron app in dev mode**

```bash
cd /Users/ded/Projects/assist/manole/ui
npm run dev
```

**Step 2: Verify startup**

Expected:
1. Electron window opens
2. StatusBar shows "Loading model..." then "Indexing..." then "Ready"
3. Chat panel shows "Ask anything about your files."

**Step 3: Send a test query**

Type "how many files do I have?" and press Enter.

Expected:
1. User message appears on the right
2. Assistant message appears on the left with streaming text
3. Agent steps (collapsible) show tool calls made
4. StatusBar stays green "Ready"

**Step 4: Test error case**

Close and reopen the app. If Python fails to spawn, verify:
1. Error message appears in the UI
2. No crash, no blank screen

**Step 5: Commit any fixes from smoke test**

```bash
git add -u
git commit -m "fix: address issues found in e2e smoke test"
```

---

### Task 9: Add FileBrowser component for directory selection

Allow users to pick a directory to index instead of hardcoding test_data.

**Files:**
- Create: `ui/src/components/FileBrowser.tsx`
- Modify: `ui/electron/main.ts` (add dialog IPC)
- Modify: `ui/electron/preload.ts` (expose dialog)
- Modify: `ui/src/App.tsx` (wire FileBrowser)
- Modify: `ui/src/lib/types.ts` (add dialog type)

**Step 1: Add dialog IPC to main.ts**

Add to `ui/electron/main.ts`:

```typescript
import { dialog } from "electron";

ipcMain.handle("dialog:openDirectory", async () => {
  const result = await dialog.showOpenDialog(mainWindow!, {
    properties: ["openDirectory"],
  });
  if (result.canceled) return null;
  return result.filePaths[0];
});
```

**Step 2: Expose in preload.ts**

Add to the `contextBridge.exposeInMainWorld` object:

```typescript
selectDirectory: () => ipcRenderer.invoke("dialog:openDirectory"),
```

**Step 3: Update types**

Add to `ui/src/lib/types.ts`:

```typescript
export interface PythonAPI {
  send: (method: string, params?: Record<string, unknown>) => Promise<Response>;
  onMessage: (callback: (response: Response) => void) => () => void;
  selectDirectory: () => Promise<string | null>;
}
```

**Step 4: Create FileBrowser component**

Create `ui/src/components/FileBrowser.tsx`:

```tsx
import { Button } from "@/components/ui/button";

interface Props {
  onSelect: (dir: string) => void;
  isLoading: boolean;
}

export function FileBrowser({ onSelect, isLoading }: Props) {
  const handleClick = async () => {
    const dir = await window.api.selectDirectory();
    if (dir) onSelect(dir);
  };

  return (
    <div className="flex h-full flex-col items-center justify-center gap-4 p-8">
      <h2 className="text-2xl font-semibold">Welcome to NeuroFind</h2>
      <p className="text-center text-muted-foreground">
        Select a folder to index and start asking questions about your files.
      </p>
      <Button onClick={handleClick} disabled={isLoading} size="lg">
        {isLoading ? "Loading..." : "Open Folder"}
      </Button>
    </div>
  );
}
```

**Step 5: Wire into App.tsx**

Update `App.tsx` to show `FileBrowser` when no directory is selected, and `ChatPanel` after init completes. Replace the auto-init `useEffect` with the directory picker flow:

```tsx
import { FileBrowser } from "./components/FileBrowser";

// In App component, replace the auto-init useEffect with:
const handleSelectDir = async (dir: string) => {
  setDataDir(dir);
  await initBackend(dir);
};

// In the render, replace <main>:
<main className="flex-1 overflow-hidden">
  {backendState === "ready" ? (
    <ChatPanel messages={messages} isLoading={isLoading} onSend={sendMessage} />
  ) : !dataDir ? (
    <FileBrowser onSelect={handleSelectDir} isLoading={backendState === "loading_model" || backendState === "indexing"} />
  ) : (
    <div className="flex h-full items-center justify-center text-muted-foreground">
      <p>{backendState === "loading_model" ? "Loading model..." : "Indexing files..."}</p>
    </div>
  )}
</main>
```

**Step 6: Verify it compiles and works**

```bash
cd /Users/ded/Projects/assist/manole/ui
npm run build && npm run dev
```

**Step 7: Commit**

```bash
cd /Users/ded/Projects/assist/manole
git add ui/
git commit -m "feat: add directory picker for indexing folders"
```

---

### Task 10: PyInstaller bundling

Freeze the Python backend into a single binary for distribution.

**Files:**
- Create: `server.spec` (PyInstaller spec)
- Modify: `ui/electron-builder.yml` (add extraResources)

**Step 1: Install PyInstaller**

```bash
cd /Users/ded/Projects/assist/manole
uv add --dev pyinstaller
```

**Step 2: Create PyInstaller spec**

Create `server.spec`:

```python
# -*- mode: python ; coding: utf-8 -*-
a = Analysis(
    ['server.py'],
    pathex=['.'],
    datas=[],
    hiddenimports=[
        'leann',
        'llama_cpp',
        'docling',
        'models',
        'agent',
        'searcher',
        'tools',
        'toolbox',
        'router',
        'rewriter',
        'parser',
        'file_reader',
        'chat',
    ],
    hookspath=[],
    runtime_hooks=[],
    excludes=['tkinter', 'matplotlib', 'IPython', 'notebook'],
)

pyz = PYZ(a.pure, a.zipped_data)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    name='manole-server',
    strip=False,
    upx=True,
    console=True,
)
```

**Step 3: Test the build**

```bash
cd /Users/ded/Projects/assist/manole
uv run pyinstaller server.spec --noconfirm
```

Expected: Creates `dist/manole-server` binary.

**Step 4: Test the binary**

```bash
echo '{"id": 0, "method": "ping"}' | ./dist/manole-server
```

Expected: JSON response with `state: "not_initialized"`.

**Step 5: Configure electron-builder extraResources**

Add to `ui/electron-builder.yml`:

```yaml
extraResources:
  - from: "../dist/manole-server"
    to: "manole-server"
  - from: "../models/"
    to: "models/"
```

**Step 6: Commit**

```bash
cd /Users/ded/Projects/assist/manole
git add server.spec ui/electron-builder.yml
git commit -m "feat: add PyInstaller spec and electron-builder resource config"
```

---

## Summary

| Task | Description | Modifies existing code? |
|------|-------------|------------------------|
| 1 | Streaming in ModelManager | Yes (backward-compatible addition) |
| 2 | server.py NDJSON adapter | No (new file) |
| 3 | Wire streaming through Agent | Yes (backward-compatible addition) |
| 4 | Scaffold electron-vite project | No (new directory) |
| 5 | Protocol types + Python bridge | No (new files) |
| 6 | usePython + useChat hooks | No (new files) |
| 7 | Chat UI components | No (new files) |
| 8 | End-to-end smoke test | Maybe (fixes) |
| 9 | FileBrowser + directory picker | No (new files) |
| 10 | PyInstaller bundling | No (new files) |

Tasks 1-3 are Python-side. Tasks 4-9 are Electron/React-side. Task 10 is packaging. They can be executed in order, with Tasks 4-6 potentially parallelized after Task 3 is done.
