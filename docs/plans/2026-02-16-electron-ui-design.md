# Electron React UI Design

**Date:** 2026-02-16
**Status:** Approved
**Approach:** Python subprocess + NDJSON over stdio

## Context

Manole/NeuroFind has a working Python backend: Agent loop, Searcher, ToolRegistry, ModelManager, all running a local 1.2B GGUF model. The core is stable and must not be modified beyond backward-compatible additions. We need a polished desktop UI.

## Architecture

Electron spawns the Python backend as a child process. Communication happens over stdin/stdout using newline-delimited JSON (NDJSON). No HTTP server, no ports, no network.

```
┌─────────────────────────────────────────────────┐
│ Electron                                        │
│  ┌──────────┐    ┌───────────────────────────┐  │
│  │ main.ts  │    │ renderer (React + shadcn)  │  │
│  │          │    │                           │  │
│  │ python.ts│◄──►│ usePython hook            │  │
│  │ (spawn,  │IPC │ useChat hook              │  │
│  │  NDJSON) │    │ ChatPanel, FileBrowser... │  │
│  └────┬─────┘    └───────────────────────────┘  │
│       │ stdio                                   │
│  ┌────▼─────┐                                   │
│  │server.py │ ← thin adapter, no business logic │
│  │          │                                   │
│  │ Agent    │                                   │
│  │ Searcher │ ← untouched core                  │
│  │ Model    │                                   │
│  └──────────┘                                   │
└─────────────────────────────────────────────────┘
```

## IPC Protocol

NDJSON over stdio. One JSON object per line.

### Requests (Electron → Python)

```json
{"id": 1, "method": "init", "params": {"dataDir": "/path/to/docs"}}
{"id": 2, "method": "query", "params": {"text": "show my invoices"}}
{"id": 3, "method": "toggle_debug", "params": {}}
{"id": 4, "method": "index_dir", "params": {"dataDir": "/new/path"}}
{"id": 5, "method": "list_indexes", "params": {}}
{"id": 0, "method": "ping"}
{"id": 6, "method": "shutdown"}
```

### Responses (Python → Electron)

```json
{"id": 1, "type": "result", "data": {"status": "ready", "indexName": "docs"}}
{"id": 2, "type": "token", "data": {"text": "I found"}}
{"id": 2, "type": "token", "data": {"text": " 3 invoices"}}
{"id": 2, "type": "agent_step", "data": {"step": 1, "tool": "semantic_search", "params": {"query": "invoices"}}}
{"id": 2, "type": "result", "data": {"text": "I found 3 invoices..."}}
{"id": null, "type": "error", "data": {"message": "Index not found"}}
```

### Unsolicited status events

```json
{"type": "status", "data": {"state": "loading_model"}}
{"type": "status", "data": {"state": "ready"}}
{"type": "progress", "data": {"stage": "indexing", "percent": 45}}
```

## Python Side: server.py

Thin adapter at project root. Reads stdin line-by-line, dispatches to existing modules, writes to stdout.

### Methods

| Method | What it does | Touches |
|--------|-------------|---------|
| `init` | Build/reuse index, load model, wire Agent | chat.py functions, ModelManager, Searcher, Agent |
| `query` | Run agent loop, stream tokens + steps | Agent.run() with streaming callback |
| `index_dir` | Index a new directory, re-wire searcher | build_index(), re-init searcher |
| `toggle_debug` | Flip debug flags | Agent/Searcher/Rewriter debug attrs |
| `list_indexes` | Return available indexes | Scan .leann/indexes/ |
| `ping` | Health check | Returns state + uptime |
| `shutdown` | Clean exit | sys.exit(0) |

### Streaming modification

The only change to existing code: add an optional `on_token` callback to `ModelManager.generate()`. The existing signature stays the same. Streaming is opt-in via a `stream=True` parameter that yields tokens and calls the callback.

### Health check flow

1. Python process starts, sends `{"type": "status", "data": {"state": "loading_model"}}` on stdout
2. Model loads, sends `{"type": "status", "data": {"state": "ready"}}`
3. Electron shows loading screen until `ready` arrives
4. Timeout: 120s, then show error

## Electron Side

### Stack

- **electron-vite** — Electron + Vite integration (main/preload/renderer)
- **React** — UI framework
- **shadcn/ui** — component library (Tailwind-based)

### File structure

```
ui/
├── package.json
├── electron.vite.config.ts
├── electron/
│   ├── main.ts            ← app lifecycle, spawns Python
│   ├── preload.ts         ← exposes IPC to renderer
│   └── python.ts          ← child process wrapper (spawn, NDJSON read/write, restart)
├── src/
│   ├── App.tsx
│   ├── components/
│   │   ├── ChatPanel.tsx      ← message list, streaming text, input bar
│   │   ├── MessageBubble.tsx  ← single message with token streaming
│   │   ├── AgentSteps.tsx     ← collapsible tool call trace
│   │   ├── FileBrowser.tsx    ← secondary panel, directory picker
│   │   └── StatusBar.tsx      ← model status, indexing progress
│   ├── hooks/
│   │   ├── usePython.ts       ← send requests, subscribe to responses
│   │   └── useChat.ts         ← conversation state, streaming accumulator
│   └── lib/
│       └── protocol.ts        ← TypeScript types for NDJSON messages
└── electron-builder.yml
```

### UI model

Chat-first interface. Primary panel is the chat window with streaming responses. Secondary panel is a file browser for selecting directories to index. StatusBar shows model state, indexing progress, debug toggle.

## Offline-First Design

After installation, the app never requires internet.

**Fully offline:**
- Model inference (llama-cpp-python, local GGUF)
- Document indexing (LEANN, local)
- Vector search (LEANN, local)
- File reading (Docling, local)
- All tool execution (filesystem operations)

No network calls anywhere in the pipeline. No telemetry, no update checks.

**Model location:** `~/.manole/models/` or app Resources. Path is configurable.

## Bundling & Distribution

### Python backend
- PyInstaller freezes `server.py` + all deps into a single binary (`manole-server`)
- GGUF model ships alongside as an asset (not inside the binary)

### Electron app
- electron-builder packages renderer + main process
- `extraResources` includes PyInstaller binary + GGUF model
- Produces `.dmg` (macOS), `.exe` installer (Windows)

### Packaged layout
```
NeuroFind.app/Contents/Resources/
├── manole-server          ← PyInstaller binary
├── models/
│   └── LFM2.5-1.2B-Instruct-Q4_0.gguf
└── app.asar
```

### Path resolution
`app.isPackaged` → production (Resources path) vs dev (spawn `python server.py` directly).

### Expected size
~1-1.5GB (model ~700MB, Python runtime + deps ~300MB, Electron ~150MB).

## Error Handling

| Scenario | Behavior |
|----------|----------|
| Python crashes | Detect child exit, show error + Restart button, preserve chat history |
| Python hangs (60s timeout) | Offer kill + restart |
| Spawn fails | Error screen with instructions |
| Directory empty/missing | Error in UI, pick another |
| Large directory indexing | Progress events → progress bar |
| GGUF missing | Setup screen instead of chat |
| Model load fails | Error with path shown |
| Out of memory | Catch from llama-cpp, report to UI |
| Query while streaming | Disable input until done |

## Testing

**Python:** Unit tests for `server.py` protocol — feed NDJSON, assert responses. Mock Model/Agent. Existing test suite untouched.

**Electron main:** Test `python.ts` with a mock Python script that echoes NDJSON.

**React:** Vitest for hooks (`useChat`, `usePython`). React Testing Library for components. Explicit streaming accumulation tests.

**Integration:** One e2e test — spawn real Python, send init + query, assert tokens stream and result terminates.

**Manual only:** PyInstaller bundling, electron-builder packaging.
