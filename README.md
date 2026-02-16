# NeuroFind

Local AI file assistant. Indexes your documents, answers questions about them. Runs entirely on your machine — no cloud, no API keys, no data leaving your disk.

## Architecture

```
ui/ (Electron + React)
  ↕ NDJSON over stdio
server.py (protocol adapter)
  ↕
agent.py → tools.py → toolbox.py
  ↕           ↕
models.py  searcher.py (LEANN vectors)
```

**Python backend** — Agent loop orchestrator with 7 tools (semantic search, grep, file metadata, directory tree, etc.), ModelManager wrapping llama-cpp-python with a local GGUF model, LEANN vector search for document retrieval.

**Electron UI** — React chat interface communicating with the Python backend over newline-delimited JSON (NDJSON) via stdio. No HTTP server, no ports.

## Prerequisites

- Python 3.13+
- [uv](https://docs.astral.sh/uv/) package manager
- Node.js 20+
- A GGUF model file (place in `models/`)

## Setup

### Python backend

```bash
# Install dependencies
uv sync

# Run tests
uv run pytest tests/ -v

# Start the CLI chat (no UI)
uv run python chat.py
```

### Electron UI

```bash
cd ui

# Install dependencies
npm install

# Development mode (hot reload)
npm run dev

# Production build
npm run build
```

The dev server starts Electron with the Python backend as a child process. The UI communicates with it over stdin/stdout using NDJSON.

## NDJSON Protocol

The Electron app spawns `python server.py` and communicates via JSON lines:

**Request:** `{"id": 1, "method": "init", "params": {"dataDir": "/path/to/docs"}}`

**Response:** `{"id": 1, "type": "result", "data": {"state": "ready"}}`

**Streaming tokens:** `{"id": 2, "type": "token", "data": {"text": "The"}}`

Available methods: `ping`, `init`, `query`, `toggle_debug`, `shutdown`, `list_indexes`.

## Building for Distribution

### 1. Freeze the Python backend

```bash
uv run pyinstaller server.spec --noconfirm
```

This creates `dist/manole-server`, a standalone binary with all Python dependencies bundled.

### 2. Test the frozen binary

```bash
echo '{"id": 0, "method": "ping"}' | ./dist/manole-server
# Expected: {"id": 0, "type": "result", "data": {"state": "not_initialized", "uptime": 0.0}}
```

### 3. Package the Electron app

```bash
cd ui
npx electron-builder --config electron-builder.yml
```

This bundles the frozen Python binary, model files, and the Electron app into a platform-specific installer (`.dmg` on macOS, `.exe` on Windows, `.AppImage` on Linux).

The `electron-builder.yml` config pulls `dist/manole-server` and `models/` into the app bundle as extra resources.

## Project Structure

```
manole/
├── agent.py          # Agent loop orchestrator
├── models.py         # ModelManager (llama-cpp-python)
├── searcher.py       # LEANN vector search
├── tools.py          # ToolRegistry (7 tools)
├── toolbox.py        # File system operations
├── router.py         # Query routing
├── rewriter.py       # Query rewriting
├── parser.py         # Response parsing
├── file_reader.py    # Document ingestion (docling)
├── server.py         # NDJSON stdio adapter
├── chat.py           # CLI chat interface
├── server.spec       # PyInstaller build spec
├── tests/            # Python tests (130 tests)
├── ui/               # Electron React app
│   ├── electron/     # Main + preload processes
│   ├── src/          # React app (components, hooks, lib)
│   └── electron-builder.yml
└── docs/plans/       # Design docs
```

## Running Tests

```bash
# All Python tests
uv run pytest tests/ -v

# Specific test file
uv run pytest tests/test_server.py -v

# Build the UI (type checking)
cd ui && npm run build
```
