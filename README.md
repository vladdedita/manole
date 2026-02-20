# Manole

> **Disclaimer:** This project was developed for learning purposes only, as part of an internal contest at [ASSIST Software](https://assist-software.net). It is not intended for production use.

Local AI file assistant. Index your documents, ask questions about them, explore file relationships on an interactive map. Runs entirely on your machine — no cloud, no API keys, no data leaving your disk.

## Quick Start

```bash
git clone https://github.com/vladdedita/manole.git
cd manole
./run.sh
```

The script detects your OS, installs prerequisites (Python 3.13, Node.js, uv) via your package manager, sets up the environment, and launches the app. Safe to run repeatedly — subsequent runs skip installs and go straight to launch.

On first launch, Manole downloads ~1.5 GB of language models. After that, it works fully offline.

## Features

- **Semantic search** — ask natural language questions about your documents. An agentic RAG pipeline retrieves relevant passages across PDFs, text files, images, and more.
- **Interactive file map** — explore file relationships as a graph. See which documents reference each other, zoom into clusters, and navigate visually.
- **Image captioning** — automatically describe images using a local vision model (Moondream2). Captions are indexed and searchable alongside text.
- **Multi-directory indexing** — open multiple folders and search across all of them at once.
- **Privacy-first** — everything runs on your machine. No cloud services, no API keys, no data exfiltration. Your files stay on your disk.
- **Offline after setup** — models download once on first launch, then the app is fully self-contained.

## Architecture

```
Electron (React + Tailwind)
  ↕ NDJSON over stdio
server.py (protocol adapter)
  ↕
agent.py → tools.py → toolbox.py
  ↕           ↕
models.py  searcher.py (LEANN vector search)
```

**Python backend** — agentic RAG pipeline with 7 tools (semantic search, grep, file metadata, directory tree, etc.), two GGUF models (text + vision) via llama-cpp-python, LEANN vector index for document retrieval.

**Electron UI** — React chat interface + file graph visualization. Communicates with the Python backend over newline-delimited JSON (NDJSON) via stdio. No HTTP server, no ports.

## Development Setup

### Prerequisites

- Python 3.13+
- [uv](https://docs.astral.sh/uv/) package manager
- Node.js 20+

### Python backend

```bash
uv sync

# Download models (first time only, ~1.5 GB)
# Models auto-download on first run, or place GGUF files manually in models/

# Run tests (290 tests)
uv run pytest tests/ -v

# Start CLI chat (no UI needed)
uv run python chat.py
```

### Electron UI

```bash
cd ui
npm install

# Development mode (starts Electron + Python backend with hot reload)
npm run dev

# Run UI tests
npx vitest run
```

### Quick start (recommended)

```bash
./run.sh
```

This handles everything: installs prerequisites, sets up the Python and Node environments, then launches Electron with the Python backend. Open a folder from the UI to start indexing.

## Running Tests

```bash
# Python tests (290 tests, ~9s)
uv run pytest tests/ -v

# UI tests (13 tests)
cd ui && npx vitest run

# Single file
uv run pytest tests/test_server.py -v
```

## Project Structure

```
manole/
  agent.py             # Agentic loop orchestrator
  models.py            # ModelManager — text + vision GGUF models
  searcher.py          # LEANN vector search wrapper
  tools.py             # ToolRegistry (7 tools)
  toolbox.py           # File system operations
  router.py            # Query intent routing
  rewriter.py          # Query rewriting
  parser.py            # Response parsing
  file_reader.py       # Document ingestion (docling)
  image_captioner.py   # Vision model image captioning
  graph.py             # File relationship graph builder
  server.py            # NDJSON stdio protocol adapter
  chat.py              # CLI chat interface
  models-manifest.json # Model metadata (filenames, repos, sizes)
  run.sh               # Setup-and-run script (installs deps, launches app)
  tests/               # Python tests
  ui/                  # Electron React app
    electron/          # Main process, Python bridge, setup manager
    src/               # React components, hooks, lib
    tests/             # UI tests (vitest)
  docs/                # Design documentation (see below)
```

## Documentation

The `docs/` directory contains the project's design history, organized by concern:

```
docs/
  plans/           # Chronological design docs (architecture, features)
  feature/         # Feature implementation tracking (roadmaps, execution logs)
  requirements/    # User stories and acceptance criteria
  ux/              # UX journey maps and user flow designs
  analysis/        # Technical investigations
  distill/         # Acceptance test scenarios
```

Key entry points:

- **How the LLM pipeline works** — [`docs/architecture.md`](docs/architecture.md) — agent loop, tools, search, models, image captioning
- **Lessons learned** — [`docs/lessons-learned.md`](docs/lessons-learned.md) — architecture decisions, technology pivots, and what we learned building with small local models
- **How the agent works** — `docs/plans/2026-02-15-agent-loop-design.md`
- **Electron UI design** — `docs/plans/2026-02-16-electron-ui-design.md`
- **Fast captioning investigation** — `docs/analysis/root-cause-analysis-slow-captioning.md`

## Kreuzberg Integration (experimental)

The [`feat/kreuzberg-integration`](https://github.com/vladdedita/manole/tree/feat/kreuzberg-integration) branch replaces the document extraction pipeline with [kreuzberg](https://github.com/kruzberg-org/kreuzberg), a Python-native extraction library. This brings several improvements over the default docling-based pipeline:

- **Incremental reindexing** — new or modified files are detected and indexed automatically without rebuilding the entire index. A file watcher monitors indexed directories in real time.
- **Faster extraction** — kreuzberg is lighter than docling and extracts text from PDFs, DOCX, PPTX, XLSX, HTML, EPUB, and more with lower overhead.
- **Manifest-based change detection** — a JSON manifest tracks file mtimes and chunk counts per indexed directory. On startup, only changed files are reprocessed (catch-up scan).
- **Live file watching** — uses [watchfiles](https://github.com/samuelcolvin/watchfiles) (Rust-based) to detect filesystem changes. Drop a new file into an indexed folder and it becomes searchable within seconds.

To try it:

```bash
git checkout feat/kreuzberg-integration
./run.sh
```

## NDJSON Protocol

The Electron app spawns `python server.py` and communicates via JSON lines:

```
→ {"id": 1, "method": "init", "params": {"dataDir": "/path/to/docs"}}
← {"id": null, "type": "status", "data": {"state": "loading_model"}}
← {"id": null, "type": "status", "data": {"state": "indexing"}}
← {"id": null, "type": "status", "data": {"state": "ready"}}
← {"id": 1, "type": "result", "data": {"status": "ready", "directoryId": "docs"}}
```

Methods: `ping`, `init`, `query`, `check_models`, `download_models`, `toggle_debug`, `shutdown`, `list_indexes`, `reindex`, `remove_directory`, `getFileGraph`.
