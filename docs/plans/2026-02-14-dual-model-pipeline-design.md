# Dual-Model Agentic RAG Pipeline Design

**Date:** 2026-02-14
**Status:** Approved

## Summary

Replace the single-model agentic RAG pipeline (LFM2.5-1.2B-Instruct doing everything) with a dual-model architecture using two purpose-built Liquid AI Nano models:

| Model | RAM | Role |
|-------|-----|------|
| LFM2-350M-Extract | ~200MB | Planner — structured JSON extraction from queries |
| LFM2-1.2B-RAG | ~2GB | Map + Reduce — grounded fact extraction and synthesis |

Both models loaded concurrently via llama-cpp-python (~2.2GB total). Leann remains for vector search and indexing.

## Architecture

```
User Query
    │
    ▼
┌─────────────────────────────────┐
│  STAGE 1: PLANNER               │  LFM2-350M-Extract
│  → keywords, filters, tool,     │  (~0.5-1s)
│    time_filter, tool_actions     │
└─────────┬───────────────────────┘
          │
          ├── tool="filesystem" ──► ToolBox → format → Response
          │
          ├── tool="hybrid" ──► ToolBox (get file paths)
          │                         │
          │                         ▼
          │                    Searcher (filtered by paths)
          │                         │
          ├── tool="semantic_search" ──► Searcher (standard)
          │                                  │
          └──────────────────────────────────┘
                                             │
                                             ▼
                                   ┌─────────────────────┐
                                   │  STAGE 3: MAP        │  LFM2-1.2B-RAG
                                   │  Per-chunk facts     │  (~1.5s × N)
                                   └─────────┬───────────┘
                                             │
                                             ▼
                                   ┌─────────────────────┐
                                   │  STAGE 4: FILTER     │  0 LLM calls
                                   │  Drop irrelevant     │
                                   └─────────┬───────────┘
                                             │
                                             ▼
                                   ┌─────────────────────┐
                                   │  STAGE 5: REDUCE     │  LFM2-1.2B-RAG
                                   │  Synthesize answer   │  (~1.5s)
                                   └─────────┬───────────┘
                                             │
                                             ▼
                                   Confidence Check (Python)
                                             │
                                             ▼
                                        Final Answer
```

## Planner Output Schema

```json
{
  "keywords": ["invoice", "Anthropic"],
  "file_filter": "pdf",
  "source_hint": "invoice",
  "tool": "semantic_search" | "filesystem" | "hybrid",
  "time_filter": "today" | "this_week" | "this_month" | null,
  "tool_actions": ["list_recent", "count", "metadata", "tree", "grep"]
}
```

Routing:
- `"filesystem"` → ToolBox executes tool_actions → format → return
- `"semantic_search"` → standard search → map → filter → reduce
- `"hybrid"` → ToolBox narrows file scope → search within those → map → filter → reduce

## Smart ToolBox

LLM-routed (via planner's tool_actions) with rich filesystem operations:

| Tool | Description |
|------|-------------|
| `count` | Count files matching filters (extension, time) |
| `list_recent` | List files by modification time |
| `metadata` | File size, created date, type info |
| `tree` | Directory structure with depth control |
| `grep` | Search file names by pattern |

Time filter support: `today` (24h), `this_week` (7d), `this_month` (30d) — pre-filters all tools.

For hybrid queries, ToolBox returns file paths instead of formatted text, used as metadata filters for search.

Operates on the source data directory (not the index).

## File Structure

```
manole/
├── main.py              # CLI entry point (unchanged)
├── chat.py              # chat_loop + CLI — thin shell, delegates to pipeline
├── models.py            # ModelManager — loads both GGUF models
├── planner.py           # Stage 1: query → structured JSON plan
├── searcher.py          # Stage 2: plan → chunks (leann + metadata filters)
├── mapper.py            # Stage 3: chunks → per-chunk facts (1.2B-RAG)
├── reducer.py           # Stage 5: facts → answer (1.2B-RAG) + confidence check
├── toolbox.py           # Smart filesystem tools (pure Python)
├── parser.py            # JSON parsing with regex fallback
├── pipeline.py          # AgenticRAG orchestrator
├── tests/
│   ├── test_parser.py
│   ├── test_planner.py
│   ├── test_toolbox.py
│   ├── test_mapper.py
│   ├── test_reducer.py
│   └── test_pipeline.py
```

## Pipeline Orchestrator (pipeline.py)

```python
class AgenticRAG:
    def __init__(self, index_path: str, data_dir: str, debug: bool = False):
        self.models = ModelManager()
        self.planner = Planner(self.models)
        self.searcher = Searcher(index_path)
        self.mapper = Mapper(self.models)
        self.reducer = Reducer(self.models)
        self.toolbox = ToolBox(data_dir)
        self.debug = debug

    def ask(self, query: str) -> str:
        plan = self.planner.plan(query)

        if plan["tool"] == "filesystem":
            result = self.toolbox.execute(plan)
            return self.reducer.format_filesystem_answer(query, result)

        if plan["tool"] == "hybrid":
            file_paths = self.toolbox.get_matching_files(plan)
            chunks = self.searcher.search(plan, file_filter_paths=file_paths)
        else:
            chunks = self.searcher.search(plan)

        if not chunks:
            chunks = self.searcher.search_unfiltered(plan)

        if not chunks:
            return "No relevant information found in your files."

        mapped = self.mapper.extract_facts(query, chunks)
        relevant = [m for m in mapped if m["relevant"]]

        if not relevant:
            return "No relevant information found in your files."

        answer = self.reducer.synthesize(query, relevant)
        return self.reducer.confidence_check(answer, relevant)
```

## Model Loading (models.py)

```python
class ModelManager:
    def __init__(self):
        self.planner_model = None   # LFM2-350M-Extract
        self.rag_model = None       # LFM2-1.2B-RAG

    def load(self):
        self.planner_model = Llama(
            model_path="models/LFM2-350M-Extract-Q4_0.gguf",
            n_ctx=2048,
            n_threads=4,
            verbose=False
        )
        self.rag_model = Llama(
            model_path="models/LFM2-1.2B-RAG-Q4_0.gguf",
            n_ctx=4096,
            n_threads=4,
            verbose=False
        )

    def plan(self, prompt: str) -> str:
        return self.planner_model(prompt, max_tokens=256)

    def extract(self, prompt: str) -> str:
        return self.rag_model(prompt, max_tokens=512)

    def synthesize(self, prompt: str) -> str:
        return self.rag_model(prompt, max_tokens=1024)
```

## Cost Per Query

| Stage | Model | LLM Calls | Est. Latency (CPU) |
|-------|-------|-----------|-------------------|
| Planner | 350M-Extract | 1 | ~0.5-1s |
| Searcher | — | 0 | ~0.1s |
| Map | 1.2B-RAG | N (max 5) | ~1.5s × N |
| Filter | — | 0 | ~0ms |
| Reduce | 1.2B-RAG | 1 | ~1.5s |
| **Semantic total** | | **2 + N** | **~9-10s** |
| **Filesystem total** | | **1** | **~0.5-1s** |
| **Hybrid total** | | **2 + N** | **~9-11s** |

## Dependencies

Add to pyproject.toml:
- `llama-cpp-python` — for GGUF model loading

## Key Decisions

1. **Approach A selected** — single Planner call with rich JSON schema (vs two-step or Python pre-router)
2. **Leann for search only** — llama-cpp-python for both LLM models
3. **Hybrid routing** — ToolBox can narrow scope, then feed into semantic search
4. **No LLM self-check** — Python token-overlap confidence scoring only
5. **Full modular split** — 8+ source files, 6 test files
