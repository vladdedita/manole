# How Manole Works

This document explains how the LLM interaction, tools, and search pipeline work inside Manole.

## Overview

Manole is an agentic RAG (Retrieval-Augmented Generation) system that runs entirely locally. A small language model (1.2B parameters) reasons about user questions, calls tools to gather information from indexed files, and synthesizes answers. A separate vision model captions images so they become searchable alongside text.

```
User Query
    |
[Query Rewriter] -- resolves pronouns, expands synonyms, classifies intent
    |
[Agent Loop] (up to 5 steps)
    |-- Model generates a tool call
    |-- Tool executes (search, file ops, etc.)
    |-- Result appended to conversation
    |-- Repeat or answer
    |
Final Answer + Source Files
```

## Models

Two GGUF models run via llama-cpp-python:

| Model | Size | Purpose |
|-------|------|---------|
| LFM2.5-1.2B-Instruct | ~660 MB | Text generation, tool calling, fact extraction, query rewriting |
| Moondream2 | ~2.8 GB + 900 MB projector | Image captioning |

Both run on CPU. Inference uses low temperature (0.1) and tight sampling (top_p=0.1, top_k=50) for deterministic, factual output. A threading lock serializes all model calls since llama-cpp-python is not thread-safe.

The text model handles multiple roles via different system prompts:
- **Agent**: Decides which tool to call and generates final answers
- **Fact extractor**: Validates search results and extracts relevant facts
- **Query rewriter**: Resolves coreference and expands search terms

## Agent Loop (agent.py)

The agent runs a multi-step reasoning loop (max 5 steps):

1. **Query rewriting** (optional) — The rewriter resolves pronouns using the last 4 conversation turns, expands the query with synonyms, and classifies intent (factual, count, list, compare, summarize, metadata).

2. **Tool call generation** — The model sees all 8 tool schemas in its system prompt and generates a tool call. The parser handles 4 output formats (the small model isn't always consistent):
   - LFM2.5 native: `<|tool_call_start|>[fn(args)]<|tool_call_end|>`
   - JSON: `{"name": "fn", "params": {...}}`
   - Bracket: `[tool_name(params)]`
   - Bare function call: `tool_name(params)`

3. **Fallback routing** — If the model doesn't produce a tool call on step 0, a keyword-based router (`router.py`) detects intent and picks a tool deterministically.

4. **Tool execution** — The tool runs and its result is appended to the conversation as a tool response message.

5. **Followup check** — After step 1+, the agent checks whether important keywords from the query were covered by tool results. If not, it calls `semantic_search` or `grep_files` for the missing terms.

6. **Answer** — When the model calls the `respond` tool (or runs out of steps), it generates a final answer grounded in the collected tool results.

## Tools (tools.py, toolbox.py)

Eight tools are available to the agent:

| Tool | What it does |
|------|-------------|
| `semantic_search(query, top_k)` | Vector search over indexed content, then LLM-validated fact extraction |
| `count_files(extension, time_filter)` | Count files, optionally filtered by extension or recency |
| `list_files(extension, limit, sort_by)` | List files sorted by date, size, or name |
| `grep_files(pattern)` | Find files by name substring |
| `file_metadata(name_hint)` | Get size, creation date, modification date for a file |
| `directory_tree(max_depth)` | Show folder structure as a tree |
| `folder_stats(sort_by, limit)` | Aggregate folder sizes and file counts |
| `disk_usage()` | Total disk usage breakdown by file type |

All filesystem tools operate on the indexed data directory only. They use pure Python (os/pathlib), no shell commands.

## Semantic Search (searcher.py)

The search pipeline is a map-filter architecture:

```
Query
  |
[Vector Search] -- LEANN index with contriever embeddings
  |
[Score Filter] -- drop chunks below 85% of top score
  |
[Map Phase] -- for each chunk, ask LLM: "extract relevant facts"
  |
[Filter] -- keep only chunks where LLM found relevant facts
  |
Facts + Source Files
```

1. **Vector search** queries the LEANN index (facebook/contriever embeddings) for the top-k most similar chunks.

2. **Score filtering** drops chunks whose similarity score is below 85% of the top result, removing noise.

3. **Fact extraction** (map phase) sends each surviving chunk to the text model with the user's question. The model extracts a JSON list of relevant facts or returns an empty list if the chunk isn't relevant.

4. **Filename fallback** — If vector search finds nothing, the searcher greps filenames for query keywords and reads matching files directly.

This two-stage approach compensates for the small embedding model's imprecision: vector search retrieves broadly, then the LLM validates relevance.

## Query Rewriting (rewriter.py)

Before the agent loop starts, the rewriter processes the query:

- **Coreference resolution** — "What about that invoice?" becomes "What is the total on invoice_2024_03.pdf?" using conversation context.
- **Term expansion** — "PDF files" becomes "PDF files documents" to improve recall.
- **Intent classification** — Categorizes as factual, count, list, compare, summarize, or metadata. This helps the router pick the right tool if the model doesn't.

## Intent Routing (router.py)

A deterministic fallback when the model doesn't produce a tool call. Uses keyword patterns:

- "how much space" / "disk usage" → `disk_usage()`
- "biggest files" / "largest" → `list_files(sort_by="size")`
- "folder structure" / "tree" → `directory_tree()`
- "how many .pdf files" → `count_files(extension=".pdf")`
- Everything else → `semantic_search(query)`

## Image Captioning (image_captioner.py)

When a directory is indexed, the captioner:

1. Scans for images (JPG, PNG, GIF, WebP, HEIC, BMP, TIFF)
2. Checks the caption cache — already-captioned images are skipped
3. For each uncached image:
   - Resizes to 768x768 max, converts to JPEG
   - Sends to Moondream2 vision model for captioning
   - Caches the caption to disk
4. Injects new captions into the LEANN index as searchable text passages

This means "find photos of the beach" works — the search matches against caption text like "Photo description: A sandy beach with turquoise water..."

Image loading is pipelined: the next image loads in a background thread while the current one is being captioned.

## File Graph (graph.py)

The interactive file map computes three types of relationships:

- **Similarity edges** — Cosine similarity between file embeddings (averaged from passage embeddings). Threshold: 0.6.
- **Reference edges** — Files that share entities: email addresses, tax IDs, IBANs, company names, or mention each other's filenames. Each entity type has a weight (email: 1.0, company: 0.7, tax ID: 0.9, IBAN: 0.8).
- **Structure edges** — Directory hierarchy (files connected to parent folders).

## NDJSON Protocol (server.py)

The Electron app communicates with the Python backend over stdio using newline-delimited JSON:

```
→ {"id": 1, "method": "query", "params": {"text": "what invoices do I have?"}}
← {"id": null, "type": "agent_step", "data": {"tool": "semantic_search", "input": "invoices"}}
← {"id": null, "type": "token", "data": {"text": "I found"}}
← {"id": null, "type": "token", "data": {"text": " 3 invoices..."}}
← {"id": 1, "type": "result", "data": {"answer": "...", "sources": [...]}}
```

Messages with `id: null` are streamed events (status updates, tokens, progress). Messages with a numeric `id` are request/response pairs. The server redirects stdout to stderr so that C library output (llama.cpp, ggml) doesn't corrupt the protocol stream.
