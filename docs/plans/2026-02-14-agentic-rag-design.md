# Agentic RAG Pipeline Design

**Date:** 2026-02-14
**Status:** Approved
**Model:** LiquidAI/LFM2.5-1.2B-Instruct (fixed)

## Problem

Basic RAG (retrieve + single LLM call) fails with the 1.2B model because:
- Noisy OCR context confuses the model
- 7 raw chunks in one prompt overwhelm the small context window
- The model either hallucates or refuses ("I cannot find this") despite relevant sources

## Solution: Sequential Agentic Pipeline

Replace the single `llm.ask(prompt)` call with a 6-stage pipeline where each stage does one focused job.

```
Query → Planner → Searcher → Map (per chunk) → Filter → Reduce → Self-Check → Answer
```

## Architecture

### Stage 1: Planner (1 LLM call)

Extracts search parameters from the user query.

**Input:** Raw user query
**Output:** JSON with keywords, file type filter, source hint
**Prompt:**
```
Analyze this user question and extract search parameters.
Question: {query}

Output a JSON object with:
- "keywords": list of 2-4 search terms
- "file_filter": file extension to filter by (e.g. "pdf", "txt") or null
- "source_hint": filename substring to filter by, or null

JSON:
```

### Stage 2: Searcher (0 LLM calls)

Calls `searcher.search()` with optional `metadata_filters` derived from the planner output.

- `source_hint: "Invoice"` → `{"source": {"contains": "Invoice"}}`
- `file_filter: "pdf"` → `{"source": {"contains": ".pdf"}}`
- Both null → plain vector search, no filters

### Stage 3: Map (N LLM calls, one per chunk)

For each retrieved chunk, asks the model a focused extraction question.

**Input:** One chunk + query
**Output:** JSON with relevance flag and extracted facts
**Prompt:**
```
Read this text and answer: does it contain information relevant to the question?

Question: {query}
Text: {chunk_text}

Output a JSON object with:
- "relevant": true or false
- "facts": list of specific facts found (dates, names, numbers, filenames). Empty list if not relevant.

JSON:
```

### Stage 4: Filter (0 LLM calls)

Python logic: drop chunks where `relevant == false`.

### Stage 5: Reduce (1 LLM call)

Synthesizes a final answer from extracted facts only (never sees raw chunks).

**Input:** Aggregated facts list + query
**Output:** Natural language answer
**Prompt:**
```
Here are facts extracted from the user's files:
{facts_list}

Question: {query}

Using ONLY these facts, write a concise answer. If the facts don't answer the question, say "No relevant information found."

Answer:
```

### Stage 6: Confidence Check (0 LLM calls)

Pure Python token overlap check — no LLM self-verification (unreliable with 1.2B models).
Computes the ratio of answer tokens that appear in the extracted facts. If overlap is below 0.2, the answer is flagged as low-confidence.

This avoids the problem of asking the same small model to judge its own output (it tends to confirm what it just said) and saves one inference call (~3-5s).

## Cost Per Query

| Stage | LLM Calls | Purpose |
|-------|-----------|---------|
| Planner | 1 | Extract keywords + filters |
| Searcher | 0 | LEANN vector search |
| Map | N (up to 5) | Extract facts per chunk |
| Filter | 0 | Drop irrelevant chunks |
| Reduce | 1 | Synthesize answer |
| Confidence | 0 | Python token overlap check |
| **Total** | **2 + N** | **Typically 7 calls** |

## JSON Parsing Strategy

The 1.2B model may produce malformed JSON. The parser:
1. Tries `json.loads()` first
2. Falls back to regex extraction (`"relevant"\s*:\s*(true|false)`)
3. If both fail, treats the chunk as relevant (safe default)

## Implementation

- **New class:** `AgenticRAG` in `chat.py` (~150 lines)
- **Integration:** `chat_loop` uses `AgenticRAG.ask()` instead of manual search+prompt
- **Debug mode:** Each stage prints intermediate results when `debug=True`
- **Fallback:** Pipeline degrades gracefully to basic RAG if any stage fails

## What Doesn't Change

- Indexing pipeline (`build_index`, `find_index_path`)
- Model choice (1.2B LFM)
- LEANN library (uses existing search + metadata_filters API)
- CLI interface (same args: directory, --reuse, --force)
