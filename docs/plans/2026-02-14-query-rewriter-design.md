# Query Rewriter Stage Design

## Problem

The current pipeline passes raw user queries directly to the planner and searcher. This causes:

1. **Poor retrieval** — "transformer" matches solar inverters, "macbook" doesn't expand to related terms
2. **Broken coreferences** — "aren't there more?" loses context without conversation history resolution
3. **No intent awareness** — "how many invoices?" and "what do the invoices say?" get the same treatment

## Solution

Add a Query Rewriter as the first stage of the pipeline, before the Planner. Uses LFM2.5-1.2B-Instruct to:

1. **Resolve coreferences** from conversation history
2. **Expand terms** for better embedding match
3. **Classify intent** (factual, count, list, compare, summarize)
4. **Generate an optimized search query** for vector retrieval

## Pipeline Flow

```
Before: Query → Plan → Search → Map → Filter → Reduce → Confidence
After:  Query → Rewrite → Plan → Search → Map → Filter → Reduce → Confidence
```

## Rewriter Output

```json
{
  "intent": "factual",
  "search_query": "engineering department budget allocation breakdown Q1 2026",
  "resolved_query": "What is the budget of the engineering department?"
}
```

### Fields

- `intent` — one of: `factual`, `count`, `list`, `compare`, `summarize`
- `search_query` — embedding-optimized query string for Contriever vector search. Expanded with synonyms and related terms. This replaces `keywords.join(" ")` as the search input.
- `resolved_query` — the user's question with coreferences resolved. This is what the planner and reducer see.

## How the Pipeline Uses It

- **Searcher** uses `search_query` instead of joining planner keywords
- **Planner** receives `resolved_query` instead of raw query — coreferences already resolved
- **Reducer** can use `intent` to format answers (bullets for `list`, number for `count`)
- **Fallback** — if rewriter LLM call fails or returns invalid JSON, use raw query (same as today)

## System Prompt

The rewriter system prompt instructs the model to:
- Read conversation history and resolve pronouns/references
- Expand abbreviations and add synonyms relevant to file/document search
- Classify the user's intent
- Generate a search-optimized query that would match document content in a vector database

The rewriter does NOT handle:
- Tool routing (semantic_search vs filesystem) — that's the planner's job
- Metadata filters (file type, source hint) — that's the planner's job
- Fact extraction — that's the mapper's job

## Examples

| Raw Query | History | Resolved Query | Search Query | Intent |
|-----------|---------|---------------|--------------|--------|
| "any invoices?" | (none) | "Are there any invoices?" | "invoice receipt payment billing" | list |
| "aren't there more?" | "Found 2 invoices" | "Are there more invoices besides the 2 found?" | "invoice receipt payment billing document" | count |
| "any macbook?" | (none) | "Are there any MacBook references?" | "MacBook Apple laptop computer hardware purchase" | list |
| "what is the budget?" | (none) | "What is the budget?" | "budget allocation spending financial plan quarterly" | factual |
| "how many PDFs?" | (none) | "How many PDF files are there?" | "PDF files documents" | count |

## Latency

Adds one LFM2.5-1.2B call (~3 seconds). Total pipeline: ~3s rewrite + ~1s plan + ~5s search + ~15s map + ~3s reduce = ~27s.

## Module

New file: `rewriter.py` with:
- `REWRITER_SYSTEM` — system prompt
- `QueryRewriter` class with `rewrite(query, context) -> dict` method
- Fallback to `{"intent": "factual", "search_query": query, "resolved_query": query}` on failure
