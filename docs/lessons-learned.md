# Manole: Lessons Learned & Architecture Decision Record

**Project:** Manole (formerly NeuroFind)
**Period:** February 2026
**Purpose:** Offline AI-powered file system search and RAG application

This document captures the significant decisions, failures, pivots, and lessons learned during the development of Manole.

---

## Table of Contents

1. [Vision Model: LFM2.5-VL → Moondream2](#1-vision-model-lfm25-vl--moondream2)
2. [RAG Strategy: Naive RAG → 6-Stage Agentic Pipeline](#2-rag-strategy-naive-rag--6-stage-agentic-pipeline)
3. [Tool Calling: Small Model Challenges](#3-tool-calling-small-model-challenges)
4. [LEANN Vector Search: Recompute Mode Trap](#4-leann-vector-search-recompute-mode-trap)
5. [Captioning Performance: f16 → Q4 + Pipelining](#5-captioning-performance-f16--q4--pipelining)
6. [Prompt Engineering for 1.2B Models](#6-prompt-engineering-for-12b-models)
7. [HNSW Noise: The get_vector_zmq Mystery](#7-hnsw-noise-the-get_vector_zmq-mystery)
8. [Summary of Key Principles](#8-summary-of-key-principles)

---

## 1. Vision Model: LFM2.5-VL → Moondream2

### The Failure (Feb 17)

LFM2.5-VL-1.6B vision model captioning failed on **every single image** with:

```
clip_init: failed to load model: load_hparams: unknown projector type: lfm2
```

**100% failure rate.** The entire vision pipeline was dead.

### Root Cause Analysis (5 Whys)

1. The GGUF mmproj file loaded (441 tensors) but was rejected at projector type validation
2. `llama-cpp-python`'s CLIP loader doesn't recognize `lfm2` as a valid projector type
3. The Python package bundles a llama.cpp snapshot that predates LFM2-VL support
4. The code used `Llava15ChatHandler` — designed for LLaVA architecture — to load an LFM2.5-VL mmproj with a fundamentally different projector
5. **Root cause: LFM2.5-VL was selected without verifying that `llama-cpp-python` supports its projector architecture.** The open GitHub issue [llama-cpp-python#2105](https://github.com/abetlen/llama-cpp-python/issues/2105) confirmed: no official LFM2-VL handler exists in any released version.

### Contributing Factor

The vision model integration was developed against **mocked tests** that never exercised the actual CLIP loader. No integration test verified that the real mmproj loads with the real handler.

### The Fix: Switch to Moondream2

| Model | Handler | Size | Status |
|-------|---------|------|--------|
| ~~LFM2.5-VL-1.6B~~ | ~~Llava15ChatHandler~~ | ~~664 MB + 814 MB mmproj~~ | ~~Broken — no handler exists~~ |
| **Moondream2** | **MoondreamChatHandler** | **2.8 GB + 900 MB projector** | **Working — official handler** |

Moondream2 was selected because:
- Dedicated `MoondreamChatHandler` exists in released llama-cpp-python
- Purpose-built for image captioning (the exact use case)
- Official GGUF available from `ggml-org/moondream2-20250414-GGUF`
- Small enough for local/offline use

### Lesson

> **Never assume library support based on model family branding.** LFM2.5 text worked perfectly, so LFM2.5-VL seemed like a safe choice. But the Python bindings lagged upstream llama.cpp by months. Always verify the specific handler/loader exists in the specific library version you're using.

> **Mocked tests don't catch integration failures.** The test suite had 100% coverage of the captioning API — all green — while the actual model couldn't load. At least one smoke test must exercise the real model-to-handler path.

---

## 2. RAG Strategy: Naive RAG → 6-Stage Agentic Pipeline

### The Problem (Feb 14)

The initial RAG approach (vector search → stuff chunks into context → ask LLM) failed with the 1.2B parameter model. The model couldn't extract information from retrieved context when given multiple chunks at once.

### The Solution: Decompose into Atomic Operations

A 6-stage pipeline was designed where each stage asks the LLM to do exactly one thing:

```
1. PLANNER    — extract search keywords + metadata filters from query (JSON)
2. SEARCHER   — LEANN vector search with filters → SearchResult chunks
3. MAP        — for EACH chunk individually: "is this relevant? extract facts" (JSON)
4. FILTER     — drop irrelevant chunks (Python logic, no LLM)
5. REDUCE     — synthesize answer from extracted facts only (not raw chunks)
6. SELF-CHECK — validate answer against one source chunk, append correction if mismatch
```

### Key Design Principle

> **The 1.2B model never sees more than one chunk at a time during the MAP stage.**

By processing chunks individually and extracting structured facts, the model's limited context window and comprehension are respected. The REDUCE stage receives clean, pre-validated facts — not noisy raw text.

### Lesson

> **Small models need simpler, single-purpose prompts.** A 1.2B model can't do "read these 5 chunks and synthesize an answer" — but it CAN do "read this one chunk and extract 3 facts as JSON." The architecture compensates for model limitations through decomposition.

---

## 3. Tool Calling: Small Model Challenges

### The Problem

LFM2.5-1.2B-Instruct supports native tool calling with `<|tool_call_start|>` tokens, but in practice the model is inconsistent about output format. Sometimes it produces native format, sometimes JSON, sometimes bare function calls.

### The Solution: 4-Format Parser with Fallback Router

The agent parser handles 4 output formats:
1. **LFM2.5 native:** `<|tool_call_start|>[fn(args)]<|tool_call_end|>`
2. **JSON:** `{"name": "fn", "params": {...}}`
3. **Bracket notation:** `[tool_name(params)]`
4. **Bare function call:** `tool_name(params)`

If none of these parse successfully, a **deterministic keyword-based router** (`router.py`) maps intent to tools:
- "how much space" → `disk_usage()`
- "biggest files" → `list_files(sort_by="size")`
- Default → `semantic_search(query)`

### Follow-up Check

After step 1+, the agent checks whether important keywords from the query were covered by tool results. If not, it calls `semantic_search` or `grep_files` for the missing terms. This catches cases where the model's tool choice was suboptimal.

### Lesson

> **Design for model inconsistency.** With a 1.2B model, you can't rely on a single output format. Build robust parsers with multiple fallback strategies, and add a deterministic escape hatch for when the model fails entirely. The keyword router handles ~15% of queries that the model can't route correctly.

---

## 4. LEANN Vector Search: Recompute Mode Trap

### The Problem (Feb 5)

LEANN searches were taking **54 seconds** per query — completely unusable for interactive chat.

### Root Cause

LEANN was built with `is_compact=True` and `is_recompute=True` (the default). In this mode, the index doesn't store full embedding vectors — it recomputes them on-demand during search via ZMQ calls. Every search triggered hundreds of `get_vector_zmq` calls.

### The Trap

Setting `recompute_embeddings=False` on the **searcher** doesn't help if the **index** was built in recompute mode. The index simply doesn't contain the vectors to skip recomputation.

### The Fix

```python
# Before (default — stores compact index, recomputes at search time):
LeannBuilder(is_compact=True, is_recompute=True)

# After (stores full embeddings, fast search):
LeannBuilder(is_compact=False, is_recompute=False)
```

Result: **Search dropped from 54 seconds to 0.012 seconds** (68x speedup). Index size grew from ~1 MB to ~17 MB — an acceptable tradeoff.

### Lesson

> **Read the library defaults critically.** LEANN's default mode optimizes for storage (compact index) at the cost of catastrophic search latency. This default makes sense for batch/offline use cases, not for interactive chat. The fix was a two-line config change, but finding it required understanding the entire embedding pipeline.

---

## 5. Captioning Performance: f16 → Q4 + Pipelining

### The Problem (Feb 18)

After moving captioning to foreground, the f16 Moondream2 model (2.64 GB) made each image take 5-15 seconds on CPU. For 20 uncached images:

```
Vision model load:    ~5-10s (lazy, first call only)
Per image (f16 CPU):  ~10s average
No downscaling:       +0.5-1s per large image (4K photos)
Sequential I/O:       No overlap between loading and inference
Total for 20 images:  ~245 seconds (4+ minutes)
```

### Root Causes (5 identified)

| Root Cause | Description |
|------------|-------------|
| **A: f16 precision** | 2.6 GB model on CPU without quantization or GPU offloading |
| **B: Sequential processing** | No I/O pipelining — total time = sum of all (load + caption) times |
| **C: No downscaling** | 12MP photos sent at full resolution; model internally resizes to 384x384 anyway |
| **D: Duplicate scan** | `_find_images()` and cache lookup called twice (server.py + captioner) |
| **E: Lazy loading** | First captioning call loads 3.5 GB from disk (model + projector) |

### Fixes Applied

1. **Image downscaling**: `img.thumbnail((768, 768))` before encoding — reduces payload 5-10x
2. **I/O pipelining**: `ThreadPoolExecutor` pre-loads next image while current is captioned
3. **Eager model loading**: Vision model loaded during init, not on first caption
4. **Skip cached re-injection**: Don't rebuild HNSW graph when all captions are already cached

### Lesson

> **When you move work from background to foreground, the performance requirements fundamentally change.** Captioning at 10 seconds/image was fine in the background. In the foreground, it was a UX disaster. Every architectural change has downstream performance implications.

---

## 6. Prompt Engineering for 1.2B Models

### The "Cat Drawings" Bug

The query rewriter had a few-shot example:
```
Question: "any cat drawings?"
→ {"search_query": "cat drawing sketch feline artwork illustration photo image"}
```

When a user asked "any images?" (with unrelated conversation context about revenue), the model **copied the example verbatim**, producing `"cat drawings images feline artwork"` instead of generic image terms.

### Root Cause

The 1.2B model treats few-shot examples as **templates to copy**, not patterns to generalize. The tokens "any" + "images" partially overlapped with "any cat drawings?", triggering near-verbatim reproduction.

### The Fix

Replace domain-specific examples with neutral ones:
```
# Before:
Question: "any cat drawings?" → "cat drawing sketch feline artwork..."

# After:
Question: "any images?" → "image photo picture photograph illustration"
```

### Broader Lesson

> **Design prompts for how small models actually behave, not how you wish they'd behave.** Key rules for 1.2B models:
> - Few-shot examples will be copied, not generalized. Use generic content.
> - One task per prompt. Never ask for multiple things.
> - Structured output (JSON) works if the schema is simple (3-5 fields max).
> - Always have a parsing fallback for when the model produces unexpected format.

---

## 7. HNSW Noise: The get_vector_zmq Mystery

### The Symptom

Every app startup produced hundreds of lines of:
```
[HNSW RNG] get_vector_zmq id=NNN cache_hit=1
```

### The Investigation

1. **Not from captioning** — the noise appeared even when all images were cached
2. **Source:** `LeannSearcher(enable_warmup=True)` traverses ALL HNSW graph nodes on init to pre-warm the vector cache
3. **Double warmup:** `server.py` created the searcher twice — once for initial load, once after captioning — causing double traversal
4. **The captioner bug:** `_inject_captions` was called with ALL captions (cached + new), triggering a full HNSW rebuild even when nothing changed

### Fixes

1. Only call `_inject_captions` when `new_captions` is non-empty
2. Skip searcher reload when no new captions were injected

### Lesson

> **Noisy logs are symptoms, not the bug.** The `get_vector_zmq` spam was a diagnostic trail that led to two real issues: redundant index rebuilds and double searcher initialization.

---

## 8. Summary of Key Principles

These principles emerged from real failures, not theory:

1. **Verify integration paths with real artifacts.** Mocked tests gave false confidence that LFM2.5-VL worked. One smoke test with a real model would have caught the failure instantly.

2. **Design for your actual model, not an ideal one.** A 1.2B model copies few-shot examples, produces inconsistent output formats, and can't process multiple chunks at once. The architecture must compensate.

3. **Background → foreground changes performance requirements.** Moving captioning from background to blocking init turned an acceptable 10s/image into a 4-minute startup.

4. **Read library defaults critically.** LEANN's default recompute mode made search 4500x slower than the non-recompute alternative.

5. **Noisy outputs are diagnostic trails.** The HNSW log spam led to finding redundant index rebuilds.

6. **Small models need guardrails, not freedom.** Deterministic fallback routing, multi-format parsing, keyword follow-up checks — these aren't hacks, they're the architecture.
