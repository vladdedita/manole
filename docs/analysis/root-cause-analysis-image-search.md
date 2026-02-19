# Root Cause Analysis: Image Search Bugs

**Date:** 2026-02-18
**Methodology:** Toyota 5 Whys (multi-causal)
**Status:** Complete

## Problem Statement

When a user asks "any images?" after a conversation about "revenue of engineering department," two bugs manifest:

1. **Rewriter Hallucination:** The query rewriter fabricates "cat drawings" context, producing `"Show me any cat drawings or images of cats in my files?"` instead of a generic image search query.
2. **No Image Results:** Even after semantic_search runs, no image caption results appear because captions are injected into the index AFTER the query completes.

---

## Bug 1: Rewriter Hallucination

### Evidence

Log output:
```
[REWRITE] Sending to model: 'Recent conversation:\n  User: break down the target revenue of the engineering department\n  Assistant: The target revenue'
[REWRITE] Raw: {
  "intent": "list",
  "search_query": "cat drawings images feline artwork illustration photos",
  "resolved_query": "Show me any cat drawings or images of cats in my files?"
}
```

The conversation history was about "revenue of engineering department." The user query "any images?" is topic-independent. The rewriter injected "cat drawings" -- a topic never mentioned by the user.

### 5 Whys Analysis

**WHY 1 (Symptom):** The rewriter output contains "cat drawings" and "feline artwork" -- terms absent from both the user query and conversation history.

- Evidence: Log shows input was `"any images?"` with revenue context. Output search_query is `"cat drawings images feline artwork illustration photos"`.

**WHY 2 (Context):** The LFM2.5-1.2B model copied the example response verbatim from the system prompt instead of generating a contextually appropriate response.

- Evidence: The system prompt in `rewriter.py` (line 29) contains this exact few-shot example:
  ```
  Question: "any cat drawings?"
  {"intent": "list", "search_query": "cat drawing sketch feline artwork illustration photo image",
   "resolved_query": "Are there any cat drawings or images of cats in my files?"}
  ```
  The model's output `"cat drawings images feline artwork illustration photos"` is nearly identical to the example's `"cat drawing sketch feline artwork illustration photo image"`. The model pattern-matched "any images?" to the closest example "any cat drawings?" and reproduced it with minimal modification.

**WHY 3 (System):** The 1.2B parameter model lacks sufficient capacity to distinguish between "follow the pattern" and "copy the content" when a user query partially matches a few-shot example. Small LLMs are known to over-index on surface-level token overlap rather than semantic intent.

- Evidence: The query "any images?" shares the tokens "any" and "images" with the example "any cat drawings?" -- a 2-of-4 content word overlap. The model treats this as a near-match and copies the example output rather than generalizing the pattern.

**WHY 4 (Design):** The few-shot examples in `REWRITER_SYSTEM` include specific, topical content (cats) that bleeds into unrelated queries. The prompt design does not isolate the structural pattern from the domain-specific content of each example.

- Evidence: `rewriter.py` lines 28-30. The "cat drawings" example is the only image-related example in the prompt. Any image query will pattern-match to it. There is no generic image example (e.g., `"any images?" -> "images photos pictures photographs"`).

**WHY 5 (Root Cause):** The rewriter system prompt contains a few-shot example with highly specific content ("cat drawings") as the only image-domain example, and the 1.2B model is too small to generalize beyond literal example copying. The prompt was designed without considering that small models treat few-shot examples as templates to copy rather than patterns to generalize.

### Root Cause Summary (Bug 1)

| Factor | Description |
|--------|-------------|
| **RC-1A: Prompt contamination** | The "cat drawings" few-shot example in `REWRITER_SYSTEM` is the sole image-domain example, causing all image queries to copy its specific content. |
| **RC-1B: Model capacity** | LFM2.5-1.2B cannot reliably generalize few-shot patterns; it copies the closest matching example verbatim. This is a known limitation of sub-2B parameter models. |

### Backwards Chain Validation (Bug 1)

If `REWRITER_SYSTEM` contains only one image example ("cat drawings") AND the model copies examples rather than generalizing, THEN any generic image query ("any images?") will produce output containing "cat drawings." This matches the observed symptom exactly.

---

## Bug 2: No Image Results (Race Condition)

### Evidence

Log output shows captions injected AFTER query completion:
```
[CAPTIONER] Injected 11 captions into index (0 cached, 11 new)
[CAPTIONER] Complete: 11/11 images captioned
```

The search returned no image results despite images existing in the directory.

### 5 Whys Analysis

**WHY 1 (Symptom):** `semantic_search("cat drawings")` returned no image caption results even though the directory contains images.

- Evidence: The captioner log shows 11 images were found and captioned, but this completed after the query ran. The search hit the index before captions were present.

**WHY 2 (Context):** Image captions had not yet been injected into the LEANN index when the search query executed.

- Evidence: In `server.py` lines 253-291, image captioning runs in a background thread (`_background_tasks`). The `handle_init` method returns `"ready"` status to the UI at line 247 BEFORE the background thread starts captioning. Queries can execute immediately after init returns.

**WHY 3 (System):** The server declares the directory "ready" for queries before background tasks (summary generation + image captioning) complete. There is no mechanism to signal that the index is incomplete or to block image-related queries until captioning finishes.

- Evidence: `server.py` line 246-251:
  ```python
  # Send ready immediately so the UI unblocks
  send(None, "directory_update", {
      "directoryId": dir_id, "state": "ready",
      "stats": stats,
  })
  ```
  The background thread starts at line 290, AFTER the ready signal. Caption injection (`_inject_captions` in `image_captioner.py` line 127) happens inside `captioner.run()` which runs in this background thread.

**WHY 4 (Design):** The architecture treats indexing and captioning as decoupled phases with no coordination. The "ready" state is binary -- either the directory is ready or not. There is no intermediate state like "ready_text_only" or "captioning_in_progress" that the agent could use to inform the user or defer image queries.

- Evidence: `server.py` state machine has only: `not_initialized`, `indexing`, `ready`. No `captioning` state. The `handle_query` method (line 299) checks only `state == "ready"` with no awareness of background task completion. The background thread reference is stored (`line 292: self.directories[dir_id]["background_thread"] = thread`) but never checked before query execution.

**WHY 5 (Root Cause):** The system was designed with the assumption that all searchable content would be available at index-build time. Image captioning was added later as a background post-processing step, but the query path was never updated to account for the possibility that the index is incomplete. There is no synchronization mechanism between the captioning pipeline and the query pipeline.

### Root Cause Summary (Bug 2)

| Factor | Description |
|--------|-------------|
| **RC-2A: Premature ready signal** | `handle_init` sends `state: "ready"` before background captioning completes, allowing queries against an incomplete index. |
| **RC-2B: No index-completeness awareness** | The query pipeline (`handle_query` / `Agent.run`) has no knowledge of whether image captions have been injected. No state distinguishes "text indexed" from "fully indexed with captions." |
| **RC-2C: Fire-and-forget background task** | The background thread is started but its completion is never awaited or checked by the query path. The thread reference is stored but unused. |

### Backwards Chain Validation (Bug 2)

If `handle_init` returns "ready" before captioning completes AND `handle_query` does not check background thread completion AND the user queries immediately, THEN the search will hit an index without captions, returning no image results. This matches the observed symptom.

---

## Cross-Validation

The two bugs are independent but compound:

- Bug 1 (hallucination) causes the wrong search query ("cat drawings" instead of generic image terms), but even a correct query would fail due to Bug 2.
- Bug 2 (race condition) means no image results regardless of query quality.
- Fixing only Bug 1 would improve query quality but still return no results if the user queries before captioning completes.
- Fixing only Bug 2 would return image results but only if the search query is reasonable (still broken by Bug 1).
- Both fixes are independently necessary.

---

## Solutions

### Immediate Mitigations (restore correct behavior)

**M-1: Remove the "cat drawings" example from the rewriter prompt.**

Replace with a generic image example that does not inject specific subject matter:

```python
# In rewriter.py REWRITER_SYSTEM, replace:
'Question: "any cat drawings?"\n'
'{"intent": "list", "search_query": "cat drawing sketch feline artwork illustration photo image", '
'"resolved_query": "Are there any cat drawings or images of cats in my files?"}\n\n'

# With:
'Question: "any images?"\n'
'{"intent": "list", "search_query": "image photo picture photograph illustration", '
'"resolved_query": "Are there any images or photos in my files?"}\n\n'
```

**M-2: Wait for captioning before returning "ready," or add a captioning-aware query path.**

Quick fix: in `handle_query`, check if the background thread is still alive and either wait for it or warn the user:

```python
# In server.py handle_query, after resolving the directory entry:
bg = entry.get("background_thread")
if bg and bg.is_alive():
    bg.join(timeout=30)  # Wait up to 30s for captioning
```

### Permanent Fixes (prevent recurrence)

**P-1: Redesign rewriter few-shot examples to use neutral content.**

All few-shot examples in `REWRITER_SYSTEM` should use generic terms that cannot contaminate outputs. Audit every example for domain-specific content that a small model might copy. Add a dedicated test that checks rewriter output does not contain tokens from examples that are irrelevant to the input query.

**P-2: Add index-completeness state tracking.**

Add a `captioning_state` field to the directory entry (`"pending"`, `"in_progress"`, `"complete"`). The agent or query handler can use this to:
- Inform the user: "Image captioning is still in progress. Text search results are available now."
- Automatically re-run the image search after captioning completes.
- Include the state in the UI so users know when full search is available.

**P-3: Make captioning synchronous for small directories, async for large ones.**

For directories with fewer than ~20 images, captioning completes in seconds. Run it synchronously before returning "ready." For larger directories, use the async path with state tracking (P-2).

### Early Detection

**D-1: Add a rewriter regression test** that sends "any images?" with unrelated conversation context and asserts the output does not contain "cat," "feline," or any example-specific tokens.

**D-2: Add an integration test** that queries for images immediately after init and verifies either (a) results are returned or (b) a clear "captioning in progress" message is shown.

---

## Summary of Root Causes

| ID | Root Cause | Bug | Fix |
|----|-----------|-----|-----|
| RC-1A | Few-shot example with specific content ("cat drawings") as sole image example | Bug 1 | P-1: Neutral few-shot examples |
| RC-1B | 1.2B model copies examples instead of generalizing patterns | Bug 1 | P-1: Design prompts for small-model behavior |
| RC-2A | Premature "ready" signal before captioning completes | Bug 2 | P-2, P-3: State tracking + sync for small dirs |
| RC-2B | No index-completeness awareness in query path | Bug 2 | P-2: Captioning state field |
| RC-2C | Fire-and-forget background thread with no coordination | Bug 2 | P-2, P-3: Thread join or state machine |

---

## Key Files

| File | Relevance |
|------|-----------|
| `/Users/ded/Projects/assist/manole/rewriter.py` | Lines 4-46: `REWRITER_SYSTEM` prompt with contaminating "cat drawings" example (RC-1A) |
| `/Users/ded/Projects/assist/manole/server.py` | Lines 246-292: Premature ready signal and fire-and-forget background captioning (RC-2A, RC-2C) |
| `/Users/ded/Projects/assist/manole/image_captioner.py` | Lines 127-144: Caption injection that runs too late (RC-2B) |
| `/Users/ded/Projects/assist/manole/agent.py` | Lines 105-127: Agent.run passes history to rewriter, no captioning awareness |
| `/Users/ded/Projects/assist/manole/models.py` | Lines 71-84: caption_image uses shared lock, serialized with text generation |
