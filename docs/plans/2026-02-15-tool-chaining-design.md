# Python-Orchestrated Tool Chaining

**Date:** 2026-02-15
**Status:** Approved

## Problem

The 1.2B model makes decent single tool choices but can't chain. After seeing a tool result, it answers immediately instead of calling another tool.

Example: "any macbook pdfs" -> `count_files(extension="pdf")` -> "25 PDFs" -> done. Never searched for "macbook".

The agent loop already supports multi-step (MAX_STEPS=5). The model just stops early.

## Decision

Python orchestration. The model stays a single-step tool caller. The agent layer detects when a result is incomplete and forces follow-up calls.

## Design

### Core mechanism: `_needs_followup()`

A method on Agent that runs after each tool result when the model gives a direct response (no tool call) at step > 0. It checks keyword coverage and picks the next tool.

**Keyword coverage:** Extract keywords from the original query (reuse `extract_keywords` from searcher.py). If key terms aren't present in accumulated tool results, we need another tool call.

**Tool selection** based on what's missing:
- Keywords not found in results -> `grep_files(pattern=keyword)` or `semantic_search(query=keywords)`
- Got file names but need content -> `semantic_search(query=original_query)`
- Got a count but query asks about specific items -> `grep_files` or `list_files`

### Agent loop change

```python
# Current: step > 0, no tool call -> return raw
# New: step > 0, no tool call -> check if followup needed
if tool_call is None:
    if step == 0:
        # ... existing fallback router logic ...
    else:
        followup = self._needs_followup(query, messages)
        if followup:
            result = self.tools.execute(followup["name"], followup["params"])
            messages.append({"role": "assistant", "content": raw})
            messages.append({"role": "tool", "content": json.dumps(...)})
            continue
        else:
            return raw  # truly done
```

### Edge cases

- **Max steps respected** — followup can't run forever (MAX_STEPS=5)
- **No infinite loops** — `_needs_followup` tracks which tools have already been called to avoid repeating
- **Doesn't override model** — only activates when model gives up (no tool call). If model chains on its own, Python doesn't interfere.

## Files to modify

- `agent.py` — add `_needs_followup()` and integrate into loop
- `tests/test_agent.py` — test chaining scenarios

## Example flows

### "any macbook pdfs"
```
Step 1: Model -> count_files(extension="pdf") -> "Found 25 .pdf files."
Step 2: Model -> "There are 25 PDF files." (direct response)
        Python: "macbook" not in results -> grep_files(pattern="macbook")
Step 3: grep result -> "macbook_ssd.pdf" -> Model synthesizes answer
```

### "how many invoice pdfs"
```
Step 1: Model -> count_files(extension="pdf") -> "Found 25 .pdf files."
Step 2: Model -> "25 PDFs" (direct response)
        Python: "invoice" not in results -> grep_files(pattern="invoice")
Step 3: grep result -> lists invoice files -> Model synthesizes count
```

### "how many eggs in carbonara" (no chaining needed)
```
Step 1: Model -> semantic_search(query="carbonara eggs")
Step 2: Search results contain "eggs" -> keyword covered -> return direct answer
```
