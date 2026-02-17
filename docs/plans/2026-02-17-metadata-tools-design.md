# Metadata Tools Design

**Date:** 2026-02-17
**Status:** Approved
**Problem:** Users ask aggregate metadata questions ("what folders take up the most space?", "largest files?", "disk usage?") and the agent has no tools to answer them. It falls back to semantic search and grep, which return irrelevant content results.

## New Tools

### 1. `folder_stats`

Aggregate per-folder statistics: total size, file count, sorted by size or count.

**Parameters:**
- `sort_by`: "size" (default) | "count"
- `limit`: integer, default 10

**Output format:**
```
Folder sizes (sorted by size):
  finances/2025/: 12.4 MB, 23 files
  design/: 3.1 MB, 8 files
  (root): 1.2 MB, 5 files
Total: 16.7 MB across 36 files
```

**Answers:** "biggest folders", "which folder has most files", "folder sizes"

### 2. `disk_usage`

Summary of the entire indexed directory: total size, total files, breakdown by extension (top 10 types by size), average file size.

**Parameters:** none

**Output format:**
```
Disk usage summary:
  Total: 16.7 MB across 36 files
  Average file size: 475.2 KB
  By type:
    .pdf: 10.2 MB (15 files)
    .txt: 3.1 MB (12 files)
    .png: 2.5 MB (5 files)
    .md: 0.9 MB (4 files)
```

**Answers:** "how much space", "disk usage", "storage overview", "what types of files"

### 3. Enhanced `list_files`

Add `sort_by` parameter to existing `list_files` tool.

**New parameter:** `sort_by`: "date" (default, existing behavior) | "size" | "name"

When `sort_by="size"`, output includes file sizes and sorts largest first.

**Answers:** "largest files", "biggest files", "smallest files"

## Router/Rewriter Changes

### Rewriter

- Add `"metadata"` to valid intents (alongside factual, count, list, compare, summarize)
- Add examples to `REWRITER_SYSTEM` prompt:
  - "what folders take up the most space?" → `{"intent": "metadata", ...}`
  - "how much storage am I using?" → `{"intent": "metadata", ...}`

### Router Fallback

Add keyword-based fallback routing for metadata queries:

```python
if intent == "metadata" or any(k in q for k in ["space", "biggest", "largest", "storage", "heavy"]):
    if any(k in q for k in ["folder", "directory"]):
        return "folder_stats", {"sort_by": "size"}
    if any(k in q for k in ["total", "usage", "overview", "summary"]):
        return "disk_usage", {}
    return "folder_stats", {"sort_by": "size"}
```

## Agent Changes

- Add `folder_stats`, `disk_usage` to `_KNOWN_TOOLS` in `agent.py`
- Add tool schemas for both new tools to `TOOL_SCHEMAS` in `agent.py`
- Update `list_files` schema to include `sort_by` parameter

## Files Modified

- `toolbox.py` — add `folder_stats()`, `disk_usage()`, enhance `list_recent_files()` with `sort_by`
- `tools.py` — add handlers and `TOOL_DEFINITIONS` for new tools, update `list_files` definition
- `agent.py` — update `TOOL_SCHEMAS`, `_KNOWN_TOOLS`
- `router.py` — add metadata keyword routing
- `rewriter.py` — add "metadata" intent with examples
- `tests/test_toolbox.py` — tests for new toolbox methods
- `tests/test_tools.py` — tests for new tool registry dispatch
- `tests/test_router.py` — tests for metadata routing (new file if needed)
- `tests/test_rewriter.py` — tests for metadata intent (if file exists)

## Testing Strategy

- Unit tests with temp directories of known file sizes
- Verify folder_stats aggregation correctness
- Verify disk_usage summary math
- Verify list_files sort_by=size ordering
- Verify router correctly maps metadata keywords
- Verify rewriter accepts "metadata" intent
