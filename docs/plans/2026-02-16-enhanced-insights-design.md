# Enhanced Directory Insights Design

**Date:** 2026-02-16

## Problem

Current insights have two issues:
1. Summary is LLM-generated from extension counts only — produces inaccurate descriptions
2. Summary text is truncated (`line-clamp-3`) with no way to read the full text

## Approach: Query Index + Enhanced Deterministic Stats

Use the already-built LEANN index to generate content-aware summaries, and enrich `_collect_stats` with more meaningful metrics.

## Enhanced Deterministic Stats

`_collect_stats` returns enriched data:

```python
{
  "fileCount": 29,
  "totalSize": 22_120_448,
  "types": {"pdf": 8, "md": 16, "yml": 1, "json": 1},
  "sizeByType": {"pdf": 21_000_000, "md": 800_000, ...},
  "largestFiles": [
    {"name": "Confluence Export.pdf", "size": 16_500_000},
    {"name": "Global and Market Journeys.pdf", "size": 3_100_000},
    {"name": "PRDs 001, 002.pdf", "size": 1_200_000}
  ],
  "avgFileSize": 763_118,
  "dirs": {"count": 3, "maxDepth": 2}
}
```

### SidePanel card displays:
- Type badges (existing): `8 PDF  16 MD  1 YML  ...`
- Total size (existing): `21.1 MB`
- Size distribution by type: `PDF 95%  MD 4%` (dominant types by size)
- Directory structure: `3 folders, 2 deep`

All deterministic — no LLM involved.

## Index-Based Summary Generation

Replace `_generate_summary(stats)` with `_generate_summary(dir_id)`:

1. Query the already-built LEANN index: `searcher.search_and_extract("What are the main topics, purpose, and content of these documents?")`
2. Feed extracted facts to the model to synthesize a 2-3 sentence summary
3. Result is content-aware — based on actual document text, not extension counts

## Summary Tooltip UX

- Summary text shows as `line-clamp-3` in the directory card (existing)
- On hover: tooltip popover shows the full summary text
- Simple CSS/div tooltip, no external library

## Data Flow

1. `_collect_stats` runs (enhanced, deterministic)
2. Index builds, searcher wires up
3. `directory_update` with `state: "ready"` + stats sent immediately — UI unblocks
4. `_generate_summary(dir_id)` queries index via searcher, model synthesizes
5. Second `directory_update` with summary sent as follow-up — card updates with summary text

Same async split as current code.
