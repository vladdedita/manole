# Incremental Reindexing with File Watching

**Date:** 2026-02-20
**Status:** Approved
**Branch:** feat/kreuzberg-integration

## Problem

When new files are added to an already-indexed directory, the system has no way to pick them up without a full rebuild. The `KreuzbergIndexer.build()` method skips entirely if the index exists, and the "Reindex" button deletes the index and rebuilds from scratch. For large directories, full rebuilds are slow and wasteful.

## Solution

Manifest-based change detection with real-time file watching. Track what's been indexed, detect changes on startup and while running, and append only new/modified content to the existing LEANN index.

## Approach: Manifest + Watcher

### Manifest

A JSON file stored at `.leann/indexes/<name>/manifest.json`:

```json
{
  "version": 1,
  "files": {
    "docs/report.pdf": {"mtime": 1708444800.0, "chunks": 12},
    "notes/meeting.md": {"mtime": 1708531200.0, "chunks": 3}
  }
}
```

Keys are paths relative to `data_dir`. The `chunks` count tracks how many index entries belong to each file (useful for future deletion support).

Written after every successful `build_index()` or `update_index()` call.

### Catch-up Scan on Startup

When `KreuzbergIndexer.build()` is called and an index already exists:

1. Load `manifest.json`
2. Walk `data_dir`, compare each file's mtime against the manifest
3. Classify files as: **new** (not in manifest), **modified** (mtime changed), **unchanged**
4. Extract and chunk only new/modified files
5. Append via `LeannBuilder.update_index()`
6. Update manifest with new entries and updated mtimes

If `manifest.json` doesn't exist (index predates this feature), fall back to existing behavior: skip the build. A full reindex generates the manifest.

### File Watcher

A `watchfiles`-based watcher running in a background thread:

- Watches each initialized directory's `data_dir`
- On file creation/modification: debounce (500ms), then extract and append via `update_index()`
- On file deletion: log it, take no action (deferred)
- Lifecycle tied to the server process: starts on directory init, stops on shutdown

Lives in `server.py` since that's where directory lifecycle is managed. Extract/append logic reuses `KreuzbergIndexer` methods.

### Modified File Behavior

When a file is modified, new chunks are appended. Old chunks from the previous version remain in the index. Search may return stale content alongside fresh content.

This is acceptable because:
- Stale chunks are a read-time nuisance, not a data integrity issue
- The `source` metadata identifies which file chunks came from
- Full reindex (existing "Reindex" button) cleans up stale chunks
- Proper deletion support is deferred to a follow-up

## KreuzbergIndexer Changes

Two new methods:

- `incremental_update(data_dir, index_path, manifest)` -- diffs filesystem against manifest, extracts new/modified files, calls `update_index()`, returns updated manifest
- `extract_and_append(file_path, data_dir, index_path)` -- single-file extract and append for the watcher

Existing `build()` writes the manifest after building.

## Dependencies

- `watchfiles` -- file system watcher, pure Python, async-capable

## Scope Boundaries

**In scope:**
- Manifest tracking (mtime-based)
- Catch-up scan on startup
- Real-time file watching during app lifetime
- New and modified file indexing

**Deferred:**
- Deletion handling (removing stale chunks from index)
- Background daemon mode (watching while app is closed)
- Content-hash-based change detection

## Risks

| Risk | Severity | Mitigation |
|------|----------|------------|
| Stale chunks from modified files pollute search | Low | Acceptable until deletion support; full reindex cleans up |
| `update_index` not thread-safe with concurrent searches | Medium | Lock around writes, same pattern as ImageCaptioner |
| Watcher misses events during heavy I/O | Low | Catch-up scan on next startup covers gaps |
| Large batch of new files overwhelms `update_index` | Low | Batch into single `update_index` call per event group |
