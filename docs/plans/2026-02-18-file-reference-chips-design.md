# File Reference Chips — Design

## Data Flow

```
Searcher.search_and_extract()          ToolRegistry.execute()
  facts_by_source dict ──────────────►  (text, sources: list[str])
  keys = filenames                            │
                                              ▼
                                       Agent.run()
                                         accumulates sources
                                         across N tool steps
                                         deduplicates
                                              │
                                              ▼
                                       (answer_text, sources)
                                              │
                                              ▼
                                       Server.handle_query()
                                         resolves filenames → absolute paths
                                         using entry["path"] (data_dir)
                                              │
                                              ▼
                                       NDJSON result message:
                                       {"type":"result","data":{"text":"...","sources":["/abs/path/doc.pdf"]}}
                                              │
                                              ▼
                                       useChat.ts reducer
                                         "response_complete" action
                                         stores sources in ChatMessage
                                              │
                                              ▼
                                       MessageBubble.tsx
                                         renders chips below answer text
                                         chip label = basename(path)
                                         chip tooltip = full path
                                         onClick → IPC "open-file"
                                              │
                                              ▼
                                       Electron main process
                                         ipcMain.handle("open-file")
                                         shell.openPath(absolutePath)
```

## Return Type Changes

| Layer | Before | After |
|-------|--------|-------|
| `Searcher.search_and_extract()` | `str` | `tuple[str, list[str]]` |
| `Searcher._filename_fallback()` | `str` | `tuple[str, list[str]]` |
| `ToolRegistry.execute()` | `str` | `tuple[str, list[str]]` |
| `Agent.run()` | `str` | `tuple[str, list[str]]` |

## Component Boundaries

**No new files.** All changes are return-type enrichments to existing functions.

- **Backend (Python):** `searcher.py`, `tools.py`, `agent.py`, `server.py` — threading `sources` through
- **Frontend (TypeScript):** `protocol.ts`, `useChat.ts`, `MessageBubble.tsx` — consuming and rendering
- **Electron (IPC):** main process + preload — opening files

## Path Resolution Strategy

Sources start as filenames (e.g. `"budget.pdf"`) from `Searcher._get_source()`. The server resolves them to absolute paths using the directory's `entry["path"]`:

1. Try `os.path.join(base_dir, filename)` — works for flat directories
2. If not found, `os.walk(base_dir)` to find nested files — handles subdirectories
3. Fallback: pass filename as-is (chip shows name, click may fail gracefully)

## UI Rendering

Chips appear:
- Below the answer text, above AgentSteps
- Only after streaming completes (`!isStreaming`)
- Only for assistant messages with non-empty sources
- Separated from text by a subtle border-top
- Styled as rounded pills with a file icon, matching existing UI tokens

## Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Where to resolve paths | Server | Single place, backend has access to `entry["path"]` |
| Source granularity | Contributing files only | User chose this in discovery |
| Click behavior | OS default app | User chose this in discovery |
| Display style | Chips below answer | User chose this in discovery |
| New NDJSON event type? | No | Piggyback on existing `result` message |
