# Multi-Directory Indexing Design

## Overview

Allow Manole to index multiple directories simultaneously, managed through a collapsible left side panel. Each directory has its own index, agent, search context, and conversation history. The model is loaded once and shared.

## Architecture: Multi-index Single Server

The `Server` class holds a `dict[str, DirectoryEntry]` instead of single agent/searcher fields. Each entry contains its own `Searcher`, `Agent`, metadata, and state. The model (`ModelManager`) loads on first `init` call and is reused across all entries.

### Backend Data Model

```python
@dataclass
class DirectoryEntry:
    dir_id: str          # slug derived from path
    path: str            # absolute path
    index_name: str
    searcher: Searcher
    agent: Agent
    state: str           # "indexing" | "ready" | "error"
    stats: dict          # file_count, types breakdown, total_size
    summary: str | None  # auto-generated after indexing completes
    error: str | None
```

### Protocol

**New unsolicited message (backend -> UI):**

```json
{ "id": null, "type": "directory_update", "data": {
    "directoryId": "string",
    "state": "indexing | ready | error",
    "stats": { "fileCount": 0, "types": {}, "totalSize": 0 },
    "summary": "string or null",
    "error": "string or null"
}}
```

**Methods (UI -> Backend):**

| Method | Params | Response |
|--------|--------|----------|
| `init` | `{ dataDir }` | `{ directoryId, indexName }` — callable multiple times |
| `query` | `{ text, directoryId?, searchAll? }` | streaming tokens + result |
| `remove_directory` | `{ directoryId }` | `{ status: "ok" }` |
| `reindex` | `{ directoryId }` | `{ status: "indexing" }` — then async `directory_update` messages |

### Indexing Flow

1. UI calls `init({ dataDir })`
2. Server creates entry, returns `directoryId` immediately
3. Model loads if first time (blocking, status pushed)
4. Indexing runs in a background thread — status updates pushed via `directory_update`
5. On completion: file stats collected, content summary auto-generated, `directory_update` with `state: "ready"` pushed
6. Agent and Searcher wired for that entry

### Query Flow

- Default: query routes to `activeDirectoryId`'s agent
- `searchAll: true`: query runs against each ready directory's agent, results merged
- Each directory maintains its own conversation history

## Frontend

### Side Panel (Collapsible Drawer)

- Slides from left, ~280px, overlays chat with backdrop
- Toggle via hamburger button in header (left of "Manole")
- Panel header: "Indexes" label + close button
- "Search All" toggle shown when 2+ directories exist
- Directory cards with: folder name, path, state dot, file stats badges, auto-summary
- Hover-reveal: Reindex and Remove icon buttons
- Active directory: accent left bar (animated with `layoutId`)
- "Add Folder" button pinned at bottom (dashed border)

### State Management

- `directories: DirectoryEntry[]` — all indexed directories
- `activeDirectoryId: string | null` — currently selected
- `searchAll: boolean` — query all vs. active only
- Per-directory conversation history (keyed ChatPanel)

### UX Flow

- Welcome screen shows when no directories added
- Adding a folder starts indexing in background, user can switch to other ready directories
- Indexing status shown inline in chat area ("Indexing in progress...") instead of full-screen blocker
- Summary generated automatically after indexing — no user action needed
- LoadingScreen component retired in favor of inline states

## Design System

Follows existing forge & parchment aesthetic:
- Cormorant Garamond display, DM Sans body, IBM Plex Mono metadata
- Aged gold accent (`#c9943e`) for active states, toggles, badges
- State dots: pulsing amber (indexing), green (ready), red (error)
- Motion animations with `[0.22, 1, 0.36, 1]` easing
- Type badges in `bg-bg-elevated` with accent count numbers
