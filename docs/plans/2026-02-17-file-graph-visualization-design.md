# File Graph Visualization Design

**Date:** 2026-02-17
**Status:** Approved

## Problem

Users index document directories with Manole but have no way to see how files relate to each other. The leann index contains passage embeddings and file metadata, but relationships between files remain invisible.

## Solution

A tabbed file graph visualization panel inside the Electron app. Three views show different relationship types between indexed files: content similarity, explicit references, and directory structure.

## Architecture

### Data Flow

1. UI sends `getFileGraph` request with `directoryId`
2. Python backend reads the leann index (passages JSONL + HNSW embeddings)
3. Backend computes all three edge types in one pass
4. Backend returns a unified `{nodes, edges}` response
5. Frontend renders with React Flow, filtering edges per active tab

### Data Model

```typescript
interface FileGraphResponse {
  nodes: FileNode[];
  edges: FileEdge[];
}

interface FileNode {
  id: string;           // file path relative to indexed directory
  name: string;         // filename
  type: string;         // extension (pdf, md, py, etc.)
  size: number;         // bytes
  dir: string;          // parent directory path
  passageCount: number; // passages this file produced
}

interface FileEdge {
  source: string;       // source node id
  target: string;       // target node id
  type: "similarity" | "reference" | "structure";
  weight: number;       // 0-1 for similarity, 1 for reference/structure
  label?: string;       // e.g. "imports", "links to", "parent"
}
```

### Edge Computation

**Similarity edges:** Average passage embeddings per file to get a file-level vector. Compute cosine similarity between all file pairs. Apply adaptive top-K (K=5) with a minimum threshold floor (0.6). Drop any neighbor below the floor.

**Reference edges:** Scan passage text for mentions of other indexed filenames/paths. Pattern match for imports, links, citations, filename mentions.

**Structure edges:** Parent-child edges from directory hierarchy.

### Backend Changes

New `getFileGraph` method in `server.py`'s NDJSON handler:

- Loads passages from `.leann/indexes/<name>/documents.leann.passages.jsonl`
- Groups passages by `file_path` metadata to build node list
- Retrieves or computes file-level embeddings from the HNSW index via `LeannSearcher`
- Generates all three edge types
- Caches result in the directory entry until reindex
- Returns unified response

### Protocol Addition

New response type in `protocol.ts`:

```typescript
export type ResponseType = "result" | "token" | "agent_step" | "error" | "status" | "progress" | "log" | "directory_update" | "file_graph";
```

## Frontend Design

### Aesthetic: "Illuminated Manuscript Atlas"

The graph panel feels like a living map of knowledge connections — nodes as manuscript seals, edges as ink traces connecting referenced pages. Integrated seamlessly with the existing forge-and-parchment theme.

### Layout

The graph replaces the chat panel area when activated. Users toggle between Chat and Map modes via a segmented control in the header.

```
┌─────────────────────────────────────────────────────────┐
│  [≡] Manole          [Chat · Map]          path  [</>]  │
├──────────┬──────────────────────────────────────────────┤
│          │  ┌─Similarity──References──Structure────────┐│
│  Side    │  │                                          ││
│  Panel   │  │     React Flow canvas                    ││
│          │  │                                          ││
│          │  ├──────────────────────────────────────────┤│
│          │  │ Bottom tray: node details + slider        ││
│          │  └──────────────────────────────────────────┘│
├──────────┴──────────────────────────────────────────────┤
│  Status bar                                              │
└─────────────────────────────────────────────────────────┘
```

### Components

**Mode Toggle** (header center): Segmented pill using `font-display` labels. Active segment gets `bg-accent-muted` with `text-accent`. Sliding indicator animated with motion spring.

**Tab Bar**: Text tabs below header — `font-mono text-[10px] uppercase tracking-widest`. Active tab gets 2px bottom border in accent color.

**Custom File Nodes**: Compact cards with:
- Extension badge (monospace, color-coded)
- Filename in `font-sans text-xs`
- `bg-bg-elevated` background, `border-border`
- Hover: `border-accent/30`, subtle gold glow
- Selected: `border-accent/50`, `bg-accent/[0.06]`
- Width scaled by passage count (80–160px)

**Edge Styles**:
- Similarity: accent color, opacity mapped to weight, curved bezier
- References: `text-secondary`, solid, directional animated dot
- Structure: `border` color, dotted, orthogonal routing

**Bottom Tray**: Thin 36px bar by default (node/edge counts). Expands to ~120px on node click showing file metadata. Similarity tab includes threshold slider.

**Threshold Slider**: Custom range input, `accent` fill, `font-mono text-[10px]` label. Default range 0.4–1.0, default value 0.6.

### File Type Colors

```
pdf   → #c9943e (accent gold)
md    → #6aad6a (success green)
txt   → #a59888 (text-secondary)
py    → #7aa2d4 (steel blue)
js/ts → #d4a85c (warm amber)
other → #635a50 (text-tertiary)
```

### Animations

- Panel swap (Chat ↔ Map): `AnimatePresence mode="wait"` with blur+fade transition
- Tab switch: nodes reposition with spring animation (stiffness: 400, damping: 30)
- Node entrance: staggered with `[0.22, 1, 0.36, 1]` ease curve
- Minimap: bottom-right, `bg-bg-secondary border-border`, 60% opacity

### States

- Loading: centered italic "Building map..." with pulsing dot
- Empty: "No passages indexed" matching SidePanel empty state
- Error: accent-bordered banner matching ChatPanel error style

## New Files

- `ui/src/components/FileGraphPanel.tsx` — main graph panel with tabs, React Flow canvas, bottom tray
- `ui/src/components/FileGraphNode.tsx` — custom React Flow node component
- `ui/src/components/FileGraphEdge.tsx` — custom React Flow edge component (optional, for animated reference edges)
- `ui/src/hooks/useFileGraph.ts` — hook to request/cache graph data via NDJSON
- Backend: new handler in `server.py`, new computation module (e.g. `graph.py`)

## Dependencies

- `@xyflow/react` (React Flow v12) — graph rendering
- `dagre` — tree layout for structure tab
- No new Python dependencies (numpy already available for cosine similarity)
