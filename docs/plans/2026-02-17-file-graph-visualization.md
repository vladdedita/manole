# File Graph Visualization Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a tabbed file graph visualization panel to the Manole Electron app showing content similarity, explicit references, and directory structure relationships between indexed files.

**Architecture:** Python backend computes a unified graph (nodes + typed edges) from leann index data via a new `getFileGraph` NDJSON method. Frontend renders with React Flow inside a new panel that swaps with the chat view via a header mode toggle. All three edge types are returned in one response; the frontend filters per active tab.

**Tech Stack:** React Flow (`@xyflow/react`), dagre (tree layout), Python numpy (cosine similarity), existing leann `PassageManager` + `backend_impl.compute_query_embedding()` APIs.

---

### Task 1: Install frontend dependencies

**Files:**
- Modify: `ui/package.json`

**Step 1: Install React Flow and dagre**

```bash
cd ui && npm install @xyflow/react @dagrejs/dagre
```

**Step 2: Verify installation**

Run: `cd ui && node -e "require('@xyflow/react'); require('@dagrejs/dagre'); console.log('OK')"`
Expected: `OK`

**Step 3: Commit**

```bash
git add ui/package.json ui/package-lock.json
git commit -m "deps: add @xyflow/react and @dagrejs/dagre for file graph"
```

---

### Task 2: Add graph types to the protocol

**Files:**
- Modify: `ui/src/lib/protocol.ts`

**Step 1: Write the type definitions**

Add to `ui/src/lib/protocol.ts` after the existing `DirectoryUpdateData` interface:

```typescript
export interface FileNode {
  id: string;
  name: string;
  type: string;
  size: number;
  dir: string;
  passageCount: number;
}

export interface FileEdge {
  source: string;
  target: string;
  type: "similarity" | "reference" | "structure";
  weight: number;
  label?: string;
}

export interface FileGraphData {
  nodes: FileNode[];
  edges: FileEdge[];
}
```

Also add `"file_graph"` to the `ResponseType` union.

**Step 2: Commit**

```bash
git add ui/src/lib/protocol.ts
git commit -m "feat: add FileGraph types to NDJSON protocol"
```

---

### Task 3: Build the Python graph computation module

**Files:**
- Create: `graph.py`
- Test: `tests/test_graph.py`

This is the core backend logic. It reads the leann index, groups passages by file, computes similarity/reference/structure edges, and returns the unified graph.

**Step 1: Write failing tests for graph computation**

Create `tests/test_graph.py`:

```python
"""Tests for file graph computation."""
import json
import pytest
import numpy as np
from unittest.mock import MagicMock, patch


def make_passage(passage_id, text, file_path, file_name=None):
    """Helper to create a passage dict matching leann JSONL format."""
    meta = {"file_path": file_path}
    if file_name:
        meta["file_name"] = file_name
    return {"id": passage_id, "text": text, "metadata": meta}


class TestBuildNodes:
    """Test node construction from passages."""

    def test_groups_passages_by_file(self):
        from graph import build_nodes
        passages = [
            make_passage("0", "hello world", "/data/doc.pdf"),
            make_passage("1", "second chunk", "/data/doc.pdf"),
            make_passage("2", "other file", "/data/notes.md"),
        ]
        nodes = build_nodes(passages, base_dir="/data")
        assert len(nodes) == 2
        by_id = {n["id"]: n for n in nodes}
        assert by_id["doc.pdf"]["passageCount"] == 2
        assert by_id["notes.md"]["passageCount"] == 1

    def test_extracts_file_metadata(self):
        from graph import build_nodes
        passages = [
            make_passage("0", "content", "/data/sub/report.pdf"),
        ]
        nodes = build_nodes(passages, base_dir="/data")
        assert len(nodes) == 1
        node = nodes[0]
        assert node["id"] == "sub/report.pdf"
        assert node["name"] == "report.pdf"
        assert node["type"] == "pdf"
        assert node["dir"] == "sub"

    def test_handles_missing_metadata(self):
        from graph import build_nodes
        passages = [{"id": "0", "text": "hello", "metadata": {}}]
        nodes = build_nodes(passages, base_dir="/data")
        assert len(nodes) == 0  # no file_path = skip


class TestSimilarityEdges:
    """Test content similarity edge computation."""

    def test_returns_edges_above_threshold(self):
        from graph import compute_similarity_edges
        # 3 files with known embeddings
        embeddings = {
            "a.pdf": np.array([1.0, 0.0, 0.0]),
            "b.pdf": np.array([0.9, 0.1, 0.0]),  # very similar to a
            "c.pdf": np.array([0.0, 0.0, 1.0]),   # orthogonal to a
        }
        edges = compute_similarity_edges(embeddings, top_k=2, threshold=0.5)
        sources_targets = {(e["source"], e["target"]) for e in edges}
        # a and b should be connected (high similarity)
        assert ("a.pdf", "b.pdf") in sources_targets or ("b.pdf", "a.pdf") in sources_targets
        # All edges should have type "similarity"
        assert all(e["type"] == "similarity" for e in edges)
        # All weights should be between 0 and 1
        assert all(0 <= e["weight"] <= 1 for e in edges)

    def test_respects_threshold(self):
        from graph import compute_similarity_edges
        embeddings = {
            "a.pdf": np.array([1.0, 0.0]),
            "b.pdf": np.array([0.0, 1.0]),  # orthogonal = 0 similarity
        }
        edges = compute_similarity_edges(embeddings, top_k=5, threshold=0.5)
        assert len(edges) == 0  # below threshold

    def test_no_self_edges(self):
        from graph import compute_similarity_edges
        embeddings = {
            "a.pdf": np.array([1.0, 0.0]),
            "b.pdf": np.array([0.9, 0.1]),
        }
        edges = compute_similarity_edges(embeddings, top_k=5, threshold=0.0)
        for e in edges:
            assert e["source"] != e["target"]


class TestReferenceEdges:
    """Test explicit reference detection."""

    def test_detects_filename_mentions(self):
        from graph import compute_reference_edges
        passages_by_file = {
            "readme.md": ["See report.pdf for details"],
            "report.pdf": ["This is the report"],
        }
        file_ids = {"readme.md", "report.pdf"}
        edges = compute_reference_edges(passages_by_file, file_ids)
        assert len(edges) >= 1
        assert edges[0]["source"] == "readme.md"
        assert edges[0]["target"] == "report.pdf"
        assert edges[0]["type"] == "reference"

    def test_no_self_references(self):
        from graph import compute_reference_edges
        passages_by_file = {
            "readme.md": ["This readme.md explains things"],
        }
        file_ids = {"readme.md"}
        edges = compute_reference_edges(passages_by_file, file_ids)
        assert len(edges) == 0


class TestStructureEdges:
    """Test directory hierarchy edges."""

    def test_parent_child_edges(self):
        from graph import compute_structure_edges
        file_ids = ["docs/a.pdf", "docs/b.pdf", "src/c.py"]
        edges = compute_structure_edges(file_ids)
        # Should have parent-child relationships
        assert len(edges) > 0
        assert all(e["type"] == "structure" for e in edges)

    def test_root_files_have_no_parent(self):
        from graph import compute_structure_edges
        file_ids = ["a.pdf", "b.pdf"]
        edges = compute_structure_edges(file_ids)
        # Root files share implicit root parent - depends on implementation
        # At minimum, no crashes
        assert isinstance(edges, list)
```

**Step 2: Run tests to verify they fail**

Run: `cd /Users/ded/Projects/assist/manole && python -m pytest tests/test_graph.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'graph'`

**Step 3: Implement `graph.py`**

Create `graph.py`:

```python
"""File graph computation from leann index data.

Reads passage data and embeddings from a leann index, groups by source file,
and computes three types of edges: content similarity, explicit references,
and directory structure.
"""
import json
import re
from pathlib import Path, PurePosixPath

import numpy as np


def build_nodes(passages: list[dict], base_dir: str) -> list[dict]:
    """Group passages by file_path and build node metadata.

    Args:
        passages: List of passage dicts with 'metadata.file_path'.
        base_dir: Absolute path of the indexed directory (for relative paths).

    Returns:
        List of FileNode dicts.
    """
    base = Path(base_dir)
    by_file: dict[str, list[dict]] = {}

    for p in passages:
        meta = p.get("metadata", {})
        file_path = meta.get("file_path")
        if not file_path:
            continue
        # Make relative to base_dir
        try:
            rel = str(PurePosixPath(Path(file_path).relative_to(base)))
        except ValueError:
            rel = PurePosixPath(file_path).name
        by_file.setdefault(rel, []).append(p)

    nodes = []
    for rel_path, file_passages in by_file.items():
        pp = PurePosixPath(rel_path)
        nodes.append({
            "id": rel_path,
            "name": pp.name,
            "type": pp.suffix.lstrip(".").lower() if pp.suffix else "",
            "size": 0,  # populated later if stat available
            "dir": str(pp.parent) if str(pp.parent) != "." else "",
            "passageCount": len(file_passages),
        })
    return nodes


def compute_similarity_edges(
    embeddings: dict[str, np.ndarray],
    top_k: int = 5,
    threshold: float = 0.6,
) -> list[dict]:
    """Compute content similarity edges between files.

    Uses cosine similarity on file-level embedding vectors.

    Args:
        embeddings: Dict mapping file_id to embedding vector.
        top_k: Max neighbors per file.
        threshold: Minimum similarity to include.

    Returns:
        List of FileEdge dicts with type="similarity".
    """
    file_ids = list(embeddings.keys())
    n = len(file_ids)
    if n < 2:
        return []

    # Build matrix and normalize
    matrix = np.stack([embeddings[fid] for fid in file_ids])
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    normalized = matrix / norms

    # Cosine similarity matrix
    sim_matrix = normalized @ normalized.T

    edges = []
    seen = set()

    for i in range(n):
        # Get top-K excluding self
        scores = sim_matrix[i].copy()
        scores[i] = -1  # exclude self
        top_indices = np.argsort(scores)[::-1][:top_k]

        for j in top_indices:
            score = float(scores[j])
            if score < threshold:
                continue
            pair = tuple(sorted((file_ids[i], file_ids[j])))
            if pair in seen:
                continue
            seen.add(pair)
            edges.append({
                "source": file_ids[i],
                "target": file_ids[j],
                "type": "similarity",
                "weight": round(score, 3),
            })

    return edges


def compute_reference_edges(
    passages_by_file: dict[str, list[str]],
    file_ids: set[str],
) -> list[dict]:
    """Detect explicit references between files by scanning passage text.

    Looks for mentions of other filenames in passage text.

    Args:
        passages_by_file: Dict mapping file_id to list of passage texts.
        file_ids: Set of all file IDs in the graph.

    Returns:
        List of FileEdge dicts with type="reference".
    """
    # Build lookup: filename -> set of file_ids that have that name
    name_to_ids: dict[str, set[str]] = {}
    for fid in file_ids:
        name = PurePosixPath(fid).name
        name_to_ids.setdefault(name, set()).add(fid)

    edges = []
    seen = set()

    for source_id, texts in passages_by_file.items():
        combined = " ".join(texts)
        for name, target_ids in name_to_ids.items():
            if len(name) < 4:  # skip very short names to avoid false positives
                continue
            # Escape for regex, word boundary match
            pattern = re.escape(name)
            if re.search(pattern, combined, re.IGNORECASE):
                for target_id in target_ids:
                    if target_id == source_id:
                        continue
                    pair = (source_id, target_id)
                    if pair in seen:
                        continue
                    seen.add(pair)
                    edges.append({
                        "source": source_id,
                        "target": target_id,
                        "type": "reference",
                        "weight": 1.0,
                        "label": f"mentions {name}",
                    })

    return edges


def compute_structure_edges(file_ids: list[str]) -> list[dict]:
    """Compute directory hierarchy edges.

    Creates edges between directories and their children.

    Args:
        file_ids: List of relative file paths.

    Returns:
        List of FileEdge dicts with type="structure".
    """
    # Collect all directories
    dirs = set()
    for fid in file_ids:
        parts = PurePosixPath(fid).parts
        for i in range(1, len(parts)):
            dirs.add("/".join(parts[:i]))

    edges = []
    seen = set()

    for fid in file_ids:
        pp = PurePosixPath(fid)
        parent = str(pp.parent) if str(pp.parent) != "." else ""
        if parent:
            pair = (parent, fid)
            if pair not in seen:
                seen.add(pair)
                edges.append({
                    "source": parent,
                    "target": fid,
                    "type": "structure",
                    "weight": 1.0,
                    "label": "contains",
                })

    # Directory-to-directory edges
    for d in dirs:
        pp = PurePosixPath(d)
        parent = str(pp.parent) if str(pp.parent) != "." else ""
        if parent:
            pair = (parent, d)
            if pair not in seen:
                seen.add(pair)
                edges.append({
                    "source": parent,
                    "target": d,
                    "type": "structure",
                    "weight": 1.0,
                    "label": "contains",
                })

    return edges


def load_passages_from_index(index_path: str) -> list[dict]:
    """Load all passages from a leann JSONL file.

    Args:
        index_path: Path to the leann index (without .meta.json suffix).

    Returns:
        List of passage dicts.
    """
    jsonl_candidates = [
        Path(f"{index_path}.passages.jsonl"),
    ]
    # Also check meta for passage_sources
    meta_path = Path(f"{index_path}.meta.json")
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        for source in meta.get("passage_sources", []):
            rel = source.get("path_relative") or source.get("path", "")
            if rel:
                jsonl_candidates.insert(0, meta_path.parent / rel)

    for jsonl_path in jsonl_candidates:
        if jsonl_path.exists():
            passages = []
            with open(jsonl_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        passages.append(json.loads(line))
            return passages

    return []


def compute_file_embeddings(
    passages: list[dict],
    leann_searcher,
    base_dir: str,
) -> dict[str, np.ndarray]:
    """Compute file-level embeddings by averaging passage embeddings.

    Uses leann's backend to compute embeddings for passage texts,
    then averages per file.

    Args:
        passages: List of passage dicts.
        leann_searcher: LeannSearcher instance (has backend_impl).
        base_dir: Base directory for relative path computation.

    Returns:
        Dict mapping relative file path to averaged embedding vector.
    """
    base = Path(base_dir)

    # Group passage texts by file
    texts_by_file: dict[str, list[str]] = {}
    for p in passages:
        meta = p.get("metadata", {})
        file_path = meta.get("file_path")
        if not file_path:
            continue
        try:
            rel = str(PurePosixPath(Path(file_path).relative_to(base)))
        except ValueError:
            rel = PurePosixPath(file_path).name
        texts_by_file.setdefault(rel, []).append(p.get("text", ""))

    # Compute embeddings per file by averaging passage embeddings
    file_embeddings = {}
    backend = leann_searcher.backend_impl

    for file_id, texts in texts_by_file.items():
        # Use a representative sample (first 5 passages) to keep it fast
        sample_texts = texts[:5]
        combined = " ".join(sample_texts)[:2000]  # cap length

        try:
            embedding = backend.compute_query_embedding(
                combined, use_server_if_available=True
            )
            if embedding is not None and len(embedding) > 0:
                file_embeddings[file_id] = embedding.flatten()
        except Exception:
            continue

    return file_embeddings


def build_file_graph(
    leann_searcher,
    base_dir: str,
    top_k: int = 5,
    threshold: float = 0.6,
) -> dict:
    """Build the complete file graph from a leann index.

    Args:
        leann_searcher: Initialized LeannSearcher instance.
        base_dir: Absolute path of the indexed directory.
        top_k: Max similarity neighbors per file.
        threshold: Minimum similarity score.

    Returns:
        Dict with 'nodes' and 'edges' lists.
    """
    # Load all passages
    index_path = leann_searcher.meta_path_str.removesuffix(".meta.json")
    passages = load_passages_from_index(index_path)

    if not passages:
        return {"nodes": [], "edges": []}

    # Build nodes
    nodes = build_nodes(passages, base_dir)
    node_ids = {n["id"] for n in nodes}

    # Populate file sizes from disk
    base = Path(base_dir)
    for node in nodes:
        try:
            stat = (base / node["id"]).stat()
            node["size"] = stat.st_size
        except (OSError, ValueError):
            pass

    # Group passage texts by file for reference detection
    passages_by_file: dict[str, list[str]] = {}
    for p in passages:
        meta = p.get("metadata", {})
        file_path = meta.get("file_path")
        if not file_path:
            continue
        try:
            rel = str(PurePosixPath(Path(file_path).relative_to(base)))
        except ValueError:
            rel = PurePosixPath(file_path).name
        passages_by_file.setdefault(rel, []).append(p.get("text", ""))

    # Compute edges
    edges = []

    # 1. Similarity edges (needs embeddings)
    try:
        file_embeddings = compute_file_embeddings(passages, leann_searcher, base_dir)
        if len(file_embeddings) >= 2:
            edges.extend(compute_similarity_edges(file_embeddings, top_k, threshold))
    except Exception:
        pass  # similarity edges are best-effort

    # 2. Reference edges
    edges.extend(compute_reference_edges(passages_by_file, node_ids))

    # 3. Structure edges
    edges.extend(compute_structure_edges(list(node_ids)))

    return {"nodes": nodes, "edges": edges}
```

**Step 4: Run tests to verify they pass**

Run: `cd /Users/ded/Projects/assist/manole && python -m pytest tests/test_graph.py -v`
Expected: All tests PASS

**Step 5: Commit**

```bash
git add graph.py tests/test_graph.py
git commit -m "feat: add file graph computation module"
```

---

### Task 4: Add `getFileGraph` handler to server.py

**Files:**
- Modify: `server.py:365-389` (dispatch method)
- Test: `tests/test_server.py` (add new test class)

**Step 1: Write failing tests**

Add to `tests/test_server.py`:

```python
class TestGetFileGraph:
    """Test file graph handler."""

    def test_get_file_graph_unknown_directory(self):
        from server import Server
        srv = Server()
        srv.state = "ready"
        result = srv.handle_get_file_graph(1, {"directoryId": "nonexistent"})
        assert result["type"] == "error"
        assert "Unknown directory" in result["data"]["message"]

    def test_get_file_graph_not_ready(self):
        from server import Server
        srv = Server()
        srv.state = "ready"
        srv.directories["test"] = {
            "dir_id": "test",
            "state": "indexing",
            "path": "/tmp/test",
        }
        result = srv.handle_get_file_graph(1, {"directoryId": "test"})
        assert result["type"] == "error"
        assert "not ready" in result["data"]["message"]

    def test_get_file_graph_returns_graph_structure(self):
        from server import Server
        srv = Server()
        srv.state = "ready"
        mock_searcher = MagicMock()
        srv.directories["test"] = {
            "dir_id": "test",
            "state": "ready",
            "path": "/tmp/test",
            "searcher": mock_searcher,
        }
        with patch("server.build_file_graph") as mock_build:
            mock_build.return_value = {
                "nodes": [{"id": "a.pdf", "name": "a.pdf", "type": "pdf", "size": 100, "dir": "", "passageCount": 3}],
                "edges": [{"source": "a.pdf", "target": "b.pdf", "type": "similarity", "weight": 0.8}],
            }
            result = srv.handle_get_file_graph(1, {"directoryId": "test"})
            assert result["type"] == "result"
            assert "nodes" in result["data"]
            assert "edges" in result["data"]
            assert len(result["data"]["nodes"]) == 1
            mock_build.assert_called_once()

    def test_get_file_graph_uses_cache(self):
        from server import Server
        srv = Server()
        srv.state = "ready"
        cached_graph = {"nodes": [], "edges": []}
        srv.directories["test"] = {
            "dir_id": "test",
            "state": "ready",
            "path": "/tmp/test",
            "searcher": MagicMock(),
            "file_graph": cached_graph,
        }
        with patch("server.build_file_graph") as mock_build:
            result = srv.handle_get_file_graph(1, {"directoryId": "test"})
            assert result["type"] == "result"
            mock_build.assert_not_called()  # should use cache

    def test_dispatch_routes_get_file_graph(self):
        from server import Server
        srv = Server()
        srv.state = "ready"
        srv.directories["test"] = {
            "dir_id": "test",
            "state": "ready",
            "path": "/tmp/test",
            "searcher": MagicMock(),
        }
        with patch("server.build_file_graph") as mock_build:
            mock_build.return_value = {"nodes": [], "edges": []}
            result = srv.dispatch({"id": 1, "method": "getFileGraph", "params": {"directoryId": "test"}})
            assert result["type"] == "result"
```

**Step 2: Run tests to verify they fail**

Run: `cd /Users/ded/Projects/assist/manole && python -m pytest tests/test_server.py::TestGetFileGraph -v`
Expected: FAIL — `AttributeError: 'Server' object has no attribute 'handle_get_file_graph'`

**Step 3: Implement the handler**

Add import at the top of `server.py` (inside `handle_get_file_graph` to avoid import at module level):

Add the handler method to the `Server` class (after `handle_reindex`, around line 353):

```python
def handle_get_file_graph(self, req_id, params: dict) -> dict:
    """Compute and return the file relationship graph."""
    from graph import build_file_graph

    dir_id = params.get("directoryId")
    if not dir_id or dir_id not in self.directories:
        return {"id": req_id, "type": "error", "data": {"message": f"Unknown directory: {dir_id}"}}

    entry = self.directories[dir_id]
    if entry.get("state") != "ready":
        return {"id": req_id, "type": "error", "data": {"message": f"Directory not ready: {dir_id}"}}

    # Return cached graph if available
    if "file_graph" in entry:
        return {"id": req_id, "type": "result", "data": entry["file_graph"]}

    searcher = entry.get("searcher")
    if not searcher:
        return {"id": req_id, "type": "error", "data": {"message": "No searcher available"}}

    self._log(f"Computing file graph for {dir_id}...")
    graph = build_file_graph(searcher.leann, entry["path"])
    entry["file_graph"] = graph
    self._log(f"File graph: {len(graph['nodes'])} nodes, {len(graph['edges'])} edges")

    return {"id": req_id, "type": "result", "data": graph}
```

Add `"getFileGraph"` to the `handlers` dict in `dispatch()` (line ~379):

```python
"getFileGraph": lambda: self.handle_get_file_graph(req_id, params),
```

**Step 4: Run tests to verify they pass**

Run: `cd /Users/ded/Projects/assist/manole && python -m pytest tests/test_server.py -v`
Expected: All tests PASS (including new and existing)

**Step 5: Commit**

```bash
git add server.py tests/test_server.py
git commit -m "feat: add getFileGraph NDJSON handler with caching"
```

---

### Task 5: Create `useFileGraph` hook

**Files:**
- Create: `ui/src/hooks/useFileGraph.ts`

**Step 1: Implement the hook**

Create `ui/src/hooks/useFileGraph.ts`:

```typescript
import { useState, useCallback } from "react";
import type { FileGraphData } from "../lib/protocol";
import { usePython } from "./usePython";

interface UseFileGraphReturn {
  graph: FileGraphData | null;
  isLoading: boolean;
  error: string | null;
  fetchGraph: (directoryId: string) => Promise<void>;
  clearGraph: () => void;
}

export function useFileGraph(): UseFileGraphReturn {
  const { send } = usePython();
  const [graph, setGraph] = useState<FileGraphData | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchGraph = useCallback(
    async (directoryId: string) => {
      setIsLoading(true);
      setError(null);
      try {
        const result = await send("getFileGraph", { directoryId });
        if (result.type === "result") {
          setGraph(result.data as unknown as FileGraphData);
        } else if (result.type === "error") {
          setError((result.data as { message: string }).message);
        }
      } catch (err) {
        setError(String(err));
      } finally {
        setIsLoading(false);
      }
    },
    [send]
  );

  const clearGraph = useCallback(() => {
    setGraph(null);
    setError(null);
  }, []);

  return { graph, isLoading, error, fetchGraph, clearGraph };
}
```

**Step 2: Commit**

```bash
git add ui/src/hooks/useFileGraph.ts
git commit -m "feat: add useFileGraph hook for graph data fetching"
```

---

### Task 6: Create custom React Flow node component

**Files:**
- Create: `ui/src/components/FileGraphNode.tsx`

**Step 1: Implement the custom node**

Create `ui/src/components/FileGraphNode.tsx`:

```tsx
import { memo } from "react";
import { Handle, Position, type NodeProps } from "@xyflow/react";

const FILE_TYPE_COLORS: Record<string, string> = {
  pdf: "#c9943e",
  md: "#6aad6a",
  txt: "#a59888",
  py: "#7aa2d4",
  js: "#d4a85c",
  ts: "#d4a85c",
  json: "#a59888",
  csv: "#7aa2d4",
};

function getTypeColor(type: string): string {
  return FILE_TYPE_COLORS[type] || "#635a50";
}

interface FileNodeData {
  name: string;
  type: string;
  passageCount: number;
  dir: string;
  size: number;
  selected?: boolean;
  [key: string]: unknown;
}

function FileGraphNodeComponent({ data, selected }: NodeProps) {
  const { name, type, passageCount } = data as unknown as FileNodeData;
  const color = getTypeColor(type as string);
  const isSelected = selected || (data as unknown as FileNodeData).selected;

  return (
    <>
      <Handle type="target" position={Position.Top} className="!bg-transparent !border-0 !w-2 !h-2" />
      <div
        className={`
          px-3 py-2 rounded-lg border transition-all duration-200
          bg-bg-elevated font-sans text-xs
          ${isSelected
            ? "border-accent/50 bg-accent/[0.06] shadow-[0_0_16px_rgba(201,148,62,0.12)]"
            : "border-border hover:border-accent/30 hover:shadow-[0_0_12px_rgba(201,148,62,0.08)]"
          }
        `}
        style={{ minWidth: 80, maxWidth: 160 }}
      >
        {/* Type badge */}
        {type && (
          <span
            className="inline-block px-1.5 py-0.5 rounded font-mono text-[9px] uppercase mb-1"
            style={{
              backgroundColor: `${color}20`,
              color: color,
            }}
          >
            {type}
          </span>
        )}

        {/* Filename */}
        <div className="text-text-primary truncate leading-tight" title={name as string}>
          {name}
        </div>

        {/* Passage count */}
        <div className="mt-1 font-mono text-[9px] text-text-tertiary">
          {passageCount} {Number(passageCount) === 1 ? "passage" : "passages"}
        </div>
      </div>
      <Handle type="source" position={Position.Bottom} className="!bg-transparent !border-0 !w-2 !h-2" />
    </>
  );
}

export const FileGraphNode = memo(FileGraphNodeComponent);
```

**Step 2: Commit**

```bash
git add ui/src/components/FileGraphNode.tsx
git commit -m "feat: add custom FileGraphNode component for React Flow"
```

---

### Task 7: Create `FileGraphPanel` component

**Files:**
- Create: `ui/src/components/FileGraphPanel.tsx`

This is the main panel with tab bar, React Flow canvas, and bottom tray.

**Step 1: Implement the panel**

Create `ui/src/components/FileGraphPanel.tsx`:

```tsx
import { useState, useCallback, useEffect, useMemo } from "react";
import { motion, AnimatePresence } from "motion/react";
import {
  ReactFlow,
  Background,
  MiniMap,
  Controls,
  useNodesState,
  useEdgesState,
  type Node,
  type Edge,
  type NodeTypes,
  MarkerType,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import dagre from "@dagrejs/dagre";
import { FileGraphNode } from "./FileGraphNode";
import type { FileGraphData, FileNode, FileEdge } from "../lib/protocol";

type TabId = "similarity" | "reference" | "structure";

const TABS: { id: TabId; label: string }[] = [
  { id: "similarity", label: "Similarity" },
  { id: "reference", label: "References" },
  { id: "structure", label: "Structure" },
];

const nodeTypes: NodeTypes = {
  file: FileGraphNode,
};

const FILE_TYPE_COLORS: Record<string, string> = {
  pdf: "#c9943e",
  md: "#6aad6a",
  txt: "#a59888",
  py: "#7aa2d4",
  js: "#d4a85c",
  ts: "#d4a85c",
};

function getEdgeStyle(edge: FileEdge, tab: TabId): Partial<Edge> {
  if (tab === "similarity") {
    return {
      style: {
        stroke: "#c9943e",
        strokeOpacity: 0.15 + edge.weight * 0.45,
        strokeWidth: 1.5,
      },
      type: "default",
    };
  }
  if (tab === "reference") {
    return {
      style: { stroke: "#a59888", strokeWidth: 1.5 },
      type: "straight",
      markerEnd: { type: MarkerType.ArrowClosed, color: "#a59888", width: 12, height: 12 },
      label: edge.label,
      labelStyle: { fontSize: 9, fill: "#635a50" },
    };
  }
  // structure
  return {
    style: { stroke: "#2a2624", strokeWidth: 1, strokeDasharray: "4 4" },
    type: "smoothstep",
  };
}

function layoutNodes(
  nodes: Node[],
  edges: Edge[],
  tab: TabId,
): Node[] {
  if (nodes.length === 0) return nodes;

  const g = new dagre.graphlib.Graph();
  g.setDefaultEdgeLabel(() => ({}));

  const isTree = tab === "structure";
  g.setGraph({
    rankdir: isTree ? "TB" : "LR",
    nodesep: isTree ? 60 : 100,
    ranksep: isTree ? 80 : 150,
    edgesep: 40,
  });

  for (const node of nodes) {
    g.setNode(node.id, { width: 140, height: 70 });
  }
  for (const edge of edges) {
    g.setEdge(edge.source, edge.target);
  }

  dagre.layout(g);

  return nodes.map((node) => {
    const pos = g.node(node.id);
    return {
      ...node,
      position: { x: pos.x - 70, y: pos.y - 35 },
    };
  });
}

interface FileGraphPanelProps {
  graph: FileGraphData | null;
  isLoading: boolean;
  error: string | null;
  onFetchGraph: () => void;
}

function formatSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

export function FileGraphPanel({ graph, isLoading, error, onFetchGraph }: FileGraphPanelProps) {
  const [activeTab, setActiveTab] = useState<TabId>("similarity");
  const [threshold, setThreshold] = useState(0.6);
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);
  const [nodes, setNodes, onNodesChange] = useNodesState([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState([]);

  // Fetch graph on mount if not loaded
  useEffect(() => {
    if (!graph && !isLoading && !error) {
      onFetchGraph();
    }
  }, [graph, isLoading, error, onFetchGraph]);

  // Filter edges by active tab and threshold
  const filteredEdges = useMemo(() => {
    if (!graph) return [];
    return graph.edges.filter((e) => {
      if (e.type !== activeTab) return false;
      if (activeTab === "similarity" && e.weight < threshold) return false;
      return true;
    });
  }, [graph, activeTab, threshold]);

  // Convert graph data to React Flow format
  useEffect(() => {
    if (!graph) return;

    const connectedNodeIds = new Set<string>();
    for (const e of filteredEdges) {
      connectedNodeIds.add(e.source);
      connectedNodeIds.add(e.target);
    }

    const rfNodes: Node[] = graph.nodes
      .filter((n) => connectedNodeIds.has(n.id))
      .map((n) => ({
        id: n.id,
        type: "file",
        position: { x: 0, y: 0 },
        data: {
          name: n.name,
          type: n.type,
          passageCount: n.passageCount,
          dir: n.dir,
          size: n.size,
          selected: n.id === selectedNodeId,
        },
      }));

    const rfEdges: Edge[] = filteredEdges
      .filter((e) => connectedNodeIds.has(e.source) && connectedNodeIds.has(e.target))
      .map((e, i) => ({
        id: `${e.source}-${e.target}-${i}`,
        source: e.source,
        target: e.target,
        ...getEdgeStyle(e, activeTab),
      }));

    const laidOut = layoutNodes(rfNodes, rfEdges, activeTab);
    setNodes(laidOut);
    setEdges(rfEdges);
  }, [graph, filteredEdges, activeTab, selectedNodeId, setNodes, setEdges]);

  const selectedNode = graph?.nodes.find((n) => n.id === selectedNodeId);
  const selectedEdgeCount = selectedNodeId
    ? filteredEdges.filter((e) => e.source === selectedNodeId || e.target === selectedNodeId).length
    : 0;

  const handleNodeClick = useCallback((_: unknown, node: Node) => {
    setSelectedNodeId((prev) => (prev === node.id ? null : node.id));
  }, []);

  // Loading state
  if (isLoading) {
    return (
      <div className="flex flex-1 items-center justify-center">
        <div className="text-center">
          <motion.span
            animate={{ opacity: [0.3, 1, 0.3] }}
            transition={{ duration: 1.5, repeat: Infinity }}
            className="inline-block h-2 w-2 rounded-full bg-warning"
          />
          <p className="mt-3 font-display text-lg italic text-text-tertiary">
            Building map...
          </p>
        </div>
      </div>
    );
  }

  // Error state
  if (error) {
    return (
      <div className="flex flex-1 items-center justify-center">
        <div className="mx-4 px-4 py-2 rounded-lg border border-accent/40 bg-accent/10 text-accent text-sm font-sans">
          {error}
        </div>
      </div>
    );
  }

  // Empty state
  if (graph && graph.nodes.length === 0) {
    return (
      <div className="flex flex-1 items-center justify-center">
        <div className="text-center">
          <div className="font-display text-lg text-text-tertiary italic">No passages indexed</div>
          <p className="mt-1 font-sans text-xs text-text-tertiary/60">
            Index a folder to see its file graph
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="flex flex-col flex-1 min-h-0">
      {/* Tab bar */}
      <div className="flex items-center gap-4 px-5 h-9 border-b border-border shrink-0">
        {TABS.map((tab) => {
          const count = graph?.edges.filter((e) => e.type === tab.id).length ?? 0;
          return (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`relative font-mono text-[10px] uppercase tracking-widest transition-colors pb-2 ${
                activeTab === tab.id
                  ? "text-accent"
                  : "text-text-tertiary hover:text-text-secondary"
              }`}
            >
              {tab.label}
              {count > 0 && (
                <span className="ml-1 text-text-tertiary">{count}</span>
              )}
              {activeTab === tab.id && (
                <motion.div
                  layoutId="graph-tab-indicator"
                  className="absolute bottom-0 left-0 right-0 h-[2px] bg-accent"
                  transition={{ type: "spring", stiffness: 400, damping: 30 }}
                />
              )}
            </button>
          );
        })}
      </div>

      {/* React Flow canvas */}
      <div className="flex-1 min-h-0">
        <ReactFlow
          nodes={nodes}
          edges={edges}
          onNodesChange={onNodesChange}
          onEdgesChange={onEdgesChange}
          onNodeClick={handleNodeClick}
          nodeTypes={nodeTypes}
          fitView
          fitViewOptions={{ padding: 0.2 }}
          proOptions={{ hideAttribution: true }}
          style={{ background: "var(--color-bg-primary)" }}
        >
          <Background color="var(--color-border)" gap={24} size={1} />
          <Controls
            showInteractive={false}
            className="!bg-bg-secondary !border-border !rounded-lg !shadow-none [&>button]:!bg-bg-secondary [&>button]:!border-border [&>button]:!text-text-secondary [&>button:hover]:!bg-bg-elevated"
          />
          <MiniMap
            nodeColor={(node) => {
              const type = (node.data as Record<string, unknown>)?.type as string;
              return FILE_TYPE_COLORS[type] || "#635a50";
            }}
            maskColor="rgba(20, 18, 16, 0.85)"
            className="!bg-bg-secondary !border !border-border !rounded-lg"
            style={{ opacity: 0.6 }}
          />
        </ReactFlow>
      </div>

      {/* Bottom tray */}
      <AnimatePresence mode="wait">
        <motion.div
          key={selectedNodeId ?? "summary"}
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: 8 }}
          transition={{ duration: 0.15 }}
          className="border-t border-border bg-bg-secondary px-5 shrink-0"
        >
          {selectedNode ? (
            <div className="py-3 flex items-center gap-4">
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-2">
                  <span className="font-sans text-sm font-medium text-text-primary truncate">
                    {selectedNode.name}
                  </span>
                  <span className="font-mono text-[10px] text-text-tertiary">
                    {formatSize(selectedNode.size)}
                  </span>
                </div>
                <div className="mt-0.5 font-mono text-[10px] text-text-tertiary truncate">
                  {selectedNode.id}
                </div>
              </div>
              <div className="flex items-center gap-3 shrink-0">
                <span className="font-mono text-[10px] text-text-tertiary">
                  {selectedNode.passageCount} passages
                </span>
                <span className="font-mono text-[10px] text-text-tertiary">
                  {selectedEdgeCount} connections
                </span>
              </div>
            </div>
          ) : (
            <div className="py-2.5 flex items-center justify-between">
              <span className="font-mono text-[10px] text-text-tertiary">
                {nodes.length} files &middot; {edges.length} edges
              </span>
              {activeTab === "similarity" && (
                <div className="flex items-center gap-2">
                  <span className="font-mono text-[10px] text-text-tertiary">
                    Threshold: {threshold.toFixed(2)}
                  </span>
                  <input
                    type="range"
                    min="0.4"
                    max="1.0"
                    step="0.05"
                    value={threshold}
                    onChange={(e) => setThreshold(parseFloat(e.target.value))}
                    className="w-24 h-1 rounded-full appearance-none bg-bg-elevated [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-3 [&::-webkit-slider-thumb]:h-3 [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:bg-accent [&::-webkit-slider-thumb]:cursor-pointer"
                  />
                </div>
              )}
            </div>
          )}
        </motion.div>
      </AnimatePresence>
    </div>
  );
}
```

**Step 2: Commit**

```bash
git add ui/src/components/FileGraphPanel.tsx
git commit -m "feat: add FileGraphPanel with tabbed React Flow visualization"
```

---

### Task 8: Integrate graph panel into App.tsx with mode toggle

**Files:**
- Modify: `ui/src/App.tsx`

**Step 1: Add mode toggle and graph panel to the app**

Add imports at the top of `App.tsx`:

```typescript
import { FileGraphPanel } from "./components/FileGraphPanel";
import { useFileGraph } from "./hooks/useFileGraph";
```

Add state and hook inside the `App` component (after existing state declarations):

```typescript
const [mode, setMode] = useState<"chat" | "map">("chat");
const { graph, isLoading: graphLoading, error: graphError, fetchGraph, clearGraph } = useFileGraph();
```

Add a `handleFetchGraph` callback:

```typescript
const handleFetchGraph = useCallback(() => {
  if (activeDirectoryId) {
    fetchGraph(activeDirectoryId);
  }
}, [activeDirectoryId, fetchGraph]);
```

Clear graph when directory changes — add effect:

```typescript
useEffect(() => {
  clearGraph();
}, [activeDirectoryId, clearGraph]);
```

**Replace the header center section** (the path display around lines 182-198) with the mode toggle + path:

```tsx
<div className="flex items-center gap-3" style={{ WebkitAppRegion: "no-drag" } as React.CSSProperties}>
  {hasDirectories && isReady && (
    <div className="flex items-center h-7 rounded-full bg-bg-elevated border border-border p-0.5">
      {(["chat", "map"] as const).map((m) => (
        <button
          key={m}
          onClick={() => setMode(m)}
          className={`relative px-3 py-0.5 font-display text-xs rounded-full transition-colors ${
            mode === m ? "text-accent" : "text-text-tertiary hover:text-text-secondary"
          }`}
        >
          {mode === m && (
            <motion.div
              layoutId="mode-indicator"
              className="absolute inset-0 rounded-full bg-accent-muted"
              transition={{ type: "spring", stiffness: 400, damping: 30 }}
            />
          )}
          <span className="relative capitalize">{m === "chat" ? "Chat" : "Map"}</span>
        </button>
      ))}
    </div>
  )}
  {hasDirectories && (
    <button
      onClick={() => setSidePanelOpen((v) => !v)}
      className="font-mono text-xs text-text-tertiary hover:text-text-secondary transition-colors truncate max-w-[300px]"
      title={searchAll ? "Searching all indexes" : activeDirectory?.path}
    >
      {searchAll ? (
        <span className="flex items-center gap-1.5">
          <span className="inline-block h-1.5 w-1.5 rounded-full bg-accent" />
          All indexes
        </span>
      ) : (
        activeDirectory?.path
      )}
    </button>
  )}
  {/* Dev panel toggle */}
  <button
    onClick={() => setDevPanelOpen((v) => !v)}
    className={`flex items-center justify-center h-7 w-7 rounded-md transition-colors ${
      devPanelOpen
        ? "bg-accent/20 text-accent"
        : "text-text-tertiary hover:text-text-secondary hover:bg-bg-elevated"
    }`}
    aria-label="Toggle developer panel"
    title="Python output"
  >
    <svg width="14" height="14" viewBox="0 0 14 14" fill="none">
      <path
        d="M4.5 4L2 7l2.5 3M9.5 4L12 7l-2.5 3M8 2.5L6 11.5"
        stroke="currentColor"
        strokeWidth="1.3"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  </button>
</div>
```

**Replace the main content AnimatePresence** (lines ~240-271) to include graph view:

```tsx
<AnimatePresence mode="wait">
  {!hasDirectories ? (
    <ChatPanel
      key="welcome"
      messages={[]}
      isLoading={false}
      error={initError}
      onSend={handleSend}
      onOpenFolder={handleOpenFolder}
    />
  ) : isInitializing ? (
    <LoadingScreen key="loading" backendState={backendState} />
  ) : isReady && mode === "map" ? (
    <motion.div
      key="map"
      initial={{ opacity: 0, filter: "blur(4px)" }}
      animate={{ opacity: 1, filter: "blur(0px)" }}
      exit={{ opacity: 0, filter: "blur(4px)" }}
      transition={{ duration: 0.3, ease: [0.22, 1, 0.36, 1] }}
      className="flex flex-col flex-1 min-h-0"
    >
      <FileGraphPanel
        graph={graph}
        isLoading={graphLoading}
        error={graphError}
        onFetchGraph={handleFetchGraph}
      />
    </motion.div>
  ) : isReady ? (
    <ChatPanel
      key={`chat-${activeDirectoryId}`}
      messages={messages}
      isLoading={isLoading}
      error={error}
      onSend={handleSend}
      onOpenFolder={handleOpenFolder}
    />
  ) : (
    <ChatPanel
      key="no-selection"
      messages={[]}
      isLoading={false}
      error={activeDirectory?.error ?? null}
      onSend={handleSend}
      onOpenFolder={handleOpenFolder}
    />
  )}
</AnimatePresence>
```

**Step 2: Verify build**

Run: `cd /Users/ded/Projects/assist/manole/ui && npm run build`
Expected: Build succeeds without TypeScript errors

**Step 3: Commit**

```bash
git add ui/src/App.tsx
git commit -m "feat: integrate file graph panel with Chat/Map mode toggle"
```

---

### Task 9: Add React Flow base CSS overrides

**Files:**
- Modify: `ui/src/assets/main.css`

**Step 1: Add React Flow theme overrides**

Append to `ui/src/assets/main.css` (after the existing styles):

```css
/* React Flow theme overrides — match forge & parchment palette */
.react-flow__node {
  cursor: pointer;
}

.react-flow__edge-path {
  stroke-linecap: round;
}

.react-flow__controls {
  gap: 2px;
}

.react-flow__controls button {
  width: 24px;
  height: 24px;
}

.react-flow__controls button svg {
  max-width: 12px;
  max-height: 12px;
}

.react-flow__minimap {
  border-radius: 8px;
}
```

**Step 2: Commit**

```bash
git add ui/src/assets/main.css
git commit -m "style: add React Flow theme overrides for Manole palette"
```

---

### Task 10: Clear graph cache on reindex

**Files:**
- Modify: `server.py` (handle_reindex method, line ~347-353)

**Step 1: Clear cached graph before re-indexing**

In `handle_reindex`, add cache invalidation before calling `handle_init`:

```python
def handle_reindex(self, req_id, params: dict) -> dict:
    """Re-index a previously added directory."""
    dir_id = params.get("directoryId")
    if not dir_id or dir_id not in self.directories:
        return {"id": req_id, "type": "error", "data": {"message": f"Unknown directory: {dir_id}"}}
    entry = self.directories[dir_id]
    entry.pop("file_graph", None)  # invalidate graph cache
    stored_path = entry["path"]
    return self.handle_init(req_id, {"dataDir": stored_path})
```

**Step 2: Add test**

Add to `tests/test_server.py` in a new or existing test class:

```python
def test_reindex_clears_graph_cache(self):
    from server import Server
    srv = Server()
    srv.directories["test"] = {
        "dir_id": "test",
        "state": "ready",
        "path": "/tmp/test",
        "file_graph": {"nodes": [], "edges": []},
        "conversation_history": [],
    }
    # Reindex will fail (no real directory) but cache should be cleared
    with patch.object(srv, "handle_init", return_value={"id": 1, "type": "result", "data": {}}):
        srv.handle_reindex(1, {"directoryId": "test"})
    assert "file_graph" not in srv.directories["test"]
```

**Step 3: Run tests**

Run: `cd /Users/ded/Projects/assist/manole && python -m pytest tests/test_server.py -v`
Expected: All PASS

**Step 4: Commit**

```bash
git add server.py tests/test_server.py
git commit -m "fix: invalidate file graph cache on reindex"
```

---

### Task 11: End-to-end verification

**Step 1: Run all Python tests**

Run: `cd /Users/ded/Projects/assist/manole && python -m pytest tests/ -v`
Expected: All tests PASS

**Step 2: Build frontend**

Run: `cd /Users/ded/Projects/assist/manole/ui && npm run build`
Expected: Build succeeds

**Step 3: Launch dev server for manual testing**

Run: `cd /Users/ded/Projects/assist/manole/ui && npm run dev`
Expected: Electron app launches. After indexing a directory:
- Header shows Chat/Map toggle
- Clicking "Map" shows the file graph panel
- Three tabs filter edges correctly
- Similarity tab has threshold slider
- Clicking a node shows details in bottom tray

**Step 4: Final commit if any fixes needed**

```bash
git add -A
git commit -m "fix: address e2e testing issues"
```
