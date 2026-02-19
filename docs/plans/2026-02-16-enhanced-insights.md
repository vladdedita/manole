# Enhanced Directory Insights Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace shallow LLM-from-stats summary with rich deterministic stats + index-queried semantic summary, and add tooltip for full summary text.

**Architecture:** Enrich `_collect_stats` with size-by-type, largest files, avg size, dir info. Replace `_generate_summary(stats)` with `_generate_summary(dir_id)` that queries the LEANN index via `searcher.search_and_extract`. Frontend: update `DirectoryEntry` type, render new stats, add hover tooltip on summary.

**Tech Stack:** Python (server.py), TypeScript/React (SidePanel.tsx, protocol.ts)

---

### Task 1: Enhance `_collect_stats` backend

**Files:**
- Modify: `server.py:63-75` (`_collect_stats` method)
- Test: `tests/test_server.py:302-319` (`TestCollectStats`)

**Step 1: Write failing tests for enhanced stats**

Add to `tests/test_server.py` in `TestCollectStats`:

```python
def test_collect_stats_enhanced(self, tmp_path):
    from server import Server
    srv = Server()
    # Create test files with known sizes
    (tmp_path / "big.pdf").write_bytes(b"x" * 10000)
    (tmp_path / "small.pdf").write_bytes(b"x" * 500)
    (tmp_path / "notes.md").write_text("hello world")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "deep").mkdir()
    (tmp_path / "sub" / "deep" / "data.csv").write_text("a,b,c")
    (tmp_path / "sub" / "readme.md").write_text("readme")

    stats = srv._collect_stats(tmp_path)

    # Existing fields still work
    assert stats["fileCount"] == 5
    assert stats["types"]["pdf"] == 2
    assert stats["types"]["md"] == 2
    assert stats["types"]["csv"] == 1

    # New: sizeByType
    assert "sizeByType" in stats
    assert stats["sizeByType"]["pdf"] == 10500
    assert stats["sizeByType"]["md"] > 0

    # New: largestFiles (top 3)
    assert "largestFiles" in stats
    assert len(stats["largestFiles"]) <= 3
    assert stats["largestFiles"][0]["name"] == "big.pdf"
    assert stats["largestFiles"][0]["size"] == 10000

    # New: avgFileSize
    assert "avgFileSize" in stats
    assert stats["avgFileSize"] == stats["totalSize"] // stats["fileCount"]

    # New: dirs
    assert "dirs" in stats
    assert stats["dirs"]["count"] == 2  # sub, sub/deep
    assert stats["dirs"]["maxDepth"] == 2
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/ded/Projects/assist/manole && .venv/bin/python -m pytest tests/test_server.py::TestCollectStats::test_collect_stats_enhanced -v`
Expected: FAIL — `sizeByType` key missing

**Step 3: Implement enhanced `_collect_stats`**

Replace `server.py:63-75`:

```python
def _collect_stats(self, data_dir: Path) -> dict:
    """Walk a directory and return file statistics."""
    file_count = 0
    total_size = 0
    types: dict[str, int] = {}
    size_by_type: dict[str, int] = {}
    largest_files: list[dict] = []
    dir_count = 0
    max_depth = 0

    base_depth = len(data_dir.parts)
    for p in data_dir.rglob("*"):
        if p.is_dir():
            dir_count += 1
            depth = len(p.parts) - base_depth
            if depth > max_depth:
                max_depth = depth
            continue
        if p.is_file():
            file_size = p.stat().st_size
            file_count += 1
            total_size += file_size
            ext = p.suffix.lstrip(".").lower()
            if ext:
                types[ext] = types.get(ext, 0) + 1
                size_by_type[ext] = size_by_type.get(ext, 0) + file_size
            largest_files.append({"name": p.name, "size": file_size})

    largest_files.sort(key=lambda f: f["size"], reverse=True)
    largest_files = largest_files[:3]

    return {
        "fileCount": file_count,
        "totalSize": total_size,
        "types": types,
        "sizeByType": size_by_type,
        "largestFiles": largest_files,
        "avgFileSize": total_size // file_count if file_count else 0,
        "dirs": {"count": dir_count, "maxDepth": max_depth},
    }
```

**Step 4: Run tests to verify they pass**

Run: `cd /Users/ded/Projects/assist/manole && .venv/bin/python -m pytest tests/test_server.py::TestCollectStats -v`
Expected: PASS (both old and new tests)

**Step 5: Commit**

```bash
git add server.py tests/test_server.py
git commit -m "feat: enrich _collect_stats with sizeByType, largestFiles, avgFileSize, dirs"
```

---

### Task 2: Replace `_generate_summary` to use index

**Files:**
- Modify: `server.py:77-89` (`_generate_summary` method)
- Modify: `server.py:203-216` (call site in `handle_init`)
- Test: `tests/test_server.py:279-299` (`TestDirectorySummary`)

**Step 1: Write failing test for index-based summary**

Replace tests in `TestDirectorySummary`:

```python
def test_generate_summary_with_index(self):
    from server import Server
    srv = Server()
    mock_model = MagicMock()
    mock_model.generate.return_value = "  A project proposal with architecture docs.  "
    srv.model = mock_model

    mock_searcher = MagicMock()
    mock_searcher.search_and_extract.return_value = "Found: proposal.md, architecture decisions, PRD documents"

    srv.directories["test"] = {
        "dir_id": "test",
        "searcher": mock_searcher,
        "state": "ready",
    }

    summary = srv._generate_summary("test")
    assert summary == "A project proposal with architecture docs."
    mock_searcher.search_and_extract.assert_called_once()
    mock_model.generate.assert_called_once()
    # Verify the model prompt includes the extracted facts
    prompt_messages = mock_model.generate.call_args[0][0]
    assert "Found: proposal.md" in prompt_messages[0]["content"]

def test_generate_summary_without_model(self):
    from server import Server
    srv = Server()
    srv.model = None
    srv.directories["test"] = {"dir_id": "test", "searcher": MagicMock(), "state": "ready"}
    summary = srv._generate_summary("test")
    assert summary == ""

def test_generate_summary_missing_directory(self):
    from server import Server
    srv = Server()
    srv.model = MagicMock()
    summary = srv._generate_summary("nonexistent")
    assert summary == ""
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/ded/Projects/assist/manole && .venv/bin/python -m pytest tests/test_server.py::TestDirectorySummary -v`
Expected: FAIL — `_generate_summary` still takes `stats` arg

**Step 3: Implement index-based `_generate_summary`**

Replace `server.py:77-89`:

```python
def _generate_summary(self, dir_id: str) -> str:
    """Query the index to generate a content-aware summary."""
    if not self.model:
        return ""
    entry = self.directories.get(dir_id)
    if not entry or "searcher" not in entry:
        return ""
    searcher = entry["searcher"]
    facts = searcher.search_and_extract(
        "What are the main topics, purpose, and content of these documents?",
        top_k=5,
    )
    prompt = (
        f"Based on these document excerpts:\n{facts}\n\n"
        "In 2-3 sentences, describe what this collection of documents is about. "
        "Be specific about the project or domain. Be concise."
    )
    messages = [{"role": "user", "content": prompt}]
    result = self.model.generate(messages)
    return (result or "").strip()
```

**Step 4: Update call site in `handle_init`**

In `server.py`, change the summary generation block (~line 203-216) — replace `self._generate_summary(stats)` with `self._generate_summary(dir_id)`:

```python
        # Generate summary (may be slow) and send as a follow-up update
        try:
            self._log(f"Generating summary for {dir_id}...")
            summary = self._generate_summary(dir_id)
            self._log(f"Summary result: {repr(summary[:100]) if summary else '(empty)'}")
            self.directories[dir_id]["summary"] = summary
            if summary:
                send(None, "directory_update", {
                    "directoryId": dir_id, "state": "ready",
                    "stats": stats, "summary": summary,
                })
                self._log(f"Sent directory_update with summary for {dir_id}")
        except Exception as exc:
            self._log(f"Summary generation failed: {exc}")
```

**Step 5: Run tests**

Run: `cd /Users/ded/Projects/assist/manole && .venv/bin/python -m pytest tests/test_server.py::TestDirectorySummary -v`
Expected: PASS

**Step 6: Commit**

```bash
git add server.py tests/test_server.py
git commit -m "feat: generate summary by querying LEANN index instead of stats"
```

---

### Task 3: Update frontend types for enhanced stats

**Files:**
- Modify: `ui/src/lib/protocol.ts:46-52` (`DirectoryUpdateData`)
- Modify: `ui/src/components/SidePanel.tsx:4-15` (`DirectoryEntry` interface)
- Modify: `ui/src/App.tsx:27-33` (subscriber type cast)

**Step 1: Update `DirectoryUpdateData` in protocol.ts**

Replace `ui/src/lib/protocol.ts:49`:

```typescript
export interface DirectoryUpdateData {
  directoryId: string;
  state: "indexing" | "ready" | "error";
  stats?: DirectoryStats;
  summary?: string;
  error?: string;
}

export interface DirectoryStats {
  fileCount: number;
  totalSize: number;
  types: Record<string, number>;
  sizeByType: Record<string, number>;
  largestFiles: { name: string; size: number }[];
  avgFileSize: number;
  dirs: { count: number; maxDepth: number };
}
```

**Step 2: Update `DirectoryEntry` in SidePanel.tsx**

Replace `ui/src/components/SidePanel.tsx:4-15`:

```typescript
import type { DirectoryStats } from "../lib/protocol";

export interface DirectoryEntry {
  id: string;
  path: string;
  state: "indexing" | "ready" | "error";
  stats?: DirectoryStats;
  summary?: string;
  error?: string;
}
```

**Step 3: Update subscriber type in App.tsx**

In `ui/src/App.tsx:27-33`, update the stats type in the cast to reference `DirectoryStats`:

```typescript
import type { Response, DirectoryStats } from "./lib/protocol";
// ...
const data = response.data as {
  directoryId: string;
  state: "indexing" | "ready" | "error";
  stats?: DirectoryStats;
  summary?: string;
  error?: string;
};
```

**Step 4: Verify TypeScript compiles**

Run: `cd /Users/ded/Projects/assist/manole/ui && npx tsc --noEmit`
Expected: Clean compile

**Step 5: Commit**

```bash
git add ui/src/lib/protocol.ts ui/src/components/SidePanel.tsx ui/src/App.tsx
git commit -m "feat: update frontend types for enhanced directory stats"
```

---

### Task 4: Render enhanced stats in SidePanel

**Files:**
- Modify: `ui/src/components/SidePanel.tsx:30-34` (`formatSize` helper)
- Modify: `ui/src/components/SidePanel.tsx:149-169` (stats rendering block)

**Step 1: Add `formatPercent` helper**

After `formatSize` in SidePanel.tsx:

```typescript
function formatPercent(part: number, total: number): string {
  if (total === 0) return "0%";
  const pct = Math.round((part / total) * 100);
  return pct < 1 ? "<1%" : `${pct}%`;
}
```

**Step 2: Replace stats rendering block**

Replace `ui/src/components/SidePanel.tsx:149-169` with:

```tsx
{entry.state === "ready" && entry.stats && (
  <motion.div
    initial={{ opacity: 0, y: 4 }}
    animate={{ opacity: 1, y: 0 }}
    transition={{ delay: 0.1, duration: 0.3 }}
    className="mt-2 space-y-1.5"
  >
    {/* Type badges */}
    <div className="flex flex-wrap items-center gap-1.5">
      {Object.entries(entry.stats.types).map(([ext, count]) => (
        <span
          key={ext}
          className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded bg-bg-elevated font-mono text-[10px] text-text-secondary"
          title={`${formatSize(entry.stats!.sizeByType[ext] ?? 0)} (${formatPercent(entry.stats!.sizeByType[ext] ?? 0, entry.stats!.totalSize)})`}
        >
          <span className="text-accent">{count}</span>
          <span className="uppercase">{ext}</span>
        </span>
      ))}
    </div>

    {/* Size + structure row */}
    <div className="flex items-center gap-2 font-mono text-[10px] text-text-tertiary">
      <span>{formatSize(entry.stats.totalSize)}</span>
      <span className="text-border">|</span>
      <span>{entry.stats.dirs.count} {entry.stats.dirs.count === 1 ? "folder" : "folders"}</span>
      {entry.stats.dirs.maxDepth > 0 && (
        <>
          <span className="text-border">|</span>
          <span>{entry.stats.dirs.maxDepth} deep</span>
        </>
      )}
    </div>
  </motion.div>
)}
```

**Step 3: Verify TypeScript compiles**

Run: `cd /Users/ded/Projects/assist/manole/ui && npx tsc --noEmit`
Expected: Clean compile

**Step 4: Commit**

```bash
git add ui/src/components/SidePanel.tsx
git commit -m "feat: render enhanced stats with size tooltips and dir structure"
```

---

### Task 5: Add summary hover tooltip

**Files:**
- Modify: `ui/src/components/SidePanel.tsx:194-204` (summary rendering block)

**Step 1: Add tooltip state and rendering**

Replace the summary block in `DirectoryCard` (`ui/src/components/SidePanel.tsx:194-204`):

```tsx
{entry.summary && (
  <div className="relative group/summary">
    <motion.p
      initial={{ opacity: 0, y: 4 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: 0.2, duration: 0.4 }}
      className="mt-2 font-sans text-[11px] leading-relaxed text-text-tertiary line-clamp-3 cursor-default"
    >
      {entry.summary}
    </motion.p>
    {/* Tooltip — full summary on hover */}
    <div className="absolute left-0 bottom-full mb-1 w-64 p-3 rounded-lg border border-border bg-bg-elevated shadow-lg font-sans text-[11px] leading-relaxed text-text-secondary opacity-0 pointer-events-none group-hover/summary:opacity-100 group-hover/summary:pointer-events-auto transition-opacity duration-200 z-50">
      {entry.summary}
    </div>
  </div>
)}
```

**Step 2: Verify TypeScript compiles**

Run: `cd /Users/ded/Projects/assist/manole/ui && npx tsc --noEmit`
Expected: Clean compile

**Step 3: Commit**

```bash
git add ui/src/components/SidePanel.tsx
git commit -m "feat: add hover tooltip to show full summary text"
```

---

### Task 6: Run full test suite

**Step 1: Run all backend tests**

Run: `cd /Users/ded/Projects/assist/manole && .venv/bin/python -m pytest tests/test_server.py -v`
Expected: All tests pass

**Step 2: Run TypeScript check**

Run: `cd /Users/ded/Projects/assist/manole/ui && npx tsc --noEmit`
Expected: Clean compile

**Step 3: Final commit if any fixups needed**
