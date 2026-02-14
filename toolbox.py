"""Smart ToolBox: LLM-routed filesystem operations with time awareness."""
from datetime import datetime, timedelta
from pathlib import Path


class ToolBox:
    """Pure Python filesystem tools operating on the indexed data directory."""

    def __init__(self, data_dir: str):
        self.root = Path(data_dir)

    def _time_cutoff(self, time_filter: str | None) -> float | None:
        if not time_filter:
            return None
        now = datetime.now()
        if time_filter == "today":
            cutoff = now - timedelta(hours=24)
        elif time_filter == "this_week":
            cutoff = now - timedelta(days=7)
        elif time_filter == "this_month":
            cutoff = now - timedelta(days=30)
        else:
            return None
        return cutoff.timestamp()

    def _list_files(self, ext_filter: str | None = None, time_filter: str | None = None) -> list[Path]:
        pattern = f"**/*.{ext_filter}" if ext_filter else "**/*"
        files = [f for f in self.root.glob(pattern) if f.is_file() and not f.name.startswith(".")]
        cutoff = self._time_cutoff(time_filter)
        if cutoff:
            files = [f for f in files if f.stat().st_mtime >= cutoff]
        return files

    def count_files(self, ext_filter: str | None = None, time_filter: str | None = None) -> str:
        files = self._list_files(ext_filter, time_filter)
        label = f".{ext_filter} " if ext_filter else ""
        return f"Found {len(files)} {label}files."

    def list_recent_files(self, ext_filter: str | None = None, time_filter: str | None = None, limit: int = 10) -> str:
        files = self._list_files(ext_filter, time_filter)
        files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
        if not files:
            return "No matching files found."
        lines = []
        for f in files[:limit]:
            mtime = datetime.fromtimestamp(f.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
            rel = f.relative_to(self.root)
            lines.append(f"  - {rel} (modified: {mtime})")
        return "Recent files:\n" + "\n".join(lines)

    def get_file_metadata(self, name_hint: str | None = None) -> str:
        files = [f for f in self.root.rglob("*") if f.is_file() and not f.name.startswith(".")]
        if name_hint:
            files = [f for f in files if name_hint.lower() in f.name.lower()]
        if not files:
            return "No matching files found."
        lines = []
        for f in files[:10]:
            stat = f.stat()
            size_kb = stat.st_size / 1024
            mtime = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M")
            rel = f.relative_to(self.root)
            lines.append(f"  - {rel}: {size_kb:.1f}KB, modified {mtime}")
        return "File metadata:\n" + "\n".join(lines)

    def tree(self, max_depth: int | None = None) -> str:
        lines = [f"{self.root.name}/"]
        self._tree_recurse(self.root, "", 0, max_depth, lines)
        return "\n".join(lines)

    def _tree_recurse(self, path: Path, prefix: str, depth: int, max_depth: int | None, lines: list):
        entries = sorted(path.iterdir(), key=lambda e: (not e.is_dir(), e.name.lower()))
        entries = [e for e in entries if not e.name.startswith(".")]
        for i, entry in enumerate(entries):
            is_last = i == len(entries) - 1
            connector = "└── " if is_last else "├── "
            lines.append(f"{prefix}{connector}{entry.name}{'/' if entry.is_dir() else ''}")
            if entry.is_dir() and (max_depth is None or depth < max_depth):
                extension = "    " if is_last else "│   "
                self._tree_recurse(entry, prefix + extension, depth + 1, max_depth, lines)

    def grep(self, pattern: str) -> str:
        files = [f for f in self.root.rglob("*") if f.is_file() and not f.name.startswith(".")]
        matches = [f for f in files if pattern.lower() in f.name.lower()]
        if not matches:
            return f"No files matching '{pattern}'."
        lines = [f"  - {f.relative_to(self.root)}" for f in matches[:20]]
        return f"Files matching '{pattern}':\n" + "\n".join(lines)


    def grep_paths(self, pattern: str, limit: int = 20) -> list[Path]:
        """Find files by name pattern. Returns Path objects."""
        files = [f for f in self.root.rglob("*") if f.is_file() and not f.name.startswith(".")]
        matches = [f for f in files if pattern.lower() in f.name.lower()]
        return matches[:limit]
