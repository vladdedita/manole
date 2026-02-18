"""Tests for ToolRegistry — tool dispatch."""
import tempfile
from pathlib import Path
from unittest.mock import MagicMock
from tools import ToolRegistry, TOOL_DEFINITIONS


class FakeSearcher:
    def __init__(self, response="No results.", sources=None):
        self.response = response
        self.sources = sources if sources is not None else []
        self.last_query = None
        self.last_top_k = None

    def search_and_extract(self, query, top_k=5):
        self.last_query = query
        self.last_top_k = top_k
        return (self.response, self.sources)


def _make_registry(search_response="No results.", search_sources=None):
    tmp = tempfile.mkdtemp()
    (Path(tmp) / "a.pdf").write_text("pdf")
    (Path(tmp) / "b.txt").write_text("txt")
    from toolbox import ToolBox
    searcher = FakeSearcher(search_response, search_sources)
    toolbox = ToolBox(tmp)
    return ToolRegistry(searcher, toolbox), searcher, tmp


def test_tool_definitions_has_nine_tools():
    """Updated: 7 original + 2 new = 9."""
    names = [t["name"] for t in TOOL_DEFINITIONS]
    assert len(names) == 9
    assert "folder_stats" in names
    assert "disk_usage" in names


def test_semantic_search_delegates():
    registry, searcher, _ = _make_registry("From budget.txt:\n  - Budget: $100k")
    text, sources = registry.execute("semantic_search", {"query": "budget", "top_k": 3})
    assert searcher.last_query == "budget"
    assert searcher.last_top_k == 3
    assert "budget" in text.lower()


def test_semantic_search_default_top_k():
    registry, searcher, _ = _make_registry()
    registry.execute("semantic_search", {"query": "test"})
    assert searcher.last_top_k == 5


def test_semantic_search_caps_top_k():
    registry, searcher, _ = _make_registry()
    registry.execute("semantic_search", {"query": "test", "top_k": 50})
    assert searcher.last_top_k == 10


def test_count_files():
    registry, _, _ = _make_registry()
    text, sources = registry.execute("count_files", {"extension": "pdf"})
    assert "1" in text
    assert sources == []


def test_list_files():
    registry, _, _ = _make_registry()
    text, sources = registry.execute("list_files", {})
    assert "a.pdf" in text or "b.txt" in text
    assert sources == []


def test_file_metadata():
    registry, _, _ = _make_registry()
    text, sources = registry.execute("file_metadata", {"name_hint": "a.pdf"})
    assert "a.pdf" in text
    assert sources == []


def test_grep_files():
    registry, _, _ = _make_registry()
    text, sources = registry.execute("grep_files", {"pattern": "pdf"})
    assert "a.pdf" in text
    assert sources == []


def test_directory_tree():
    registry, _, _ = _make_registry()
    text, sources = registry.execute("directory_tree", {"max_depth": 1})
    assert "a.pdf" in text or "b.txt" in text
    assert sources == []


def test_folder_stats_dispatch():
    registry, _, _ = _make_registry()
    text, sources = registry.execute("folder_stats", {"sort_by": "size"})
    assert "Folder" in text or "No files" in text
    assert sources == []


def test_disk_usage_dispatch():
    registry, _, _ = _make_registry()
    text, sources = registry.execute("disk_usage", {})
    assert "Disk" in text or "No files" in text
    assert sources == []


def test_list_files_with_sort_by():
    registry, _, _ = _make_registry()
    text, sources = registry.execute("list_files", {"sort_by": "size"})
    assert text  # should not crash
    assert sources == []


def test_unknown_tool():
    registry, _, _ = _make_registry()
    text, sources = registry.execute("nonexistent_tool", {})
    assert "Unknown tool" in text
    assert sources == []


def test_semantic_search_returns_sources():
    """execute() returns (text, sources) when semantic_search produces sources."""
    searcher = MagicMock()
    searcher.search_and_extract.return_value = ("From doc.pdf:\n  - fact", ["doc.pdf"])
    toolbox = MagicMock()
    registry = ToolRegistry(searcher, toolbox)

    text, sources = registry.execute("semantic_search", {"query": "test"})

    assert sources == ["doc.pdf"]
    assert "doc.pdf" in text
