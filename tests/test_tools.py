"""Tests for ToolRegistry — tool dispatch."""
import tempfile
from pathlib import Path
from tools import ToolRegistry, TOOL_DEFINITIONS


class FakeSearcher:
    def __init__(self, response="No results."):
        self.response = response
        self.last_query = None
        self.last_top_k = None

    def search_and_extract(self, query, top_k=5):
        self.last_query = query
        self.last_top_k = top_k
        return self.response


def _make_registry(search_response="No results."):
    tmp = tempfile.mkdtemp()
    (Path(tmp) / "a.pdf").write_text("pdf")
    (Path(tmp) / "b.txt").write_text("txt")
    from toolbox import ToolBox
    searcher = FakeSearcher(search_response)
    toolbox = ToolBox(tmp)
    return ToolRegistry(searcher, toolbox), searcher, tmp


def test_tool_definitions_has_seven_tools():
    names = [t["name"] for t in TOOL_DEFINITIONS]
    assert len(names) == 7
    assert "semantic_search" in names
    assert "count_files" in names
    assert "respond" in names


def test_semantic_search_delegates():
    registry, searcher, _ = _make_registry("From budget.txt:\n  - Budget: $100k")
    result = registry.execute("semantic_search", {"query": "budget", "top_k": 3})
    assert searcher.last_query == "budget"
    assert searcher.last_top_k == 3
    assert "budget" in result.lower()


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
    result = registry.execute("count_files", {"extension": "pdf"})
    assert "1" in result


def test_list_files():
    registry, _, _ = _make_registry()
    result = registry.execute("list_files", {})
    assert "a.pdf" in result or "b.txt" in result


def test_file_metadata():
    registry, _, _ = _make_registry()
    result = registry.execute("file_metadata", {"name_hint": "a.pdf"})
    assert "a.pdf" in result


def test_grep_files():
    registry, _, _ = _make_registry()
    result = registry.execute("grep_files", {"pattern": "pdf"})
    assert "a.pdf" in result


def test_directory_tree():
    registry, _, _ = _make_registry()
    result = registry.execute("directory_tree", {"max_depth": 1})
    assert "a.pdf" in result or "b.txt" in result


def test_unknown_tool():
    registry, _, _ = _make_registry()
    result = registry.execute("nonexistent_tool", {})
    assert "Unknown tool" in result
