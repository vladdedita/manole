"""Tests for Smart ToolBox — filesystem operations."""
import os
import tempfile
import time
from pathlib import Path
from toolbox import ToolBox


def _make_test_dir():
    """Create a temp directory with test files."""
    tmp = tempfile.mkdtemp()
    # Create files
    (Path(tmp) / "invoice.pdf").write_text("pdf content")
    (Path(tmp) / "notes.txt").write_text("some notes")
    (Path(tmp) / "code.py").write_text("print('hello')")
    # Create subdirectory
    sub = Path(tmp) / "subdir"
    sub.mkdir()
    (sub / "deep.pdf").write_text("deep pdf")
    return tmp


def test_count_all_files():
    tmp = _make_test_dir()
    tb = ToolBox(tmp)
    result = tb.count_files()
    assert "4" in result  # 4 files total


def test_count_with_extension_filter():
    tmp = _make_test_dir()
    tb = ToolBox(tmp)
    result = tb.count_files(ext_filter="pdf")
    assert "2" in result  # invoice.pdf + deep.pdf


def test_list_recent_files():
    tmp = _make_test_dir()
    tb = ToolBox(tmp)
    result = tb.list_recent_files()
    assert "invoice.pdf" in result
    assert "notes.txt" in result


def test_list_recent_with_limit():
    tmp = _make_test_dir()
    tb = ToolBox(tmp)
    result = tb.list_recent_files(limit=2)
    lines = [l for l in result.strip().split("\n") if l.strip().startswith("-")]
    assert result.count(".") >= 2


def test_list_recent_with_extension_filter():
    tmp = _make_test_dir()
    tb = ToolBox(tmp)
    result = tb.list_recent_files(ext_filter="pdf")
    assert "invoice.pdf" in result
    assert "notes.txt" not in result


def test_metadata_for_file():
    tmp = _make_test_dir()
    tb = ToolBox(tmp)
    result = tb.get_file_metadata(name_hint="invoice")
    assert "invoice.pdf" in result
    assert "KB" in result or "bytes" in result.lower()


def test_tree_structure():
    tmp = _make_test_dir()
    tb = ToolBox(tmp)
    result = tb.tree()
    assert "subdir" in result
    assert "invoice.pdf" in result


def test_tree_with_depth():
    tmp = _make_test_dir()
    tb = ToolBox(tmp)
    result = tb.tree(max_depth=0)
    assert "subdir" in result
    assert "deep.pdf" not in result


def test_grep_filenames():
    tmp = _make_test_dir()
    tb = ToolBox(tmp)
    result = tb.grep("invoice")
    assert "invoice.pdf" in result


def test_grep_no_match():
    tmp = _make_test_dir()
    tb = ToolBox(tmp)
    result = tb.grep("nonexistent")
    assert "No files" in result or "0" in result


def test_execute_routes_count():
    tmp = _make_test_dir()
    tb = ToolBox(tmp)
    plan = {"tool_actions": ["count"], "file_filter": "pdf", "time_filter": None}
    result = tb.execute(plan)
    assert "2" in result


def test_execute_routes_list_recent():
    tmp = _make_test_dir()
    tb = ToolBox(tmp)
    plan = {"tool_actions": ["list_recent"], "file_filter": None, "time_filter": None}
    result = tb.execute(plan)
    assert "invoice.pdf" in result


def test_execute_routes_tree():
    tmp = _make_test_dir()
    tb = ToolBox(tmp)
    plan = {"tool_actions": ["tree"], "file_filter": None, "time_filter": None}
    result = tb.execute(plan)
    assert "subdir" in result


def test_get_matching_files_returns_paths():
    tmp = _make_test_dir()
    tb = ToolBox(tmp)
    plan = {"tool_actions": ["list_recent"], "file_filter": "pdf", "time_filter": None}
    paths = tb.get_matching_files(plan)
    assert all(isinstance(p, str) for p in paths)
    assert any("invoice.pdf" in p for p in paths)
    assert not any("notes.txt" in p for p in paths)


def test_time_filter_today():
    tmp = _make_test_dir()
    tb = ToolBox(tmp)
    plan = {"tool_actions": ["list_recent"], "file_filter": None, "time_filter": "today"}
    result = tb.execute(plan)
    assert "invoice.pdf" in result
