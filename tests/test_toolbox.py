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


def test_grep_includes_count():
    """grep result should include explicit match count for small models."""
    tmp = _make_test_dir()
    tb = ToolBox(tmp)
    result = tb.grep("invoice")
    assert "Found 1" in result


def test_grep_no_match():
    tmp = _make_test_dir()
    tb = ToolBox(tmp)
    result = tb.grep("nonexistent")
    assert "No files" in result or "0" in result




def test_grep_paths_returns_path_objects():
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "invoice_001.pdf").write_text("inv1")
        (Path(tmp) / "invoice_002.pdf").write_text("inv2")
        (Path(tmp) / "readme.txt").write_text("readme")
        tb = ToolBox(tmp)
        paths = tb.grep_paths("invoice")
    assert len(paths) == 2
    assert all(isinstance(p, Path) for p in paths)
    names = {p.name for p in paths}
    assert "invoice_001.pdf" in names
    assert "invoice_002.pdf" in names


def test_grep_paths_no_match():
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "readme.txt").write_text("hello")
        tb = ToolBox(tmp)
        paths = tb.grep_paths("nonexistent")
    assert paths == []


def test_grep_paths_limit():
    with tempfile.TemporaryDirectory() as tmp:
        for i in range(10):
            (Path(tmp) / f"doc_{i}.txt").write_text(f"doc {i}")
        tb = ToolBox(tmp)
        paths = tb.grep_paths("doc", limit=3)
    assert len(paths) == 3
