"""Tests for fallback router — thin safety net."""
from router import route, _detect_extension


def test_tree_query():
    name, params = route("show me the directory structure")
    assert name == "directory_tree"
    assert params["max_depth"] == 2


def test_folder_query():
    name, params = route("what folders do I have?")
    assert name == "directory_tree"


def test_file_metadata_query():
    name, params = route("when was invoice.pdf modified?")
    assert name == "file_metadata"
    assert params.get("name_hint") is not None


def test_file_size_query():
    name, params = route("what is the file size of budget.pdf?")
    assert name == "file_metadata"


def test_semantic_search_default():
    name, params = route("what is the target revenue?")
    assert name == "semantic_search"
    assert params["query"] == "what is the target revenue?"


def test_semantic_search_for_content_questions():
    name, params = route("find all invoices with amounts over $100")
    assert name == "semantic_search"


def test_count_falls_through_to_semantic():
    """Router no longer handles count — model should decide."""
    name, _ = route("how many PDF files do I have?")
    assert name == "semantic_search"


def test_list_falls_through_to_semantic():
    """Router no longer handles list — model should decide."""
    name, _ = route("list files")
    assert name == "semantic_search"


def test_detect_extension_pdf():
    assert _detect_extension("how many PDF files") == "pdf"


def test_detect_extension_pdfs_plural():
    assert _detect_extension("how many pdfs") == "pdf"


def test_detect_extension_txt():
    assert _detect_extension("count text files") == "txt"


def test_detect_extension_none():
    assert _detect_extension("how many files") is None


def test_intent_ignored():
    """Intent param is accepted but no longer drives routing."""
    name, _ = route("how many pdf files do I have?", intent="count")
    assert name == "semantic_search"


def test_metadata_intent_folder_query():
    name, params = route("what folders are biggest?", intent="metadata")
    assert name == "folder_stats"
    assert params["sort_by"] == "size"


def test_space_keyword_routes_to_folder_stats():
    name, params = route("what takes up the most space?")
    assert name == "folder_stats"


def test_largest_folder_query():
    name, params = route("which folder is the largest?")
    assert name == "folder_stats"


def test_disk_usage_query():
    name, params = route("total disk usage overview")
    assert name == "disk_usage"


def test_storage_summary_query():
    name, params = route("give me a storage summary")
    assert name == "disk_usage"


def test_top_files_by_size_routes_to_list_files():
    """'top 3 files by size' should use list_files, not folder_stats."""
    name, params = route("what are the top 3 files by size?", intent="metadata")
    assert name == "list_files"
    assert params["sort_by"] == "size"


def test_biggest_files_routes_to_list_files():
    name, params = route("what are the biggest files?")
    assert name == "list_files"
    assert params["sort_by"] == "size"


def test_largest_files_routes_to_list_files():
    name, params = route("show me the largest files")
    assert name == "list_files"
    assert params["sort_by"] == "size"
