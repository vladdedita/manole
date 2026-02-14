"""Tests for Python fallback router."""
from router import route, _detect_extension


def test_count_query_routes_to_count_files():
    name, params = route("how many PDF files do I have?")
    assert name == "count_files"
    assert params["extension"] == "pdf"


def test_count_query_no_extension():
    name, params = route("how many files?")
    assert name == "count_files"
    assert params["extension"] is None


def test_tree_query():
    name, params = route("show me the directory structure")
    assert name == "directory_tree"
    assert params["max_depth"] == 2


def test_folder_query():
    name, params = route("what folders do I have?")
    assert name == "directory_tree"


def test_list_files_query():
    name, params = route("list files")
    assert name == "list_files"


def test_recent_files_query():
    name, params = route("show me recent files")
    assert name == "list_files"


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


def test_detect_extension_pdf():
    assert _detect_extension("how many PDF files") == "pdf"


def test_detect_extension_txt():
    assert _detect_extension("count text files") == "txt"


def test_detect_extension_none():
    assert _detect_extension("how many files") is None


def test_case_insensitive():
    name, _ = route("HOW MANY PDF FILES?")
    assert name == "count_files"
