"""Fallback router — thin safety net for when model fails to produce a tool call."""
import re

_EXT_MAP = {
    "pdf": "pdf", "pdfs": "pdf",
    "txt": "txt", "text": "txt",
    "py": "py", "python": "py",
    "md": "md", "markdown": "md", "csv": "csv", "json": "json", "xml": "xml",
    "doc": "doc", "docx": "docx", "xls": "xls", "xlsx": "xlsx",
    "png": "png", "jpg": "jpg", "jpeg": "jpeg",
}


def _detect_extension(query: str) -> str | None:
    q = query.lower()
    for keyword, ext in _EXT_MAP.items():
        if keyword in q.split():
            return ext
    return None


def _extract_name_hint(query: str) -> str | None:
    match = re.search(r'[\w-]+\.\w{2,4}', query)
    if match:
        return match.group(0)
    match = re.search(r'["\']([^"\']+)["\']', query)
    if match:
        return match.group(1)
    words = query.lower().replace("?", "").split()
    stop = {"the", "a", "an", "is", "was", "of", "for", "my", "what", "when", "how", "file", "size"}
    nouns = [w for w in words if w not in stop and len(w) > 2]
    return nouns[-1] if nouns else None


def route(query: str, intent: str | None = None, debug: bool = False) -> tuple[str, dict]:
    q = query.lower()

    # Metadata queries: folder sizes, disk usage, storage
    size_keywords = ["space", "biggest", "largest", "storage", "heavy", "disk usage"]
    count_keywords = ["most", "least", "fewest"]
    is_metadata = intent == "metadata" or any(k in q for k in size_keywords)
    is_count_query = any(k in q for k in count_keywords)

    if is_metadata or is_count_query:
        if debug:
            matched_size = [k for k in size_keywords if k in q]
            matched_count = [k for k in count_keywords if k in q]
            print(f"  [ROUTER] Keywords matched: size={matched_size} count={matched_count} | intent={intent}")

        if any(k in q for k in ["total", "usage", "overview", "summary"]):
            if debug:
                print("  [ROUTER] → disk_usage()")
            return "disk_usage", {}

        # Distinguish file-level vs folder-level queries
        file_words = ["file", "files", "document", "documents"]
        folder_words = ["folder", "folders", "directory", "directories"]
        mentions_files = any(w in q for w in file_words)
        mentions_folders = any(w in q for w in folder_words)
        ext = _detect_extension(q)

        if debug:
            print(f"  [ROUTER] mentions_files={mentions_files} mentions_folders={mentions_folders} ext={ext}")

        # "folder with the most/least X files" → folder_stats with count + extension
        if is_count_query and mentions_folders:
            order = "asc" if any(k in q for k in ["least", "fewest"]) else "desc"
            params = {"sort_by": "count", "order": order}
            if ext:
                params["extension"] = ext
            if debug:
                print(f"  [ROUTER] → folder_stats({params})")
            return "folder_stats", params

        # "biggest/largest files" → list_files sorted by size
        if mentions_files and not mentions_folders:
            params = {"sort_by": "size"}
            if ext:
                params["extension"] = ext
            if debug:
                print(f"  [ROUTER] → list_files({params})")
            return "list_files", params

        if debug:
            print("  [ROUTER] → folder_stats(sort_by=size)")
        return "folder_stats", {"sort_by": "size"}

    # Unambiguous filesystem keywords only
    if any(k in q for k in ["folder", "tree", "directory", "structure"]):
        if debug:
            print("  [ROUTER] → directory_tree(max_depth=2)")
        return "directory_tree", {"max_depth": 2}
    if any(k in q for k in ["file size", "how big", "how large", "how old", "when was", "modified", "created"]):
        hint = _extract_name_hint(q)
        if debug:
            print(f"  [ROUTER] → file_metadata(name_hint={hint!r})")
        return "file_metadata", {"name_hint": hint}

    # Everything else → semantic search (model should have handled it)
    if debug:
        print(f"  [ROUTER] → semantic_search (fallback)")
    return "semantic_search", {"query": query}
