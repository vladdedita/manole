"""Python fallback router — keyword heuristics for tool routing."""
import re

_EXT_MAP = {
    "pdf": "pdf", "txt": "txt", "text": "txt", "py": "py", "python": "py",
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


def route(query: str, intent: str | None = None) -> tuple[str, dict]:
    q = query.lower()

    # Intent-based routing (from rewriter) takes priority
    if intent == "count":
        return "count_files", {"extension": _detect_extension(q)}
    if intent == "list":
        ext = _detect_extension(q)
        if ext:
            return "list_files", {"extension": ext, "limit": 10}

    # Keyword-based fallback
    if any(k in q for k in ["how many", "count"]):
        return "count_files", {"extension": _detect_extension(q)}
    if any(k in q for k in ["file types", "folder", "tree", "directory", "structure"]):
        return "directory_tree", {"max_depth": 2}
    if any(k in q for k in ["list files", "recent files", "what files", "show files", "show me files", "list my"]):
        return "list_files", {"extension": _detect_extension(q), "limit": 10}
    if any(k in q for k in ["file size", "when was", "modified", "created", "how big", "how large", "how old"]):
        return "file_metadata", {"name_hint": _extract_name_hint(q)}
    return "semantic_search", {"query": query}
