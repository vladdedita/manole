"""Tool definitions and registry for the agent loop."""

TOOL_DEFINITIONS = [
    {
        "name": "semantic_search",
        "description": (
            "Search inside file contents by meaning. "
            "Use when the user asks about information WITHIN files "
            "(invoices, budgets, notes, specific data). "
            "Returns extracted facts from matching file chunks."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "What to search for in file contents"},
                "top_k": {"type": "integer", "description": "Number of results (default 5, max 10)"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "count_files",
        "description": "Count how many files exist. Use for 'how many files/documents/PDFs' questions.",
        "parameters": {
            "type": "object",
            "properties": {
                "extension": {"type": "string", "description": "Filter by extension (e.g. 'pdf', 'txt') or null for all"},
            },
        },
    },
    {
        "name": "list_files",
        "description": "List files sorted by modification date. Use for 'what files', 'recent files', 'show me files' questions.",
        "parameters": {
            "type": "object",
            "properties": {
                "extension": {"type": "string", "description": "Filter by extension or null"},
                "limit": {"type": "integer", "description": "Max files to return (default 10)"},
                "sort_by": {"type": "string", "description": "'date' (default), 'size', or 'name'"},
            },
        },
    },
    {
        "name": "file_metadata",
        "description": "Get file size, creation date, modification date. Use for questions about specific file details.",
        "parameters": {
            "type": "object",
            "properties": {
                "name_hint": {"type": "string", "description": "Filename substring to match"},
            },
            "required": ["name_hint"],
        },
    },
    {
        "name": "grep_files",
        "description": "Find files by name pattern. Use when looking for files with specific names.",
        "parameters": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Substring to match in filenames"},
            },
            "required": ["pattern"],
        },
    },
    {
        "name": "directory_tree",
        "description": "Show folder structure. Use for 'what folders', 'directory structure', 'file organization' questions.",
        "parameters": {
            "type": "object",
            "properties": {
                "max_depth": {"type": "integer", "description": "How deep to show (default 2)"},
            },
        },
    },
    {
        "name": "folder_stats",
        "description": "Show folder sizes and file counts. Use for 'biggest folder', 'folder sizes', 'which folder has most files'.",
        "parameters": {
            "type": "object",
            "properties": {
                "sort_by": {"type": "string", "description": "'size' (default) or 'count'"},
                "limit": {"type": "integer", "description": "Max folders to show (default 10)"},
            },
        },
    },
    {
        "name": "disk_usage",
        "description": "Show total disk usage summary with breakdown by file type. Use for 'how much space', 'storage', 'disk usage'.",
        "parameters": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "respond",
        "description": (
            "Return a final answer to the user. "
            "ONLY call this when you have enough information to answer. "
            "If you need more information, call another tool first."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "answer": {"type": "string", "description": "Your complete answer to the user"},
            },
            "required": ["answer"],
        },
    },
]


class ToolRegistry:
    """Maps tool names to execution functions."""

    def __init__(self, searcher, toolbox):
        self.searcher = searcher
        self.toolbox = toolbox
        self._handlers = {
            "semantic_search": self._semantic_search,
            "count_files": self._count_files,
            "list_files": self._list_files,
            "file_metadata": self._file_metadata,
            "grep_files": self._grep_files,
            "directory_tree": self._directory_tree,
            "folder_stats": self._folder_stats,
            "disk_usage": self._disk_usage,
        }

    def execute(self, tool_name: str, params: dict) -> str:
        handler = self._handlers.get(tool_name)
        if not handler:
            return f"Unknown tool: {tool_name}"
        return handler(params)

    def _semantic_search(self, params: dict) -> str:
        query = params.get("query", "")
        top_k = min(params.get("top_k", 5), 10)
        return self.searcher.search_and_extract(query, top_k=top_k)

    def _count_files(self, params: dict) -> str:
        return self.toolbox.count_files(ext_filter=params.get("extension"))

    def _list_files(self, params: dict) -> str:
        return self.toolbox.list_recent_files(
            ext_filter=params.get("extension"),
            limit=params.get("limit", 10),
            sort_by=params.get("sort_by", "date"),
        )

    def _file_metadata(self, params: dict) -> str:
        return self.toolbox.get_file_metadata(name_hint=params.get("name_hint"))

    def _grep_files(self, params: dict) -> str:
        return self.toolbox.grep(params.get("pattern", ""))

    def _directory_tree(self, params: dict) -> str:
        return self.toolbox.tree(max_depth=params.get("max_depth", 2))

    def _folder_stats(self, params: dict) -> str:
        return self.toolbox.folder_stats(
            sort_by=params.get("sort_by", "size"),
            limit=params.get("limit", 10),
        )

    def _disk_usage(self, params: dict) -> str:
        return self.toolbox.disk_usage()
