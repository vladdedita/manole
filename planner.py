"""Stage 1: Planner — extract structured search parameters from user queries."""
from parser import parse_json

PLANNER_PROMPT = (
    "Extract search parameters from the user's question as JSON.\n\n"
    "Output fields:\n"
    '- "keywords": list of 2-4 search terms\n'
    '- "file_filter": file extension like "pdf", "txt", "py", or null\n'
    '- "source_hint": filename substring to filter by, or null\n'
    '- "tool": "semantic_search", "filesystem", or "hybrid"\n'
    '- "time_filter": "today", "this_week", "this_month", or null\n'
    '- "tool_actions": list from ["count", "list_recent", "metadata", "tree", "grep"], or []\n'
    "\n"
    "Examples:\n\n"
    'Question: "find my Anthropic invoices"\n'
    'JSON: {{"keywords": ["invoice", "Anthropic"], "file_filter": "pdf", '
    '"source_hint": "Invoice", "tool": "semantic_search", '
    '"time_filter": null, "tool_actions": []}}\n\n'
    'Question: "how many PDF files do I have?"\n'
    'JSON: {{"keywords": ["PDF", "files", "count"], "file_filter": "pdf", '
    '"source_hint": null, "tool": "filesystem", '
    '"time_filter": null, "tool_actions": ["count"]}}\n\n'
    'Question: "summarize files I modified today"\n'
    'JSON: {{"keywords": ["modified", "today", "summary"], "file_filter": null, '
    '"source_hint": null, "tool": "hybrid", '
    '"time_filter": "today", "tool_actions": ["list_recent"]}}\n\n'
    'Question: "what is my folder structure?"\n'
    'JSON: {{"keywords": ["folder", "structure", "directory"], "file_filter": null, '
    '"source_hint": null, "tool": "filesystem", '
    '"time_filter": null, "tool_actions": ["tree"]}}\n\n'
    'Question: "notes about machine learning"\n'
    'JSON: {{"keywords": ["machine learning", "notes", "AI"], "file_filter": null, '
    '"source_hint": null, "tool": "semantic_search", '
    '"time_filter": null, "tool_actions": []}}\n\n'
    "Question: {query}\n"
    "JSON:"
)

_DEFAULT_PLAN = {
    "keywords": [],
    "file_filter": None,
    "source_hint": None,
    "tool": "semantic_search",
    "time_filter": None,
    "tool_actions": [],
}


class Planner:
    """Extracts structured search plan from user queries using 350M-Extract model."""

    def __init__(self, models, debug: bool = False):
        self.models = models
        self.debug = debug

    def plan(self, query: str) -> dict:
        prompt = PLANNER_PROMPT.format(query=query)
        raw = self.models.plan(prompt)

        if self.debug:
            print(f"  [PLAN] Raw: {raw}")

        parsed = parse_json(raw)
        if parsed is None:
            if self.debug:
                print("  [PLAN] Parse failed, using defaults")
            return dict(_DEFAULT_PLAN)

        # Fill in missing fields with defaults
        result = {}
        for key, default in _DEFAULT_PLAN.items():
            result[key] = parsed.get(key, default)

        # Validate tool field
        if result["tool"] not in ("semantic_search", "filesystem", "hybrid"):
            result["tool"] = "semantic_search"

        # Validate time_filter
        if result["time_filter"] not in ("today", "this_week", "this_month", None):
            result["time_filter"] = None

        if self.debug:
            print(f"  [PLAN] {result}")

        return result
