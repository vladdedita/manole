"""Stage 1: Planner — extract structured search parameters from user queries."""
from parser import parse_json

PLANNER_SYSTEM = (
    "You extract search parameters from questions as JSON. "
    "ALWAYS reply with JSON only, never answer the question.\n"
    "\n"
    "Output fields:\n"
    '- "keywords": list of 2-4 search terms from the question (include ALL important nouns)\n'
    '- "file_filter": file extension like "pdf", "txt", "py", or null\n'
    '- "source_hint": filename substring to filter by, or null\n'
    '- "tool": "semantic_search" or "filesystem"\n'
    "\n"
    "Rules:\n"
    "- Use filesystem ONLY for counting files, listing directories, or file metadata\n"
    "- Use semantic_search for everything else\n"
    "- Include ALL important nouns as keywords\n"
    "\n"
    "Examples:\n\n"
    'Question: "find invoices"\n'
    '{"keywords": ["invoice"], "file_filter": "pdf", '
    '"source_hint": null, "tool": "semantic_search"}\n\n'
    'Question: "how many PDF files do I have?"\n'
    '{"keywords": ["PDF", "files", "count"], "file_filter": "pdf", '
    '"source_hint": null, "tool": "filesystem"}\n\n'
    'Question: "what is the engineering department budget?"\n'
    '{"keywords": ["engineering", "department", "budget"], "file_filter": null, '
    '"source_hint": null, "tool": "semantic_search"}\n\n'
    "Reply with a single JSON object only. Do not explain."
)

_STOP_WORDS = frozenset(
    "a an the is are was were do does did in on at to for of and or any my"
    " how many what which where when who some all this that it me we have"
    " give list show find get".split()
)


def _keywords_from_query(query: str) -> list[str]:
    """Extract meaningful keywords from raw query text as fallback."""
    words = [w.strip("?!.,/") for w in query.lower().split()]
    return [w for w in words if w and w not in _STOP_WORDS][:4]


_DEFAULT_PLAN = {
    "keywords": [],
    "file_filter": None,
    "source_hint": None,
    "tool": "semantic_search",
}


class Planner:
    """Extracts structured search plan from user queries using 1.2B-RAG model."""

    def __init__(self, models, debug: bool = False):
        self.models = models
        self.debug = debug

    def plan(self, query: str, context: str = "") -> dict:
        user_msg = query
        if context:
            user_msg = f"{context}\n\nQuestion: {query}"
        raw = self.models.plan(PLANNER_SYSTEM, user_msg)

        if self.debug:
            print(f"  [PLAN] Raw: {raw}")

        parsed = parse_json(raw)
        if parsed is None:
            if self.debug:
                print("  [PLAN] Parse failed, using query keywords")
            fallback = dict(_DEFAULT_PLAN)
            fallback["keywords"] = _keywords_from_query(query)
            return fallback

        # Fill in missing fields with defaults
        result = {}
        for key, default in _DEFAULT_PLAN.items():
            result[key] = parsed.get(key, default)

        # Validate keywords — use query fallback if empty/invalid
        kw = result.get("keywords", [])
        if not isinstance(kw, list) or not all(isinstance(k, str) for k in kw):
            kw = []
        if not kw:
            kw = _keywords_from_query(query)
        result["keywords"] = kw

        # Validate tool field
        if result["tool"] not in ("semantic_search", "filesystem"):
            result["tool"] = "semantic_search"

        # Fix string "null" → None for nullable fields
        for key in ("file_filter", "source_hint"):
            if result.get(key) == "null":
                result[key] = None

        if self.debug:
            print(f"  [PLAN] {result}")

        return result
