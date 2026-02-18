"""Query rewriter — coreference resolution, term expansion, intent classification."""
from parser import parse_json

REWRITER_SYSTEM = (
    "You rewrite user queries for searching local files. "
    "Given a question and optional conversation history, produce JSON with:\n"
    '- "intent": one of "factual", "count", "list", "compare", "summarize", "metadata"\n'
    '  Use "count" when the user asks "how many" of something.\n'
    '  Use "metadata" for questions about file sizes, folder sizes, disk usage, storage space.\n'
    '- "search_query": expanded query optimized for vector search '
    "(add synonyms, related terms, full forms of abbreviations)\n"
    '- "resolved_query": rewrite the user\'s question with pronouns and references '
    "resolved. ALWAYS frame it as looking in the user's files — never answer from general knowledge.\n"
    "  For metadata queries: preserve whether the user asks about FILES or FOLDERS. "
    "If they ask about sizes or ranking, say 'sorted by size' explicitly. "
    "Example: 'top files by size' → 'List files sorted by file size'. "
    "'biggest folders' → 'Show folders sorted by size'.\n\n"
    "Examples:\n\n"
    'Question: "any invoices?"\n'
    '{"intent": "list", "search_query": "invoice receipt payment billing", '
    '"resolved_query": "Are there any invoices in my files?"}\n\n'
    'Question: "how many PDFs?"\n'
    '{"intent": "count", "search_query": "PDF files documents", '
    '"resolved_query": "How many PDF files are there?"}\n\n'
    'Question: "how many eggs in carbonara"\n'
    '{"intent": "factual", "search_query": "carbonara recipe eggs ingredients", '
    '"resolved_query": "How many eggs does the carbonara recipe call for according to my files?"}\n\n'
    'Question: "any cat drawings?"\n'
    '{"intent": "list", "search_query": "cat drawing sketch feline artwork illustration photo image", '
    '"resolved_query": "Are there any cat drawings or images of cats in my files?"}\n\n'
    "With conversation history:\n"
    'Recent conversation:\n'
    '  User: any animal pictures?\n'
    '  Assistant: I found some image files. Would you like details?\n'
    'Question: "yes"\n'
    '{"intent": "list", "search_query": "animal pictures images photos", '
    '"resolved_query": "Show me the animal pictures found in my files."}\n\n'
    'Question: "what folders take up the most space?"\n'
    '{"intent": "metadata", "search_query": "folder size space storage disk usage", '
    '"resolved_query": "Which folders take up the most space in my files?"}\n\n'
    'Question: "how much storage am I using?"\n'
    '{"intent": "metadata", "search_query": "total disk usage storage space", '
    '"resolved_query": "How much total storage space are my files using?"}\n\n'
    "IMPORTANT: Always search the user's files. Never answer from general knowledge.\n"
    "Reply with a single JSON object only."
)

_VALID_INTENTS = frozenset({"factual", "count", "list", "compare", "summarize", "metadata"})


def _fallback(query: str) -> dict:
    return {"intent": "factual", "search_query": query, "resolved_query": query}


class QueryRewriter:
    """Rewrites queries for better retrieval using LFM2.5-1.2B-Instruct."""

    def __init__(self, model, debug: bool = False):
        self.model = model
        self.debug = debug

    def rewrite(self, query: str, context: str = "") -> dict:
        user_msg = query
        if context:
            user_msg = f"{context}\n\nQuestion: {query}"

        messages = [
            {"role": "system", "content": REWRITER_SYSTEM},
            {"role": "user", "content": user_msg},
        ]
        if self.debug:
            print(f"  [REWRITE] Sending to model: {user_msg[:120]!r}")
        raw = self.model.generate(messages, max_tokens=256)

        if self.debug:
            print(f"  [REWRITE] Raw: {raw}")

        parsed = parse_json(raw, debug=self.debug)
        if parsed is None:
            if self.debug:
                print("  [REWRITE] Parse failed, using raw query")
            return _fallback(query)

        intent = parsed.get("intent", "factual")
        if intent not in _VALID_INTENTS:
            intent = "factual"

        search_query = parsed.get("search_query") or query
        resolved_query = parsed.get("resolved_query") or query

        result = {
            "intent": intent,
            "search_query": search_query,
            "resolved_query": resolved_query,
        }

        if self.debug:
            print(f"  [REWRITE] {result}")

        return result
