"""Stage 0: Query Rewriter — coreference resolution, term expansion, intent classification."""
from parser import parse_json

REWRITER_SYSTEM = (
    "You rewrite user queries for document search. "
    "Given a question and optional conversation history, produce JSON with:\n"
    "- \"intent\": one of \"factual\", \"count\", \"list\", \"compare\", \"summarize\"\n"
    "- \"search_query\": expanded query optimized for vector search "
    "(add synonyms, related terms, full forms of abbreviations)\n"
    "- \"resolved_query\": rewrite the user's question with pronouns and references resolved into a clear, self-contained question\n\n"
    "Examples:\n\n"
    "Question: \"any invoices?\"\n"
    "{\"intent\": \"list\", \"search_query\": \"invoice receipt payment billing\", "
    "\"resolved_query\": \"Are there any invoices?\"}\n\n"
    "Question: \"how many PDFs?\"\n"
    "{\"intent\": \"count\", \"search_query\": \"PDF files documents\", "
    "\"resolved_query\": \"How many PDF files are there?\"}\n\n"
    "Reply with a single JSON object only."
)

_VALID_INTENTS = frozenset({"factual", "count", "list", "compare", "summarize"})


def _fallback(query: str) -> dict:
    return {"intent": "factual", "search_query": query, "resolved_query": query}


class QueryRewriter:
    """Rewrites queries for better retrieval using LFM2.5-1.2B-Instruct."""

    def __init__(self, models, debug: bool = False):
        self.models = models
        self.debug = debug

    def rewrite(self, query: str, context: str = "") -> dict:
        user_msg = query
        if context:
            user_msg = f"{context}\n\nQuestion: {query}"

        raw = self.models.rewrite(REWRITER_SYSTEM, user_msg)

        if self.debug:
            print(f"  [REWRITE] Raw: {raw}")

        parsed = parse_json(raw)
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
