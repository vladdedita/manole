"""Searcher — vector search with internal map-filter."""
from parser import parse_json

MAP_SYSTEM = (
    "You are a data extraction assistant. The user will give you a question and a text passage. "
    "Decide if the text DIRECTLY answers the question. "
    "If the text is about a DIFFERENT topic, set relevant to false. "
    "If relevant, extract the specific data points as short factual strings.\n\n"
    "Example 1:\n"
    'Question: What is the invoice total?\n'
    'Text: Invoice #123, Amount: $500, Due: Jan 15\n'
    '{"relevant": true, "facts": ["Invoice #123", "Amount: $500", "Due: Jan 15"]}\n\n'
    "Example 2:\n"
    'Question: What is the invoice total?\n'
    'Text: Meeting notes: discussed new hire onboarding and team lunch plans\n'
    '{"relevant": false, "facts": []}\n\n'
    "Example 3:\n"
    'Question: any macbook invoice?\n'
    'Text: Sprint review: deployed helm charts, updated CI pipeline, new model released\n'
    '{"relevant": false, "facts": []}\n\n'
    'Reply with JSON only: {"relevant": true/false, "facts": [...]}'
)

MAX_FACTS_PER_CHUNK = 10

_STOPWORDS = frozenset({
    "a", "an", "the", "is", "was", "are", "were", "be", "been",
    "do", "does", "did", "has", "have", "had", "it", "its",
    "of", "for", "in", "on", "to", "at", "by", "my", "me",
    "what", "when", "where", "how", "who", "which", "any",
    "and", "or", "not", "no", "but", "if", "so", "can",
    "all", "each", "every", "this", "that", "there", "here",
    "from", "with", "about", "into", "over", "after", "before",
    "show", "find", "get", "tell", "give", "list",
})


def extract_keywords(query: str) -> list[str]:
    """Extract searchable keywords from a query string."""
    words = query.lower().replace("?", "").replace("!", "").replace(".", "").split()
    return [w for w in words if w not in _STOPWORDS and len(w) > 2]


class Searcher:
    """LeannSearcher wrapper with internal fact extraction."""

    def __init__(self, leann_searcher, model, debug: bool = False):
        self.leann = leann_searcher
        self.model = model
        self.debug = debug

    def search_and_extract(self, query: str, top_k: int = 5) -> str:
        """Search + map-filter in one call. Returns formatted facts string."""
        chunks = self.leann.search(query, top_k=top_k)
        if not chunks:
            return "No matching content found."

        # Score pre-filter: drop chunks well below the top score
        if len(chunks) > 1:
            threshold = chunks[0].score * 0.85
            before = len(chunks)
            chunks = [c for c in chunks if c.score >= threshold]
            if self.debug and len(chunks) < before:
                print(f"  [SEARCH] Score filter: {len(chunks)}/{before} above {threshold:.2f}")

        # Map: extract facts per chunk
        facts_by_source = {}
        for chunk in chunks:
            extracted = self._extract_facts(query, chunk)
            if extracted["relevant"] and extracted["facts"]:
                source = self._get_source(chunk)
                facts_by_source.setdefault(source, []).extend(extracted["facts"])

        if not facts_by_source:
            return "Search returned results but none were relevant to the query."

        # Format for agent context
        lines = []
        for source, facts in facts_by_source.items():
            lines.append(f"From {source}:")
            for fact in facts:
                lines.append(f"  - {fact}")
        return "\n".join(lines)

    def _extract_facts(self, query: str, chunk) -> dict:
        """Ask the model if this chunk is relevant and extract facts."""
        source = self._get_source(chunk)
        meta = chunk.metadata or {}
        context_parts = [f"File: {source}"]
        for key, val in meta.items():
            if key not in ("source", "file_name", "file_path", "id") and val is not None:
                context_parts.append(f"{key}: {val}")
        context_parts.append(f"Chunk: {chunk.id}")
        context_parts.append(f"Relevance score: {chunk.score:.2f}")
        context = " | ".join(context_parts)

        messages = [
            {"role": "system", "content": MAP_SYSTEM},
            {"role": "user", "content": f"Question: {query}\n\n[{context}]\n{chunk.text[:1200]}"},
        ]
        raw = self.model.generate(messages, max_tokens=256)

        if self.debug:
            print(f"  [SEARCH] {source}: raw={raw[:100]}")

        parsed = parse_json(raw)
        if parsed is None:
            if self.debug:
                print(f"  [SEARCH] {source}: parse failed, treating as irrelevant")
            return {"relevant": False, "facts": []}

        facts = parsed.get("facts", [])
        if not isinstance(facts, list):
            facts = []
        facts = [self._normalize_fact(f) for f in facts[:MAX_FACTS_PER_CHUNK]]
        facts = [f for f in facts if f is not None]

        relevant = parsed.get("relevant", False)
        if self.debug:
            print(f"  [SEARCH] {source}: relevant={relevant}, facts={len(facts)}")

        return {"relevant": relevant, "facts": facts}

    @staticmethod
    def _get_source(chunk) -> str:
        meta = chunk.metadata or {}
        return (
            meta.get("source")
            or meta.get("file_name")
            or meta.get("file")
            or meta.get("filename")
            or chunk.id
        )

    @staticmethod
    def _normalize_fact(f) -> str | None:
        result = None
        if isinstance(f, str) and f.strip():
            result = f.strip()
        elif isinstance(f, dict):
            name = f.get("name", "")
            value = f.get("value", "")
            if name and value:
                result = f"{name}: {value}"
            else:
                vals = [str(v) for v in f.values() if v]
                if vals:
                    result = ": ".join(vals)
        if result and len(result) >= 3:
            return result
        return None
