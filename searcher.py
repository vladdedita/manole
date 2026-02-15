"""Searcher — vector search with internal map-filter."""
from parser import parse_json

MAP_SYSTEM = (
    "You are a data extraction assistant. The user will give you a question and a text passage. "
    "Extract any data points from the text that could help answer the question. "
    "Return them as short factual strings. If the text has nothing useful, return an empty list.\n\n"
    "Example 1:\n"
    'Question: What is the invoice total?\n'
    'Text: Invoice #123, Amount: $500, Due: Jan 15\n'
    '{"facts": ["Invoice #123", "Amount: $500", "Due: Jan 15"]}\n\n'
    "Example 2:\n"
    'Question: What is the invoice total?\n'
    'Text: Meeting notes: discussed new hire onboarding and team lunch plans\n'
    '{"facts": []}\n\n'
    'Reply with JSON only: {"facts": [...]}'
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
    # File format words — too broad for filename grep
    "pdf", "pdfs", "txt", "csv", "doc", "docx", "xls", "xlsx",
    "png", "jpg", "jpeg", "json", "xml", "file", "files",
    "document", "documents",
})


def extract_keywords(query: str) -> list[str]:
    """Extract searchable keywords from a query string."""
    words = query.lower().replace("?", "").replace("!", "").replace(".", "").split()
    return [w for w in words if w not in _STOPWORDS and len(w) > 2]


class Searcher:
    """LeannSearcher wrapper with internal fact extraction."""

    def __init__(self, leann_searcher, model, file_reader=None, toolbox=None, debug: bool = False):
        self.leann = leann_searcher
        self.model = model
        self.file_reader = file_reader
        self.toolbox = toolbox
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
        # Trust vector search scores for relevance — the model's job is fact extraction.
        # The 1.2B model often misjudges relevance but still extracts useful facts.
        facts_by_source = {}
        for chunk in chunks:
            extracted = self._extract_facts(query, chunk)
            if extracted["facts"]:
                source = self._get_source(chunk)
                facts_by_source.setdefault(source, []).extend(extracted["facts"])

        if not facts_by_source:
            if self.file_reader and self.toolbox:
                return self._filename_fallback(query)
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
                print(f"  [SEARCH] {source}: parse failed")
            return {"facts": []}

        # Model may return a list directly instead of {"facts": [...]}
        if isinstance(parsed, list):
            facts = parsed
        else:
            facts = parsed.get("facts", [])
        if not isinstance(facts, list):
            facts = []
        facts = [self._normalize_fact(f) for f in facts[:MAX_FACTS_PER_CHUNK]]
        facts = [f for f in facts if f is not None]

        if self.debug:
            print(f"  [SEARCH] {source}: facts={len(facts)}")

        return {"facts": facts}

    def _filename_fallback(self, query: str) -> str:
        """Grep filenames for query keywords, read matches, extract facts."""
        keywords = extract_keywords(query)
        if not keywords:
            return "Search returned results but none were relevant to the query."

        if self.debug:
            print(f"  [SEARCH] Filename fallback: keywords={keywords}")

        # Collect unique matching file paths across all keywords
        seen = set()
        matching_paths = []
        for keyword in keywords:
            for path in self.toolbox.grep_paths(keyword, limit=3):
                path_str = str(path)
                if path_str not in seen:
                    seen.add(path_str)
                    matching_paths.append(path)

        matching_paths = matching_paths[:3]  # cap total files

        if not matching_paths:
            if self.debug:
                print("  [SEARCH] Filename fallback: no matching files")
            return "Search returned results but none were relevant to the query."

        if self.debug:
            print(f"  [SEARCH] Filename fallback: reading {[p.name for p in matching_paths]}")

        # Read each file and extract facts.
        # Filename match already establishes relevance — always include extracted facts.
        facts_by_source = {}
        for path in matching_paths:
            text = self.file_reader.read(str(path))
            if text.startswith("File not found") or text.startswith("Failed to read") or text.startswith("No text content"):
                if self.debug:
                    print(f"  [SEARCH] Filename fallback: {path.name}: {text[:80]}")
                continue

            # Create a fake chunk-like object for _extract_facts
            fake_chunk = type("Chunk", (), {
                "id": path.name,
                "text": text,
                "score": 0.0,
                "metadata": {"file_name": path.name, "file_path": str(path)},
            })()

            extracted = self._extract_facts(query, fake_chunk)
            # Trust filename match as relevance — include facts even if model says irrelevant
            if extracted["facts"]:
                facts_by_source.setdefault(path.name, []).extend(extracted["facts"])
            else:
                # Model extracted nothing (e.g. foreign language text) — surface the file anyway
                facts_by_source.setdefault(path.name, []).append(f"File found: {path.name}")

        if not facts_by_source:
            return "Search returned results but none were relevant to the query."

        lines = []
        for source, facts in facts_by_source.items():
            lines.append(f"From {source}:")
            for fact in facts:
                lines.append(f"  - {fact}")
        return "\n".join(lines)

    @staticmethod
    def _get_source(chunk) -> str:
        meta = chunk.metadata or {}
        name = (
            meta.get("source")
            or meta.get("file_name")
            or meta.get("file")
            or meta.get("filename")
        )
        if not name and meta.get("file_path"):
            from pathlib import PurePosixPath
            name = PurePosixPath(meta["file_path"]).name
        return name or chunk.id

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
