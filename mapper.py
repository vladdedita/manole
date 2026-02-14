"""Stage 3: Mapper — per-chunk relevance + fact extraction."""
from parser import parse_json

MAP_SYSTEM = (
    "You are a data extraction assistant. The user will give you a question and a text passage. "
    "Decide if the text DIRECTLY answers the question. "
    "Set relevant to false unless the text specifically discusses the topic asked about. "
    "If relevant, extract the specific data points as short factual strings. "
    "Extract ALL matching data points, do not skip any. "
    "Reply with JSON: {\"relevant\": true/false, \"facts\": [\"fact1\", \"fact2\"]}"
)

EXTRACT_SYSTEM = (
    "You are a data extraction assistant. "
    "Extract specific data points from the text as short strings. "
    "Reply with JSON: {\"relevant\": true/false, \"facts\": [\"string1\", \"string2\"]}"
)

MAX_FACTS = 10


def _normalize_fact(f) -> str | None:
    """Convert a fact to a string. Handles dicts like {"name": "X", "value": "Y"}."""
    result = None
    if isinstance(f, str) and f.strip():
        result = f.strip()
    elif isinstance(f, dict):
        # Handle {"name": "Total Budget", "value": "$450,000"} style
        name = f.get("name", "")
        value = f.get("value", "")
        if name and value:
            result = f"{name}: {value}"
        else:
            # Fallback: join all values
            vals = [str(v) for v in f.values() if v]
            if vals:
                result = ": ".join(vals)
    # Discard facts that are too short to be meaningful (single numbers, letters)
    if result and len(result) >= 3:
        return result
    return None


def _parse_map_result(raw: str, source: str, chunk_text: str) -> dict | None:
    """Parse mapper output into result dict. Returns None on failure."""
    parsed = parse_json(raw)
    if parsed is None:
        return None
    facts = parsed.get("facts", [])
    if not isinstance(facts, list):
        facts = []
    facts = [s for f in facts[:MAX_FACTS] if (s := _normalize_fact(f)) is not None]
    return {
        "relevant": parsed.get("relevant", True),
        "facts": facts,
        "source": source,
    }


class Mapper:
    """Extracts facts from chunks. Primary: 1.2B-RAG, fallback: 350M-Extract."""

    def __init__(self, models, debug: bool = False):
        self.models = models
        self.debug = debug

    @staticmethod
    def _build_context(chunk) -> str:
        """Build context header from chunk metadata."""
        meta = chunk.metadata or {}
        parts = []
        source = meta.get("source", "")
        if source:
            parts.append(f"File: {source}")
        # Include any other metadata fields (page, chunk_index, etc.)
        for key, val in meta.items():
            if key != "source" and key != "id" and val is not None:
                parts.append(f"{key}: {val}")
        parts.append(f"Chunk: {chunk.id}")
        parts.append(f"Relevance score: {chunk.score:.2f}")
        return " | ".join(parts)

    def map_chunk(self, query: str, chunk) -> dict:
        meta = chunk.metadata or {}
        source = meta.get("source") or meta.get("file_name") or meta.get("file") or meta.get("filename") or chunk.id
        if self.debug:
            print(f"  [MAP] {chunk.id} metadata keys: {list(meta.keys())}")
        context = self._build_context(chunk)
        user_msg = f"Question: {query}\n\n[{context}]\n{chunk.text[:1200]}"

        # Primary: 1.2B-RAG
        raw = self.models.map_chunk(MAP_SYSTEM, user_msg)
        result = _parse_map_result(raw, source, chunk.text)
        if result is not None:
            if self.debug:
                print(f"  [MAP] {source}: relevant={result['relevant']}, facts={len(result['facts'])}: {result['facts']}")
            return result

        # Fallback: 350M-Extract
        if self.debug:
            print(f"  [MAP] {source}: 1.2B parse failed, trying 350M")
        raw = self.models.extract(EXTRACT_SYSTEM, user_msg)
        result = _parse_map_result(raw, source, chunk.text)
        if result is not None:
            if self.debug:
                print(f"  [MAP] {source}: 350M ok, relevant={result['relevant']}, facts={len(result['facts'])}: {result['facts']}")
            return result

        # Final fallback: treat as relevant with raw text
        if self.debug:
            print(f"  [MAP] {source}: both models failed, using raw text")
        return {"relevant": True, "facts": [chunk.text[:200]], "source": source}

    def extract_facts(self, query: str, chunks: list) -> list[dict]:
        if self.debug:
            print(f"  [MAP] Processing {len(chunks)} chunks...")
        return [self.map_chunk(query, chunk) for chunk in chunks]
