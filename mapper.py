"""Stage 3: Mapper — per-chunk fact extraction using 1.2B-RAG model."""
from parser import parse_json

MAP_PROMPT = (
    "Does this text answer the question? Extract facts or say not relevant.\n\n"
    "Question: {query}\n"
    "Text: {chunk_text}\n\n"
    'If relevant, output: {{"relevant": true, "facts": ["fact1", "fact2"]}}\n'
    'If NOT relevant, output: {{"relevant": false, "facts": []}}\n\n'
    "JSON:\n"
)


class Mapper:
    """Extracts facts from individual chunks using the 1.2B-RAG model."""

    def __init__(self, models, debug: bool = False):
        self.models = models
        self.debug = debug

    def map_chunk(self, query: str, chunk) -> dict:
        prompt = MAP_PROMPT.format(query=query, chunk_text=chunk.text[:500])
        raw = self.models.extract(prompt)
        parsed = parse_json(raw)
        source = chunk.metadata.get("source", chunk.id)
        if parsed is None:
            if self.debug:
                print(f"  [MAP] {source}: parse failed, treating as relevant")
            return {"relevant": True, "facts": [chunk.text[:200]], "source": source}
        result = {
            "relevant": parsed.get("relevant", True),
            "facts": parsed.get("facts", []),
            "source": source,
        }
        if self.debug:
            print(f"  [MAP] {source}: relevant={result['relevant']}, facts={len(result['facts'])}")
        return result

    def extract_facts(self, query: str, chunks: list) -> list[dict]:
        if self.debug:
            print(f"  [MAP] Processing {len(chunks)} chunks...")
        return [self.map_chunk(query, chunk) for chunk in chunks]
