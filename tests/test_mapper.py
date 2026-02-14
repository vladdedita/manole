"""Tests for Mapper — per-chunk fact extraction using 1.2B-RAG model."""
import json
from dataclasses import dataclass, field
from mapper import Mapper, MAP_PROMPT


@dataclass
class FakeSearchResult:
    id: str
    text: str
    score: float
    metadata: dict = field(default_factory=dict)


class FakeModelManager:
    def __init__(self, responses: list[str]):
        self.responses = list(responses)
        self.prompts = []

    def extract(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.responses.pop(0) if self.responses else ""


def test_map_prompt_contains_placeholders():
    assert "{query}" in MAP_PROMPT
    assert "{chunk_text}" in MAP_PROMPT


def test_map_relevant_chunk():
    response = json.dumps({"relevant": True, "facts": ["Invoice #123", "Amount: $50"]})
    models = FakeModelManager([response])
    mapper = Mapper(models)
    chunk = FakeSearchResult(id="0", text="Invoice #123 for $50", score=0.9, metadata={"source": "a.pdf"})
    result = mapper.map_chunk("find invoices", chunk)
    assert result["relevant"] is True
    assert "Invoice #123" in result["facts"]
    assert result["source"] == "a.pdf"


def test_map_irrelevant_chunk():
    response = json.dumps({"relevant": False, "facts": []})
    models = FakeModelManager([response])
    mapper = Mapper(models)
    chunk = FakeSearchResult(id="0", text="Nice weather", score=0.3, metadata={"source": "b.txt"})
    result = mapper.map_chunk("find invoices", chunk)
    assert result["relevant"] is False
    assert result["facts"] == []


def test_map_garbage_defaults_to_relevant():
    models = FakeModelManager(["I have no idea"])
    mapper = Mapper(models)
    chunk = FakeSearchResult(id="0", text="Some text here", score=0.5, metadata={})
    result = mapper.map_chunk("query", chunk)
    assert result["relevant"] is True
    assert chunk.text[:200] in result["facts"]


def test_map_chunk_truncates_long_text():
    models = FakeModelManager([json.dumps({"relevant": True, "facts": ["fact"]})])
    mapper = Mapper(models)
    long_text = "x" * 1000
    chunk = FakeSearchResult(id="0", text=long_text, score=0.9, metadata={})
    mapper.map_chunk("query", chunk)
    assert len(models.prompts[0]) < len(long_text) + len(MAP_PROMPT)


def test_extract_facts_multiple_chunks():
    responses = [
        json.dumps({"relevant": True, "facts": ["fact1"]}),
        json.dumps({"relevant": False, "facts": []}),
        json.dumps({"relevant": True, "facts": ["fact2", "fact3"]}),
    ]
    models = FakeModelManager(responses)
    mapper = Mapper(models)
    chunks = [
        FakeSearchResult(id=str(i), text=f"text{i}", score=0.9, metadata={"source": f"f{i}.pdf"})
        for i in range(3)
    ]
    mapped = mapper.extract_facts("query", chunks)
    assert len(mapped) == 3
    assert mapped[0]["relevant"] is True
    assert mapped[1]["relevant"] is False
    assert mapped[2]["facts"] == ["fact2", "fact3"]
