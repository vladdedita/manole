"""Tests for Mapper — per-chunk fact extraction with 1.2B/350M fallback."""
import json
from dataclasses import dataclass, field
from mapper import Mapper, MAX_FACTS


@dataclass
class FakeSearchResult:
    id: str
    text: str
    score: float
    metadata: dict = field(default_factory=dict)


class FakeModelManager:
    def __init__(self, map_responses=None, extract_responses=None):
        self.map_responses = list(map_responses or [])
        self.extract_responses = list(extract_responses or [])

    def map_chunk(self, system: str, user: str) -> str:
        return self.map_responses.pop(0) if self.map_responses else ""

    def extract(self, system: str, user: str) -> str:
        return self.extract_responses.pop(0) if self.extract_responses else ""


def test_map_relevant_chunk():
    response = json.dumps({"relevant": True, "facts": ["Invoice #123", "Amount: $50"]})
    models = FakeModelManager(map_responses=[response])
    mapper = Mapper(models)
    chunk = FakeSearchResult(id="0", text="Invoice #123 for $50", score=0.9, metadata={"source": "a.pdf"})
    result = mapper.map_chunk("find invoices", chunk)
    assert result["relevant"] is True
    assert "Invoice #123" in result["facts"]
    assert result["source"] == "a.pdf"


def test_map_irrelevant_chunk():
    response = json.dumps({"relevant": False, "facts": []})
    models = FakeModelManager(map_responses=[response])
    mapper = Mapper(models)
    chunk = FakeSearchResult(id="0", text="Nice weather", score=0.3, metadata={"source": "b.txt"})
    result = mapper.map_chunk("find invoices", chunk)
    assert result["relevant"] is False
    assert result["facts"] == []


def test_fallback_to_350m_on_1_2b_failure():
    """When 1.2B fails to parse, 350M-Extract is tried as fallback."""
    extract_response = json.dumps({"relevant": True, "facts": ["fallback fact"]})
    models = FakeModelManager(
        map_responses=["garbage output"],
        extract_responses=[extract_response],
    )
    mapper = Mapper(models)
    chunk = FakeSearchResult(id="0", text="Some text", score=0.5, metadata={"source": "c.pdf"})
    result = mapper.map_chunk("query", chunk)
    assert result["relevant"] is True
    assert result["facts"] == ["fallback fact"]


def test_raw_text_fallback_when_both_fail():
    """When both models fail, raw chunk text is used."""
    models = FakeModelManager(
        map_responses=["garbage"],
        extract_responses=["also garbage"],
    )
    mapper = Mapper(models)
    chunk = FakeSearchResult(id="0", text="Some important text", score=0.5, metadata={})
    result = mapper.map_chunk("query", chunk)
    assert result["relevant"] is True
    assert "Some important text" in result["facts"][0]


def test_facts_capped_at_max():
    many_facts = [f"fact{i}" for i in range(25)]
    response = json.dumps({"relevant": True, "facts": many_facts})
    models = FakeModelManager(map_responses=[response])
    mapper = Mapper(models)
    chunk = FakeSearchResult(id="0", text="text", score=0.9, metadata={})
    result = mapper.map_chunk("query", chunk)
    assert len(result["facts"]) == MAX_FACTS


def test_map_chunk_truncates_long_text():
    models = FakeModelManager(map_responses=[json.dumps({"relevant": True, "facts": ["fact"]})])
    mapper = Mapper(models)
    long_text = "x" * 1000
    chunk = FakeSearchResult(id="0", text=long_text, score=0.9, metadata={})
    mapper.map_chunk("query", chunk)
    # No crash with long text


def test_extract_facts_multiple_chunks():
    responses = [
        json.dumps({"relevant": True, "facts": ["fact1"]}),
        json.dumps({"relevant": False, "facts": []}),
        json.dumps({"relevant": True, "facts": ["fact2", "fact3"]}),
    ]
    models = FakeModelManager(map_responses=responses)
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
