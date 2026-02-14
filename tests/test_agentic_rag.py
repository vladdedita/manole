"""Tests for AgenticRAG pipeline using mock LLM and searcher."""
import json
from dataclasses import dataclass
from unittest.mock import MagicMock


@dataclass
class FakeSearchResult:
    """Mimics leann SearchResult."""
    id: str
    text: str
    score: float
    metadata: dict


class FakeLLM:
    """Records prompts and returns canned responses."""
    def __init__(self, responses: list[str]):
        self.responses = list(responses)
        self.prompts = []

    def ask(self, prompt: str, **kwargs) -> str:
        self.prompts.append(prompt)
        if self.responses:
            return self.responses.pop(0)
        return ""


class FakeSearcher:
    """Returns pre-configured search results."""
    def __init__(self, results: list[FakeSearchResult]):
        self.results = results
        self.last_filters = None

    def search(self, query, top_k=7, metadata_filters=None, **kwargs):
        self.last_filters = metadata_filters
        return self.results[:top_k]


def test_placeholder():
    """Verify test infrastructure works."""
    assert FakeLLM(["hello"]).ask("hi") == "hello"
    assert FakeSearcher([]).search("q") == []


import sys
sys.path.insert(0, "/Users/ded/Projects/assist/manole")
from chat import parse_json, PLANNER_PROMPT, MAP_PROMPT, REDUCE_PROMPT


def test_parse_json_valid():
    result = parse_json('{"relevant": true, "facts": ["Invoice #123"]}')
    assert result == {"relevant": True, "facts": ["Invoice #123"]}


def test_parse_json_with_surrounding_text():
    result = parse_json('Here is the JSON:\n{"relevant": false, "facts": []}\nDone.')
    assert result == {"relevant": False, "facts": []}


def test_parse_json_malformed_fallback_relevant():
    result = parse_json('"relevant": true, some garbage')
    assert result["relevant"] is True


def test_parse_json_malformed_fallback_not_relevant():
    result = parse_json('The answer is "relevant": false and nothing else')
    assert result["relevant"] is False


def test_parse_json_total_garbage():
    result = parse_json("I don't understand the question")
    assert result is None


def test_planner_prompt_formats():
    result = PLANNER_PROMPT.format(query="find invoices")
    assert "find invoices" in result
    assert "keywords" in result


def test_map_prompt_formats():
    result = MAP_PROMPT.format(query="find invoices", chunk_text="Invoice #123")
    assert "find invoices" in result
    assert "Invoice #123" in result


def test_reduce_prompt_formats():
    result = REDUCE_PROMPT.format(facts_list="- Invoice #123", query="find invoices")
    assert "Invoice #123" in result


from chat import AgenticRAG


def _make_results(*texts):
    """Helper: create FakeSearchResults from text strings."""
    return [
        FakeSearchResult(id=str(i), text=t, score=0.9 - i * 0.1, metadata={"source": f"file{i}.pdf"})
        for i, t in enumerate(texts)
    ]


def test_planner_extracts_filters():
    planner_response = json.dumps({
        "keywords": ["invoice", "anthropic"],
        "file_filter": "pdf",
        "source_hint": "Invoice",
    })
    llm = FakeLLM([planner_response])
    searcher = FakeSearcher(_make_results("chunk1"))
    rag = AgenticRAG(searcher, llm, top_k=3, debug=False)

    plan = rag._plan("find my Anthropic invoices")
    assert plan["source_hint"] == "Invoice"
    assert plan["file_filter"] == "pdf"


def test_planner_bad_json_returns_empty_plan():
    llm = FakeLLM(["I don't understand"])
    searcher = FakeSearcher([])
    rag = AgenticRAG(searcher, llm, top_k=3, debug=False)

    plan = rag._plan("find invoices")
    assert plan["keywords"] == []
    assert plan["source_hint"] is None
    assert plan["file_filter"] is None


def test_search_applies_metadata_filters():
    planner_response = json.dumps({
        "keywords": ["invoice"],
        "file_filter": None,
        "source_hint": "Invoice",
    })
    llm = FakeLLM([planner_response])
    results = _make_results("Invoice #123")
    searcher = FakeSearcher(results)
    rag = AgenticRAG(searcher, llm, top_k=5, debug=False)

    plan = rag._plan("invoices")
    search_results = rag._search("invoices", plan)
    assert searcher.last_filters == {"source": {"contains": "Invoice"}}
    assert len(search_results) == 1
