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


def test_map_extracts_facts_from_relevant_chunk():
    map_response = json.dumps({"relevant": True, "facts": ["Invoice #123", "Amount: $50"]})
    llm = FakeLLM([map_response])
    searcher = FakeSearcher([])
    rag = AgenticRAG(searcher, llm, top_k=5, debug=False)

    chunk = FakeSearchResult(id="0", text="Invoice #123 for $50", score=0.9, metadata={})
    result = rag._map_chunk("find invoices", chunk)
    assert result["relevant"] is True
    assert "Invoice #123" in result["facts"]


def test_map_marks_irrelevant_chunk():
    map_response = json.dumps({"relevant": False, "facts": []})
    llm = FakeLLM([map_response])
    rag = AgenticRAG(FakeSearcher([]), llm, top_k=5, debug=False)

    chunk = FakeSearchResult(id="0", text="Nice weather today", score=0.3, metadata={})
    result = rag._map_chunk("find invoices", chunk)
    assert result["relevant"] is False


def test_map_garbage_response_defaults_to_relevant():
    llm = FakeLLM(["I have no idea what you want"])
    rag = AgenticRAG(FakeSearcher([]), llm, top_k=5, debug=False)

    chunk = FakeSearchResult(id="0", text="Some text", score=0.5, metadata={})
    result = rag._map_chunk("query", chunk)
    assert result["relevant"] is True
    assert chunk.text in result["facts"]


def test_filter_removes_irrelevant():
    mapped = [
        {"relevant": True, "facts": ["fact1"], "source": "a.pdf"},
        {"relevant": False, "facts": [], "source": "b.pdf"},
        {"relevant": True, "facts": ["fact2"], "source": "c.pdf"},
    ]
    rag = AgenticRAG(FakeSearcher([]), FakeLLM([]), debug=False)
    filtered = rag._filter(mapped)
    assert len(filtered) == 2
    assert all(m["relevant"] for m in filtered)


from chat import confidence_score


def test_reduce_synthesizes_answer():
    llm = FakeLLM(["There are 2 invoices: #123 and #456."])
    rag = AgenticRAG(FakeSearcher([]), llm, debug=False)

    relevant = [
        {"relevant": True, "facts": ["Invoice #123", "Amount: $50"], "source": "a.pdf"},
        {"relevant": True, "facts": ["Invoice #456", "Amount: $75"], "source": "b.pdf"},
    ]
    answer = rag._reduce("find invoices", relevant)
    assert "2 invoices" in answer


def test_reduce_no_relevant_chunks():
    llm = FakeLLM(["No relevant information found."])
    rag = AgenticRAG(FakeSearcher([]), llm, debug=False)
    answer = rag._reduce("find invoices", [])
    assert "No relevant" in answer


def test_confidence_high_overlap():
    facts = ["Invoice #123", "Amount: $50", "Date: Dec 4"]
    answer = "Invoice #123 for $50 dated Dec 4"
    score = confidence_score(answer, facts)
    assert score >= 0.5


def test_confidence_low_overlap():
    facts = ["Invoice #123", "Amount: $50"]
    answer = "The weather is nice today and I like cats"
    score = confidence_score(answer, facts)
    assert score < 0.3


def test_confidence_empty_facts():
    score = confidence_score("some answer", [])
    assert score == 0.0


def test_confidence_check_flags_low_confidence():
    llm = FakeLLM(["The sky is blue and birds fly south."])
    rag = AgenticRAG(FakeSearcher([]), llm, debug=False)

    relevant = [
        {"relevant": True, "facts": ["Invoice #123", "Amount: $50"], "source": "a.pdf"},
    ]
    answer = rag._reduce("find invoices", relevant)
    result = rag._confidence_check(answer, relevant)
    assert "(low confidence)" in result.lower()


def test_full_pipeline_invoice_query():
    """End-to-end: query about invoices returns structured answer."""
    # LLM responses in order: planner, map x2, reduce (no self-check LLM call)
    responses = [
        json.dumps({"keywords": ["invoice"], "file_filter": "pdf", "source_hint": "Invoice"}),
        json.dumps({"relevant": True, "facts": ["Invoice EFCDCDB4-0005", "Amount: $21.78", "Date: Dec 4, 2025"]}),
        json.dumps({"relevant": False, "facts": []}),
        "Found 1 invoice: EFCDCDB4-0005 for $21.78 dated December 4, 2025.",
    ]
    llm = FakeLLM(responses)
    results = _make_results(
        "Invoice number EFCDCDB4-0005 Date December 4, 2025 Amount $21.78",
        "National Park admission ticket",
    )
    searcher = FakeSearcher(results)
    rag = AgenticRAG(searcher, llm, top_k=5, debug=False)

    answer = rag.ask("any invoices in my files?")
    assert "EFCDCDB4-0005" in answer
    assert "$21.78" in answer


def test_full_pipeline_no_results():
    """When no chunks are relevant, returns 'no information found'."""
    responses = [
        json.dumps({"keywords": ["quantum"], "file_filter": None, "source_hint": None}),
        json.dumps({"relevant": False, "facts": []}),
    ]
    llm = FakeLLM(responses)
    results = _make_results("Invoice #123")
    searcher = FakeSearcher(results)
    rag = AgenticRAG(searcher, llm, top_k=5, debug=False)

    answer = rag.ask("tell me about quantum physics")
    assert "No relevant" in answer
