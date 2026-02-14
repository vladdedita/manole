"""Tests for Searcher — search with internal map-filter."""
import json
from dataclasses import dataclass, field
from unittest.mock import MagicMock
from searcher import Searcher


@dataclass
class FakeSearchResult:
    id: str
    text: str
    score: float
    metadata: dict = field(default_factory=dict)


class FakeLeann:
    def __init__(self, results):
        self.results = results
        self.last_query = None

    def search(self, query, top_k=5, **kwargs):
        self.last_query = query
        return self.results[:top_k]


def _make_model(responses):
    model = MagicMock()
    model.generate = MagicMock(side_effect=list(responses))
    return model


def _make_results(*texts, sources=None, scores=None):
    if sources is None:
        sources = [f"file{i}.txt" for i in range(len(texts))]
    if scores is None:
        scores = [0.95 - i * 0.05 for i in range(len(texts))]
    return [
        FakeSearchResult(
            id=str(i), text=t, score=scores[i],
            metadata={"file_name": sources[i]},
        )
        for i, t in enumerate(texts)
    ]


def test_search_and_extract_returns_formatted_facts():
    results = _make_results("Budget doc with $450k total")
    model = _make_model([json.dumps({"relevant": True, "facts": ["Total Budget: $450,000"]})])
    leann = FakeLeann(results)
    searcher = Searcher(leann, model)
    output = searcher.search_and_extract("budget")
    assert "From file0.txt:" in output
    assert "$450,000" in output


def test_search_no_results():
    model = _make_model([])
    leann = FakeLeann([])
    searcher = Searcher(leann, model)
    output = searcher.search_and_extract("quantum physics")
    assert "No matching content" in output


def test_irrelevant_chunks_filtered():
    results = _make_results("invoice data", "weather report")
    model = _make_model([
        json.dumps({"relevant": True, "facts": ["Invoice #123"]}),
        json.dumps({"relevant": False, "facts": []}),
    ])
    leann = FakeLeann(results)
    searcher = Searcher(leann, model)
    output = searcher.search_and_extract("invoices")
    assert "Invoice #123" in output
    assert "weather" not in output.lower()


def test_all_irrelevant_returns_message():
    results = _make_results("random text")
    model = _make_model([json.dumps({"relevant": False, "facts": []})])
    leann = FakeLeann(results)
    searcher = Searcher(leann, model)
    output = searcher.search_and_extract("quantum")
    assert "none were relevant" in output.lower()


def test_score_prefilter():
    results = _make_results("good", "ok", "bad", scores=[0.95, 0.80, 0.50])
    model = _make_model([
        json.dumps({"relevant": True, "facts": ["fact1"]}),
        json.dumps({"relevant": True, "facts": ["fact2"]}),
    ])
    leann = FakeLeann(results)
    searcher = Searcher(leann, model)
    output = searcher.search_and_extract("test")
    assert model.generate.call_count == 2


def test_parse_failure_defaults_to_irrelevant():
    results = _make_results("some text")
    model = _make_model(["not valid json at all"])
    leann = FakeLeann(results)
    searcher = Searcher(leann, model)
    output = searcher.search_and_extract("test")
    assert "none were relevant" in output.lower()


def test_file_name_used_as_source():
    results = [FakeSearchResult(
        id="42", text="budget data", score=0.9,
        metadata={"file_name": "budget_q1_2026.txt"},
    )]
    model = _make_model([json.dumps({"relevant": True, "facts": ["Budget: $100k"]})])
    leann = FakeLeann(results)
    searcher = Searcher(leann, model)
    output = searcher.search_and_extract("budget")
    assert "budget_q1_2026.txt" in output


def test_fallback_source_from_id():
    results = [FakeSearchResult(id="99", text="data", score=0.9, metadata={})]
    model = _make_model([json.dumps({"relevant": True, "facts": ["some fact"]})])
    leann = FakeLeann(results)
    searcher = Searcher(leann, model)
    output = searcher.search_and_extract("test")
    assert "99" in output


def test_top_k_passed_to_leann():
    leann = FakeLeann([])
    model = _make_model([])
    searcher = Searcher(leann, model)
    searcher.search_and_extract("test", top_k=3)
    assert leann.last_query == "test"


def test_multiple_sources_grouped():
    results = _make_results("data1", "data2", sources=["a.pdf", "b.pdf"])
    model = _make_model([
        json.dumps({"relevant": True, "facts": ["fact A"]}),
        json.dumps({"relevant": True, "facts": ["fact B"]}),
    ])
    leann = FakeLeann(results)
    searcher = Searcher(leann, model)
    output = searcher.search_and_extract("test")
    assert "From a.pdf:" in output
    assert "From b.pdf:" in output
    assert "fact A" in output
    assert "fact B" in output
