"""Tests for Searcher — leann vector search with metadata filters."""
from dataclasses import dataclass, field
from searcher import Searcher


@dataclass
class FakeSearchResult:
    id: str
    text: str
    score: float
    metadata: dict = field(default_factory=dict)


class FakeLeannSearcher:
    """Mimics LeannSearcher.search()."""
    def __init__(self, results: list[FakeSearchResult]):
        self.results = results
        self.last_kwargs = {}

    def search(self, query, top_k=5, metadata_filters=None, **kwargs):
        self.last_kwargs = {"query": query, "top_k": top_k, "metadata_filters": metadata_filters}
        if metadata_filters:
            source_filter = metadata_filters.get("source", {})
            contains = source_filter.get("contains", "")
            if contains:
                return [r for r in self.results if contains.lower() in r.metadata.get("source", "").lower()][:top_k]
        return self.results[:top_k]


def _make_results(*texts, sources=None):
    if sources is None:
        sources = [f"file{i}.pdf" for i in range(len(texts))]
    return [
        FakeSearchResult(id=str(i), text=t, score=0.9 - i * 0.1, metadata={"source": sources[i]})
        for i, t in enumerate(texts)
    ]


def test_search_basic():
    results = _make_results("chunk1", "chunk2", "chunk3")
    searcher = Searcher(FakeLeannSearcher(results))
    plan = {"keywords": ["test"], "source_hint": None, "file_filter": None}
    found = searcher.search(plan, top_k=5)
    assert len(found) == 3


def test_search_applies_source_hint():
    results = _make_results("invoice data", "recipe data", sources=["Invoice_001.pdf", "recipe.txt"])
    searcher = Searcher(FakeLeannSearcher(results))
    plan = {"keywords": ["invoice"], "source_hint": "Invoice", "file_filter": None}
    found = searcher.search(plan, top_k=5)
    assert len(found) == 1
    assert "Invoice" in found[0].metadata["source"]


def test_search_applies_file_filter():
    results = _make_results("data1", "data2", sources=["doc.pdf", "notes.txt"])
    searcher = Searcher(FakeLeannSearcher(results))
    plan = {"keywords": ["data"], "source_hint": None, "file_filter": "pdf"}
    found = searcher.search(plan, top_k=5)
    assert len(found) == 1
    assert found[0].metadata["source"].endswith(".pdf")


def test_search_source_hint_takes_precedence_over_file_filter():
    results = _make_results("data", sources=["Invoice_001.pdf"])
    searcher = Searcher(FakeLeannSearcher(results))
    plan = {"keywords": ["invoice"], "source_hint": "Invoice", "file_filter": "pdf"}
    found = searcher.search(plan, top_k=5)
    assert len(found) == 1


def test_search_unfiltered_fallback():
    results = _make_results("chunk1", "chunk2")
    searcher = Searcher(FakeLeannSearcher(results))
    plan = {"keywords": ["test"]}
    found = searcher.search_unfiltered(plan, top_k=5)
    assert len(found) == 2


def test_search_respects_top_k():
    results = _make_results("a", "b", "c", "d", "e")
    searcher = Searcher(FakeLeannSearcher(results))
    plan = {"keywords": ["test"], "source_hint": None, "file_filter": None}
    found = searcher.search(plan, top_k=2)
    assert len(found) == 2


def test_search_with_file_filter_paths():
    results = _make_results("data1", "data2", "data3", sources=["a.pdf", "b.txt", "c.pdf"])
    leann = FakeLeannSearcher(results)
    searcher = Searcher(leann)
    plan = {"keywords": ["data"], "source_hint": None, "file_filter": None}
    found = searcher.search(plan, top_k=5, file_filter_paths=["/path/to/a.pdf", "/path/to/c.pdf"])
    assert all("pdf" in r.metadata["source"] for r in found)
