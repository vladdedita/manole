"""Tests for Searcher — search with internal map-filter."""
import json
from dataclasses import dataclass, field
from pathlib import Path
from unittest.mock import MagicMock
from searcher import Searcher, MAP_SYSTEM, extract_keywords


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
    text, sources = searcher.search_and_extract("budget")
    assert "From file0.txt:" in text
    assert "$450,000" in text


def test_search_no_results():
    model = _make_model([])
    leann = FakeLeann([])
    searcher = Searcher(leann, model)
    text, sources = searcher.search_and_extract("quantum physics")
    assert "No matching content" in text


def test_irrelevant_chunks_filtered():
    results = _make_results("invoice data", "weather report")
    model = _make_model([
        json.dumps({"relevant": True, "facts": ["Invoice #123"]}),
        json.dumps({"relevant": False, "facts": []}),
    ])
    leann = FakeLeann(results)
    searcher = Searcher(leann, model)
    text, sources = searcher.search_and_extract("invoices")
    assert "Invoice #123" in text
    assert "weather" not in text.lower()


def test_all_irrelevant_returns_message():
    results = _make_results("random text")
    model = _make_model([json.dumps({"relevant": False, "facts": []})])
    leann = FakeLeann(results)
    searcher = Searcher(leann, model)
    text, sources = searcher.search_and_extract("quantum")
    assert "none were relevant" in text.lower()


def test_score_prefilter():
    # Threshold is 0.95 * 0.85 = 0.8075, so 0.82 passes but 0.50 doesn't
    results = _make_results("good", "ok", "bad", scores=[0.95, 0.82, 0.50])
    model = _make_model([
        json.dumps({"relevant": True, "facts": ["fact1"]}),
        json.dumps({"relevant": True, "facts": ["fact2"]}),
    ])
    leann = FakeLeann(results)
    searcher = Searcher(leann, model)
    text, sources = searcher.search_and_extract("test")
    assert model.generate.call_count == 2


def test_parse_failure_defaults_to_irrelevant():
    results = _make_results("some text")
    model = _make_model(["not valid json at all"])
    leann = FakeLeann(results)
    searcher = Searcher(leann, model)
    text, sources = searcher.search_and_extract("test")
    assert "none were relevant" in text.lower()


def test_file_name_used_as_source():
    results = [FakeSearchResult(
        id="42", text="budget data", score=0.9,
        metadata={"file_name": "budget_q1_2026.txt"},
    )]
    model = _make_model([json.dumps({"relevant": True, "facts": ["Budget: $100k"]})])
    leann = FakeLeann(results)
    searcher = Searcher(leann, model)
    text, sources = searcher.search_and_extract("budget")
    assert "budget_q1_2026.txt" in text


def test_fallback_source_from_id():
    results = [FakeSearchResult(id="99", text="data", score=0.9, metadata={})]
    model = _make_model([json.dumps({"relevant": True, "facts": ["some fact"]})])
    leann = FakeLeann(results)
    searcher = Searcher(leann, model)
    text, sources = searcher.search_and_extract("test")
    assert "99" in text


def test_top_k_passed_to_leann():
    leann = FakeLeann([])
    model = _make_model([])
    searcher = Searcher(leann, model)
    searcher.search_and_extract("test", top_k=3)
    assert leann.last_query == "test"


def test_map_prompt_has_empty_facts_example():
    """Few-shot example with empty facts is critical for small models."""
    assert '"facts": []' in MAP_SYSTEM


def test_multiple_sources_grouped():
    results = _make_results("data1", "data2", sources=["a.pdf", "b.pdf"])
    model = _make_model([
        json.dumps({"relevant": True, "facts": ["fact A"]}),
        json.dumps({"relevant": True, "facts": ["fact B"]}),
    ])
    leann = FakeLeann(results)
    searcher = Searcher(leann, model)
    text, sources = searcher.search_and_extract("test")
    assert "From a.pdf:" in text
    assert "From b.pdf:" in text
    assert "fact A" in text
    assert "fact B" in text


def test_extract_keywords_basic():
    assert extract_keywords("any macbook invoice?") == ["macbook", "invoice"]


def test_extract_keywords_filters_stopwords():
    result = extract_keywords("what is the file size")
    assert "what" not in result
    assert "the" not in result
    assert "file" not in result
    assert "size" in result


def test_extract_keywords_lowercase():
    assert extract_keywords("MacBook PDF") == ["macbook"]


def test_extract_keywords_short_words_removed():
    result = extract_keywords("is it an ok file")
    assert "is" not in result
    assert "it" not in result
    assert "an" not in result
    assert "ok" not in result
    assert "file" not in result


def test_extract_keywords_filters_file_extensions():
    """File format words are stopwords — too broad for filename grep."""
    result = extract_keywords("any macbook pdf files")
    assert result == ["macbook"]


class FakeFileReader:
    """Fake FileReader that returns predefined text for any path."""
    def __init__(self, text="Invoice #999 from Dante International, Amount: $500"):
        self.text = text
        self.read_calls = []

    def read(self, path):
        self.read_calls.append(path)
        return self.text


class FakeToolBox:
    """Fake ToolBox with grep_paths support."""
    def __init__(self, paths=None):
        self.paths = paths or []

    def grep_paths(self, pattern, limit=3):
        return [p for p in self.paths if pattern.lower() in p.name.lower()][:limit]


def test_filename_fallback_when_chunks_irrelevant():
    """When all chunks are irrelevant, fallback greps filenames and reads files."""
    results = _make_results("unrelated meeting notes")
    model = _make_model([
        json.dumps({"relevant": False, "facts": []}),
        json.dumps({"relevant": True, "facts": ["Invoice #999", "Dante International"]}),
    ])
    leann = FakeLeann(results)
    file_reader = FakeFileReader()
    toolbox = FakeToolBox(paths=[Path("/data/macbook_ssd.pdf")])

    searcher = Searcher(leann, model, file_reader=file_reader, toolbox=toolbox)
    text, sources = searcher.search_and_extract("macbook invoice")

    assert "Invoice #999" in text
    assert "macbook_ssd.pdf" in text
    assert len(file_reader.read_calls) == 1


def test_filename_fallback_no_matching_files():
    """When no filenames match keywords, returns standard no-results message."""
    results = _make_results("unrelated text")
    model = _make_model([
        json.dumps({"relevant": False, "facts": []}),
    ])
    leann = FakeLeann(results)
    file_reader = FakeFileReader()
    toolbox = FakeToolBox(paths=[])

    searcher = Searcher(leann, model, file_reader=file_reader, toolbox=toolbox)
    text, sources = searcher.search_and_extract("macbook invoice")

    assert "none were relevant" in text.lower()


def test_filename_fallback_not_triggered_when_chunks_relevant():
    """Filename fallback should NOT run when chunk search finds results."""
    results = _make_results("Invoice #123 for MacBook Pro")
    model = _make_model([
        json.dumps({"relevant": True, "facts": ["Invoice #123", "MacBook Pro"]}),
    ])
    leann = FakeLeann(results)
    file_reader = FakeFileReader()
    toolbox = FakeToolBox(paths=[Path("/data/macbook_ssd.pdf")])

    searcher = Searcher(leann, model, file_reader=file_reader, toolbox=toolbox)
    text, sources = searcher.search_and_extract("macbook invoice")

    assert "Invoice #123" in text
    assert len(file_reader.read_calls) == 0


def test_filename_fallback_without_file_reader():
    """Without file_reader, fallback is skipped gracefully."""
    results = _make_results("unrelated text")
    model = _make_model([
        json.dumps({"relevant": False, "facts": []}),
    ])
    leann = FakeLeann(results)

    searcher = Searcher(leann, model)
    text, sources = searcher.search_and_extract("macbook invoice")

    assert "none were relevant" in text.lower()


def test_filename_fallback_includes_facts_even_when_model_says_irrelevant():
    """Filename match = relevance signal. Facts included even if model says relevant=False."""
    results = _make_results("unrelated meeting notes")
    model = _make_model([
        json.dumps({"relevant": False, "facts": []}),
        # Model says irrelevant (e.g. Romanian text) but still extracts facts
        json.dumps({"relevant": False, "facts": ["Factura #999", "Total: 500 RON"]}),
    ])
    leann = FakeLeann(results)
    file_reader = FakeFileReader(text="Factura fiscala nr. 999, Total: 500 RON")
    toolbox = FakeToolBox(paths=[Path("/data/macbook_ssd.pdf")])

    searcher = Searcher(leann, model, file_reader=file_reader, toolbox=toolbox)
    text, sources = searcher.search_and_extract("macbook invoice")

    assert "Factura #999" in text
    assert "macbook_ssd.pdf" in text


def test_filename_fallback_surfaces_file_when_no_facts_extracted():
    """When model extracts zero facts (e.g. foreign language), file is still surfaced."""
    results = _make_results("unrelated text")
    model = _make_model([
        json.dumps({"relevant": False, "facts": []}),
        # Model returns empty facts (can't parse Romanian text)
        json.dumps({"relevant": False, "facts": []}),
    ])
    leann = FakeLeann(results)
    file_reader = FakeFileReader(text="Factura fiscala nr. 999")
    toolbox = FakeToolBox(paths=[Path("/data/macbook_ssd.pdf")])

    searcher = Searcher(leann, model, file_reader=file_reader, toolbox=toolbox)
    text, sources = searcher.search_and_extract("macbook invoice")

    assert "macbook_ssd.pdf" in text
    assert "File found" in text


def test_filename_fallback_caps_at_3_files():
    """Fallback reads at most 3 matching files."""
    results = _make_results("unrelated")
    model = _make_model([
        json.dumps({"relevant": False, "facts": []}),
        json.dumps({"relevant": True, "facts": ["fact1"]}),
        json.dumps({"relevant": True, "facts": ["fact2"]}),
        json.dumps({"relevant": True, "facts": ["fact3"]}),
    ])
    leann = FakeLeann(results)
    file_reader = FakeFileReader()
    toolbox = FakeToolBox(paths=[
        Path(f"/data/macbook_{i}.pdf") for i in range(5)
    ])

    searcher = Searcher(leann, model, file_reader=file_reader, toolbox=toolbox)
    output = searcher.search_and_extract("macbook")

    assert len(file_reader.read_calls) == 3


def test_search_and_extract_returns_sources():
    """search_and_extract returns (text, sources) tuple with source filenames."""
    results = _make_results("Budget doc", sources=["budget.pdf"])
    model = _make_model([json.dumps({"relevant": True, "facts": ["Budget: $450k"]})])
    leann = FakeLeann(results)
    searcher = Searcher(leann, model)

    text, sources = searcher.search_and_extract("budget")

    assert isinstance(sources, list)
    assert "budget.pdf" in sources
    assert "budget.pdf" in text


def test_search_and_extract_no_results_returns_empty_sources():
    """When no chunks match, sources list is empty."""
    leann = FakeLeann([])
    model = _make_model([])
    searcher = Searcher(leann, model)

    text, sources = searcher.search_and_extract("anything")

    assert sources == []
    assert "No matching" in text
