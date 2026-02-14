"""Integration tests for AgenticRAG pipeline orchestrator."""
import json
from dataclasses import dataclass, field
from pipeline import AgenticRAG


@dataclass
class FakeSearchResult:
    id: str
    text: str
    score: float
    metadata: dict = field(default_factory=dict)


class FakeModelManager:
    """Returns canned responses for plan/map_chunk/synthesize calls."""
    def __init__(self, plan_responses=None, extract_responses=None, synth_responses=None):
        self.plan_responses = list(plan_responses or [])
        self.extract_responses = list(extract_responses or [])
        self.synth_responses = list(synth_responses or [])
        self.plan_prompts = []
        self.extract_prompts = []
        self.synth_prompts = []

    def plan(self, system, user):
        self.plan_prompts.append(user)
        return self.plan_responses.pop(0) if self.plan_responses else "{}"

    def map_chunk(self, system, user):
        self.extract_prompts.append(user)
        return self.extract_responses.pop(0) if self.extract_responses else "{}"

    def synthesize(self, system, user):
        self.synth_prompts.append(user)
        return self.synth_responses.pop(0) if self.synth_responses else ""

    def rewrite(self, system, user):
        """Default: pass-through rewrite (no expansion)."""
        query = user.split("Question: ")[-1] if "Question: " in user else user
        return json.dumps({
            "intent": "factual",
            "search_query": query,
            "resolved_query": query,
        })

    def load(self):
        pass


class FakeLeannSearcher:
    def __init__(self, results):
        self.results = results
        self.last_query = None

    def search(self, query, top_k=5, metadata_filters=None, **kwargs):
        self.last_query = query
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


def test_semantic_search_pipeline():
    """Full semantic search flow: plan -> search -> map -> filter -> reduce."""
    plan_json = json.dumps({
        "keywords": ["invoice"], "file_filter": "pdf", "source_hint": "Invoice",
        "tool": "semantic_search",     })
    map_responses = [
        json.dumps({"relevant": True, "facts": ["Invoice #123", "Amount: $50"]}),
        json.dumps({"relevant": False, "facts": []}),
    ]
    synth_response = "Found 1 invoice: #123 for $50."

    models = FakeModelManager(
        plan_responses=[plan_json],
        extract_responses=map_responses,
        synth_responses=[synth_response],
    )
    results = _make_results("Invoice #123 for $50", "Weather report", sources=["Invoice_001.pdf", "weather.txt"])
    leann = FakeLeannSearcher(results)

    rag = AgenticRAG(models=models, leann_searcher=leann, data_dir="/tmp/test")
    answer = rag.ask("find invoices")
    assert "#123" in answer
    assert "$50" in answer


def test_filesystem_pipeline():
    """Filesystem queries bypass search entirely."""
    import tempfile
    from pathlib import Path

    tmp = tempfile.mkdtemp()
    (Path(tmp) / "a.pdf").write_text("pdf1")
    (Path(tmp) / "b.pdf").write_text("pdf2")
    (Path(tmp) / "c.txt").write_text("txt1")

    plan_json = json.dumps({
        "keywords": ["PDF", "count"], "file_filter": "pdf", "source_hint": None,
        "tool": "filesystem",     })
    synth_response = "You have 2 PDF files."

    models = FakeModelManager(
        plan_responses=[plan_json],
        synth_responses=[synth_response],
    )
    leann = FakeLeannSearcher([])

    rag = AgenticRAG(models=models, leann_searcher=leann, data_dir=tmp)
    answer = rag.ask("how many PDFs?")
    assert "2" in answer


def test_no_results_returns_message():
    plan_json = json.dumps({
        "keywords": ["quantum"], "file_filter": None, "source_hint": None,
        "tool": "semantic_search",     })
    models = FakeModelManager(plan_responses=[plan_json])
    leann = FakeLeannSearcher([])

    rag = AgenticRAG(models=models, leann_searcher=leann, data_dir="/tmp/test")
    answer = rag.ask("quantum physics?")
    assert "No relevant" in answer


def test_all_chunks_irrelevant_returns_message():
    plan_json = json.dumps({
        "keywords": ["quantum"], "file_filter": None, "source_hint": None,
        "tool": "semantic_search",     })
    map_response = json.dumps({"relevant": False, "facts": []})

    models = FakeModelManager(
        plan_responses=[plan_json],
        extract_responses=[map_response],
    )
    results = _make_results("Cooking recipe")
    leann = FakeLeannSearcher(results)

    rag = AgenticRAG(models=models, leann_searcher=leann, data_dir="/tmp/test")
    answer = rag.ask("quantum physics?")
    assert "No relevant" in answer


def test_search_fallback_when_filters_return_empty():
    """When filtered search returns 0 results, retry unfiltered."""
    plan_json = json.dumps({
        "keywords": ["invoice"], "file_filter": None, "source_hint": "nonexistent",
        "tool": "semantic_search",     })
    map_response = json.dumps({"relevant": True, "facts": ["some fact"]})
    synth_response = "Found something via fallback."

    models = FakeModelManager(
        plan_responses=[plan_json],
        extract_responses=[map_response],
        synth_responses=[synth_response],
    )
    results = _make_results("actual data")
    leann = FakeLeannSearcher(results)

    rag = AgenticRAG(models=models, leann_searcher=leann, data_dir="/tmp/test")
    answer = rag.ask("invoices")
    assert "fallback" in answer.lower() or "Found" in answer


def test_debug_mode_does_not_crash():
    plan_json = json.dumps({
        "keywords": ["test"], "file_filter": None, "source_hint": None,
        "tool": "semantic_search",     })
    map_response = json.dumps({"relevant": True, "facts": ["fact"]})
    synth_response = "answer"

    models = FakeModelManager(
        plan_responses=[plan_json],
        extract_responses=[map_response],
        synth_responses=[synth_response],
    )
    results = _make_results("text")
    leann = FakeLeannSearcher(results)

    rag = AgenticRAG(models=models, leann_searcher=leann, data_dir="/tmp/test", debug=True)
    answer = rag.ask("test query")
    assert answer  # just verify it doesn't crash


def test_rewriter_feeds_search_query_to_searcher():
    """Rewriter's search_query should be used for vector search."""
    plan_json = json.dumps({
        "keywords": ["budget"], "file_filter": None, "source_hint": None,
        "tool": "semantic_search",
    })
    map_response = json.dumps({"relevant": True, "facts": ["Budget: $100k"]})
    synth_response = "The budget is $100k."

    models = FakeModelManager(
        plan_responses=[plan_json],
        extract_responses=[map_response],
        synth_responses=[synth_response],
    )
    rewrite_response = json.dumps({
        "intent": "factual",
        "search_query": "engineering department budget allocation quarterly",
        "resolved_query": "What is the engineering department budget?",
    })
    models._rewrite_responses = [rewrite_response]
    models.rewrite = lambda system, user: models._rewrite_responses.pop(0)

    results = _make_results("Budget doc: $100k for engineering")
    leann = FakeLeannSearcher(results)

    rag = AgenticRAG(models=models, leann_searcher=leann, data_dir="/tmp/test")
    answer = rag.ask("what is the budget?")

    assert leann.last_query == "engineering department budget allocation quarterly"


def test_rewriter_resolved_query_passed_to_planner():
    """Planner should receive resolved_query, not raw query."""
    plan_json = json.dumps({
        "keywords": ["invoice"], "file_filter": None, "source_hint": None,
        "tool": "semantic_search",
    })
    map_response = json.dumps({"relevant": True, "facts": ["Invoice #1"]})
    synth_response = "Found invoice."

    models = FakeModelManager(
        plan_responses=[plan_json],
        extract_responses=[map_response],
        synth_responses=[synth_response],
    )
    rewrite_response = json.dumps({
        "intent": "count",
        "search_query": "invoice receipt payment",
        "resolved_query": "Are there more invoices besides the 2 found?",
    })
    models._rewrite_responses = [rewrite_response]
    models.rewrite = lambda system, user: models._rewrite_responses.pop(0)

    results = _make_results("Invoice data")
    leann = FakeLeannSearcher(results)

    rag = AgenticRAG(models=models, leann_searcher=leann, data_dir="/tmp/test")
    rag.ask("aren't there more?")

    assert "Are there more invoices" in models.plan_prompts[0]
