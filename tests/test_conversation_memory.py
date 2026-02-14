"""Tests for conversation memory — follow-up queries use recent context."""
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
    def __init__(self, plan_responses=None, extract_responses=None, synth_responses=None):
        self.plan_responses = list(plan_responses or [])
        self.extract_responses = list(extract_responses or [])
        self.synth_responses = list(synth_responses or [])
        self.plan_prompts = []
        self.extract_prompts = []
        self.synth_prompts = []

    def plan(self, prompt):
        self.plan_prompts.append(prompt)
        return self.plan_responses.pop(0) if self.plan_responses else "{}"

    def extract(self, prompt):
        self.extract_prompts.append(prompt)
        return self.extract_responses.pop(0) if self.extract_responses else "{}"

    def synthesize(self, prompt):
        self.synth_prompts.append(prompt)
        return self.synth_responses.pop(0) if self.synth_responses else ""

    def load(self):
        pass


class FakeLeannSearcher:
    def __init__(self, results):
        self.results = results

    def search(self, query, top_k=5, metadata_filters=None, **kwargs):
        return self.results[:top_k]


def _make_results(*texts):
    return [
        FakeSearchResult(id=str(i), text=t, score=0.9, metadata={"source": f"f{i}.pdf"})
        for i, t in enumerate(texts)
    ]


def test_conversation_history_included_in_planner_prompt():
    """After first query, planner prompt should include conversation context."""
    plan1 = json.dumps({
        "keywords": ["invoice"], "file_filter": "pdf", "source_hint": None,
        "tool": "semantic_search", "time_filter": None, "tool_actions": [],
    })
    plan2 = json.dumps({
        "keywords": ["invoice", "more"], "file_filter": "pdf", "source_hint": None,
        "tool": "semantic_search", "time_filter": None, "tool_actions": [],
    })
    map_resp = json.dumps({"relevant": True, "facts": ["Invoice #1", "Invoice #2"]})

    models = FakeModelManager(
        plan_responses=[plan1, plan2],
        extract_responses=[map_resp, map_resp, map_resp],
        synth_responses=["Found 2 invoices.", "Actually found 3 invoices."],
    )
    results = _make_results("Invoice #1 data", "Invoice #2 data")
    leann = FakeLeannSearcher(results)
    rag = AgenticRAG(models=models, leann_searcher=leann, data_dir="/tmp/test")

    # First query
    rag.ask("any invoices?")

    # Second query — planner should see conversation history
    rag.ask("aren't there more than 2?")

    # The second planner prompt should contain the first Q&A
    second_plan_prompt = models.plan_prompts[1]
    assert "any invoices?" in second_plan_prompt or "invoices" in second_plan_prompt.lower()
    assert "Found 2 invoices" in second_plan_prompt


def test_conversation_history_included_in_reduce_prompt():
    """Reduce prompt should include conversation context for follow-ups."""
    plan_resp = json.dumps({
        "keywords": ["invoice"], "file_filter": None, "source_hint": None,
        "tool": "semantic_search", "time_filter": None, "tool_actions": [],
    })
    map_resp = json.dumps({"relevant": True, "facts": ["Invoice data"]})

    models = FakeModelManager(
        plan_responses=[plan_resp, plan_resp],
        extract_responses=[map_resp, map_resp],
        synth_responses=["Found 2 invoices.", "There are actually 3."],
    )
    results = _make_results("Invoice data")
    leann = FakeLeannSearcher(results)
    rag = AgenticRAG(models=models, leann_searcher=leann, data_dir="/tmp/test")

    rag.ask("any invoices?")
    rag.ask("tell me more about them")

    # Reduce prompt for second query should have conversation context
    second_synth_prompt = models.synth_prompts[1]
    assert "any invoices?" in second_synth_prompt
    assert "Found 2 invoices" in second_synth_prompt


def test_conversation_history_max_window():
    """Only the last N exchanges are kept."""
    plan_resp = json.dumps({
        "keywords": ["test"], "file_filter": None, "source_hint": None,
        "tool": "semantic_search", "time_filter": None, "tool_actions": [],
    })
    map_resp = json.dumps({"relevant": True, "facts": ["fact"]})

    # 5 queries, window of 3
    models = FakeModelManager(
        plan_responses=[plan_resp] * 5,
        extract_responses=[map_resp] * 5,
        synth_responses=[f"Answer {i}" for i in range(5)],
    )
    results = _make_results("data")
    leann = FakeLeannSearcher(results)
    rag = AgenticRAG(models=models, leann_searcher=leann, data_dir="/tmp/test", history_window=3)

    for i in range(5):
        rag.ask(f"query {i}")

    # The 5th planner prompt should have queries 1-3 (not query 0)
    last_plan_prompt = models.plan_prompts[4]
    assert "query 0" not in last_plan_prompt
    assert "query 1" in last_plan_prompt or "query 2" in last_plan_prompt


def test_empty_history_on_first_query():
    """First query should work fine with no history."""
    plan_resp = json.dumps({
        "keywords": ["test"], "file_filter": None, "source_hint": None,
        "tool": "semantic_search", "time_filter": None, "tool_actions": [],
    })
    map_resp = json.dumps({"relevant": True, "facts": ["fact"]})

    models = FakeModelManager(
        plan_responses=[plan_resp],
        extract_responses=[map_resp],
        synth_responses=["answer"],
    )
    results = _make_results("data")
    leann = FakeLeannSearcher(results)
    rag = AgenticRAG(models=models, leann_searcher=leann, data_dir="/tmp/test")

    answer = rag.ask("first question")
    assert answer  # no crash
    # Planner prompt should NOT have "Recent conversation" header
    assert "Recent conversation" not in models.plan_prompts[0]
