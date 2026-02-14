"""Tests for Planner stage — query to structured JSON plan."""
import json
from planner import Planner, PLANNER_SYSTEM, _keywords_from_query


class FakeModelManager:
    """Returns canned responses for plan() calls."""
    def __init__(self, responses: list[str]):
        self.responses = list(responses)
        self.system_prompts = []
        self.user_prompts = []

    def plan(self, system: str, user: str) -> str:
        self.system_prompts.append(system)
        self.user_prompts.append(user)
        return self.responses.pop(0) if self.responses else ""


def test_planner_system_contains_examples():
    """System prompt must have few-shot examples."""
    assert "semantic_search" in PLANNER_SYSTEM
    assert "filesystem" in PLANNER_SYSTEM


def test_planner_system_forbids_answering():
    """System prompt must tell model to never answer the question."""
    assert "never answer" in PLANNER_SYSTEM.lower() or "ALWAYS reply with JSON" in PLANNER_SYSTEM


def test_planner_extracts_semantic_search_plan():
    response = json.dumps({
        "keywords": ["invoice"],
        "file_filter": "pdf",
        "tool": "semantic_search",
    })
    models = FakeModelManager([response])
    plan = Planner(models).plan("find invoices")

    assert plan["tool"] == "semantic_search"
    assert plan["keywords"] == ["invoice"]
    assert plan["file_filter"] == "pdf"


def test_planner_extracts_filesystem_plan():
    response = json.dumps({
        "keywords": ["PDF", "files", "count"],
        "file_filter": "pdf",
        "tool": "filesystem",
    })
    models = FakeModelManager([response])
    plan = Planner(models).plan("how many PDF files do I have?")

    assert plan["tool"] == "filesystem"


def test_planner_garbage_falls_back_to_query_keywords():
    """When model answers conversationally, extract keywords from query."""
    models = FakeModelManager(["The budget is not available in the files."])
    plan = Planner(models).plan("what is the engineering budget?")

    assert plan["tool"] == "semantic_search"
    assert "engineering" in plan["keywords"]
    assert "budget" in plan["keywords"]


def test_planner_empty_keywords_falls_back_to_query():
    """When model returns empty keywords, extract from query."""
    response = json.dumps({"keywords": [], "tool": "semantic_search"})
    models = FakeModelManager([response])
    plan = Planner(models).plan("find invoices")

    assert plan["keywords"] == ["invoices"]


def test_planner_invalid_tool_defaults_to_semantic():
    response = json.dumps({"keywords": ["test"], "tool": "inventory"})
    models = FakeModelManager([response])
    plan = Planner(models).plan("test query")

    assert plan["tool"] == "semantic_search"


def test_planner_partial_json_extracts_what_it_can():
    response = '{"keywords": ["invoice"], "tool": "semantic_search"}'
    models = FakeModelManager([response])
    plan = Planner(models).plan("invoices")

    assert plan["keywords"] == ["invoice"]
    assert plan["tool"] == "semantic_search"
    assert plan["file_filter"] is None


def test_keywords_from_query_filters_stopwords():
    kw = _keywords_from_query("any invoices in the files?")
    assert "invoices" in kw
    assert "files" in kw
    assert "any" not in kw
    assert "the" not in kw
    assert "in" not in kw


def test_context_passed_in_user_message():
    """Conversation context should be in the user message, not system."""
    models = FakeModelManager(['{"keywords": ["budget"]}'])
    planner = Planner(models)
    planner.plan("what is the budget?", context="Recent conversation:\n  User: any invoices?")

    assert "Recent conversation" in models.user_prompts[0]
    assert "budget" in models.user_prompts[0]
    # System should NOT contain conversation context
    assert "Recent conversation" not in models.system_prompts[0]
