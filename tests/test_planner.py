"""Tests for Planner stage — query to structured JSON plan."""
import json
from planner import Planner, PLANNER_PROMPT


class FakeModelManager:
    """Returns canned responses for plan() calls."""
    def __init__(self, responses: list[str]):
        self.responses = list(responses)
        self.prompts = []

    def plan(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.responses.pop(0) if self.responses else ""


def test_planner_prompt_contains_examples():
    """Prompt must have few-shot examples for all three tool types."""
    assert "semantic_search" in PLANNER_PROMPT
    assert "filesystem" in PLANNER_PROMPT
    assert "hybrid" in PLANNER_PROMPT


def test_planner_extracts_semantic_search_plan():
    response = json.dumps({
        "keywords": ["invoice", "Anthropic"],
        "file_filter": "pdf",
        "source_hint": "Invoice",
        "tool": "semantic_search",
        "time_filter": None,
        "tool_actions": [],
    })
    models = FakeModelManager([response])
    planner = Planner(models)
    plan = planner.plan("find my Anthropic invoices")

    assert plan["tool"] == "semantic_search"
    assert plan["keywords"] == ["invoice", "Anthropic"]
    assert plan["file_filter"] == "pdf"
    assert plan["source_hint"] == "Invoice"


def test_planner_extracts_filesystem_plan():
    response = json.dumps({
        "keywords": ["PDF", "files", "count"],
        "file_filter": "pdf",
        "source_hint": None,
        "tool": "filesystem",
        "time_filter": None,
        "tool_actions": ["count"],
    })
    models = FakeModelManager([response])
    planner = Planner(models)
    plan = planner.plan("how many PDF files do I have?")

    assert plan["tool"] == "filesystem"
    assert "count" in plan["tool_actions"]


def test_planner_extracts_hybrid_plan():
    response = json.dumps({
        "keywords": ["modified", "today", "summary"],
        "file_filter": None,
        "source_hint": None,
        "tool": "hybrid",
        "time_filter": "today",
        "tool_actions": ["list_recent"],
    })
    models = FakeModelManager([response])
    planner = Planner(models)
    plan = planner.plan("summarize files I modified today")

    assert plan["tool"] == "hybrid"
    assert plan["time_filter"] == "today"


def test_planner_garbage_output_returns_safe_defaults():
    models = FakeModelManager(["I don't understand the question"])
    planner = Planner(models)
    plan = planner.plan("find invoices")

    assert plan["tool"] == "semantic_search"
    assert plan["keywords"] == []
    assert plan["file_filter"] is None
    assert plan["source_hint"] is None
    assert plan["time_filter"] is None
    assert plan["tool_actions"] == []


def test_planner_partial_json_extracts_what_it_can():
    """350M model outputs JSON with missing fields."""
    response = '{"keywords": ["invoice"], "tool": "semantic_search"}'
    models = FakeModelManager([response])
    planner = Planner(models)
    plan = planner.plan("invoices")

    assert plan["keywords"] == ["invoice"]
    assert plan["tool"] == "semantic_search"
    assert plan["file_filter"] is None  # missing field gets default
