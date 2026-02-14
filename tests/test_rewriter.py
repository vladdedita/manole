"""Tests for QueryRewriter — coreference resolution, term expansion, intent classification."""
import json
from rewriter import QueryRewriter, REWRITER_SYSTEM


class FakeModelManager:
    """Returns canned responses for rewrite() calls."""
    def __init__(self, responses: list[str]):
        self.responses = list(responses)
        self.system_prompts = []
        self.user_prompts = []

    def rewrite(self, system: str, user: str) -> str:
        self.system_prompts.append(system)
        self.user_prompts.append(user)
        return self.responses.pop(0) if self.responses else ""


def test_rewriter_system_mentions_intent():
    assert "intent" in REWRITER_SYSTEM.lower()


def test_rewriter_system_mentions_search_query():
    assert "search_query" in REWRITER_SYSTEM


def test_rewriter_parses_valid_json():
    response = json.dumps({
        "intent": "factual",
        "search_query": "engineering department budget allocation",
        "resolved_query": "What is the engineering department budget?",
    })
    models = FakeModelManager([response])
    result = QueryRewriter(models).rewrite("what is the budget?")

    assert result["intent"] == "factual"
    assert "budget" in result["search_query"]
    assert "budget" in result["resolved_query"]


def test_rewriter_resolves_coreference():
    response = json.dumps({
        "intent": "count",
        "search_query": "invoice receipt payment billing document",
        "resolved_query": "Are there more invoices besides the 2 found?",
    })
    context = "Recent conversation:\n  User: any invoices?\n  Assistant: Found 2 invoices."
    models = FakeModelManager([response])
    result = QueryRewriter(models).rewrite("aren't there more?", context=context)

    assert result["intent"] == "count"
    assert "invoice" in result["search_query"]
    assert "invoice" in result["resolved_query"].lower()


def test_rewriter_context_passed_in_user_message():
    models = FakeModelManager([json.dumps({
        "intent": "factual",
        "search_query": "budget",
        "resolved_query": "What is the budget?",
    })])
    QueryRewriter(models).rewrite("what is the budget?", context="Recent conversation:\n  User: hi")

    assert "Recent conversation" in models.user_prompts[0]
    assert "budget" in models.user_prompts[0]
    assert "Recent conversation" not in models.system_prompts[0]


def test_rewriter_fallback_on_garbage():
    """When model returns non-JSON, fall back to raw query."""
    models = FakeModelManager(["I don't understand"])
    result = QueryRewriter(models).rewrite("find invoices")

    assert result["intent"] == "factual"
    assert result["search_query"] == "find invoices"
    assert result["resolved_query"] == "find invoices"


def test_rewriter_fallback_on_empty():
    models = FakeModelManager([""])
    result = QueryRewriter(models).rewrite("test query")

    assert result["intent"] == "factual"
    assert result["search_query"] == "test query"
    assert result["resolved_query"] == "test query"


def test_rewriter_invalid_intent_defaults_to_factual():
    response = json.dumps({
        "intent": "philosophical",
        "search_query": "meaning of life",
        "resolved_query": "What is the meaning of life?",
    })
    models = FakeModelManager([response])
    result = QueryRewriter(models).rewrite("meaning of life")

    assert result["intent"] == "factual"


def test_rewriter_missing_fields_fallback():
    """Partial JSON — missing search_query should fall back to raw query."""
    response = json.dumps({"intent": "list"})
    models = FakeModelManager([response])
    result = QueryRewriter(models).rewrite("find invoices")

    assert result["intent"] == "list"
    assert result["search_query"] == "find invoices"
    assert result["resolved_query"] == "find invoices"
