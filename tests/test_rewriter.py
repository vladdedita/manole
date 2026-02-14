"""Tests for QueryRewriter — coreference resolution, term expansion, intent classification."""
import json
from unittest.mock import MagicMock
from rewriter import QueryRewriter, REWRITER_SYSTEM, _VALID_INTENTS


def _mock_model(response: str):
    model = MagicMock()
    model.generate = MagicMock(return_value=response)
    return model


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
    model = _mock_model(response)
    result = QueryRewriter(model).rewrite("what is the budget?")

    assert result["intent"] == "factual"
    assert "budget" in result["search_query"]
    assert "budget" in result["resolved_query"].lower()


def test_rewriter_resolves_coreference():
    response = json.dumps({
        "intent": "count",
        "search_query": "invoice receipt payment billing document",
        "resolved_query": "Are there more invoices besides the 2 found?",
    })
    model = _mock_model(response)
    context = "Recent conversation:\n  User: any invoices?\n  Assistant: Found 2 invoices."
    result = QueryRewriter(model).rewrite("aren't there more?", context=context)

    assert result["intent"] == "count"
    assert "invoice" in result["search_query"]
    assert "invoice" in result["resolved_query"].lower()


def test_rewriter_context_passed_in_user_message():
    response = json.dumps({
        "intent": "factual",
        "search_query": "budget",
        "resolved_query": "What is the budget?",
    })
    model = _mock_model(response)
    QueryRewriter(model).rewrite("what is the budget?", context="Recent conversation:\n  User: hi")

    call_args = model.generate.call_args
    messages = call_args[0][0] if call_args[0] else call_args.kwargs.get("messages", [])
    user_msg = [m["content"] for m in messages if m["role"] == "user"][0]
    assert "Recent conversation" in user_msg
    assert "budget" in user_msg


def test_rewriter_fallback_on_garbage():
    model = _mock_model("I don't understand")
    result = QueryRewriter(model).rewrite("find invoices")

    assert result["intent"] == "factual"
    assert result["search_query"] == "find invoices"
    assert result["resolved_query"] == "find invoices"


def test_rewriter_fallback_on_empty():
    model = _mock_model("")
    result = QueryRewriter(model).rewrite("test query")

    assert result["intent"] == "factual"
    assert result["search_query"] == "test query"
    assert result["resolved_query"] == "test query"


def test_rewriter_invalid_intent_defaults_to_factual():
    response = json.dumps({
        "intent": "philosophical",
        "search_query": "meaning of life",
        "resolved_query": "What is the meaning of life?",
    })
    model = _mock_model(response)
    result = QueryRewriter(model).rewrite("meaning of life")

    assert result["intent"] == "factual"


def test_rewriter_missing_fields_fallback():
    response = json.dumps({"intent": "list"})
    model = _mock_model(response)
    result = QueryRewriter(model).rewrite("find invoices")

    assert result["intent"] == "list"
    assert result["search_query"] == "find invoices"
    assert result["resolved_query"] == "find invoices"


def test_rewriter_uses_max_tokens_256():
    model = _mock_model(json.dumps({
        "intent": "factual",
        "search_query": "test",
        "resolved_query": "test",
    }))
    QueryRewriter(model).rewrite("test")

    call_args = model.generate.call_args
    assert call_args.kwargs.get("max_tokens") == 256 or (len(call_args[0]) > 1 and call_args[0][1] == 256)
