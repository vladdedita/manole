"""Tests for JSON parsing with regex fallback."""
from parser import parse_json


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


def test_parse_json_planner_output():
    """Planner returns keywords/filters JSON."""
    raw = '{"keywords": ["invoice", "Anthropic"], "file_filter": "pdf", "source_hint": "Invoice"}'
    result = parse_json(raw)
    assert result["keywords"] == ["invoice", "Anthropic"]
    assert result["file_filter"] == "pdf"


def test_parse_json_planner_with_preamble():
    """350M model might add text before JSON."""
    raw = 'Here is the extracted JSON:\n{"keywords": ["invoice"], "file_filter": null, "source_hint": null}'
    result = parse_json(raw)
    assert result["keywords"] == ["invoice"]


def test_parse_json_nested_braces():
    """Handle JSON with nested structures."""
    raw = '{"keywords": ["test"], "tool_actions": ["count"]}'
    result = parse_json(raw)
    assert result["keywords"] == ["test"]
    assert result["tool_actions"] == ["count"]


def test_parse_json_full_planner_schema():
    """Handle the full planner output with all fields."""
    raw = 'Here is the JSON:\n{"keywords": ["invoice", "Anthropic"], "file_filter": "pdf", "source_hint": "Invoice", "tool": "semantic_search", "time_filter": null, "tool_actions": []}'
    result = parse_json(raw)
    assert result["keywords"] == ["invoice", "Anthropic"]
    assert result["tool"] == "semantic_search"
    assert result["tool_actions"] == []
