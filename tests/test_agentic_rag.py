"""Tests for AgenticRAG pipeline using mock LLM and searcher."""
import json
from dataclasses import dataclass
from unittest.mock import MagicMock


@dataclass
class FakeSearchResult:
    """Mimics leann SearchResult."""
    id: str
    text: str
    score: float
    metadata: dict


class FakeLLM:
    """Records prompts and returns canned responses."""
    def __init__(self, responses: list[str]):
        self.responses = list(responses)
        self.prompts = []

    def ask(self, prompt: str, **kwargs) -> str:
        self.prompts.append(prompt)
        if self.responses:
            return self.responses.pop(0)
        return ""


class FakeSearcher:
    """Returns pre-configured search results."""
    def __init__(self, results: list[FakeSearchResult]):
        self.results = results
        self.last_filters = None

    def search(self, query, top_k=7, metadata_filters=None, **kwargs):
        self.last_filters = metadata_filters
        return self.results[:top_k]


def test_placeholder():
    """Verify test infrastructure works."""
    assert FakeLLM(["hello"]).ask("hi") == "hello"
    assert FakeSearcher([]).search("q") == []


import sys
sys.path.insert(0, "/Users/ded/Projects/assist/manole")
from chat import parse_json


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
