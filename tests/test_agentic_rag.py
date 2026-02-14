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
