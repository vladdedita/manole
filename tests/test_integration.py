"""Integration tests — full agent loop with mocked model and real filesystem."""
import json
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from unittest.mock import MagicMock

from agent import Agent
from tools import ToolRegistry
from toolbox import ToolBox
from searcher import Searcher
from router import route


@dataclass
class FakeSearchResult:
    id: str
    text: str
    score: float
    metadata: dict = field(default_factory=dict)


class FakeLeann:
    def __init__(self, results):
        self.results = results

    def search(self, query, top_k=5, **kwargs):
        return self.results[:top_k]


class RouterWrapper:
    @staticmethod
    def route(query, intent=None):
        return route(query, intent=intent)


def _setup(model_responses, search_results=None, files=None):
    """Set up agent with mocked model and optional real filesystem."""
    model = MagicMock()
    model.generate = MagicMock(side_effect=list(model_responses))

    tmp = tempfile.mkdtemp()
    for name, content in (files or {}).items():
        p = Path(tmp) / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)

    leann = FakeLeann(search_results or [])
    searcher = Searcher(leann, model)
    toolbox = ToolBox(tmp)
    registry = ToolRegistry(searcher, toolbox)

    agent = Agent(model, registry, RouterWrapper())
    return agent, model


def test_filesystem_count_via_native_tool_call():
    """Model uses native tool calling for filesystem query."""
    agent, model = _setup(
        model_responses=[
            '<|tool_call_start|>count_files(extension="pdf")<|tool_call_end|>',
            "You have 2 PDF files.",
        ],
        files={"a.pdf": "pdf1", "b.pdf": "pdf2", "c.txt": "txt1"},
    )
    answer = agent.run("how many PDFs?")
    assert "2" in answer


def test_filesystem_count_via_fallback_router():
    """When model doesn't produce tool call, fallback router handles it."""
    # Flow: agent step 0 → no tool call → fallback router routes "how many PDF files?"
    # → count_files(extension="pdf") → result → agent step 1 → final answer
    agent, model = _setup(
        model_responses=[
            "I'll help you count your files.",  # no tool call → fallback router
            "You have 2 PDF files.",  # final answer after seeing tool result
        ],
        files={"a.pdf": "pdf1", "b.pdf": "pdf2", "c.txt": "txt1"},
    )
    answer = agent.run("how many PDF files?")
    assert "2" in answer


def test_semantic_search_with_facts():
    """semantic_search extracts facts and model synthesizes answer.

    Call order with shared model.generate mock:
    1. Agent step 0: model.generate → response[0] (no tool call, triggers fallback router)
    2. Fallback router → semantic_search → searcher.search_and_extract →
       _extract_facts calls model.generate → response[1] (JSON extraction result)
    3. Agent step 1: model.generate → response[2] (final answer)
    """
    results = [
        FakeSearchResult(
            id="1", text="Total Budget: $450,000. Revenue targets: Project Alpha $180k.",
            score=0.95, metadata={"file_name": "budget.txt"},
        )
    ]
    agent, model = _setup(
        model_responses=[
            # 1. Agent step 0: no tool call → fallback router fires semantic_search
            "I'll search for budget information.",
            # 2. Searcher._extract_facts for the one chunk
            json.dumps({"relevant": True, "facts": ["Total Budget: $450,000", "Project Alpha: $180k"]}),
            # 3. Agent step 1: final answer after seeing extracted facts
            "The total budget is $450,000 with Project Alpha at $180k.",
        ],
        search_results=results,
    )
    answer = agent.run("what is the budget?")
    assert answer is not None
    assert "450,000" in answer


def test_directory_tree():
    """directory_tree shows folder structure."""
    agent, _ = _setup(
        model_responses=[
            '<|tool_call_start|>directory_tree(max_depth=2)<|tool_call_end|>',
            "Your files are organized in one folder with PDFs and text files.",
        ],
        files={"docs/a.pdf": "pdf", "docs/b.txt": "txt"},
    )
    answer = agent.run("what is the folder structure?")
    assert answer is not None


def test_conversation_follow_up():
    """Follow-up questions use conversation history.

    Flow: agent step 0 → no tool call → fallback router → semantic_search
    (but no search results, so searcher returns immediately without calling model.generate)
    → agent step 1 → final answer.
    """
    agent, model = _setup(
        model_responses=[
            "Let me check on that.",  # step 0: no tool call → fallback router → semantic_search (no results)
            "There are more than 2 based on the previous search.",  # step 1: final answer
        ],
    )
    history = [
        {"role": "user", "content": "find invoices"},
        {"role": "assistant", "content": "Found 2 invoices."},
    ]
    answer = agent.run("aren't there more?", history=history)
    # Verify history was passed — check model received history messages
    call_args = model.generate.call_args_list[0]
    messages = call_args[0][0] if call_args[0] else call_args.kwargs.get("messages", [])
    user_contents = [m["content"] for m in messages if m["role"] == "user"]
    assert "find invoices" in user_contents
    assert answer is not None


def test_grep_files():
    """grep_files finds files by name pattern."""
    agent, _ = _setup(
        model_responses=[
            '<|tool_call_start|>grep_files(pattern="invoice")<|tool_call_end|>',
            "Found 2 invoice files: invoice_001.pdf and invoice_002.pdf.",
        ],
        files={"invoice_001.pdf": "inv1", "invoice_002.pdf": "inv2", "readme.txt": "readme"},
    )
    answer = agent.run("find invoice files")
    assert "invoice" in answer.lower()
