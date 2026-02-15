"""Tests for Agent — orchestrator agent loop."""
import json
from unittest.mock import MagicMock
from agent import Agent


class FakeToolRegistry:
    def __init__(self, responses=None):
        self.responses = dict(responses or {})
        self.calls = []

    def execute(self, tool_name, params):
        self.calls.append((tool_name, params))
        return self.responses.get(tool_name, f"Result for {tool_name}")


class FakeRouter:
    def __init__(self, tool_name="semantic_search", params=None):
        self.tool_name = tool_name
        self.params = params or {"query": "test"}
        self.called = False

    def route(self, query, intent=None):
        self.called = True
        return self.tool_name, self.params


def _make_model(responses):
    model = MagicMock()
    model.generate = MagicMock(side_effect=list(responses))
    return model


def test_model_tool_call_semantic_search():
    """Model produces a tool call, agent executes it, then model responds."""
    model = _make_model([
        '<|tool_call_start|>semantic_search(query="budget")<|tool_call_end|>',
        "The budget is $450,000.",
    ])
    tools = FakeToolRegistry({"semantic_search": "From budget.txt:\n  - Budget: $450k"})
    router = FakeRouter()

    agent = Agent(model, tools, router)
    answer = agent.run("what is the budget?")

    assert answer == "The budget is $450,000."
    assert tools.calls[0] == ("semantic_search", {"query": "budget"})
    assert not router.called


def test_fallback_router_on_step_0():
    """When model doesn't produce a tool call on step 0, fallback router kicks in."""
    model = _make_model([
        "I'll help you find that information.",  # no tool call
        "You have 5 PDF files.",  # answer after seeing tool result
    ])
    tools = FakeToolRegistry({"count_files": "Found 5 .pdf files."})
    router = FakeRouter(tool_name="count_files", params={"extension": "pdf"})

    agent = Agent(model, tools, router)
    answer = agent.run("how many PDFs?")

    assert router.called
    assert "5" in answer


def test_direct_answer_on_later_step():
    """On step > 0, no tool call means model is answering directly."""
    model = _make_model([
        '<|tool_call_start|>count_files(extension="pdf")<|tool_call_end|>',
        "You have 3 PDF files.",  # direct answer, no tool call
    ])
    tools = FakeToolRegistry({"count_files": "Found 3 .pdf files."})
    router = FakeRouter()

    agent = Agent(model, tools, router)
    answer = agent.run("how many PDFs?")

    assert "3" in answer


def test_respond_tool():
    """Model can use respond() to return an answer explicitly."""
    model = _make_model([
        '<|tool_call_start|>count_files(extension="txt")<|tool_call_end|>',
        '<|tool_call_start|>respond(answer="You have 2 text files.")<|tool_call_end|>',
    ])
    tools = FakeToolRegistry({"count_files": "Found 2 .txt files."})
    router = FakeRouter()

    agent = Agent(model, tools, router)
    answer = agent.run("how many text files?")

    assert answer == "You have 2 text files."


def test_max_steps_forces_synthesis():
    """After MAX_STEPS, agent forces a synthesis turn."""
    # Model keeps calling tools without answering
    tool_calls = ['<|tool_call_start|>semantic_search(query="test")<|tool_call_end|>'] * 5
    model = _make_model(tool_calls + ["Final forced answer."])
    tools = FakeToolRegistry({"semantic_search": "some results"})
    router = FakeRouter()

    agent = Agent(model, tools, router)
    answer = agent.run("complex question")

    assert answer == "Final forced answer."
    assert model.generate.call_count == 6  # 5 tool calls + 1 forced synthesis


def test_conversation_history_passed():
    """Recent history is included in messages."""
    model = _make_model([
        "The follow-up answer.",  # no tool call, triggers fallback router
        "The follow-up answer.",  # answer after tool result
    ])
    tools = FakeToolRegistry()
    router = FakeRouter()

    history = [
        {"role": "user", "content": "find invoices"},
        {"role": "assistant", "content": "Found 3 invoices."},
    ]

    agent = Agent(model, tools, router)
    agent.run("aren't there more?", history=history)

    # Check that history was included in messages
    call_args = model.generate.call_args_list[0]
    messages = call_args[0][0] if call_args[0] else call_args.kwargs.get("messages", [])
    user_contents = [m["content"] for m in messages if m["role"] == "user"]
    assert "find invoices" in user_contents
    assert "aren't there more?" in user_contents


def test_history_capped_at_4_messages():
    """Only last 4 history messages (2 turns) are included."""
    model = _make_model([
        '<|tool_call_start|>semantic_search(query="test")<|tool_call_end|>',
        "answer",
    ])
    tools = FakeToolRegistry({"semantic_search": "results"})
    router = FakeRouter()

    history = [
        {"role": "user", "content": f"q{i}"}
        for i in range(10)
    ]

    agent = Agent(model, tools, router)
    agent.run("final question", history=history)

    # Verify the initial messages list before any tool results were appended:
    # system + 4 history + 1 current query = 6
    # We check that only 4 history messages (last 4) were included
    call_args = model.generate.call_args_list[0]
    messages = call_args[0][0] if call_args[0] else call_args.kwargs.get("messages", [])
    history_msgs = [m for m in messages if m["role"] == "user" and m["content"].startswith("q")]
    assert len(history_msgs) == 4
    assert history_msgs[0]["content"] == "q6"
    assert history_msgs[-1]["content"] == "q9"


def test_json_tool_call_format():
    """Agent handles JSON-formatted tool calls as fallback."""
    model = _make_model([
        json.dumps({"name": "count_files", "params": {"extension": "pdf"}}),
        "You have 5 PDFs.",
    ])
    tools = FakeToolRegistry({"count_files": "Found 5 .pdf files."})
    router = FakeRouter()

    agent = Agent(model, tools, router)
    answer = agent.run("how many PDFs?")

    assert "5" in answer
    assert tools.calls[0] == ("count_files", {"extension": "pdf"})


def test_debug_mode():
    """Debug mode shouldn't crash."""
    model = _make_model([
        "direct answer",  # no tool call, triggers fallback router
        "direct answer",  # answer after tool result
    ])
    tools = FakeToolRegistry()
    router = FakeRouter("semantic_search", {"query": "test"})

    agent = Agent(model, tools, router, debug=True)
    answer = agent.run("test")
    assert answer  # just verify no crash


def test_unknown_tool_returns_error_to_model():
    """If model calls an unknown tool, error message goes back to context."""
    model = _make_model([
        '<|tool_call_start|>magic_tool(query="hi")<|tool_call_end|>',
        "Sorry, I couldn't use that tool.",
    ])
    tools = FakeToolRegistry()
    router = FakeRouter()

    agent = Agent(model, tools, router)
    answer = agent.run("do magic")
    # Agent should still work — unknown tool result goes back as message
    assert answer is not None


def test_bracket_format_tool_call():
    """Model outputs [tool_name(params)] bracket format."""
    model = _make_model([
        '[semantic_search(query="invoices")]',
        "Found 3 invoices.",
    ])
    tools = FakeToolRegistry({"semantic_search": "invoice1.pdf\ninvoice2.pdf\ninvoice3.pdf"})
    router = FakeRouter()

    agent = Agent(model, tools, router)
    answer = agent.run("find invoices")

    assert tools.calls[0] == ("semantic_search", {"query": "invoices"})
    assert not router.called


def test_bare_function_call():
    """Model outputs bare tool_name(params) without any wrapping."""
    model = _make_model([
        'count_files(extension="pdf")',
        "You have 5 PDFs.",
    ])
    tools = FakeToolRegistry({"count_files": "Found 5 .pdf files."})
    router = FakeRouter()

    agent = Agent(model, tools, router)
    answer = agent.run("how many PDFs?")

    assert tools.calls[0] == ("count_files", {"extension": "pdf"})
    assert not router.called


def test_followup_grep_when_keyword_missing():
    """Python injects grep_files when model stops but query keyword not in results."""
    model = _make_model([
        '[count_files(extension="pdf")]',
        "There are 25 PDF files.",
        "Found macbook_ssd.pdf — a MacBook-related PDF.",
    ])
    tools = FakeToolRegistry({
        "count_files": "Found 25 .pdf files.",
        "grep_files": "macbook_ssd.pdf",
    })
    router = FakeRouter()

    agent = Agent(model, tools, router)
    answer = agent.run("any macbook pdfs")

    tool_calls = [(name, params) for name, params in tools.calls]
    assert ("count_files", {"extension": "pdf"}) in tool_calls
    assert any(name == "grep_files" and "macbook" in params.get("pattern", "") for name, params in tool_calls)
    assert not router.called


def test_bare_unknown_tool_not_parsed():
    """Bare format only matches known tools, not arbitrary words."""
    model = _make_model([
        'thinking(about="stuff")',  # not a known tool
        "Here is my answer.",
    ])
    tools = FakeToolRegistry()
    router = FakeRouter()

    agent = Agent(model, tools, router)
    agent.run("test")

    # Should have fallen through to router since 'thinking' is not a known tool
    assert router.called


def test_bracket_format_directory_tree():
    """Bracket format works for directory_tree tool."""
    model = _make_model([
        '[directory_tree(max_depth=2)]',
        "Here are your folders.",
    ])
    tools = FakeToolRegistry({"directory_tree": "Documents/\n  invoices/\n  photos/"})
    router = FakeRouter()

    agent = Agent(model, tools, router)
    answer = agent.run("what folders do I have?")

    assert tools.calls[0] == ("directory_tree", {"max_depth": 2})
    assert not router.called


def test_tool_call_extracted_from_prose():
    """Model outputs tool call buried in prose — parser extracts it."""
    model = _make_model([
        'Reply with the tool call: count_files(extension="pdf")\n\nThere are PDF files.',
        "You have 5 PDFs.",
    ])
    tools = FakeToolRegistry({"count_files": "Found 5 .pdf files."})
    router = FakeRouter()

    agent = Agent(model, tools, router)
    answer = agent.run("how many PDFs?")

    assert tools.calls[0] == ("count_files", {"extension": "pdf"})
    assert not router.called


def test_followup_semantic_search_after_grep():
    """When grep was already used, followup tries semantic_search."""
    model = _make_model([
        '[grep_files(pattern="invoice")]',
        "Found 5 invoice files.",
        "The total across all invoices is $2,500.",
    ])
    tools = FakeToolRegistry({
        "grep_files": "invoice_001.pdf\ninvoice_002.pdf",
        "semantic_search": "From invoice_001.pdf: Total: $1,200\nFrom invoice_002.pdf: Total: $1,300",
    })
    router = FakeRouter()

    agent = Agent(model, tools, router)
    answer = agent.run("what is the revenue from invoices")

    tool_calls = [(name, params) for name, params in tools.calls]
    assert any(name == "grep_files" for name, params in tool_calls)
    assert any(name == "semantic_search" for name, params in tool_calls)


def test_no_followup_when_keywords_covered():
    """No followup when all query keywords appear in tool results."""
    model = _make_model([
        '[semantic_search(query="carbonara eggs")]',
        "The recipe calls for 4 eggs.",
    ])
    tools = FakeToolRegistry({
        "semantic_search": "From pasta_carbonara.txt: 4 eggs, pecorino romano, guanciale",
    })
    router = FakeRouter()

    agent = Agent(model, tools, router)
    answer = agent.run("how many eggs in carbonara")

    assert answer == "The recipe calls for 4 eggs."
    assert len(tools.calls) == 1


def test_followup_stops_after_both_tools_tried():
    """Followup doesn't loop forever — stops when grep and semantic_search both tried."""
    model = _make_model([
        '[count_files(extension="pdf")]',
        "There are 25 PDFs.",
        "Still 25 PDFs.",
        "I found some results.",
    ])
    tools = FakeToolRegistry({
        "count_files": "Found 25 .pdf files.",
        "grep_files": "No matching files found.",
        "semantic_search": "No results found.",
    })
    router = FakeRouter()

    agent = Agent(model, tools, router)
    answer = agent.run("any macbook pdfs")

    assert answer == "I found some results."
    tool_names = [name for name, _ in tools.calls]
    assert tool_names.count("grep_files") == 1
    assert tool_names.count("semantic_search") == 1
