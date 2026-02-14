# Agentic RAG Pipeline Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the single-shot RAG call in chat.py with a 5-stage agentic pipeline (Plan → Search → Map → Filter → Reduce) plus a Python-based confidence check that makes the 1.2B LFM model produce accurate, grounded answers.

**Architecture:** A new `AgenticRAG` class in `chat.py` orchestrates the pipeline. Each stage has a focused prompt template and a parsing function. The class takes a LEANN searcher and LLM instance, calls the model multiple times with small focused prompts, and returns a final answer. Debug mode prints intermediate results at each stage.

**Tech Stack:** Python 3.13, leann (LeannChat/LeannSearcher), LiquidAI/LFM2.5-1.2B-Instruct via HuggingFace

---

### Task 1: Add pytest and create test infrastructure

**Files:**
- Modify: `pyproject.toml`
- Create: `tests/__init__.py`
- Create: `tests/test_agentic_rag.py`

**Step 1: Add pytest dependency**

Add to `pyproject.toml` under `[project]`:
```toml
[project.optional-dependencies]
dev = ["pytest>=8.0"]
```

**Step 2: Install dev dependencies**

Run: `cd /Users/ded/Projects/assist/manole && uv pip install -e ".[dev]"`
Expected: pytest installed successfully

**Step 3: Create test file with mock infrastructure**

Create `tests/__init__.py` (empty).

Create `tests/test_agentic_rag.py`:
```python
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
```

**Step 4: Run tests to verify infrastructure**

Run: `cd /Users/ded/Projects/assist/manole && uv run pytest tests/ -v`
Expected: 1 test PASS

**Step 5: Commit**

```bash
git add pyproject.toml tests/
git commit -m "test: add pytest infrastructure with mock LLM/searcher for agentic RAG"
```

---

### Task 2: Implement `parse_json` helper with robust fallback

**Files:**
- Modify: `chat.py` (add function after imports, before `get_index_name`)
- Modify: `tests/test_agentic_rag.py`

**Step 1: Write failing tests for JSON parsing**

Add to `tests/test_agentic_rag.py`:
```python
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
```

**Step 2: Run tests to verify they fail**

Run: `cd /Users/ded/Projects/assist/manole && uv run pytest tests/test_agentic_rag.py -k parse_json -v`
Expected: FAIL (ImportError, `parse_json` doesn't exist yet)

**Step 3: Implement `parse_json` in `chat.py`**

Add after the imports in `chat.py`:
```python
import json
import re


def parse_json(text: str) -> dict | None:
    """Parse JSON from LLM output with fallback regex extraction."""
    # Try direct parse
    try:
        return json.loads(text.strip())
    except (json.JSONDecodeError, ValueError):
        pass

    # Try to find a JSON object in the text
    match = re.search(r'\{[^{}]+\}', text)
    if match:
        try:
            return json.loads(match.group())
        except (json.JSONDecodeError, ValueError):
            pass

    # Fallback: extract "relevant" field via regex
    rel_match = re.search(r'"relevant"\s*:\s*(true|false)', text, re.IGNORECASE)
    if rel_match:
        relevant = rel_match.group(1).lower() == "true"
        facts_match = re.search(r'"facts"\s*:\s*\[([^\]]*)\]', text)
        facts = []
        if facts_match:
            facts = [f.strip().strip('"') for f in facts_match.group(1).split(",") if f.strip()]
        return {"relevant": relevant, "facts": facts}

    return None
```

**Step 4: Run tests to verify they pass**

Run: `cd /Users/ded/Projects/assist/manole && uv run pytest tests/test_agentic_rag.py -k parse_json -v`
Expected: 5 tests PASS

**Step 5: Commit**

```bash
git add chat.py tests/test_agentic_rag.py
git commit -m "feat: add robust JSON parser with regex fallback for LLM output"
```

---

### Task 3: Implement prompt templates as module constants

**Files:**
- Modify: `chat.py` (replace existing `RAG_PROMPT` with all 4 stage prompts)

**Step 1: Write test for prompt template formatting**

Add to `tests/test_agentic_rag.py`:
```python
from chat import PLANNER_PROMPT, MAP_PROMPT, REDUCE_PROMPT


def test_planner_prompt_formats():
    result = PLANNER_PROMPT.format(query="find invoices")
    assert "find invoices" in result
    assert "keywords" in result


def test_map_prompt_formats():
    result = MAP_PROMPT.format(query="find invoices", chunk_text="Invoice #123")
    assert "find invoices" in result
    assert "Invoice #123" in result


def test_reduce_prompt_formats():
    result = REDUCE_PROMPT.format(facts_list="- Invoice #123", query="find invoices")
    assert "Invoice #123" in result
```

**Step 2: Run tests to verify they fail**

Run: `cd /Users/ded/Projects/assist/manole && uv run pytest tests/test_agentic_rag.py -k prompt -v`
Expected: FAIL (ImportError)

**Step 3: Replace RAG_PROMPT with stage-specific prompts in `chat.py`**

Replace the existing `RAG_PROMPT` constant and add the others. Place after `DEBUG_SOURCES = True`:

```python
PLANNER_PROMPT = (
    "Analyze this user question and extract search parameters.\n"
    "Question: {query}\n\n"
    "Output a JSON object with:\n"
    '- "keywords": list of 2-4 search terms\n'
    '- "file_filter": file extension to filter by (e.g. "pdf", "txt") or null\n'
    '- "source_hint": filename substring to filter by, or null\n\n'
    "JSON:\n"
)

MAP_PROMPT = (
    "Read this text and answer: does it contain information relevant to the question?\n\n"
    "Question: {query}\n"
    "Text: {chunk_text}\n\n"
    "Output a JSON object with:\n"
    '- "relevant": true or false\n'
    '- "facts": list of specific facts found (dates, names, numbers, filenames). '
    "Empty list if not relevant.\n\n"
    "JSON:\n"
)

REDUCE_PROMPT = (
    "Here are facts extracted from the user's files:\n"
    "{facts_list}\n\n"
    "Question: {query}\n\n"
    "Using ONLY these facts, write a concise answer. "
    'If the facts don\'t answer the question, say "No relevant information found."\n\n'
    "Answer:\n"
)

```

**Step 4: Run tests to verify they pass**

Run: `cd /Users/ded/Projects/assist/manole && uv run pytest tests/test_agentic_rag.py -k prompt -v`
Expected: 3 tests PASS

**Step 5: Commit**

```bash
git add chat.py tests/test_agentic_rag.py
git commit -m "feat: add prompt templates for all agentic RAG pipeline stages"
```

---

### Task 4: Implement `AgenticRAG` class — planner and search stages

**Files:**
- Modify: `chat.py` (add class after prompt constants, before `chat_loop`)
- Modify: `tests/test_agentic_rag.py`

**Step 1: Write failing tests for planner + search**

Add to `tests/test_agentic_rag.py`:
```python
from chat import AgenticRAG


def _make_results(*texts):
    """Helper: create FakeSearchResults from text strings."""
    return [
        FakeSearchResult(id=str(i), text=t, score=0.9 - i * 0.1, metadata={"source": f"file{i}.pdf"})
        for i, t in enumerate(texts)
    ]


def test_planner_extracts_filters():
    planner_response = json.dumps({
        "keywords": ["invoice", "anthropic"],
        "file_filter": "pdf",
        "source_hint": "Invoice",
    })
    llm = FakeLLM([planner_response])
    searcher = FakeSearcher(_make_results("chunk1"))
    rag = AgenticRAG(searcher, llm, top_k=3, debug=False)

    plan = rag._plan("find my Anthropic invoices")
    assert plan["source_hint"] == "Invoice"
    assert plan["file_filter"] == "pdf"


def test_planner_bad_json_returns_empty_plan():
    llm = FakeLLM(["I don't understand"])
    searcher = FakeSearcher([])
    rag = AgenticRAG(searcher, llm, top_k=3, debug=False)

    plan = rag._plan("find invoices")
    assert plan["keywords"] == []
    assert plan["source_hint"] is None
    assert plan["file_filter"] is None


def test_search_applies_metadata_filters():
    planner_response = json.dumps({
        "keywords": ["invoice"],
        "file_filter": None,
        "source_hint": "Invoice",
    })
    llm = FakeLLM([planner_response])
    results = _make_results("Invoice #123")
    searcher = FakeSearcher(results)
    rag = AgenticRAG(searcher, llm, top_k=5, debug=False)

    plan = rag._plan("invoices")
    search_results = rag._search("invoices", plan)
    assert searcher.last_filters == {"source": {"contains": "Invoice"}}
    assert len(search_results) == 1
```

**Step 2: Run tests to verify they fail**

Run: `cd /Users/ded/Projects/assist/manole && uv run pytest tests/test_agentic_rag.py -k "planner or search_applies" -v`
Expected: FAIL (ImportError, `AgenticRAG` doesn't exist)

**Step 3: Implement AgenticRAG with `_plan` and `_search`**

Add to `chat.py` after the prompt constants:
```python
class AgenticRAG:
    """Multi-stage agentic RAG pipeline for small language models."""

    def __init__(self, searcher, llm, top_k=5, debug=True):
        self.searcher = searcher
        self.llm = llm
        self.top_k = top_k
        self.debug = debug

    def _log(self, stage: str, msg: str):
        if self.debug:
            print(f"  [{stage}] {msg}")

    def _plan(self, query: str) -> dict:
        """Stage 1: Extract search parameters from query."""
        self._log("PLAN", f"Analyzing query: {query}")
        response = self.llm.ask(PLANNER_PROMPT.format(query=query), temperature=0.0)
        parsed = parse_json(response)

        default = {"keywords": [], "file_filter": None, "source_hint": None}
        if parsed is None:
            self._log("PLAN", "Failed to parse planner output, using defaults")
            return default

        result = {
            "keywords": parsed.get("keywords", []),
            "file_filter": parsed.get("file_filter"),
            "source_hint": parsed.get("source_hint"),
        }
        self._log("PLAN", f"Result: {result}")
        return result

    def _search(self, query: str, plan: dict) -> list:
        """Stage 2: Search with optional metadata filters."""
        metadata_filters = None
        source_hint = plan.get("source_hint")
        file_filter = plan.get("file_filter")

        if source_hint or file_filter:
            metadata_filters = {}
            if source_hint:
                metadata_filters["source"] = {"contains": source_hint}
            elif file_filter:
                metadata_filters["source"] = {"contains": f".{file_filter}"}

        self._log("SEARCH", f"Filters: {metadata_filters}")
        results = self.searcher.search(
            query, top_k=self.top_k, metadata_filters=metadata_filters
        )
        self._log("SEARCH", f"Found {len(results)} chunks")
        return results
```

**Step 4: Run tests to verify they pass**

Run: `cd /Users/ded/Projects/assist/manole && uv run pytest tests/test_agentic_rag.py -k "planner or search_applies" -v`
Expected: 3 tests PASS

**Step 5: Commit**

```bash
git add chat.py tests/test_agentic_rag.py
git commit -m "feat: add AgenticRAG planner and search stages"
```

---

### Task 5: Implement map and filter stages

**Files:**
- Modify: `chat.py` (add methods to `AgenticRAG`)
- Modify: `tests/test_agentic_rag.py`

**Step 1: Write failing tests**

Add to `tests/test_agentic_rag.py`:
```python
def test_map_extracts_facts_from_relevant_chunk():
    map_response = json.dumps({"relevant": True, "facts": ["Invoice #123", "Amount: $50"]})
    # planner + map responses
    planner_resp = json.dumps({"keywords": ["invoice"], "file_filter": None, "source_hint": None})
    llm = FakeLLM([map_response])
    searcher = FakeSearcher([])
    rag = AgenticRAG(searcher, llm, top_k=5, debug=False)

    chunk = FakeSearchResult(id="0", text="Invoice #123 for $50", score=0.9, metadata={})
    result = rag._map_chunk("find invoices", chunk)
    assert result["relevant"] is True
    assert "Invoice #123" in result["facts"]


def test_map_marks_irrelevant_chunk():
    map_response = json.dumps({"relevant": False, "facts": []})
    llm = FakeLLM([map_response])
    rag = AgenticRAG(FakeSearcher([]), llm, top_k=7, debug=False)

    chunk = FakeSearchResult(id="0", text="Nice weather today", score=0.3, metadata={})
    result = rag._map_chunk("find invoices", chunk)
    assert result["relevant"] is False


def test_map_garbage_response_defaults_to_relevant():
    llm = FakeLLM(["I have no idea what you want"])
    rag = AgenticRAG(FakeSearcher([]), llm, top_k=7, debug=False)

    chunk = FakeSearchResult(id="0", text="Some text", score=0.5, metadata={})
    result = rag._map_chunk("query", chunk)
    assert result["relevant"] is True
    assert chunk.text in result["facts"]


def test_filter_removes_irrelevant():
    mapped = [
        {"relevant": True, "facts": ["fact1"], "source": "a.pdf"},
        {"relevant": False, "facts": [], "source": "b.pdf"},
        {"relevant": True, "facts": ["fact2"], "source": "c.pdf"},
    ]
    rag = AgenticRAG(FakeSearcher([]), FakeLLM([]), debug=False)
    filtered = rag._filter(mapped)
    assert len(filtered) == 2
    assert all(m["relevant"] for m in filtered)
```

**Step 2: Run tests to verify they fail**

Run: `cd /Users/ded/Projects/assist/manole && uv run pytest tests/test_agentic_rag.py -k "map or filter" -v`
Expected: FAIL

**Step 3: Implement `_map_chunk` and `_filter` methods**

Add to `AgenticRAG` class:
```python
    def _map_chunk(self, query: str, chunk) -> dict:
        """Stage 3: Extract facts from a single chunk."""
        prompt = MAP_PROMPT.format(query=query, chunk_text=chunk.text[:500])
        response = self.llm.ask(prompt, temperature=0.0)
        parsed = parse_json(response)

        source = chunk.metadata.get("source", chunk.id)

        if parsed is None:
            self._log("MAP", f"  {source}: parse failed, treating as relevant")
            return {"relevant": True, "facts": [chunk.text[:200]], "source": source}

        result = {
            "relevant": parsed.get("relevant", True),
            "facts": parsed.get("facts", []),
            "source": source,
        }
        self._log("MAP", f"  {source}: relevant={result['relevant']}, facts={len(result['facts'])}")
        return result

    def _filter(self, mapped: list[dict]) -> list[dict]:
        """Stage 4: Drop irrelevant chunks."""
        relevant = [m for m in mapped if m["relevant"]]
        self._log("FILTER", f"Kept {len(relevant)}/{len(mapped)} chunks")
        return relevant
```

**Step 4: Run tests to verify they pass**

Run: `cd /Users/ded/Projects/assist/manole && uv run pytest tests/test_agentic_rag.py -k "map or filter" -v`
Expected: 4 tests PASS

**Step 5: Commit**

```bash
git add chat.py tests/test_agentic_rag.py
git commit -m "feat: add map and filter stages to agentic RAG pipeline"
```

---

### Task 6: Implement reduce and confidence check stages

**Files:**
- Modify: `chat.py` (add methods to `AgenticRAG`)
- Modify: `tests/test_agentic_rag.py`

**Step 1: Write failing tests**

Add to `tests/test_agentic_rag.py`:
```python
from chat import AgenticRAG, confidence_score


def test_reduce_synthesizes_answer():
    llm = FakeLLM(["There are 2 invoices: #123 and #456."])
    rag = AgenticRAG(FakeSearcher([]), llm, debug=False)

    relevant = [
        {"relevant": True, "facts": ["Invoice #123", "Amount: $50"], "source": "a.pdf"},
        {"relevant": True, "facts": ["Invoice #456", "Amount: $75"], "source": "b.pdf"},
    ]
    answer = rag._reduce("find invoices", relevant)
    assert "2 invoices" in answer


def test_reduce_no_relevant_chunks():
    llm = FakeLLM(["No relevant information found."])
    rag = AgenticRAG(FakeSearcher([]), llm, debug=False)
    answer = rag._reduce("find invoices", [])
    assert "No relevant" in answer


def test_confidence_high_overlap():
    facts = ["Invoice #123", "Amount: $50", "Date: Dec 4"]
    answer = "Invoice #123 for $50 dated Dec 4"
    score = confidence_score(answer, facts)
    assert score >= 0.5


def test_confidence_low_overlap():
    facts = ["Invoice #123", "Amount: $50"]
    answer = "The weather is nice today and I like cats"
    score = confidence_score(answer, facts)
    assert score < 0.3


def test_confidence_empty_facts():
    score = confidence_score("some answer", [])
    assert score == 0.0


def test_confidence_check_flags_low_confidence():
    llm = FakeLLM(["The sky is blue and birds fly south."])
    rag = AgenticRAG(FakeSearcher([]), llm, debug=False)

    relevant = [
        {"relevant": True, "facts": ["Invoice #123", "Amount: $50"], "source": "a.pdf"},
    ]
    answer = rag._reduce("find invoices", relevant)
    result = rag._confidence_check(answer, relevant)
    assert "(low confidence)" in result.lower()
```

**Step 2: Run tests to verify they fail**

Run: `cd /Users/ded/Projects/assist/manole && uv run pytest tests/test_agentic_rag.py -k "reduce or confidence" -v`
Expected: FAIL

**Step 3: Implement `confidence_score`, `_reduce`, and `_confidence_check`**

Add `confidence_score` as a module-level function in `chat.py` (after `parse_json`):
```python
def confidence_score(answer: str, facts: list[str]) -> float:
    """Compute token overlap between answer and source facts. Returns 0.0-1.0."""
    if not facts:
        return 0.0
    answer_tokens = set(answer.lower().split())
    if not answer_tokens:
        return 0.0
    fact_tokens = set()
    for fact in facts:
        fact_tokens.update(fact.lower().split())
    if not fact_tokens:
        return 0.0
    overlap = answer_tokens & fact_tokens
    return len(overlap) / len(answer_tokens)
```

Add to `AgenticRAG` class:
```python
    def _reduce(self, query: str, relevant: list[dict]) -> str:
        """Stage 5: Synthesize answer from extracted facts."""
        if not relevant:
            return "No relevant information found."

        facts_list = ""
        for item in relevant:
            source = item.get("source", "?")
            facts_list += f"\nFrom {source}:\n"
            for fact in item["facts"]:
                facts_list += f"  - {fact}\n"

        prompt = REDUCE_PROMPT.format(facts_list=facts_list, query=query)
        self._log("REDUCE", f"Synthesizing from {len(relevant)} sources")
        answer = self.llm.ask(prompt, temperature=0.1)
        return answer.strip()

    def _confidence_check(self, answer: str, relevant: list[dict]) -> str:
        """Stage 6: Python-based confidence check via token overlap. No LLM call."""
        all_facts = []
        for item in relevant:
            all_facts.extend(item.get("facts", []))

        score = confidence_score(answer, all_facts)
        self._log("CHECK", f"Confidence: {score:.2f}")

        if score < 0.2:
            self._log("CHECK", "Low confidence — answer may not be grounded in sources")
            return f"{answer}\n\n(Low confidence — answer may not reflect source documents)"

        return answer
```

**Step 4: Run tests to verify they pass**

Run: `cd /Users/ded/Projects/assist/manole && uv run pytest tests/test_agentic_rag.py -k "reduce or confidence" -v`
Expected: 6 tests PASS

**Step 5: Commit**

```bash
git add chat.py tests/test_agentic_rag.py
git commit -m "feat: add reduce stage and Python-based confidence check (no LLM self-check)"
```

---

### Task 7: Implement `ask()` method and full pipeline test

**Files:**
- Modify: `chat.py` (add `ask` method to `AgenticRAG`)
- Modify: `tests/test_agentic_rag.py`

**Step 1: Write failing test for full pipeline**

Add to `tests/test_agentic_rag.py`:
```python
def test_full_pipeline_invoice_query():
    """End-to-end: query about invoices returns structured answer."""
    # LLM responses in order: planner, map x2, reduce (no self-check LLM call)
    responses = [
        json.dumps({"keywords": ["invoice"], "file_filter": "pdf", "source_hint": "Invoice"}),
        json.dumps({"relevant": True, "facts": ["Invoice EFCDCDB4-0005", "Amount: $21.78", "Date: Dec 4, 2025"]}),
        json.dumps({"relevant": False, "facts": []}),
        "Found 1 invoice: EFCDCDB4-0005 for $21.78 dated December 4, 2025.",
    ]
    llm = FakeLLM(responses)
    results = _make_results(
        "Invoice number EFCDCDB4-0005 Date December 4, 2025 Amount $21.78",
        "National Park admission ticket",
    )
    searcher = FakeSearcher(results)
    rag = AgenticRAG(searcher, llm, top_k=5, debug=False)

    answer = rag.ask("any invoices in my files?")
    assert "EFCDCDB4-0005" in answer
    assert "$21.78" in answer


def test_full_pipeline_no_results():
    """When no chunks are relevant, returns 'no information found'."""
    responses = [
        json.dumps({"keywords": ["quantum"], "file_filter": None, "source_hint": None}),
        json.dumps({"relevant": False, "facts": []}),
    ]
    llm = FakeLLM(responses)
    results = _make_results("Invoice #123")
    searcher = FakeSearcher(results)
    rag = AgenticRAG(searcher, llm, top_k=5, debug=False)

    answer = rag.ask("tell me about quantum physics")
    assert "No relevant" in answer
```

**Step 2: Run tests to verify they fail**

Run: `cd /Users/ded/Projects/assist/manole && uv run pytest tests/test_agentic_rag.py -k "full_pipeline" -v`
Expected: FAIL (`ask` method doesn't exist)

**Step 3: Implement `ask` method**

Add to `AgenticRAG` class:
```python
    def ask(self, query: str) -> str:
        """Run the full agentic RAG pipeline."""
        t0 = time.time()

        # Stage 1: Plan
        plan = self._plan(query)

        # Stage 2: Search
        results = self._search(query, plan)
        if not results:
            # Retry without filters
            self._log("SEARCH", "No results with filters, retrying without")
            results = self.searcher.search(query, top_k=self.top_k)

        if not results:
            return "No relevant information found."

        # Stage 3: Map
        self._log("MAP", f"Processing {len(results)} chunks...")
        mapped = [self._map_chunk(query, chunk) for chunk in results]

        # Stage 4: Filter
        relevant = self._filter(mapped)

        # Stage 5: Reduce
        answer = self._reduce(query, relevant)

        # Stage 6: Confidence check (Python token overlap, no LLM call)
        answer = self._confidence_check(answer, relevant)

        elapsed = time.time() - t0
        self._log("DONE", f"Pipeline completed in {elapsed:.1f}s")
        return answer
```

**Step 4: Run tests to verify they pass**

Run: `cd /Users/ded/Projects/assist/manole && uv run pytest tests/test_agentic_rag.py -k "full_pipeline" -v`
Expected: 2 tests PASS

**Step 5: Run all tests**

Run: `cd /Users/ded/Projects/assist/manole && uv run pytest tests/ -v`
Expected: All tests PASS

**Step 6: Commit**

```bash
git add chat.py tests/test_agentic_rag.py
git commit -m "feat: complete AgenticRAG pipeline with ask() orchestration"
```

---

### Task 8: Integrate AgenticRAG into chat_loop

**Files:**
- Modify: `chat.py` (rewrite `chat_loop` to use `AgenticRAG`)

**Step 1: No new test needed — this is UI integration**

The `chat_loop` function is interactive (stdin/stdout), so we test it manually.

**Step 2: Rewrite `chat_loop`**

Replace the `chat_loop` function in `chat.py` with:
```python
def chat_loop(index_name: str):
    print("\nLoading LFM2.5-1.2B-Instruct...")
    t0 = time.time()

    index_path = find_index_path(index_name)
    print(f"Using index: {index_path}")

    chat = LeannChat(
        index_path,
        llm_config={
            "type": "hf",
            "model": "LiquidAI/LFM2.5-1.2B-Instruct",
        },
    )

    rag = AgenticRAG(chat.searcher, chat.llm, top_k=5, debug=DEBUG_SOURCES)

    print(f"Ready in {time.time() - t0:.1f}s")
    print("=" * 50)
    print("Ask anything about your files. Type 'quit' to exit.")
    print("Type 'debug' to toggle pipeline trace.")
    print("=" * 50)

    while True:
        try:
            query = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not query:
            continue
        if query.lower() in ("quit", "exit", "q"):
            print("Bye!")
            break
        if query.lower() == "debug":
            rag.debug = not rag.debug
            print(f"Pipeline trace: {'ON' if rag.debug else 'OFF'}")
            continue

        t0 = time.time()
        response = rag.ask(query)
        elapsed = time.time() - t0

        print(f"\n{response}")
        print(f"\n({elapsed:.1f}s)")
```

**Step 3: Remove the old `RAG_PROMPT` constant**

Delete the `RAG_PROMPT` variable (it's replaced by the 4 stage-specific prompts).

**Step 4: Run all tests to verify nothing broke**

Run: `cd /Users/ded/Projects/assist/manole && uv run pytest tests/ -v`
Expected: All tests PASS

**Step 5: Commit**

```bash
git add chat.py
git commit -m "feat: integrate AgenticRAG pipeline into chat_loop"
```

---

### Task 9: Manual smoke test

**Step 1: Run the chat application**

Run: `cd /Users/ded/Projects/assist/manole && uv run python chat.py --reuse test_data`
(Or whatever index name you've already built.)

**Step 2: Test with the invoice query**

Type: `any invoices in my files?`

Expected output should show:
- `[PLAN]` stage with extracted keywords
- `[SEARCH]` with filter info
- `[MAP]` per-chunk relevance assessment
- `[FILTER]` showing how many chunks kept/dropped
- `[REDUCE]` synthesizing answer
- `[CHECK]` verification result
- Final answer mentioning specific invoice numbers

**Step 3: Test with an irrelevant query**

Type: `tell me about quantum physics`

Expected: Pipeline should filter out all chunks and return "No relevant information found."

**Step 4: Test debug toggle**

Type: `debug` to turn off trace, then ask another question. Verify clean output.

**Step 5: Commit any prompt tweaks if needed**

If the prompts need adjustment based on actual model output, tweak and commit.
