# Query Rewriter Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a Query Rewriter as the first stage of the agentic RAG pipeline to resolve coreferences, expand search terms, and classify user intent.

**Architecture:** New `rewriter.py` module with `QueryRewriter` class that calls LFM2.5-1.2B-Instruct to produce `{intent, search_query, resolved_query}`. Integrates before the Planner in pipeline.py. Searcher updated to accept a raw search string instead of joining keywords.

**Tech Stack:** llama-cpp-python (via existing ModelManager), pytest

---

### Task 1: Add `rewrite()` method to ModelManager

**Files:**
- Modify: `models.py`
- Test: `tests/test_models.py`

**Step 1: Write the failing test**

Add to `tests/test_models.py`:

```python
def test_rewrite_calls_instruct_model():
    mgr = ModelManager.__new__(ModelManager)
    mock_model = MagicMock()
    mock_model.create_chat_completion.return_value = _mock_chat_response('{"intent": "factual"}')
    mgr.instruct_model = mock_model
    mgr.extract_model = MagicMock()

    result = mgr.rewrite("system", "user")
    mock_model.create_chat_completion.assert_called_once()
    assert "factual" in result
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/ded/Projects/assist/manole && uv run pytest tests/test_models.py::test_rewrite_calls_instruct_model -v`
Expected: FAIL with `AttributeError: 'ModelManager' object has no attribute 'rewrite'`

**Step 3: Write minimal implementation**

Add to `models.py` in the `ModelManager` class, after the `plan()` method:

```python
def rewrite(self, system: str, user: str) -> str:
    """LFM2.5-1.2B: query rewriting."""
    return self._chat(self.instruct_model, system, user, max_tokens=256, **_INSTRUCT_PARAMS)
```

**Step 4: Run test to verify it passes**

Run: `cd /Users/ded/Projects/assist/manole && uv run pytest tests/test_models.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add models.py tests/test_models.py
git commit -m "feat: add rewrite() method to ModelManager"
```

---

### Task 2: Create `rewriter.py` with `QueryRewriter` class

**Files:**
- Create: `rewriter.py`
- Create: `tests/test_rewriter.py`

**Step 1: Write the failing tests**

Create `tests/test_rewriter.py`:

```python
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
```

**Step 2: Run tests to verify they fail**

Run: `cd /Users/ded/Projects/assist/manole && uv run pytest tests/test_rewriter.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'rewriter'`

**Step 3: Write minimal implementation**

Create `rewriter.py`:

```python
"""Stage 0: Query Rewriter — coreference resolution, term expansion, intent classification."""
from parser import parse_json

REWRITER_SYSTEM = (
    "You rewrite user queries for document search. "
    "Given a question and optional conversation history, produce JSON with:\n"
    "- \"intent\": one of \"factual\", \"count\", \"list\", \"compare\", \"summarize\"\n"
    "- \"search_query\": expanded query optimized for vector search "
    "(add synonyms, related terms, full forms of abbreviations)\n"
    "- \"resolved_query\": the user's question with pronouns and references resolved\n\n"
    "Examples:\n\n"
    "Question: \"any invoices?\"\n"
    "{\"intent\": \"list\", \"search_query\": \"invoice receipt payment billing\", "
    "\"resolved_query\": \"Are there any invoices?\"}\n\n"
    "Question: \"how many PDFs?\"\n"
    "{\"intent\": \"count\", \"search_query\": \"PDF files documents\", "
    "\"resolved_query\": \"How many PDF files are there?\"}\n\n"
    "Reply with a single JSON object only."
)

_VALID_INTENTS = frozenset({"factual", "count", "list", "compare", "summarize"})


def _fallback(query: str) -> dict:
    return {"intent": "factual", "search_query": query, "resolved_query": query}


class QueryRewriter:
    """Rewrites queries for better retrieval using LFM2.5-1.2B-Instruct."""

    def __init__(self, models, debug: bool = False):
        self.models = models
        self.debug = debug

    def rewrite(self, query: str, context: str = "") -> dict:
        user_msg = query
        if context:
            user_msg = f"{context}\n\nQuestion: {query}"

        raw = self.models.rewrite(REWRITER_SYSTEM, user_msg)

        if self.debug:
            print(f"  [REWRITE] Raw: {raw}")

        parsed = parse_json(raw)
        if parsed is None:
            if self.debug:
                print("  [REWRITE] Parse failed, using raw query")
            return _fallback(query)

        intent = parsed.get("intent", "factual")
        if intent not in _VALID_INTENTS:
            intent = "factual"

        search_query = parsed.get("search_query") or query
        resolved_query = parsed.get("resolved_query") or query

        result = {
            "intent": intent,
            "search_query": search_query,
            "resolved_query": resolved_query,
        }

        if self.debug:
            print(f"  [REWRITE] {result}")

        return result
```

**Step 4: Run tests to verify they pass**

Run: `cd /Users/ded/Projects/assist/manole && uv run pytest tests/test_rewriter.py -v`
Expected: ALL PASS (8 tests)

**Step 5: Commit**

```bash
git add rewriter.py tests/test_rewriter.py
git commit -m "feat: add QueryRewriter stage with coreference resolution and intent classification"
```

---

### Task 3: Update Searcher to accept `search_query` string

The Searcher currently joins `plan["keywords"]` into a string. After the rewriter, the pipeline will pass a pre-built `search_query` string. The Searcher needs to accept either.

**Files:**
- Modify: `searcher.py`
- Modify: `tests/test_searcher.py`

**Step 1: Write the failing test**

Add to `tests/test_searcher.py`:

```python
def test_search_with_search_query_string():
    """When plan has search_query, use it instead of joining keywords."""
    leann = FakeLeannSearcher([
        FakeSearchResult(id="1", text="budget data", score=0.9, metadata={"source": "budget.txt"}),
    ])
    searcher = Searcher(leann)
    plan = {"keywords": ["old", "keywords"], "file_filter": None, "source_hint": None}
    results = searcher.search(plan, search_query="engineering department budget allocation")

    assert len(results) == 1
    assert leann.last_query == "engineering department budget allocation"


def test_search_without_search_query_uses_keywords():
    """When no search_query provided, fall back to joining keywords."""
    leann = FakeLeannSearcher([
        FakeSearchResult(id="1", text="data", score=0.9, metadata={"source": "file.txt"}),
    ])
    searcher = Searcher(leann)
    plan = {"keywords": ["invoice", "payment"], "file_filter": None, "source_hint": None}
    results = searcher.search(plan)

    assert leann.last_query == "invoice payment"
```

Update the `FakeLeannSearcher` in `tests/test_searcher.py` to capture the query:

```python
class FakeLeannSearcher:
    def __init__(self, results):
        self.results = results
        self.last_query = None

    def search(self, query, top_k=5, metadata_filters=None, **kwargs):
        self.last_query = query
        # ... existing filter logic ...
```

**Step 2: Run tests to verify they fail**

Run: `cd /Users/ded/Projects/assist/manole && uv run pytest tests/test_searcher.py::test_search_with_search_query_string -v`
Expected: FAIL with `TypeError: search() got an unexpected keyword argument 'search_query'`

**Step 3: Write minimal implementation**

Update `searcher.py` — add `search_query` parameter to both `search()` and `search_unfiltered()`:

```python
def search(self, plan: dict, top_k: int = 5, search_query: str | None = None, file_filter_paths: list[str] | None = None) -> list:
    query = search_query or " ".join(plan.get("keywords", []))
    if not query:
        query = "document"
    metadata_filters = self._build_filters(plan)
    results = self.leann.search(query, top_k=top_k, metadata_filters=metadata_filters)
    if file_filter_paths:
        path_basenames = {p.rsplit("/", 1)[-1].lower() for p in file_filter_paths}
        results = [
            r for r in results
            if r.metadata.get("source", "").rsplit("/", 1)[-1].lower() in path_basenames
        ]
    return results

def search_unfiltered(self, plan: dict, top_k: int = 5, search_query: str | None = None) -> list:
    query = search_query or " ".join(plan.get("keywords", []))
    if not query:
        query = "document"
    return self.leann.search(query, top_k=top_k)
```

**Step 4: Run all searcher tests**

Run: `cd /Users/ded/Projects/assist/manole && uv run pytest tests/test_searcher.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add searcher.py tests/test_searcher.py
git commit -m "feat: searcher accepts search_query string as alternative to joining keywords"
```

---

### Task 4: Integrate Rewriter into pipeline.py

**Files:**
- Modify: `pipeline.py`
- Modify: `tests/test_pipeline.py`

**Step 1: Write the failing test**

Add to `tests/test_pipeline.py`:

```python
def test_rewriter_feeds_search_query_to_searcher():
    """Rewriter's search_query should be used for vector search."""
    plan_json = json.dumps({
        "keywords": ["budget"], "file_filter": None, "source_hint": None,
        "tool": "semantic_search",
    })
    map_response = json.dumps({"relevant": True, "facts": ["Budget: $100k"]})
    synth_response = "The budget is $100k."

    models = FakeModelManager(
        plan_responses=[plan_json],
        extract_responses=[map_response],
        synth_responses=[synth_response],
    )
    # Add rewrite method to FakeModelManager
    rewrite_response = json.dumps({
        "intent": "factual",
        "search_query": "engineering department budget allocation quarterly",
        "resolved_query": "What is the engineering department budget?",
    })
    models._rewrite_responses = [rewrite_response]
    models.rewrite = lambda system, user: models._rewrite_responses.pop(0)

    results = _make_results("Budget doc: $100k for engineering")
    leann = FakeLeannSearcher(results)

    rag = AgenticRAG(models=models, leann_searcher=leann, data_dir="/tmp/test")
    answer = rag.ask("what is the budget?")

    # Verify the search used the rewriter's expanded query
    assert leann.last_query == "engineering department budget allocation quarterly"


def test_rewriter_resolved_query_passed_to_planner():
    """Planner should receive resolved_query, not raw query."""
    plan_json = json.dumps({
        "keywords": ["invoice"], "file_filter": None, "source_hint": None,
        "tool": "semantic_search",
    })
    map_response = json.dumps({"relevant": True, "facts": ["Invoice #1"]})
    synth_response = "Found invoice."

    models = FakeModelManager(
        plan_responses=[plan_json],
        extract_responses=[map_response],
        synth_responses=[synth_response],
    )
    rewrite_response = json.dumps({
        "intent": "count",
        "search_query": "invoice receipt payment",
        "resolved_query": "Are there more invoices besides the 2 found?",
    })
    models._rewrite_responses = [rewrite_response]
    models.rewrite = lambda system, user: models._rewrite_responses.pop(0)

    results = _make_results("Invoice data")
    leann = FakeLeannSearcher(results)

    rag = AgenticRAG(models=models, leann_searcher=leann, data_dir="/tmp/test")
    rag.ask("aren't there more?")

    # Planner should have received the resolved query
    assert "Are there more invoices" in models.plan_prompts[0]
```

Update the `FakeLeannSearcher` in `tests/test_pipeline.py` to capture query:

```python
class FakeLeannSearcher:
    def __init__(self, results):
        self.results = results
        self.last_query = None

    def search(self, query, top_k=5, metadata_filters=None, **kwargs):
        self.last_query = query
        # ... existing filter logic ...
```

**Step 2: Run tests to verify they fail**

Run: `cd /Users/ded/Projects/assist/manole && uv run pytest tests/test_pipeline.py::test_rewriter_feeds_search_query_to_searcher -v`
Expected: FAIL (pipeline doesn't use rewriter yet)

**Step 3: Write minimal implementation**

Update `pipeline.py`:

```python
"""AgenticRAG: dual-model agentic RAG pipeline orchestrator."""
import time

from planner import Planner
from rewriter import QueryRewriter
from searcher import Searcher
from mapper import Mapper
from reducer import Reducer
from toolbox import ToolBox


class AgenticRAG:
    """Orchestrates the full Rewrite -> Plan -> Search -> Map -> Filter -> Reduce pipeline."""

    def __init__(
        self,
        models,
        leann_searcher,
        data_dir: str,
        top_k: int = 10,
        debug: bool = False,
        history_window: int = 3,
    ):
        self.rewriter = QueryRewriter(models, debug=debug)
        self.planner = Planner(models, debug=debug)
        self.searcher = Searcher(leann_searcher)
        self.mapper = Mapper(models, debug=debug)
        self.reducer = Reducer(models, debug=debug)
        self.toolbox = ToolBox(data_dir)
        self.top_k = top_k
        self.debug = debug
        self.history: list[tuple[str, str]] = []
        self.history_window = history_window

    def _build_context(self) -> str:
        if not self.history:
            return ""
        lines = ["Recent conversation:"]
        for q, a in self.history:
            lines.append(f"  User: {q}")
            lines.append(f"  Assistant: {a[:200]}")
        return "\n".join(lines)

    def _log(self, msg: str):
        if self.debug:
            print(f"  [PIPELINE] {msg}")

    def ask(self, query: str) -> str:
        """Run the full agentic RAG pipeline."""
        t0 = time.time()
        context = self._build_context()

        # Stage 0: Rewrite
        rewrite = self.rewriter.rewrite(query, context=context)
        resolved_query = rewrite["resolved_query"]
        search_query = rewrite["search_query"]
        self._log(f"Rewrite: intent={rewrite['intent']}, search_query={search_query}")

        # Stage 1: Plan (uses resolved query, not raw)
        plan = self.planner.plan(resolved_query, context=context)
        self._log(f"Plan: tool={plan['tool']}")

        # Filesystem path: ToolBox -> format -> return
        if plan["tool"] == "filesystem":
            result = self.toolbox.execute(plan)
            answer = self.reducer.format_filesystem_answer(resolved_query, result)
            self._log(f"Filesystem answer in {time.time() - t0:.1f}s")
            self.history.append((query, answer))
            self.history = self.history[-self.history_window:]
            return answer

        # Semantic search path (uses rewriter's search_query)
        chunks = self.searcher.search(plan, top_k=self.top_k, search_query=search_query)

        # Fallback: retry without filters
        if not chunks:
            self._log("No results with filters, retrying unfiltered")
            chunks = self.searcher.search_unfiltered(plan, top_k=self.top_k, search_query=search_query)

        if not chunks:
            answer = "No relevant information found in your files."
            self.history.append((query, answer))
            self.history = self.history[-self.history_window:]
            return answer

        # Stage 3: Map (uses resolved query for relevance check)
        mapped = self.mapper.extract_facts(resolved_query, chunks)

        # Stage 4: Filter
        relevant = [m for m in mapped if m["relevant"]]
        self._log(f"Filter: {len(relevant)}/{len(mapped)} chunks relevant")

        if not relevant:
            answer = "No relevant information found in your files."
            self.history.append((query, answer))
            self.history = self.history[-self.history_window:]
            return answer

        # Stage 5: Reduce
        answer = self.reducer.synthesize(resolved_query, relevant, context=context)

        # Stage 6: Confidence check
        answer = self.reducer.confidence_check(answer, relevant)

        self._log(f"Pipeline completed in {time.time() - t0:.1f}s")
        self.history.append((query, answer))
        self.history = self.history[-self.history_window:]
        return answer
```

**Step 4: Run all pipeline tests**

Run: `cd /Users/ded/Projects/assist/manole && uv run pytest tests/test_pipeline.py -v`
Expected: ALL PASS

Note: Existing tests that don't provide a `rewrite()` method on `FakeModelManager` will fail. Fix by adding a default rewrite method to `FakeModelManager` in `tests/test_pipeline.py`:

```python
class FakeModelManager:
    def __init__(self, plan_responses=None, extract_responses=None, synth_responses=None):
        # ... existing __init__ ...
        pass

    def rewrite(self, system, user):
        """Default: pass-through rewrite (no expansion)."""
        # Extract query from user message (may contain context prefix)
        query = user.split("Question: ")[-1] if "Question: " in user else user
        return json.dumps({
            "intent": "factual",
            "search_query": query,
            "resolved_query": query,
        })

    # ... rest unchanged ...
```

**Step 5: Run full test suite**

Run: `cd /Users/ded/Projects/assist/manole && uv run pytest -v`
Expected: ALL PASS

**Step 6: Commit**

```bash
git add pipeline.py tests/test_pipeline.py
git commit -m "feat: integrate QueryRewriter as Stage 0 in pipeline"
```

---

### Task 5: Update conversation memory tests

**Files:**
- Modify: `tests/test_conversation_memory.py`

**Step 1: Read current conversation memory tests**

Read `tests/test_conversation_memory.py` to understand what needs updating.

**Step 2: Add rewrite method to FakeModelManager**

The `FakeModelManager` in `tests/test_conversation_memory.py` needs the same default `rewrite()` method as in Task 4. Add it.

**Step 3: Run tests**

Run: `cd /Users/ded/Projects/assist/manole && uv run pytest tests/test_conversation_memory.py -v`
Expected: ALL PASS

**Step 4: Run full test suite**

Run: `cd /Users/ded/Projects/assist/manole && uv run pytest -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add tests/test_conversation_memory.py
git commit -m "test: update conversation memory tests for rewriter integration"
```

---

### Task 6: Final verification

**Step 1: Run full test suite**

Run: `cd /Users/ded/Projects/assist/manole && uv run pytest -v`
Expected: ALL PASS

**Step 2: Verify no import errors**

Run: `cd /Users/ded/Projects/assist/manole && uv run python -c "from pipeline import AgenticRAG; print('OK')"`
Expected: `OK`

**Step 3: Verify rewriter module standalone**

Run: `cd /Users/ded/Projects/assist/manole && uv run python -c "from rewriter import QueryRewriter, REWRITER_SYSTEM; print('OK')"`
Expected: `OK`
