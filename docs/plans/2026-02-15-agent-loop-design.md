# Agent Loop Migration Design

## Problem

The current unidirectional pipeline (Rewrite → Plan → Search → Map → Filter → Reduce) has three structural problems:

1. **Rigid routing** — Every query walks the same path. "How many PDFs?" goes through vector search + 5 LLM map calls when it should be one `os.listdir()`.
2. **Single-shot search** — If the first search misses results, the pipeline gives up or does one blind retry. It can't reason about what's missing and search again differently.
3. **No adaptation** — The model can't say "I found 4 invoices but the filesystem shows 10 invoice files — let me search for the rest." The pipeline doesn't allow multi-step reasoning.

## Solution

Replace the fixed pipeline with an orchestrator agent loop. The model decides what to do at each step by calling tools or responding directly.

**Model:** LFM2.5-1.2B-Instruct (single model, GGUF Q4_0, 696MB). Drop the 350M-Extract entirely.

**Approach:** Agent loop with Python fallback router. Try native tool calling first; if parsing fails, a keyword-based Python router handles step 1 routing. This gives us the architectural benefits regardless of tool calling reliability.

## Architecture

```
User Query
    │
    ▼
┌──────────────────────────────────────────────────┐
│              AGENT LOOP (max 5 steps)             │
│                                                   │
│  Step 1: Route query to a tool                    │
│    ├── Try: model native tool calling             │
│    └── Fallback: Python heuristic router          │
│                                                   │
│  Step 2: Execute tool, append result to messages  │
│                                                   │
│  Step 3: Model sees results + generates           │
│    ├── Another tool call → loop back to Step 2    │
│    ├── Direct text → return as answer             │
│    └── respond() tool → return answer param       │
│                                                   │
│  Max steps reached → force synthesis from context │
└──────────────────────────────────────────────────┘
```

## Components

### models.py — Single Model

Drop the dual-model setup. One model, one `generate(messages)` method.

```python
class ModelManager:
    DEFAULT_MODEL_PATH = "models/LFM2.5-1.2B-Instruct-Q4_0.gguf"

    def __init__(self, model_path=None, n_threads=4):
        self.model_path = model_path or self.DEFAULT_MODEL_PATH
        self.n_threads = n_threads
        self.model = None

    def load(self):
        from llama_cpp import Llama
        self.model = Llama(
            model_path=self.model_path,
            n_ctx=8192,
            n_threads=self.n_threads,
            verbose=False,
        )

    def generate(self, messages: list[dict], max_tokens=1024) -> str:
        self.model.reset()
        response = self.model.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.1,
            top_k=50, top_p=0.1, repeat_penalty=1.05,
        )
        return response["choices"][0]["message"]["content"]
```

Key changes:
- No `extract_model`, no 350M
- `n_ctx=8192` (up from 4096) for multi-turn agent conversation
- Single `generate(messages)` replaces `plan()`, `rewrite()`, `map_chunk()`, `extract()`, `synthesize()`
- Callers build their own message lists

### tools.py — Tool Definitions + Registry

Seven tools available to the agent:

| Tool | Purpose |
|------|---------|
| `semantic_search(query, top_k)` | Search file contents by meaning, returns extracted facts |
| `count_files(extension)` | Count files by extension |
| `list_files(extension, limit)` | List files sorted by modification date |
| `file_metadata(name_hint)` | Get file size, dates for matching files |
| `grep_files(pattern)` | Find files by name pattern |
| `directory_tree(max_depth)` | Show folder structure |
| `respond(answer)` | Return final answer to user |

`ToolRegistry` maps tool names to execution functions, delegating to `searcher` and `toolbox`:

```python
class ToolRegistry:
    def __init__(self, searcher, toolbox):
        self.searcher = searcher
        self.toolbox = toolbox
        self._handlers = {
            "semantic_search": self._semantic_search,
            "count_files": self._count_files,
            "list_files": self._list_files,
            "file_metadata": self._file_metadata,
            "grep_files": self._grep_files,
            "directory_tree": self._directory_tree,
        }

    def execute(self, tool_name: str, params: dict) -> str:
        handler = self._handlers.get(tool_name)
        if not handler:
            return f"Unknown tool: {tool_name}"
        return handler(params)
```

### router.py — Python Fallback Router

Safety net for when native tool calling fails. Keyword heuristics route the query:

```python
def route(query: str) -> tuple[str, dict]:
    q = query.lower()
    if any(k in q for k in ["how many", "count"]):
        ext = _detect_extension(q)
        return "count_files", {"extension": ext}
    if any(k in q for k in ["file types", "folder", "tree", "directory", "structure"]):
        return "directory_tree", {"max_depth": 2}
    if any(k in q for k in ["list files", "recent files", "what files", "show files"]):
        ext = _detect_extension(q)
        return "list_files", {"extension": ext}
    if any(k in q for k in ["file size", "when was", "modified", "created"]):
        return "file_metadata", {"name_hint": _extract_name_hint(q)}
    return "semantic_search", {"query": query}
```

Used on step 0 when model doesn't produce a parseable tool call.

### searcher.py — Search with Internal Map-Filter

The searcher becomes self-contained: search → score-filter → extract facts → format. The agent never sees raw chunks.

```python
class Searcher:
    def __init__(self, leann_searcher, model):
        self.leann = leann_searcher
        self.model = model

    def search_and_extract(self, query: str, top_k: int = 5) -> str:
        chunks = self.leann.search(query, top_k=top_k)
        if not chunks:
            return "No matching content found."

        # Score pre-filter (carried over from current pipeline)
        if len(chunks) > 1:
            threshold = chunks[0].score * 0.8
            chunks = [c for c in chunks if c.score >= threshold]

        # Map: extract facts per chunk using the same 1.2B model
        facts_by_source = {}
        for chunk in chunks:
            extracted = self._extract_facts(query, chunk)
            if extracted["relevant"] and extracted["facts"]:
                source = self._get_source(chunk)
                facts_by_source.setdefault(source, []).extend(extracted["facts"])

        if not facts_by_source:
            return "Search returned results but none were relevant to the query."

        # Format for agent context
        lines = []
        for source, facts in facts_by_source.items():
            lines.append(f"From {source}:")
            for fact in facts:
                lines.append(f"  - {fact}")
        return "\n".join(lines)
```

Key changes:
- Absorbs current `mapper.py` logic into `_extract_facts()`
- Score pre-filter (0.8 threshold) carries over
- `_get_source()` uses `file_name` metadata (as recently fixed)
- Returns formatted text string, not objects
- Relevance defaults to `False` on parse failure (not `True`)

### agent.py — The Agent Loop

```python
SYSTEM_PROMPT = """You are a personal file assistant. You help users find information in their local files.

You have access to tools to search file contents and inspect the filesystem.

Rules:
- Call semantic_search when the user asks about information INSIDE files
- Call filesystem tools for questions ABOUT files themselves
- You can call multiple tools if needed to get a complete answer
- If a search returns no results, try a different query or tool before giving up
- Keep answers concise and grounded in what the tools return
- NEVER make up information that wasn't in tool results"""


class Agent:
    MAX_STEPS = 5

    def __init__(self, model, tool_registry, router, debug=False):
        self.model = model
        self.tools = tool_registry
        self.router = router
        self.debug = debug

    def run(self, query: str, history: list[dict] = None) -> str:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        if history:
            messages.extend(history[-4:])  # last 2 turns
        messages.append({"role": "user", "content": query})

        for step in range(self.MAX_STEPS):
            raw = self.model.generate(messages)
            tool_call = self._parse_tool_call(raw)

            if tool_call is None:
                if step == 0:
                    # First step, no tool call — use fallback router
                    tool_name, params = self.router.route(query)
                    result = self.tools.execute(tool_name, params)
                    messages.append({"role": "assistant", "content": raw})
                    messages.append({"role": "tool", "name": tool_name, "content": result})
                    continue
                else:
                    # Later steps — model is answering from context
                    return raw

            if tool_call["name"] == "respond":
                return tool_call["params"].get("answer", raw)

            result = self.tools.execute(tool_call["name"], tool_call["params"])
            messages.append({"role": "assistant", "content": raw})
            messages.append({"role": "tool", "name": tool_call["name"], "content": result})

        # Max steps — force synthesis
        messages.append({
            "role": "user",
            "content": "Give a concise final answer based on the information above."
        })
        return self.model.generate(messages)
```

Key behaviors:
- Step 0 fallback: if model doesn't produce a tool call on the first step, Python router kicks in
- Later steps: no tool call = model is answering directly from accumulated context
- `respond()` tool: explicit "I'm done" signal, returns the answer parameter
- Max steps: force a synthesis turn from everything gathered
- Conversation memory: last 2 turns (4 messages) prepended to message list

### toolbox.py — Filesystem Tools

Mostly unchanged. Remove the `execute(plan)` method (no longer needed — `ToolRegistry` calls individual methods directly). Keep all filesystem methods as-is.

### chat.py — Thin CLI Shell

Mostly unchanged. Swap `AgenticRAG` for `Agent`. Conversation history moves from `rag.history` to a local list of message dicts.

### parser.py — JSON Extraction

Kept as-is. Still needed for parsing tool call JSON and fact extraction JSON from model output.

## Module Migration

| Current Module | New Module | Change |
|---------------|------------|--------|
| `pipeline.py` | `agent.py` | Replaced by agent loop |
| `planner.py` | `router.py` | Replaced by fallback router + native tool calling |
| `rewriter.py` | *(deleted)* | Model sees conversation history directly |
| `mapper.py` | `searcher.py` | Absorbed into `_extract_facts()` |
| `reducer.py` | *(deleted)* | Model synthesizes answers in the agent loop |
| `models.py` | `models.py` | Simplified to single model |
| `searcher.py` | `searcher.py` | Gains internal map-filter |
| `toolbox.py` | `toolbox.py` | Remove `execute()`, keep individual methods |
| `parser.py` | `parser.py` | Unchanged |
| `chat.py` | `chat.py` | Swap pipeline for agent |
| *(new)* | `tools.py` | Tool definitions + registry |
| *(new)* | `router.py` | Python fallback router |

## Query Flow Examples

### "how many PDF files?"
```
Step 1: Model calls count_files(extension="pdf")
        → "Found 12 .pdf files."
Step 2: Model responds: "You have 12 PDF files."
Total: 2 LLM calls, 0 search calls
```

### "target revenue for engineering department?"
```
Step 1: Model calls semantic_search(query="target revenue engineering department")
        → Searcher: search → score-filter → extract facts → format
        → "From budget_q1_2026.txt:
             - Total Budget: $450,000
             - Revenue targets: Project Alpha $180,000, Project Beta $95,000, Consulting services $45,000"
Step 2: Model responds with answer from facts
Total: 2 agent LLM calls + N map calls inside search
```

### "list all invoices and their totals" (multi-step)
```
Step 1: Model calls semantic_search(query="invoice total amount")
        → Returns facts for 4 invoices
Step 2: Model calls grep_files(pattern="invoice")
        → Finds 10 invoice files
Step 3: Model sees gap (4/10), calls semantic_search(query="invoice amount due")
        → Returns facts for 3 more invoices
Step 4: Model responds with combined list
Total: 4 agent calls + search map calls
```

### "aren't there more than 2?" (follow-up)
```
History includes previous Q&A about invoices.
Step 1: Model sees context, calls semantic_search(query="invoices")
Step 2: Model responds with updated list
Total: 2 agent calls
```

## Risks and Mitigations

| Risk | Mitigation |
|------|-----------|
| 1.2B model can't produce tool calls | Python fallback router handles step 1 routing |
| Fact extraction still flaky | Same MAP_SYSTEM prompt, score pre-filter. If extraction fails, return raw chunk text |
| Agent loops forever | MAX_STEPS=5 hard limit, forced synthesis on exit |
| Context window overflow | n_ctx=8192, history capped at 4 messages, facts capped per chunk |
| Model ignores tool results and hallucinates | System prompt: "NEVER make up information that wasn't in tool results" |

## Success Criteria

1. Filesystem queries ("how many PDFs?") answered in 2 LLM calls with no search
2. Semantic queries ("target revenue?") answered correctly using extracted facts
3. Multi-step queries ("list all invoices and totals") can call multiple tools
4. Follow-up questions resolved from conversation history
5. All tests pass with mocked model
