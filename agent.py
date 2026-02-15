"""Agent loop orchestrator — model decides what tools to call at each step."""
import json
import re
from parser import parse_json
from searcher import extract_keywords

TOOL_SCHEMAS = [
    {
        "name": "semantic_search",
        "description": "Search inside file contents for information",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "Search query"}},
            "required": ["query"],
        },
    },
    {
        "name": "count_files",
        "description": "Count files, optionally filtered by extension",
        "parameters": {
            "type": "object",
            "properties": {"extension": {"type": "string", "description": "File extension filter, e.g. pdf"}},
        },
    },
    {
        "name": "list_files",
        "description": "List recent files, optionally filtered by extension",
        "parameters": {
            "type": "object",
            "properties": {
                "extension": {"type": "string", "description": "File extension filter"},
                "limit": {"type": "integer", "description": "Max files to return"},
            },
        },
    },
    {
        "name": "grep_files",
        "description": "Find files by name pattern",
        "parameters": {
            "type": "object",
            "properties": {"pattern": {"type": "string", "description": "Filename pattern to search for"}},
            "required": ["pattern"],
        },
    },
    {
        "name": "file_metadata",
        "description": "Get file size and dates",
        "parameters": {
            "type": "object",
            "properties": {"name_hint": {"type": "string", "description": "Filename or partial name"}},
            "required": ["name_hint"],
        },
    },
    {
        "name": "directory_tree",
        "description": "Show folder structure",
        "parameters": {
            "type": "object",
            "properties": {"max_depth": {"type": "integer", "description": "Max depth to show"}},
        },
    },
]

SYSTEM_PROMPT = (
    "You are a file assistant. Answer questions using ONLY tool results.\n"
    "NEVER answer from general knowledge. Always call a tool first.\n"
    "List of tools: " + json.dumps(TOOL_SCHEMAS)
)


class Agent:
    """Orchestrator agent loop with tool calling."""

    MAX_STEPS = 5

    def __init__(self, model, tool_registry, router, rewriter=None, debug=False):
        self.model = model
        self.tools = tool_registry
        self.router = router
        self.rewriter = rewriter
        self.debug = debug

    def run(self, query: str, history: list[dict] = None) -> str:
        """Run the agent loop for a user query."""
        # Rewrite query for better intent detection and search
        rewrite = None
        if self.rewriter:
            context = ""
            if history:
                context = "\n".join(
                    f"  {'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}"
                    for m in history[-4:]
                )
                if context:
                    context = f"Recent conversation:\n{context}"
            rewrite = self.rewriter.rewrite(query, context=context)

        # Use resolved query for the model, search_query for semantic search
        effective_query = rewrite["resolved_query"] if rewrite else query

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        if history:
            messages.extend(history[-4:])

        messages.append({"role": "user", "content": effective_query})

        for step in range(self.MAX_STEPS):
            if self.debug:
                print(f"  [AGENT] Step {step + 1}/{self.MAX_STEPS}")

            raw = self.model.generate(messages)

            if self.debug:
                print(f"  [AGENT] Response: {raw[:200]}")

            tool_call = self._parse_tool_call(raw)

            if tool_call is None:
                if step == 0:
                    # First step, no tool call — use fallback router
                    intent = rewrite["intent"] if rewrite else None
                    search_query = rewrite["search_query"] if rewrite else query
                    tool_name, params = self.router.route(query, intent=intent)
                    # Use expanded search_query for semantic search
                    if tool_name == "semantic_search":
                        params["query"] = search_query
                    if self.debug:
                        print(f"  [AGENT] Fallback router: {tool_name}({params})")
                    result = self.tools.execute(tool_name, params)
                    messages.append({"role": "assistant", "content": raw})
                    messages.append({
                        "role": "tool",
                        "content": json.dumps({"tool": tool_name, "result": result}),
                    })
                    continue
                else:
                    followup = self._needs_followup(query, messages)
                    if followup:
                        tool_name = followup["name"]
                        tool_params = followup["params"]
                        if self.debug:
                            print(f"  [AGENT] Followup: {tool_name}({tool_params})")
                        result = self.tools.execute(tool_name, tool_params)
                        if self.debug:
                            print(f"  [AGENT] Followup result: {result[:200]}")
                        messages.append({"role": "assistant", "content": raw})
                        messages.append({
                            "role": "tool",
                            "content": json.dumps({"tool": tool_name, "result": result}),
                        })
                        continue
                    if self.debug:
                        print("  [AGENT] Direct response (no tool call)")
                    return raw

            tool_name = tool_call["name"]
            tool_params = tool_call.get("params", {})

            if self.debug:
                print(f"  [AGENT] Tool: {tool_name}({tool_params})")

            if tool_name == "respond":
                return tool_params.get("answer", raw)

            result = self.tools.execute(tool_name, tool_params)

            if self.debug:
                print(f"  [AGENT] Result: {result[:200]}")

            messages.append({"role": "assistant", "content": raw})
            messages.append({
                "role": "tool",
                "content": json.dumps({"tool": tool_name, "result": result}),
            })

        # Max steps reached — force synthesis
        if self.debug:
            print("  [AGENT] Max steps reached, forcing response")

        messages.append({
            "role": "user",
            "content": "Give a concise final answer based on the information above.",
        })
        return self.model.generate(messages)

    _KNOWN_TOOLS = frozenset({
        "semantic_search", "count_files", "list_files", "grep_files",
        "file_metadata", "directory_tree", "respond",
    })

    # Words too generic to trigger a followup grep/search
    _FOLLOWUP_STOPWORDS = frozenset({
        "many", "much", "some", "any", "all", "most", "few", "more", "less",
        "have", "has", "had", "get", "got", "find", "show", "list", "give",
        "what", "which", "where", "when", "how", "why", "who",
        "there", "here", "this", "that", "these", "those",
        "file", "files", "folder", "folders", "directory", "directories", "structure",
        "count", "number", "total", "size",
        "can", "could", "would", "should", "will", "might",
        "about", "just", "only", "also", "even", "still",
        "aren't", "isn't", "don't", "doesn't", "didn't", "won't",
        "final", "question", "answer", "help", "please", "thanks",
        "test", "magic", "stuff", "thing", "things",
    })

    def _needs_followup(self, query: str, messages: list[dict]) -> dict | None:
        """Check if query keywords are covered by tool results. Return next tool call if not."""
        keywords = extract_keywords(query)
        # Filter out generic words that wouldn't make good grep/search targets
        keywords = [kw for kw in keywords if kw not in self._FOLLOWUP_STOPWORDS]
        if not keywords:
            return None

        result_text = ""
        tools_used = set()
        for msg in messages:
            if msg["role"] == "tool":
                result_text += msg["content"].lower() + " "
                try:
                    parsed = json.loads(msg["content"])
                    if isinstance(parsed, dict) and "tool" in parsed:
                        tools_used.add(parsed["tool"])
                except (json.JSONDecodeError, TypeError):
                    pass
            elif msg["role"] == "assistant":
                result_text += msg["content"].lower() + " "

        def _covered(kw: str, text: str) -> bool:
            """Check if keyword is covered in text, with basic stem matching."""
            if kw in text:
                return True
            # Strip trailing 's' for basic plural handling
            stem = kw.rstrip("s")
            if len(stem) >= 3 and stem in text:
                return True
            return False

        missing = [kw for kw in keywords if not _covered(kw, result_text)]
        if not missing:
            return None

        if "grep_files" not in tools_used:
            return {"name": "grep_files", "params": {"pattern": missing[0]}}

        if "semantic_search" not in tools_used:
            return {"name": "semantic_search", "params": {"query": " ".join(missing)}}

        return None

    def _parse_tool_call(self, response: str) -> dict | None:
        """Parse tool call from model output.

        Tries:
        1. LFM2.5 native format: <|tool_call_start|>[fn(args)]<|tool_call_end|>
        2. JSON format: {"name": "fn", "params": {...}}
        3. Bracket-wrapped: [tool_name(params)]
        4. Known tool call anywhere in response: tool_name(params)
        """
        # Try LFM2.5 native format (brackets inside special tokens)
        tc_match = re.search(
            r'<\|tool_call_start\|>\[?(.*?)\]?<\|tool_call_end\|>',
            response,
            re.DOTALL,
        )
        if tc_match:
            result = self._parse_native_tool_call(tc_match.group(1))
            if result:
                return result

        # Try JSON format
        parsed = parse_json(response)
        if parsed and "name" in parsed:
            return {
                "name": parsed["name"],
                "params": parsed.get("params", parsed.get("parameters", {})),
            }

        # Try bracket-wrapped format: [tool_name(params)]
        bracket_match = re.search(r'\[(\w+\(.*?\))\]', response, re.DOTALL)
        if bracket_match:
            result = self._parse_native_tool_call(bracket_match.group(1))
            if result:
                return result

        # Try bare function call anywhere in response
        bare_match = re.search(
            r'(' + '|'.join(self._KNOWN_TOOLS) + r')\(([^)]*)\)',
            response,
        )
        if bare_match:
            call_str = bare_match.group(0)
            result = self._parse_native_tool_call(call_str)
            if result:
                return result

        return None

    @staticmethod
    def _parse_native_tool_call(raw: str) -> dict | None:
        """Parse LFM2.5's Pythonic function call format.

        Example: semantic_search(query="invoices", top_k=5)
        """
        match = re.match(r'(\w+)\((.*)\)', raw.strip(), re.DOTALL)
        if not match:
            return None

        name = match.group(1)
        params_str = match.group(2)

        params = {}
        for param_match in re.finditer(
            r'(\w+)\s*=\s*(".*?"|\'.*?\'|\d+|None|True|False)',
            params_str,
        ):
            key = param_match.group(1)
            value = param_match.group(2).strip("\"'")
            if value == "None":
                value = None
            elif value.isdigit():
                value = int(value)
            elif value in ("True", "False"):
                value = value == "True"
            params[key] = value

        return {"name": name, "params": params}
