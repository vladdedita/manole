"""Agent loop orchestrator — model decides what tools to call at each step."""
import re
from parser import parse_json

SYSTEM_PROMPT = (
    "You are a personal file assistant. You help users find information in their local files.\n\n"
    "You have access to tools to search file contents and inspect the filesystem.\n\n"
    "Rules:\n"
    "- Call semantic_search when the user asks about information INSIDE files\n"
    "- Call filesystem tools (count_files, list_files, grep_files, directory_tree) "
    "for questions ABOUT files themselves\n"
    "- You can call multiple tools if needed to get a complete answer\n"
    "- If a search returns no results, try a different query or tool before giving up\n"
    "- Keep answers concise and grounded in what the tools return\n"
    "- NEVER make up information that wasn't in tool results"
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
                    messages.append({"role": "tool", "name": tool_name, "content": result})
                    continue
                else:
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
            messages.append({"role": "tool", "name": tool_name, "content": result})

        # Max steps reached — force synthesis
        if self.debug:
            print("  [AGENT] Max steps reached, forcing response")

        messages.append({
            "role": "user",
            "content": "Give a concise final answer based on the information above.",
        })
        return self.model.generate(messages)

    def _parse_tool_call(self, response: str) -> dict | None:
        """Parse tool call from model output.

        Tries:
        1. LFM2.5 native format: <|tool_call_start|>fn(args)<|tool_call_end|>
        2. JSON format: {"name": "fn", "params": {...}}
        """
        # Try LFM2.5 native format
        tc_match = re.search(
            r'<\|tool_call_start\|>(.*?)<\|tool_call_end\|>',
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
