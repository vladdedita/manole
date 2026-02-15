# Style and Conventions

- Python 3.13, type hints used sparingly
- Module-level constants for prompt templates (PLANNER_PROMPT, MAP_PROMPT, etc.)
- Functions use snake_case, classes use PascalCase
- Docstrings on public functions/methods, brief one-liners
- No linter/formatter configured yet
- Tests use pytest with mock objects (FakeLLM, FakeSearcher, FakeSearchResult)
- Imports from chat module use sys.path insertion in tests

## LLM Integration — LFM2.5

The project uses LiquidAI LFM2.5-1.2B-Instruct via llama-cpp-python (GGUF).
Always follow the official LFM2.5 documentation and model card for:
- **Tool calling format**: Use `"List of tools: [...]"` with JSON schemas in system prompt to activate native tool-calling mode
- **Tool call output**: Model outputs `<|tool_call_start|>[fn(params)]<|tool_call_end|>` (Pythonic calls with brackets inside special tokens)
- **Tool results**: Use `role: "tool"` with `json.dumps()` content — the chat template renders `<|im_start|>tool` which the model was trained on
- **Generation params**: temperature=0.1, top_k=50, top_p=0.1, repetition_penalty=1.05

Reference: https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct and https://docs.liquid.ai/lfm/key-concepts/tool-use

Do NOT invent ad-hoc prompt formats or few-shot workarounds — use the model's native protocol.
