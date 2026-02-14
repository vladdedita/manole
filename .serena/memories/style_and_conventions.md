# Style and Conventions

- Python 3.13, type hints used sparingly
- Module-level constants for prompt templates (PLANNER_PROMPT, MAP_PROMPT, etc.)
- Functions use snake_case, classes use PascalCase
- Docstrings on public functions/methods, brief one-liners
- No linter/formatter configured yet
- Tests use pytest with mock objects (FakeLLM, FakeSearcher, FakeSearchResult)
- Imports from chat module use sys.path insertion in tests
