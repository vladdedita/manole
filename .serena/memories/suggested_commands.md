# Suggested Commands

## Testing
```bash
uv run pytest tests/ -v              # Run all tests
uv run pytest tests/ -k "keyword" -v # Run specific tests
```

## Running the App
```bash
uv run python chat.py /path/to/dir       # Index directory and start chat
uv run python chat.py --reuse index_name  # Reuse existing index
uv run python chat.py --force /path       # Force rebuild index
```

## Dependencies
```bash
uv sync --extra dev    # Install with dev dependencies (pytest)
uv sync                # Install production dependencies only
```

## Utility Commands (macOS/Darwin)
```bash
git status    # Check git state
git log --oneline -10  # Recent commits
```
