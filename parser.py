"""JSON parsing with multi-level regex fallback for small LLM output."""
import json
import re


def parse_json(text: str) -> dict | None:
    """Parse JSON from LLM output with fallback regex extraction.

    Tries three strategies:
    1. Direct JSON parse
    2. Extract first {...} block from surrounding text
    3. Regex extraction of 'relevant' and 'facts' fields
    """
    # Try direct parse
    try:
        return json.loads(text.strip())
    except (json.JSONDecodeError, ValueError):
        pass

    # Try to find a JSON object in the text (DOTALL handles newlines,
    # .*? is non-greedy but allows nested brackets/arrays)
    match = re.search(r'\{.*\}', text, re.DOTALL)
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
