"""JSON parsing with multi-level regex fallback for small LLM output."""
import json
import re


def parse_json(text: str, debug: bool = False) -> dict | None:
    """Parse JSON from LLM output with fallback regex extraction.

    Tries three strategies:
    1. Direct JSON parse
    2. Extract first {...} block from surrounding text
    3. Regex extraction of 'relevant' and 'facts' fields
    """
    # Try direct parse
    try:
        result = json.loads(text.strip())
        if debug:
            print("  [PARSER] Strategy: direct JSON parse")
        return result
    except (json.JSONDecodeError, ValueError):
        pass

    # Try each { as a potential JSON start, parse greedily from there
    for i, ch in enumerate(text):
        if ch == '{':
            # Find the last } after this position
            last_brace = text.rfind('}', i)
            while last_brace > i:
                try:
                    result = json.loads(text[i:last_brace + 1])
                    if debug:
                        print(f"  [PARSER] Strategy: brace extraction at pos {i}")
                    return result
                except (json.JSONDecodeError, ValueError):
                    # Shrink: try the next } inward
                    last_brace = text.rfind('}', i, last_brace)
            break  # only try the first { — that's where the real JSON is

    # Fallback: extract "relevant" field via regex
    rel_match = re.search(r'"relevant"\s*:\s*(true|false)', text, re.IGNORECASE)
    if rel_match:
        relevant = rel_match.group(1).lower() == "true"
        facts_match = re.search(r'"facts"\s*:\s*\[([^\]]*)\]', text)
        facts = []
        if facts_match:
            facts = [f.strip().strip('"') for f in facts_match.group(1).split(",") if f.strip()]
        if debug:
            print(f"  [PARSER] Strategy: regex fallback (relevant={relevant}, {len(facts)} facts)")
        return {"relevant": relevant, "facts": facts}

    if debug:
        print(f"  [PARSER] All strategies failed for: {text[:80]!r}")
    return None
