"""
core/parse.py — Shared LLM output parsing helper.

All agents that parse structured JSON from LLM responses use parse_llm_json.
This centralises markdown-fence stripping so no agent duplicates the logic.
"""

from __future__ import annotations

import json
import logging
import re

logger = logging.getLogger(__name__)


def parse_llm_json(raw: str) -> dict:
    """
    Strip optional markdown code fences then parse the string as JSON.

    Handles both:
      - Raw JSON:       {"key": "value"}
      - Fenced JSON:    ```json\\n{"key": "value"}\\n```
                        ```\\n{"key": "value"}\\n```

    Returns the parsed dict.
    Raises ValueError (with the raw string included) if parsing still fails
    after stripping.
    """
    text = raw.strip()
    # Remove opening fence: ```json or ```
    text = re.sub(r"^```(?:json)?\s*\n?", "", text)
    # Remove closing fence
    text = re.sub(r"\n?```\s*$", "", text.rstrip())

    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Failed to parse LLM output: {exc}\nRaw output: {raw}"
        ) from exc


# ---------------------------------------------------------------------------
# Isolated test — run with: python -m core.parse
# ---------------------------------------------------------------------------

def test_parse_llm_json():
    # ── Test 1: plain JSON string parses correctly ────────────────
    result = parse_llm_json('{"key": "value", "num": 42}')
    assert result == {"key": "value", "num": 42}
    print("PASS: plain JSON parses correctly")

    # ── Test 2: JSON wrapped in triple-backtick fences ────────────
    result = parse_llm_json('```\n{"key": "value"}\n```')
    assert result == {"key": "value"}
    print("PASS: triple-backtick fenced JSON parses correctly")

    # ── Test 3: JSON wrapped in triple-backtick json fences ───────
    result = parse_llm_json('```json\n{"key": "value"}\n```')
    assert result == {"key": "value"}
    print("PASS: triple-backtick json fenced JSON parses correctly")

    # ── Test 4: malformed JSON raises ValueError with raw string ──
    raw_bad = '{"key": value_without_quotes}'
    try:
        parse_llm_json(raw_bad)
        assert False, "Should have raised ValueError"
    except ValueError as exc:
        assert raw_bad in str(exc), \
            f"Raw string must appear in the error message, got: {exc}"
    print("PASS: malformed JSON raises ValueError containing the raw string")

    # ── Test 5: empty string raises ValueError ────────────────────
    try:
        parse_llm_json("")
        assert False, "Should have raised ValueError"
    except ValueError:
        pass
    print("PASS: empty string raises ValueError")

    print("\nPASS: all parse_llm_json tests passed")


if __name__ == "__main__":
    test_parse_llm_json()
