"""
core/parse.py — Shared LLM output parsing helper.

All agents that parse structured JSON from LLM responses use parse_llm_json.
This centralises markdown-fence stripping so no agent duplicates the logic.
"""

from __future__ import annotations

import json
import re


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
