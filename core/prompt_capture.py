"""
core/prompt_capture.py — Optional full-prompt capture for every agent LLM call.

Not part of AgentState. A module-level dict, keyed by session_id, stores the
literal system and user messages sent to each agent's LLM during a /chat run.
Auditor merges each attempt's captures into retry_history; api.py surfaces
them top-level when mode is "full".

Design mirrors core/session_context.py — module-level dict + explicit public
accessors, so ARCHITECTURE.md's "no new AgentState fields" rule is respected.

Config levels (config.yaml → tracing.capture_prompts):
  minimal           — capture() is a no-op.
  full              — captures persist; only the latest (agent, call_name)
                      snapshot per query is exposed via get_latest_prompts().
  full_with_retries — every attempt's captures exposed via
                      get_prompts_for_attempt(attempt).

Lifecycle: api.py calls clear_captures(session_id) at the start of every /chat.
"""

from __future__ import annotations

import logging
from typing import Literal

from core.llm_config import get_config

logger = logging.getLogger(__name__)


Mode = Literal["minimal", "full", "full_with_retries"]


# session_id -> attempt_number -> list of capture dicts
# capture dict shape: {"agent": str, "call": str, "system": str, "user": str}
_captures: dict[str, dict[int, list[dict]]] = {}


def get_mode() -> Mode:
    """Read the current capture mode from config.yaml."""
    cfg = get_config()
    mode = cfg.get("tracing", {}).get("capture_prompts", "minimal")
    if mode not in ("minimal", "full", "full_with_retries"):
        logger.warning("Unknown tracing.capture_prompts value %r — defaulting to minimal", mode)
        return "minimal"
    return mode  # type: ignore[return-value]


def is_enabled() -> bool:
    """True when any capture level other than minimal is configured."""
    return get_mode() != "minimal"


def capture(
    session_id: str,
    attempt: int,
    agent: str,
    call: str,
    system: str,
    user: str,
) -> None:
    """Store a prompt snapshot. No-op when mode is minimal.

    attempt: 1-based retry index at the moment the LLM call is made.
             Chat agent passes 0 since its calls sit outside the retry loop.
    call:    identifies which LLM call inside the agent (e.g. "reasoning",
             "formatter", "exhaust", "main"). Lets the Chat agent's three
             distinct calls stay separable.
    """
    if not is_enabled():
        return

    if not session_id:
        return

    entry = {
        "agent":  agent,
        "call":   call,
        "system": system,
        "user":   user,
    }
    by_attempt = _captures.setdefault(session_id, {})
    by_attempt.setdefault(attempt, []).append(entry)
    logger.debug(
        "[%s] captured prompt — agent=%s call=%s attempt=%d system_len=%d user_len=%d",
        session_id, agent, call, attempt, len(system), len(user),
    )


def get_prompts_for_attempt(session_id: str, attempt: int) -> list[dict]:
    """Return a copy of captures recorded for this (session, attempt), in order.

    Empty list if capture was disabled or the attempt has no captures.
    Used by the Auditor to attach prompts to each retry_history entry.
    """
    return [dict(e) for e in _captures.get(session_id, {}).get(attempt, [])]


def get_latest_prompts(session_id: str) -> list[dict]:
    """Return one capture per (agent, call_name) — the last snapshot wins.

    Used by api.py when mode is "full" to expose a single top-level
    agent_prompts list rather than per-attempt detail.
    """
    by_attempt = _captures.get(session_id, {})
    # Walk in attempt order so later attempts overwrite earlier ones.
    latest: dict[tuple[str, str], dict] = {}
    for attempt in sorted(by_attempt.keys()):
        for entry in by_attempt[attempt]:
            latest[(entry["agent"], entry["call"])] = dict(entry)
    return list(latest.values())


def clear_captures(session_id: str) -> None:
    """Drop all captures for a session. No-op if absent."""
    _captures.pop(session_id, None)


# ---------------------------------------------------------------------------
# Isolated test — run with: python -m core.prompt_capture
# ---------------------------------------------------------------------------

def test_prompt_capture():
    from unittest.mock import patch

    sid = "test-pc-001"

    # ── Test 1: minimal mode is a no-op ────────────────────────────
    with patch(f"{__name__}.get_mode", return_value="minimal"):
        clear_captures(sid)
        capture(sid, 1, "planner", "main", "sys", "usr")
        assert get_prompts_for_attempt(sid, 1) == []
        assert get_latest_prompts(sid) == []
    print("PASS: minimal mode does not record captures")

    # ── Test 2: full mode records; per-attempt retrieval works ─────
    with patch(f"{__name__}.get_mode", return_value="full"):
        clear_captures(sid)
        capture(sid, 1, "planner",     "main", "p-sys-1", "p-usr-1")
        capture(sid, 1, "synthesizer", "main", "s-sys-1", "s-usr-1")
        capture(sid, 1, "auditor",     "main", "a-sys-1", "a-usr-1")

        attempt_1 = get_prompts_for_attempt(sid, 1)
        assert len(attempt_1) == 3
        agents = [e["agent"] for e in attempt_1]
        assert agents == ["planner", "synthesizer", "auditor"]
        assert attempt_1[0] == {"agent": "planner", "call": "main",
                                "system": "p-sys-1", "user": "p-usr-1"}
    print("PASS: captures stored and returned in insertion order per attempt")

    # ── Test 3: later attempts append, don't overwrite ─────────────
    with patch(f"{__name__}.get_mode", return_value="full_with_retries"):
        capture(sid, 2, "planner", "main", "p-sys-2", "p-usr-2")
        assert len(get_prompts_for_attempt(sid, 1)) == 3
        assert len(get_prompts_for_attempt(sid, 2)) == 1
    print("PASS: distinct attempts stored independently")

    # ── Test 4: get_latest_prompts last-wins by (agent, call) ──────
    with patch(f"{__name__}.get_mode", return_value="full"):
        latest = get_latest_prompts(sid)
        by_agent = {e["agent"]: e for e in latest}
        # Planner appears in attempts 1 and 2; attempt 2 should win.
        assert by_agent["planner"]["system"] == "p-sys-2"
        # Synthesizer only in attempt 1; that's what we get.
        assert by_agent["synthesizer"]["system"] == "s-sys-1"
    print("PASS: get_latest_prompts returns last-wins view keyed by (agent, call)")

    # ── Test 5: clear drops everything for a session ───────────────
    clear_captures(sid)
    assert get_prompts_for_attempt(sid, 1) == []
    assert get_prompts_for_attempt(sid, 2) == []
    assert get_latest_prompts(sid) == []
    print("PASS: clear_captures removes all session state")

    # ── Test 6: callers cannot mutate the store by side-effect ─────
    with patch(f"{__name__}.get_mode", return_value="full"):
        capture(sid, 1, "planner", "main", "sys", "usr")
        entry = get_prompts_for_attempt(sid, 1)[0]
        entry["system"] = "MUTATED"
        fresh = get_prompts_for_attempt(sid, 1)[0]
        assert fresh["system"] == "sys"
    clear_captures(sid)
    print("PASS: returned lists are copies; no side-effect mutation")

    # ── Test 7: empty session_id is ignored ────────────────────────
    with patch(f"{__name__}.get_mode", return_value="full"):
        capture("", 1, "planner", "main", "sys", "usr")
        assert get_prompts_for_attempt("", 1) == []
    print("PASS: empty session_id capture is ignored")

    # ── Test 8: unknown mode falls back to minimal ─────────────────
    with patch(f"{__name__}.get_config", return_value={"tracing": {"capture_prompts": "weird"}}):
        assert get_mode() == "minimal"
        assert is_enabled() is False
    print("PASS: unknown mode string falls back to minimal")

    print("\nPASS: all prompt_capture tests passed")


if __name__ == "__main__":
    test_prompt_capture()
