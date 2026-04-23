"""
core/scope_result.py — Per-session record of the Chat agent's scope call.

Not part of AgentState. A module-level dict, keyed by session_id, stores the
most recent scope verdict + short reply produced by the Chat agent's scope
filter call. api.py reads it at response time to surface `scope_result` in
the /chat trace.

Design mirrors core/session_context.py — a module-level dict with explicit
public accessors, so ARCHITECTURE.md's "no new AgentState fields" rule is
respected.

Semantics:
  - Cleared at the start of every /chat request by api.py.
  - Written exactly once per /chat run, by the Chat agent, immediately
    after the scope LLM call parses.
  - Stays unset (→ get returns None) when the fast-path greeting regex
    short-circuits before the scope call fires, or on any path that does
    not make the scope call (retry exhaustion, format+deliver).

Value shape:
  {
    "scope":    "in_scope" | "out_of_scope",
    "response": "" | "short friendly reply when out_of_scope"
  }
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)


# Per-session scope call result. Absent key → scope call did not fire this turn.
_results: dict[str, dict] = {}


def get_scope_result(session_id: str) -> Optional[dict]:
    """Return the stored scope result for the session, or None if unset.

    Returns a copy so callers cannot mutate the store by side effect.
    """
    value = _results.get(session_id)
    if value is None:
        return None
    return dict(value)


def set_scope_result(session_id: str, scope: str, response: str) -> None:
    """Record the scope call outcome for the session.

    Overwrites any previous value. `scope` must be "in_scope" or
    "out_of_scope"; callers outside `agents/chat.py` should not be writing
    here.
    """
    if scope not in ("in_scope", "out_of_scope"):
        logger.warning(
            "[%s] set_scope_result received invalid scope %r — coercing to in_scope",
            session_id, scope,
        )
        scope = "in_scope"
    _results[session_id] = {"scope": scope, "response": response or ""}
    logger.debug(
        "[%s] scope_result recorded: scope=%s has_response=%s",
        session_id, scope, bool(response),
    )


def clear_scope_result(session_id: str) -> None:
    """Drop the stored scope result for the session. No-op if absent.

    Called by api.py at the start of every /chat request so the next run
    starts from a clean slate — a fast-path greeting on the next request
    must not carry forward the previous request's scope verdict.
    """
    _results.pop(session_id, None)


# ---------------------------------------------------------------------------
# Isolated test — run with: python -m core.scope_result
# ---------------------------------------------------------------------------

def test_scope_result():
    sid = "test-sr-001"
    clear_scope_result(sid)

    # 1. Unset session returns None
    assert get_scope_result(sid) is None

    # 2. set_scope_result stores the value
    set_scope_result(sid, "in_scope", "")
    assert get_scope_result(sid) == {"scope": "in_scope", "response": ""}

    # 3. Overwrite replaces prior value
    set_scope_result(sid, "out_of_scope", "I only answer questions about the data.")
    assert get_scope_result(sid) == {
        "scope":    "out_of_scope",
        "response": "I only answer questions about the data.",
    }

    # 4. get returns a copy — mutating it must not affect the store
    snapshot = get_scope_result(sid)
    snapshot["scope"] = "MUTATED"
    assert get_scope_result(sid)["scope"] == "out_of_scope"

    # 5. Sessions are isolated
    set_scope_result("other-sid", "in_scope", "")
    assert get_scope_result(sid)["scope"] == "out_of_scope"
    assert get_scope_result("other-sid")["scope"] == "in_scope"

    # 6. clear removes the entry; re-read is None
    clear_scope_result(sid)
    assert get_scope_result(sid) is None
    # Clearing an absent session is a no-op
    clear_scope_result("never-existed")

    # 7. Invalid scope coerces to in_scope and logs a warning
    set_scope_result(sid, "garbage", "should not happen")
    result = get_scope_result(sid)
    assert result["scope"]    == "in_scope"
    assert result["response"] == "should not happen"

    # 8. None response is normalized to empty string
    set_scope_result(sid, "in_scope", None)  # type: ignore[arg-type]
    assert get_scope_result(sid)["response"] == ""

    clear_scope_result(sid)
    clear_scope_result("other-sid")
    print("PASS: all scope_result tests passed")


if __name__ == "__main__":
    test_scope_result()
