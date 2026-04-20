"""
core/session_context.py — Per-session user-fact store.

Not part of AgentState. The Chat agent's reasoning call extracts stable
user-identifying facts (name, department, role, team, ID) on every turn and
merges them here, keyed by session_id. Subsequent reasoning calls read this
store to resolve self-references like "my salary" without re-asking.

Design mirrors core/manifest.py — a module-level dict with explicit
public accessors. No AgentState field is added, so ARCHITECTURE.md's state
schema rules are respected.

Lifecycle: entries persist for the lifetime of the process. Callers may
invoke clear_session_context(session_id) when a session ends (e.g. the
frontend's "New Session" button, or administrative cleanup).
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


_contexts: dict[str, dict] = {}


def get_session_context(session_id: str) -> dict:
    """Return the session's context dict, or an empty dict if none exists.

    Returns a copy so callers can't mutate the store by side-effect.
    """
    return dict(_contexts.get(session_id, {}))


def update_session_context(session_id: str, updates: dict) -> None:
    """Shallow-merge `updates` into the session's context dict.

    Keys in `updates` overwrite any existing values — this is how
    corrections propagate ("actually I'm Noa, not Dan" replaces user_name).
    Empty or None values are merged in unchanged; callers decide whether to
    clear a key explicitly.
    """
    if not updates:
        return
    current = _contexts.setdefault(session_id, {})
    current.update(updates)
    logger.debug("[%s] session context updated: %s", session_id, updates)


def clear_session_context(session_id: str) -> None:
    """Remove all stored facts for the given session. No-op if absent."""
    _contexts.pop(session_id, None)


# ---------------------------------------------------------------------------
# Isolated test — run with: python -m core.session_context
# ---------------------------------------------------------------------------

def test_session_context():
    sid = "test-sid-001"
    clear_session_context(sid)

    # 1. Empty on first read
    assert get_session_context(sid) == {}

    # 2. Update adds keys
    update_session_context(sid, {"user_name": "Dan", "department": "Engineering"})
    ctx = get_session_context(sid)
    assert ctx == {"user_name": "Dan", "department": "Engineering"}

    # 3. Update overwrites on key collision (correction path)
    update_session_context(sid, {"user_name": "Noa"})
    ctx = get_session_context(sid)
    assert ctx["user_name"] == "Noa"
    assert ctx["department"] == "Engineering"

    # 4. get_session_context returns a copy — mutating it must not affect the store
    ctx["user_name"] = "Mutated"
    assert get_session_context(sid)["user_name"] == "Noa"

    # 5. Empty updates are a no-op
    update_session_context(sid, {})
    assert get_session_context(sid)["user_name"] == "Noa"

    # 6. Sessions are isolated
    update_session_context("other-sid", {"user_name": "Other"})
    assert get_session_context(sid)["user_name"] == "Noa"
    assert get_session_context("other-sid")["user_name"] == "Other"

    # 7. Clear removes the entry
    clear_session_context(sid)
    assert get_session_context(sid) == {}
    # Clear on absent session is a no-op
    clear_session_context("never-existed")

    clear_session_context("other-sid")
    print("PASS: all session_context tests passed")


if __name__ == "__main__":
    test_session_context()
